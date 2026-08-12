# Active-FL on `fed-infra`, with Temporal-managed worker fleets

**Date:** 2026-08-12
**Status:** Approved design, pending implementation plan
**Repos affected:** `active-fed` (primary), `fed-twin` (retrofit), `fed-infra` (new)

---

## 1. Context

`active-fed` and `fed-twin` are two personal projects that solve different problems on
overlapping infrastructure.

| | fed-twin | active-fed |
|---|---|---|
| FL transport | Flower gRPC, server on Master pod | MinIO object bus, no FL server |
| Algorithm | REINFORCE, `PolicyNet` 4→32→16→2 | PPO + GAE, `ActorCritic` 4→64→64→(2,1) |
| Env variation | md5(`twin_id`) → ±15% physics | seeded wrapper, ±20% physics |
| Metrics path | regex-scraped from `kubectl logs` → CSV | logged directly to MLflow |
| Layout | flat modules copied to image root | real `src` package, hatchling, mypy, ~1070 lines of tests |
| Deploy modes | single-cluster **+ Karmada multi-cluster** | single-cluster only |
| Baselines | fed vs single-twin pipelines | fedavg vs active weight/data modes |

They share **almost no code**. What they share is an *infrastructure pattern*: kind →
Kubeflow Pipelines (plus a long ARM patch sequence) → MLflow → MinIO, and in fed-twin's
case Karmada on top.

Three problems motivate this work:

1. **active-fed has no multi-cluster mode.** fed-twin has a working Karmada setup;
   active-fed cannot demonstrate geographically distributed federated learning.
2. **The bootstrap is unstable, and duplicated.** Both repos independently patch four
   KFP images plus MinIO and argoexec to survive on ARM
   (`fed-twin/setup/install_single_cluster_local.sh:75-104`,
   `active-fed/setup/install_local.sh:34-46`). Fixes made in one repo do not reach the other.
3. **Worker orchestration is opaque and brittle.** In `active-fed`, N workers run inside a
   single PyTorchJob that the `train_workers` component busy-polls with a hard 20-minute
   ceiling and no retry (`src/pipelines/active_fl_pipeline.py:169-193`). A failed worker
   is invisible; a slow round is fatal.

---

## 2. Decisions

Each decision below was made explicitly during design. Rejected alternatives are recorded
because the reasoning matters more than the outcome.

### D1 — Shared `fed-infra` repo, consumed as a git submodule by **both** repos simultaneously

Chosen over vendoring into each repo, and over converting active-fed first.

Simultaneous conversion is riskier — one bad `fed-infra` commit can block both projects —
but it forces the abstraction to be genuinely repo-agnostic from day one, because two
consumers with materially different needs (Flower vs PPO, Temporal vs no Temporal)
exercise it immediately. Vendoring was rejected because the duplication is what created
problem 2 above.

### D2 — Deployment topologies: single-cluster and multi-cluster only

No "visual" pipeline variants, and no dedicated single-agent K8s baseline pipeline.
`weight_mode` × `active_data_mode` remain runtime parameters (4 combinations), so the
matrix is 4 combinations × 2 topologies = 8 runs, driven by config rather than by
additional compiled pipelines.

fed-twin needs visual pipelines because its functional pipeline collapses all rounds into
a single KFP component. active-fed's pipeline already renders a visible
`train → aggregate → evaluate` chain per round, so the visual variant would add far less
than it does for fed-twin.

### D3 — KFP drives; Temporal executes the worker fleet beneath one node

This reverses an earlier decision in which Temporal owned the round loop and submitted KFP
runs. The final direction is:

- **KFP** sequences rounds and owns the DAG plus artifact lineage. All three components
  execute real work.
- **Temporal** owns the worker fleet *within* a round. The `train_workers` component starts
  a Temporal workflow and blocks on it, streaming per-worker status into the node's logs.

The scopes are disjoint, so there is no two-schedulers conflict: KFP decides what runs
next across rounds, Temporal decides how a fleet of pods is launched, retried, and
observed within a round.

Rejected alternatives:

- *Temporal owns the round loop, submits one KFP run per round* — gives finer durability
  but produces N tiny 3-node DAGs and no whole-experiment graph.
- *Temporal owns everything, KFP dropped* — cleanest ownership and best durability, but
  loses the KFP UI and artifact lineage, which are a hard requirement.
- *KFP owns everything, Temporal only watches pods via the k8s API* — makes Temporal a
  monitoring system, which is not what it is for, and leaves it unable to act per-pod.

**Accepted cost:** run-level durability. If the KFP run itself dies, it is resubmitted with
`start_round` recovered from MinIO (see §6). Work lost is bounded to one round.

### D4 — Temporal server runs in-cluster via Helm

Deployed by `fed-infra` as an optional component alongside MLflow and MinIO, with
PostgreSQL persistence. One environment, one teardown, and the multi-cluster case needs no
special handling because the host cluster already hosts every shared service.

### D5 — Observability uses built-in UIs; no custom dashboard

Temporal Web (per-worker workflow state, failure reasons, live heartbeat progress),
KFP UI (round DAG, artifacts), MLflow UI (ML metrics), Kubernetes Dashboard (pod phase,
restarts, events, logs), and Karmada Dashboard in multi-cluster mode. All are deployed by
`fed-infra` as components.

Grafana + kube-state-metrics is explicitly deferred: it adds historical resource metrics,
which is not the current need. A custom single-pane dashboard is rejected for now as code
we would own and maintain in exchange for convenience only.

---

## 3. Architecture

### 3.1 Repository topology

```
fed-infra/                        # new repo — contains no consumer-specific strings
  bin/fed-infra-up                # entrypoint: reads consumer's infra.env
  bin/fed-infra-down
  lib/common.sh                  # log, die, retry, require_cmd, wait_rollout
  lib/kind.sh                    # kind_ensure_cluster, kind_load_image (digest-checked)
  lib/kfp.sh                     # kfp_install, kfp_patch_arm
  lib/minio.sh                   # minio_install, minio_ensure_bucket
  lib/mlflow.sh                  # mlflow_install
  lib/temporal.sh                # temporal_install (helm + postgres)
  lib/karmada.sh                 # karmada_init, karmada_join (incl. IP + secret rewrite)
  lib/dashboard.sh               # k8s_dashboard_install, karmada_dashboard_install
  lib/nodeport.sh                # expose_nodeport
  manifests/*.yaml.tpl           # envsubst templates; namespace, creds, ports injected
  kind/{single,multi-host,member}.yaml.tpl

active-fed/  vendor/fed-infra @ pinned SHA  +  infra.env
fed-twin/    vendor/fed-infra @ pinned SHA  +  infra.env
```

**Invariant:** `fed-infra` must never contain the literal strings `active-fed` or
`fed-twin`. Identity is declared by the consumer. This is enforced by a CI grep.

**Consumer contract** (`infra.env`):

```sh
FED_CLUSTER_NAME=active-fed
FED_NAMESPACE=active-fed
FED_PROFILE=single                # single | multi
FED_MEMBER_COUNT=3                # multi profile only
FED_COMPONENTS=kfp,minio,mlflow,temporal,k8s-dashboard
FED_IMAGES="active-fed-worker:v1 active-fed-aggregator:v1 active-fed-temporal-worker:v1"
FED_KFP_VERSION=2.4.0
FED_NODEPORT_KFP=30080
FED_NODEPORT_MLFLOW=30500
FED_NODEPORT_MINIO_API=30900
FED_NODEPORT_MINIO_CONSOLE=30901
FED_NODEPORT_TEMPORAL_UI=30733
```

fed-twin's `infra.env` sets `FED_COMPONENTS=kfp,minio,mlflow,karmada` — Temporal is simply a
component it does not enable. This componentisation is what makes one shared repo viable
for two consumers with different needs.

### 3.2 Control plane layering

```
KFP run  (one per experiment, rounds unrolled from start_round)
 └── round r
      ├── train_workers            KFP component ─┐
      │      starts TrainRoundWorkflow            │ blocks, streams status
      │        └── WorkerWorkflow × N  (Temporal) │  ← per-pod retry / failure / progress
      │              └── launch_and_watch_pod     │
      ├── score_and_aggregate      KFP component  │
      └── evaluate_global          KFP component ─┘
```

`train_workers` becomes a thin Temporal client:

```python
@component(base_image="active-fed-aggregator:v1")   # + temporalio
def train_workers(fl_round: int, num_workers: int, ..., worker_report: Output[Artifact]):
    client = await Client.connect(TEMPORAL_ADDR)
    handle = await client.start_workflow(
        TrainRoundWorkflow, spec,
        id=f"train-{kfp_run_id}-r{fl_round}",     # deterministic → retry re-attaches
        task_queue="active-fed",
    )
    while not done:
        print(render(await handle.query("status")))   # per-pod state → KFP node logs
    report = await handle.result()
```

The deterministic workflow ID resolves a real bug: today `job_name` embeds
`uuid.uuid4()` (`active_fl_pipeline.py:57`), so a retried component would launch a *second*
PyTorchJob and race 2N workers on the same MinIO keys. Temporal's workflow-ID reuse policy
makes retry idempotent by construction.

### 3.3 Temporal workflows (`src/orchestration/`)

```python
@workflow.defn
class TrainRoundWorkflow:
    @workflow.run
    async def run(self, spec: RoundSpec) -> RoundReport:
        results = await asyncio.gather(*[
            workflow.execute_child_workflow(
                WorkerWorkflow, WorkerSpec(spec, worker_id=i),
                id=f"{workflow.info().workflow_id}-w{i}")
            for i in range(spec.num_workers)
        ], return_exceptions=True)
        return RoundReport.from_results(results)      # tolerates partial fleet failure

    @workflow.query
    def status(self) -> dict[int, WorkerStatus]: ...

@workflow.defn
class WorkerWorkflow:                                  # one durable entity per pod
    @workflow.run
    async def run(self, spec: WorkerSpec) -> WorkerResult:
        return await workflow.execute_activity(
            launch_and_watch_pod, spec,
            heartbeat_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

    @workflow.query
    def status(self) -> WorkerStatus: ...
```

Activities: `launch_and_watch_pod` (creates the Job, watches pod phase, heartbeats
progress), `cleanup_worker_job`.

**Progress signal.** The activity heartbeats observed progress, in priority order:
(1) worker pod phase transitions from the k8s watch, and (2) appearance of
`round_N/workers/worker_i_metrics.json` in MinIO. A stalled worker therefore trips its
heartbeat timeout rather than hanging until a wall-clock deadline. Heartbeat details are
visible in Temporal Web as pending-activity state — this is the per-pod progress view.

**Where the Temporal worker process runs.** A `Deployment` in the consumer namespace
(`active-fed/k8s/temporal-worker.yaml`), built from the aggregator image, polling task
queue `active-fed`. Its ServiceAccount needs `create/delete/get/list/watch` on `jobs`,
`pytorchjobs`, `pods`, and `pods/log` — an extension of the existing ClusterRole in
`active-fed/k8s/rbac.yaml`. This deployment is active-fed-specific and does **not** belong
in `fed-infra`.

### 3.4 Single-cluster flow

Aggregator, workers, and all shared services run in one kind cluster. `WorkerWorkflow`
launches one k8s `Job` per worker (replacing the single N-replica PyTorchJob), which gives
genuine per-pod identity, retry, and failure attribution.

`train_worker.py` requires **no changes**: it already reads
`int(os.environ.get("RANK", args.worker_id))` (`src/agent/train_worker.py:130`), so each
Job sets `RANK={i}`.

### 3.5 Multi-cluster flow (Karmada)

- **Host cluster:** KFP, MLflow, MinIO, Temporal + PostgreSQL, Karmada control plane, the
  Temporal worker Deployment.
- **Member cluster `member{i}`:** PPO worker pods only.
- **Dispatch:** `launch_and_watch_pod` applies a `Job` plus a `PropagationPolicy` pinned to
  `member{i}`, against the Karmada apiserver (kubeconfig mounted from a secret).
- **Data path:** workers reach the host's MinIO and MLflow over NodePort, the same
  mechanism fed-twin uses at `fed_twin_multi_cluster_pipeline.py:63`.
- **Completion:** detected by the appearance of
  `round_N/workers/worker_i_metrics.json` in MinIO.

This deletes fed-twin's entire member-kubeconfig-secret and cross-cluster log-following
machinery (`fed-twin/src/automate_run.py:126-254`). fed-twin needs it because Flower
clients only *print* their metrics — the log stream **is** its data channel. active-fed's
workers already write weights, deltas, and metrics to MinIO
(`src/agent/train_worker.py:83-115`), so object storage is the completion signal.

Because `physics_seed` derives from `worker_id`, one member cluster maps to exactly one
physical variation, which is the geo-distributed fleet narrative.

**Known limitation.** Pod-level watch in multi-cluster mode is weaker than in
single-cluster mode. Member-cluster pods are reachable from the host only through the
Karmada aggregated API server, which is slower and less reliable than a direct watch.
Mitigation: `launch_and_watch_pod` treats the Karmada aggregated API as best-effort
enrichment and uses MinIO progress as the authoritative liveness signal. Per-pod *failure
reason* may therefore be coarser in multi-cluster mode. This is accepted.

### 3.6 Observability

| Surface | Deployed by | Answers |
|---|---|---|
| KFP UI | fed-infra | Round DAG, node logs, artifact lineage |
| Temporal Web | fed-infra | Which worker failed and why; retry counts; live progress |
| MLflow UI | fed-infra | Reward curves, client scores, acceptance rate, active-data usage |
| Kubernetes Dashboard | fed-infra | Pod phase, restarts, events, exec |
| Karmada Dashboard | fed-infra (multi only) | Propagation state, member health |

**Cross-linking.** MLflow runs are tagged with `kfp_run_id`, `temporal_workflow_id`, and
`topology`. The Temporal workflow ID is printed into KFP node logs, and the KFP run ID is
carried in the workflow input. From any one surface, the other two are reachable.

---

## 4. Changes to `active-fed`

| Area | Change |
|---|---|
| `src/pipelines/active_fl_pipeline.py` | `train_workers` rewritten as a Temporal client; add `start_round` parameter; unroll `range(start_round, fl_rounds)`; `.set_retry()` on all tasks; add `init_global_model` at pipeline head |
| `src/orchestration/` | **New.** `workflows.py`, `activities.py`, `types.py`, `worker_main.py` |
| `k8s/temporal-worker.yaml` | **New.** Temporal worker Deployment + SA |
| `k8s/rbac.yaml` | Extend ClusterRole with `jobs`, `pods/log`, `create`/`delete` verbs |
| `k8s/{minio,mlflow-server}.yaml` | Deleted — provided by `fed-infra` templates, including the namespace they currently create |
| `src/pipelines/run_pipeline.py` | Computes `start_round` by scanning MinIO for the highest `round_N/global.pt`; submits per topology; stops swallowing post-processing failures |
| `setup/install_local.sh`, `teardown_local.sh` | Reduced to `fed-infra-up` / `fed-infra-down` wrappers |
| `setup/kind-cluster.yaml` | Deleted — templated by `fed-infra` |
| `infra.env` | **New.** Consumer contract |
| `config/k8s.yaml` | Add `topology: single\|multi`, `members: N`, `min_workers: N` (quorum for a round to proceed), `worker_launcher: temporal\|pytorchjob` (migration flag, see §9), `temporal.*` |
| `docker/Dockerfile.aggregator` | Add `temporalio` |
| `Makefile` | `local-setup` → profile-aware; add `multi-setup`, `temporal-ui` |
| `config/local.yaml`, `src/experiment/` | **Unchanged.** The local runner stays dependency-free |

## 5. Changes to `fed-twin`

Infrastructure only. No pipeline, Flower, or ML changes.

| Area | Change |
|---|---|
| `setup/install_single_cluster_local.sh` | Reduced to `fed-infra-up` wrapper with `FED_PROFILE=single` |
| `setup/install_multi_cluster_local.sh` | Reduced to `fed-infra-up` wrapper with `FED_PROFILE=multi` |
| `setup/teardown_*.sh` | Reduced to `fed-infra-down` wrappers |
| `setup/kind-*.yaml` | Deleted — templated by `fed-infra` |
| `k8s/mlflow-server.yaml` | Deleted — provided by `fed-infra` |
| `infra.env` | **New.** `FED_COMPONENTS=kfp,minio,mlflow,karmada` (no Temporal) |

---

## 6. Failure and recovery model

Recovery happens at the lowest layer that can see the failure. Escalating a pod crash to
the top is slow, loses context, and redoes work that did not fail.

| Layer | Mechanism | Owner | Covers |
|---|---|---|---|
| Pod | `restartPolicy: OnFailure`, `backoffLimit` | Training Operator / Job controller | Worker OOM or crash |
| Worker | `WorkerWorkflow` retry policy, heartbeat timeout | Temporal | Worker wedged or repeatedly failing |
| Component | `.set_retry()` on KFP tasks | KFP / Argo | Aggregator or eval pod dies |
| Run | Resubmit with `start_round` from MinIO | Operator / `run_pipeline.py` | Whole run lost |

**Idempotency underpins every layer.** A restarted worker re-fetches `round_N/global.pt`
and overwrites its own `worker_i_*` objects. A retried `score_and_aggregate` re-reads the
same round-N inputs and rewrites `round_{N+1}/global.pt` under fixed evaluation seeds.
A retried `train_workers` component re-attaches to the existing Temporal workflow by its
deterministic ID. All converge to the same state.

**Partial fleet failure is tolerated.** `TrainRoundWorkflow` gathers with
`return_exceptions=True`; if at least `min_workers` (config/k8s.yaml, default 2) succeed,
the round proceeds with the survivors and records the shortfall in `RoundReport`. Below
quorum the workflow fails and the KFP task retry takes over. This matches the existing
behaviour of `collect_worker_updates`, which already returns fewer clients than requested
rather than failing (`src/aggregator/collect.py:74-77`).

**Run-level resume.** `start_round` is computed by `run_pipeline.py` before submission: it
scans MinIO for the highest existing `round_N/global.pt` and unrolls the pipeline from
there. A resubmitted run therefore covers only the remaining rounds. This deliberately does
not rely on KFP caching, which is currently left at its default in
`src/pipelines/run_pipeline.py:141` and is too implicit to build resume on.

---

## 7. Bugs fixed as part of this work

1. **Round-0 divergence (correctness, silent).** In the K8s path, round 0 has no
   `round_0/global.pt`, so `_fetch_global_weights` returns `None`
   (`src/agent/train_worker.py:76-79`) and every worker keeps its *own independently
   random* `ActorCritic` — no seed is set in the worker entrypoint. Round 0 therefore
   averages N unrelated random networks. The local runner does not have this bug: it builds
   `global_weights` once and calls `set_weights` on every worker
   (`src/experiment/local_runner.py:148`). Local and K8s results are consequently not
   comparable at round 0 today. Fixed by an `init_global_model` step that writes a single
   seeded `round_0/global.pt` before round 0.

2. **Non-idempotent job naming.** `uuid.uuid4()` in `job_name`
   (`active_fl_pipeline.py:57`) means a retried component launches a second fleet.
   Resolved by the deterministic Temporal workflow ID.

3. **`restartPolicy: Never`** (`active_fl_pipeline.py:68,105`) — a crashed worker vanishes
   and the aggregator silently proceeds with N−1 clients. Changed to `OnFailure` with a
   `backoffLimit`.

4. **Hard 20-minute timeout, no retry** (`active_fl_pipeline.py:169-193`) — replaced by
   Temporal heartbeat-based liveness.

5. **Silently swallowed post-processing failures.** `subprocess.run(..., check=False)` in
   `src/pipelines/run_pipeline.py:163-193` discards fetch and plot errors. Changed to
   surface failures.

---

## 8. Testing

**`fed-infra`**
- `shellcheck` on all scripts.
- `bats` unit tests for each `lib/*.sh` function with `kubectl`/`helm`/`kind` stubbed on
  `PATH`, asserting emitted commands.
- `fed-infra-up --dry-run` renders every manifest without a cluster; CI diffs the output
  against golden files for both consumers' `infra.env`.
- CI grep asserting the repo contains no consumer-specific strings.
- Nightly kind smoke test for the `single` profile.

**`active-fed`**
- All existing tests (~1070 lines) stay green; they cover pure logic and are unaffected.
- New workflow tests via `temporalio.testing.WorkflowEnvironment` with time-skipping and
  mocked activities: full-fleet success, partial failure below/above `min_workers`,
  worker retry-then-succeed, heartbeat timeout. Milliseconds, no cluster.
- Activity tests against a faked Kubernetes client and a MinIO stub.
- `make compile-pipeline` in CI already guards pipeline compilation.

**`fed-twin`**
- `make single-cluster-setup` must complete green **before** any `fed-infra` SHA bump is
  allowed to land in either consumer. This is the gate that makes simultaneous conversion
  survivable.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| A bad `fed-infra` commit blocks both repos | Consumers pin a SHA; bumps gated on both smoke tests |
| Memory: kind + KFP + Karmada + Temporal + PostgreSQL | Cap multi-cluster at 2–3 members (`local.yaml` currently says `num_workers: 5`); document a resource budget; Temporal PostgreSQL sized down for local use |
| Temporal is a new failure domain | Temporal is optional per `FED_COMPONENTS`. During P1–P2 the pre-Temporal PyTorchJob path stays selectable via `worker_launcher: pytorchjob` in `config/k8s.yaml`, so a Temporal outage cannot block experiments. The flag and the dead path are removed at the end of P2 rather than kept indefinitely |
| Degraded pod-level detail in multi-cluster | Documented in §3.5; MinIO progress is the authoritative signal |
| One `Job` per worker is heavier than one N-replica PyTorchJob | Acceptable at this scale (≤5 workers); revisit if worker counts grow |

---

## 10. Out of scope

- Visual pipeline variants and a dedicated single-agent K8s baseline pipeline (D2).
- Grafana, kube-state-metrics, and any custom dashboard (D5).
- Changes to `active-fed`'s local runner, `config/local.yaml`, or any ML/algorithmic code.
- Changes to fed-twin's pipelines, Flower core, or analysis scripts.
- Publishing images to a registry; local kind loading only, as today.

---

## 11. Phasing

Both repos convert simultaneously (D1), but the work is ordered:

- **P0 — `fed-infra` foundation.** Create the repo; extract `common`, `kind`, `kfp`,
  `minio`, `mlflow`, `nodeport`; write both `infra.env` files; convert both single-cluster
  setup paths. *Gate: `make local-setup` (active-fed) and `make single-cluster-setup`
  (fed-twin) both green.*
- **P1 — Temporal, single-cluster.** `temporal` component in `fed-infra`; `src/orchestration/`;
  Temporal worker Deployment and RBAC; rewrite `train_workers` as a Temporal client.
  *Gate: a 2-round run completes with per-worker state visible in Temporal Web.*
- **P2 — Correctness and recovery.** All five fixes from §7, plus `start_round` resume.
  Ends by deleting the `worker_launcher` migration flag and the PyTorchJob path.
  *Gate: a killed worker pod recovers without operator intervention; round-0 weights are
  identical across workers; a run resubmitted after a mid-experiment kill resumes at the
  last completed round.*
- **P3 — Multi-cluster.** `karmada` component in `fed-infra` (used by both repos); Job +
  `PropagationPolicy` dispatch; MinIO completion detection. *Gate: a 2-round, 2-member run
  completes; fed-twin's multi-cluster path still works.*
- **P4 — Observability.** Kubernetes Dashboard and Karmada Dashboard components; MLflow
  cross-link tags. *Gate: from an MLflow run, both the KFP run and the Temporal workflow
  are reachable.*
