# Karmada Multi-Cluster Implementation Plan (Phase P3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `active-fed` a second deployment topology — aggregator and shared services on a host cluster, PPO workers propagated to member clusters via Karmada — and convert `fed-twin`'s existing multi-cluster bootstrap onto the shared `fed-infra` library.

**Architecture:** `fed-infra` gains a `karmada` component and a `multi` profile that creates one host cluster plus N member clusters. In `active-fed`, the Temporal activity from P1 gains a dispatch mode: instead of creating a `Job` in the local cluster, it applies a `Job` plus a `PropagationPolicy` to the Karmada apiserver, pinned to one member cluster. Completion is detected by watching MinIO for the worker's metrics object — not by following logs across clusters.

**Tech Stack:** Existing plus Karmada v1.17.0 (`karmadactl`), and the Karmada aggregated API for best-effort pod introspection.

## Scope re-verified against the post-P2 tree (2026-08-18)

Checked before execution, because this plan was written before P1 and P2 landed and P2's plan had already drifted:

- `launch_and_watch_pod`, `_ensure_job` and `collect_worker_updates` all exist as assumed.
- **`launch_and_watch_pod` now RAISES `WorkerJobFailed` on failure rather than returning `WorkerResult(succeeded=False)`.** This was P1's final fix — Temporal retries on exceptions only, so returning made `RetryPolicy(maximum_attempts=3)` dead. Task 4's MinIO-completion path **must preserve the raise**; converting it back to a return would silently disable retry again.
- `build_job_manifest` ships `backoffLimit: 0` with `restartPolicy: Never` so Temporal owns retry exclusively. **Do not change either.**
- `WorkerSpec`/`RoundSpec` do not yet carry `topology`/`member_cluster` — Task 3 adds them as planned.
- `fed-twin/setup/install_multi_cluster_local.sh` already sources `vendor/fed-infra` and calls `fed_mlflow_install` (a P0 final-review fix). Task 6 builds on that; do not revert it.
- `fed-infra` has no `karmada` component, no multi-cluster kind templates, and `fed_up` is single-profile only — all as this plan assumes.

**Environment caveat:** the colima VM currently reports ~5.8 GiB, not the 9.7 GiB assumed when this plan was written. Eight concurrent workers already OOM-killed one during P2's gate. Task 7 needs host + members; restore VM memory or cap `FED_MEMBER_COUNT` at 1 before attempting it. Tasks 1-6 need no cluster.

## Global Constraints

- **Prerequisite:** Phases P0–P2 complete and gated. Temporal owns the fleet; round-0 init and `start_round` resume are in place.
- **`fed-infra` MUST NOT contain the strings `active-fed` or `fed-twin`.** Enforced by `tests/agnostic.bats`.
- `lib/*.sh` are sourced and MUST NOT set shell options; `bin/*` and `tests/stubs/*` MUST. Dry-run guards use `if/then/return 0`, never `[ … ] && return 0`. Both enforced by tests.
- **Any new `FED_*` variable used inside a template MUST be added to `FED_TEMPLATE_VARS` in `lib/render.sh`** or it renders as an empty string with no error. This bit us once already.
- Test-harness facts: `STUB_*_FAIL_GLOB` can contain spaces but **cannot** use `|` alternation; use `*` to fail everything. `STUB_KUBECTL_FAIL_ONCE_GLOB`/`_COUNT` exist for fail-once-then-succeed. `tests/stubs/kind` captures stdin only for `--config -`.
- **Do not change** the local runner, `config/local.yaml`, or RL/ML algorithm code.
- Commit messages: Conventional Commits, no trailers. Work on `main`; no push.

## Resource warning — read before starting

Karmada on kind means **one cluster per member plus the host**. `config/local.yaml` currently says `num_workers: 5`; five members plus a host on a 4-CPU / 9.7 GiB colima VM will not fit alongside KFP, Temporal and PostgreSQL.

**Cap multi-cluster runs at 2 members.** Task 1 makes `FED_MEMBER_COUNT` explicit and Task 7's gate uses 2. If a run thrashes, reduce to 1 member before concluding anything is broken.

## Why `active-fed`'s multi-cluster is simpler than `fed-twin`'s

`fed-twin` must inject member kubeconfigs as a Kubernetes secret and follow logs across clusters (`src/automate_run.py:126-254`) because Flower clients only *print* their metrics — the log stream **is** its data channel.

`active-fed` has no such constraint: workers already push weights, deltas and metrics to MinIO (`src/agent/train_worker.py:83-115`) and log to MLflow directly. So completion is detected by polling MinIO for `round_N/workers/worker_i_metrics.json`, and the entire kubeconfig-secret mechanism is unnecessary. Do not port it.

**Accepted limitation:** per-pod detail is weaker in multi-cluster mode. Member pods are reachable from the host only through the Karmada aggregated API, which is slower and less reliable than a direct watch. Treat it as best-effort enrichment; MinIO progress is the authoritative liveness signal.

---

## File Structure

**`fed-infra` — new**

| Path | Responsibility |
|---|---|
| `lib/karmada.sh` | `fed_karmada_init`, `fed_karmada_join`, `fed_karmada_wait_cluster` |
| `kind/multi-host.yaml.tpl` | Host cluster: KFP/MLflow/MinIO/Temporal NodePorts |
| `kind/member.yaml.tpl` | Member cluster: no NodePorts, workers only |
| `tests/karmada.bats` | Unit tests with stubs |
| `tests/stubs/karmadactl` | New stub |
| `tests/fixtures/consumer-c.env` | A `multi` profile fixture |
| `tests/golden/consumer-c/` | Golden manifests for the multi profile |

**`fed-infra` — modified:** `lib/config.sh` (`FED_MEMBER_COUNT`, `FED_KARMADA_VERSION`, `FED_KARMADA_CONFIG`), `lib/components.sh` (profile branch), `lib/render.sh` (whitelist), both `bin/*`.

**`active-fed` — new/modified**

| Path | Change |
|---|---|
| `src/orchestration/dispatch.py` | New — `LocalJobDispatcher` and `KarmadaJobDispatcher` behind one interface |
| `src/orchestration/activities.py` | Select a dispatcher from `WorkerSpec.topology`; MinIO-based completion |
| `src/orchestration/types.py` | `WorkerSpec`/`RoundSpec` gain `topology` and `member_cluster` |
| `infra.env.multi` | New — the `multi` profile consumer contract |
| `config/k8s.yaml` | `topology: single\|multi`, `members: 2` |
| `Makefile` | `multi-setup`, `multi-teardown` |

**`fed-twin` — modified:** `setup/install_multi_cluster_local.sh` and `teardown_multi_cluster_local.sh` reduced to `fed-infra` wrappers; `setup/kind-multi-cluster-host.yaml` deleted; `infra.env.multi` added.

---

## Task 1: `fed-infra` — Karmada component

**Files:**
- Create: `fed-infra/lib/karmada.sh`, `tests/karmada.bats`, `tests/stubs/karmadactl`
- Modify: `lib/config.sh`, `lib/render.sh`

**Interfaces:**
- Produces: `fed_karmada_init(host_cluster)`, `fed_karmada_join(cluster_name, kube_context)`, `fed_karmada_wait_cluster(cluster_name)`. New config: `FED_MEMBER_COUNT` (default `2`), `FED_MEMBER_PREFIX` (default `member`), `FED_KARMADA_VERSION` (default `v1.17.0`), `FED_KARMADA_CONFIG` (default `${HOME}/.karmada/karmada-apiserver.config`).

**Port from `fed-twin`, with the fragile parts kept:** `fed-twin/setup/install_multi_cluster_local.sh:111-146` contains a `join_and_patch` function that joins a cluster then rewrites both the `Cluster` object's `apiEndpoint` and the kubeconfig stored in its secret, replacing `127.0.0.1`/`localhost` with the container's Docker-network IP. That rewrite is **essential** — without it the Karmada control plane cannot reach members from inside the host container. Port it faithfully; it is the single most failure-prone part of this phase.

- [x] **Step 1: Write `tests/stubs/karmadactl`**

Same shape as the other stubs: `set -euo pipefail`, log argv to `$STUB_LOG`, honour `STUB_KARMADACTL_FAIL_GLOB` with the `# shellcheck disable=SC2254` comment, emit `STUB_KARMADACTL_OUT`, exit 0.

- [x] **Step 2: Write `tests/karmada.bats`**

Cover: `fed_karmada_init` skips when the `karmada-system` namespace already exists; it passes `--karmada-data`/`--karmada-pki`/`--cert-external-ip`; `fed_karmada_join` skips an already-joined cluster; join patches both the `Cluster` apiEndpoint and the secret; all three functions are no-ops under `FED_DRY_RUN=1`; each fails fast when its underlying command fails.

- [x] **Step 3: Implement `lib/karmada.sh`**

Port from `fed-twin/setup/install_multi_cluster_local.sh:88-155`, converting to library conventions: dry-run guard first, `|| return 1` after every external command, idempotent existence probes, no `set -euo pipefail`. Keep the Python-based base64 secret rewrite — it is doing real work that `sed` on base64 cannot.

- [x] **Step 4: Add config defaults and whitelist entries**

`FED_MEMBER_COUNT`, `FED_MEMBER_PREFIX`, `FED_KARMADA_VERSION`, `FED_KARMADA_CONFIG` in `fed_config_defaults` + the export list. Add `${FED_MEMBER_COUNT}` and `${FED_MEMBER_PREFIX}` to `FED_TEMPLATE_VARS`.

- [x] **Step 5: Run tests and commit**

```bash
make check
git add lib/karmada.sh lib/config.sh lib/render.sh tests/
git commit -m "feat: karmada control plane component with member join and endpoint patching"
```

---

## Task 2: `fed-infra` — multi profile

**Files:**
- Create: `kind/multi-host.yaml.tpl`, `kind/member.yaml.tpl`, `tests/fixtures/consumer-c.env`, `tests/golden/consumer-c/`
- Modify: `lib/components.sh`, `bin/fed-infra-up`, `bin/fed-infra-down`, `tests/golden.bats`

**Interfaces:**
- Produces: `fed_up` branches on `FED_PROFILE`. For `multi`: create the host cluster, create `FED_MEMBER_COUNT` member clusters, install `karmada`, join every cluster, then install the remaining components **on the host only**. `fed_down` deletes host and all members.

- [x] **Step 1: Write the templates**

`multi-host.yaml.tpl` mirrors `single-cluster.yaml.tpl` including all NodePort mappings — the host runs every shared service. `member.yaml.tpl` is minimal: a control-plane node with **no** `extraPortMappings`, since members only run worker pods and reach the host over its NodePorts.

- [x] **Step 2: Branch `fed_up` on profile**

Keep the single-profile path exactly as it is. Add a `multi` branch that creates the host, loops `FED_MEMBER_COUNT` members via `fed_kind_ensure_cluster "${FED_MEMBER_PREFIX}${i}" member.yaml.tpl`, loads `FED_IMAGES` into **every** cluster (workers run on members), then switches context to the host before installing kfp/training/temporal/minio/mlflow.

Guard the whole branch under dry-run the same way as everything else, and add a `consumer-c` golden fixture so the multi profile's rendered manifests are diffed like the others.

- [x] **Step 3: Extend `fed_down`**

Delete members first, then the host. Deleting the host first orphans member clusters that still reference it.

- [x] **Step 4: Run tests and commit**

```bash
make check
git add kind/ lib/components.sh bin/ tests/
git commit -m "feat: multi-cluster profile with host and member kind clusters"
```

---

## Task 3: `active-fed` — dispatcher abstraction

**Files:**
- Create: `active-fed/src/orchestration/dispatch.py`, `tests/test_dispatch.py`
- Modify: `src/orchestration/types.py`

**Interfaces:**
- `WorkerSpec` and `RoundSpec` gain `topology: str = "single"` and `member_cluster: str = ""`.
- `dispatch.py` produces:

```python
class JobDispatcher(Protocol):
    def ensure_job(self, spec: WorkerSpec) -> str: ...      # returns job name; idempotent
    def delete_job(self, spec: WorkerSpec) -> None: ...

class LocalJobDispatcher:   # current behaviour, batch/v1 Job in the local cluster
class KarmadaJobDispatcher: # Job + PropagationPolicy applied to the Karmada apiserver

def dispatcher_for(spec: WorkerSpec) -> JobDispatcher: ...
def build_propagation_policy(spec: WorkerSpec) -> dict: ...   # pure, testable
```

- [x] **Step 1: Write the failing tests**

Test `build_propagation_policy` as a pure function: it targets the Job by the deterministic name from P1; `clusterAffinity.clusterNames` contains exactly `spec.member_cluster`; the policy name is deterministic and a valid Kubernetes name; `dispatcher_for` returns `LocalJobDispatcher` for `topology="single"` and `KarmadaJobDispatcher` for `"multi"`; and a `multi` spec with an empty `member_cluster` raises rather than silently propagating everywhere.

That last case matters: an empty `clusterNames` list in Karmada means *all* clusters, so a missing member name would run every worker on every member.

- [x] **Step 2: Implement**

`LocalJobDispatcher` wraps the P1 code path unchanged. `KarmadaJobDispatcher` applies both the Job and the `PropagationPolicy` against the Karmada apiserver, using a kubeconfig path from `FED_KARMADA_CONFIG` mounted into the Temporal worker pod. Assign members round-robin: `member_cluster = f"{prefix}{worker_id % member_count + 1}"`, computed in `RoundSpec.worker_spec`.

Because `physics_seed` derives from `worker_id`, one member cluster maps to one physical variation — which is the geo-distributed fleet narrative the project is demonstrating.

- [x] **Step 3: Run tests and commit**

---

## Task 4: `active-fed` — MinIO-based completion

**Files:**
- Modify: `src/orchestration/activities.py`
- Create: `tests/test_completion_watch.py`

**Interfaces:**
- Produces: `wait_for_worker_artifact(minio_client, bucket, fl_round, worker_id, timeout_s, poll_s) -> bool`, used by `launch_and_watch_pod` when `topology == "multi"`.

**Why:** in multi-cluster the worker pod is in another cluster and the host cannot reliably watch it. But the worker writes `round_N/workers/worker_i_metrics.json` on success, so that object's appearance **is** the completion signal — and it is the same signal the aggregator already depends on (`src/aggregator/collect.py:47-56`).

- [x] **Step 1: Write the failing tests**

Fake MinIO client. Cover: returns True as soon as the object appears; returns False on timeout; heartbeats each poll; tolerates transient `S3Error` without aborting; and — importantly — does **not** treat the presence of `worker_i_weights.pt` alone as completion, since the worker writes weights before metrics and a partial upload must not be read as success.

- [x] **Step 2: Implement and wire into the activity**

In `launch_and_watch_pod`, branch on topology: `single` keeps the Job-status watch from P1; `multi` uses `wait_for_worker_artifact`, with the Karmada aggregated API consulted only for best-effort failure enrichment inside a `try`/`except` that never masks the real outcome.

- [x] **Step 3: Run tests and commit**

---

## Task 5: `active-fed` — multi consumer contract

**Files:**
- Create: `active-fed/infra.env.multi`
- Modify: `config/k8s.yaml`, `Makefile`, `k8s/temporal-worker.yaml`

- [x] **Step 1: Write `infra.env.multi`**

`FED_PROFILE=multi`, `FED_CLUSTER_NAME=active-fed-host`, `FED_MEMBER_COUNT=2`, `FED_MEMBER_PREFIX=active-fed-member`, `FED_COMPONENTS=kfp,training,temporal,minio,mlflow,karmada`, same S3/NodePort values as the single profile.

- [x] **Step 2: Mount the Karmada kubeconfig into the Temporal worker**

The worker pod runs on the host and must reach the Karmada apiserver. Add a Secret created from `${FED_KARMADA_CONFIG}` and mount it, setting `FED_KARMADA_CONFIG` in the container env to the mount path.

- [x] **Step 3: Add `config/k8s.yaml` keys and Makefile targets, then commit**

---

## Task 6: `fed-twin` — convert multi-cluster onto the library

**Files:**
- Modify: `fed-twin/setup/install_multi_cluster_local.sh`, `teardown_multi_cluster_local.sh`
- Create: `fed-twin/infra.env.multi`
- Delete: `fed-twin/setup/kind-multi-cluster-host.yaml`

**Behaviour must be preserved exactly.** Read the current script and compare every value: cluster names (`multi-cluster-host`, `multi-cluster-member{i}`), member count derived from `config/config.json`'s `num_workers`, the Karmada image pre-fetch, the inotify `sysctl` bumps on every node, the dashboard install and its NodePort 32000, and the admin token generation.

**Note P0 already touched this file.** The final P0 review found it referenced two deleted files and it was fixed to call `fed_mlflow_build_image`/`fed_mlflow_install`. Build on that; do not revert it.

The Karmada Dashboard and the admin-token step are `fed-twin`-specific presentation concerns — leave them in the consumer script rather than moving them into `fed-infra`, unless P4 decides otherwise.

- [x] **Steps:** write `infra.env.multi`; reduce the script to image builds + `fed-infra-up --env infra.env.multi` + the consumer-specific dashboard/token/propagation pieces; delete the superseded kind config; verify with a dry-run; confirm no `.py` changed; commit.

---

## Task 7: Phase gate — real multi-cluster run

**Cap at 2 members.** Expect this to be slow and memory-hungry; a full bring-up may take 45+ minutes.

- [ ] **Step 1: `active-fed` multi bring-up**

```bash
make multi-teardown || true
make multi-setup
kind get clusters          # host + 2 members
kubectl --kubeconfig ~/.karmada/karmada-apiserver.config get clusters
```

Expected: three kind clusters; all three joined and `Ready` in Karmada.

- [ ] **Step 2: Run a 2-round, 2-worker pipeline**

Confirm: `PropagationPolicy` objects exist on the Karmada apiserver; each member cluster runs exactly one worker pod (`kubectl --context kind-active-fed-member1 get pods`); the aggregator collects both workers' updates; MLflow records the round.

- [ ] **Step 3: Prove members are genuinely separate**

Confirm no worker pod ran on the host cluster: `kubectl --context kind-active-fed-host get pods -l app=active-fl-worker` returns nothing.

- [ ] **Step 4: Prove MinIO-based completion works without cross-cluster log access**

Confirm the aggregator succeeded without any kubeconfig secret existing — `kubectl get secret -n active-fed | grep -c karm` should be 0 apart from the one mounted into the Temporal worker.

- [ ] **Step 5: `fed-twin` multi-cluster still works**

`make multi-cluster-teardown && make multi-cluster-setup`, then `./run_pipeline.sh fed_twin_multi_cluster` reaches `Succeeded`.

- [ ] **Step 6: Confirm the gate**

P3 is complete when: both consumers' multi-cluster paths come up on `fed-infra`; `active-fed` runs a federated round with workers on separate member clusters; no cross-cluster log-following or kubeconfig-secret machinery exists in `active-fed`; `fed-twin`'s multi-cluster pipeline still succeeds; both single-cluster paths still pass their P0/P1 gates; `fed-infra`'s `make check` is green including the new `consumer-c` goldens.

---

## Out of scope for P3

- Dashboards and MLflow cross-link tags → **P4**
- Any change to the local runner, `config/local.yaml`, or RL/ML code.
- Multi-cluster *visual* pipelines — explicitly rejected during design.
