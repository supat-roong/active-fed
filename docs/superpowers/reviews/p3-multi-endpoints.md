# P3 — multi-cluster MinIO/MLflow endpoint fix

## The problem

Under `topology="multi"`, Karmada propagates worker Jobs onto **member** kind
clusters, while MinIO and MLflow run only on the **host** cluster. Every
`WorkerSpec` carries `minio_endpoint`/`mlflow_tracking_uri` as in-cluster DNS
names (`minio-service.active-fed.svc.cluster.local:9000`,
`http://mlflow-service.active-fed.svc.cluster.local:5000`). A member cluster is
a separate Kubernetes cluster with its own DNS, so those names never resolve
there — confirmed live on member1 as a `NameResolutionError`, surfacing only
after the worker had already started training (i.e. dispatch itself
"succeeded").

All kind clusters in this environment share one Docker bridge network, so a
member pod can reach the host's MinIO/MLflow **NodePorts** via the host's own
node IP (verified live: `curl http://172.18.0.2:30500/` and
`curl http://172.18.0.2:30900/minio/health/live` both 200). The fix rewrites
both endpoints to `<host-node-internal-ip>:<nodeport>` for `topology="multi"`
only, resolving the node IP dynamically (never from config — it moved from
`172.18.0.3` to `172.18.0.2` across a single VM restart in the live
environment) and reading the two NodePort numbers from config, threaded the
same way `topology`/`members`/`member_prefix` already are.

## Where the rewrite is applied, and why

`_rewrite_endpoints_for_multi(spec) -> WorkerSpec` and its helper
`_resolve_host_node_ip(core) -> str` live in `src/orchestration/activities.py`,
and are called as the first line of `_launch_and_watch_pod_multi`, before
`dispatcher_for(spec).ensure_job(spec)` (which is what eventually calls
`build_job_manifest(spec)`):

```python
spec = _rewrite_endpoints_for_multi(spec)
not_before = datetime.now(timezone.utc)
dispatcher = dispatcher_for(spec)
name = await dispatcher.ensure_job(spec)
```

Reasoning for this exact location, considered against the alternatives:

- **Not in `dispatch.py`.** `KarmadaJobDispatcher` only ever builds clients
  pointed at the **Karmada apiserver** (`_karmada_clients()`, via
  `FED_KARMADA_CONFIG`) — a distinct control plane from the host cluster the
  Temporal worker pod itself runs on. A node's IP is only meaningful read from
  the API of the cluster it belongs to, and the task specifically calls for
  reusing `activities._k8s_batch_and_core()` (the *local*-cluster client) —
  which `dispatch.py` deliberately never imports, to keep the "who talks to
  which cluster" boundary in one place per topology (see that module's own
  docstring on `_k8s_batch_and_core`). Putting the rewrite in `dispatch.py`
  would have meant building a second, redundant local-cluster client there,
  undermining that existing boundary.
- **Not in the KFP pipeline component (`train_workers` in
  `active_fl_pipeline.py`).** That component runs on the host cluster too, but
  it only ever *starts* a Temporal workflow and blocks on its result — it
  never builds a `WorkerSpec` or touches Kubernetes directly. Task said
  explicitly: dispatch time, inside the Temporal worker.
- **`_launch_and_watch_pod_multi`, not `launch_and_watch_pod` itself.** Every
  other multi-only concern (dispatcher selection, `not_before` capture, the
  Karmada-status fast-fail wiring) already lives in this function, gated once
  by `launch_and_watch_pod`'s `if spec.topology == "multi":` branch. Doing the
  rewrite here means it runs **exactly once per activity invocation** (i.e.
  once per worker-Job dispatch attempt — Temporal may invoke the activity
  again on retry, and each retry legitimately re-resolves the node IP, since
  it may have changed), and every downstream use of `spec` in this function —
  the dispatcher, `_minio_client_for` for polling, the Karmada diagnostics —
  automatically sees the rewritten endpoints. Because it runs before
  `dispatcher.ensure_job(spec)`, the rewritten `minio_endpoint`/
  `mlflow_tracking_uri` are exactly what `build_job_manifest(spec)` embeds
  into the worker Job's env — verified directly by
  `test_launch_and_watch_pod_multi_topology_dispatches_and_waits_for_artifact`,
  which asserts on the spec the fake dispatcher actually received.
- **`topology="single"` is untouched.** `_rewrite_endpoints_for_multi`/
  `_resolve_host_node_ip` are private functions only ever called from
  `_launch_and_watch_pod_multi`, which `launch_and_watch_pod`'s single-topology
  branch never reaches. `WorkerSpec`/`RoundSpec` frozen dataclasses gained two
  new fields (`minio_nodeport`, `mlflow_nodeport`, default `0`) but nothing
  reads them outside the multi rewrite.

`WorkerSpec` is a frozen dataclass (Temporal replays workflow arguments, so
immutability keeps replay deterministic); `_rewrite_endpoints_for_multi` uses
`dataclasses.replace` to build a new instance rather than mutate in place.

## Config → run_pipeline.py → pipeline dsl → RoundSpec threading

Followed the existing `member_prefix` pattern exactly:

- `config/k8s-multi.yaml`: added `orchestration.minio_nodeport: 30900` /
  `mlflow_nodeport: 30500` (not added to `config/k8s.yaml` — `topology="single"`
  never reads them, so there's no equivalent single-profile contract to keep
  in sync, and the task's "do not touch config/k8s.yaml" instruction was
  read as covering this).
- `src/pipelines/run_pipeline.py`: `DEFAULT_MINIO_NODEPORT = 30900` /
  `DEFAULT_MLFLOW_NODEPORT = 30500`, read via
  `orch.get("minio_nodeport", DEFAULT_MINIO_NODEPORT)`, added to the KFP run
  `arguments` dict.
- `src/pipelines/active_fl_pipeline.py`: `minio_nodeport: int = 30900` /
  `mlflow_nodeport: int = 30500` added as dsl parameters on both
  `active_fl_pipeline` and the `train_workers` component, threaded into
  `RoundSpec(...)`.
- `src/orchestration/types.py`: `RoundSpec`/`WorkerSpec` both gained
  `minio_nodeport: int = 0` / `mlflow_nodeport: int = 0`;
  `RoundSpec.worker_spec()` passes them through unchanged, exactly like
  `member_count`/`member_prefix`.

New agreement tests (mirroring `test_topology_and_members_default_layers_agree`,
`test_member_prefix_matches_the_multi_infra_contract`,
`test_multi_config_agrees_with_the_multi_infra_contract`), all in
`tests/test_active_fl_pipeline.py`:

- `test_compiled_pipeline_carries_nodeports` — both params actually reach the
  compiled pipeline IR (not silently decorative).
- `test_nodeport_default_layers_agree` — `config/k8s-multi.yaml` (layer 1) and
  the compiled dsl parameter default (layer 3) both equal
  `run_pipeline.DEFAULT_MINIO_NODEPORT`/`DEFAULT_MLFLOW_NODEPORT`.
- `test_nodeports_match_the_multi_infra_contract` — the `DEFAULT_*` constants
  equal `FED_NODEPORT_MINIO_API`/`FED_NODEPORT_MLFLOW` in `infra.env.multi`.
- `test_multi_config_nodeports_agree_with_the_multi_infra_contract` —
  `config/k8s-multi.yaml`'s values equal the same two `infra.env.multi`
  variables.

## Node-lookup failure behavior

`_resolve_host_node_ip` raises `RuntimeError`, naming exactly what could not be
resolved (node-listing call itself failing, zero nodes, no `Ready` node, or a
`Ready` node exposing no `InternalIP` address) — never falling back to the
original in-cluster DNS name. Left retryable (not added to
`workflows.py`'s `non_retryable_error_types`): a node-listing failure or a
briefly-`NotReady` node is plausibly transient.

`_rewrite_endpoints_for_multi` also validates `minio_nodeport`/
`mlflow_nodeport` are both configured (nonzero) *before* doing the node
lookup, raising `ValueError` (not `RuntimeError`) — a deterministic
misconfiguration exactly like `job_name_for`'s `kfp_run_id` check, so
`workflows.py`'s existing `non_retryable_error_types=["ValueError"]` fails it
fast instead of burning ~40s of Temporal retry backoff on an outcome that
cannot change.

Both exceptions are raised **before** the `try/except` in
`_launch_and_watch_pod_multi` that wraps `wait_for_worker_artifact`, so they
propagate straight out of `launch_and_watch_pod` as raised exceptions — never
caught, never turned into a returned `WorkerResult(succeeded=False, ...)`.
Verified directly by
`test_launch_and_watch_pod_multi_topology_raises_when_node_lookup_fails`,
which calls the full activity entrypoint (not just the internal helpers).

## TDD evidence

Worked bottom-up (types → activities → config/pipeline threading), writing
each layer's tests first and confirming the RED failure reason before
implementing:

1. **types.py** (`tests/test_orchestration_types.py`): 3 new tests
   (`test_worker_spec_defaults_nodeports_to_zero`,
   `test_round_spec_defaults_nodeports_to_zero`,
   `test_round_spec_threads_nodeports_into_worker_spec`) failed with
   `AttributeError`/`TypeError: unexpected keyword argument` before the
   dataclass fields existed.
2. **activities.py** (`tests/test_orchestration_activities.py`): 11 new/changed
   tests failed with `ImportError: cannot import name '_resolve_host_node_ip'`
   / `'_rewrite_endpoints_for_multi'`, or (for the end-to-end dispatch
   assertion) `AssertionError: 'minio-service:9000' == '172.18.0.2:30900'`,
   before the functions and wiring existed.
3. **config/run_pipeline/pipeline** (`tests/test_active_fl_pipeline.py`): 4 new
   tests failed with `ImportError: cannot import name 'DEFAULT_MINIO_NODEPORT'`
   and `AssertionError: 'minio_nodeport' in {...}` before the constants/dsl
   params existed.

All were then made to pass with minimal implementation, and the full suite
was re-run after each layer. A pre-existing regression guard
(`_forbid_k8s_batch_and_core`, used across 5 multi-topology tests) asserted
`_k8s_batch_and_core` is *never* called for `topology="multi"` — true before
this fix, and no longer true after it (the node lookup legitimately needs the
host cluster's local `CoreV1Api`). Replaced with
`_fake_k8s_batch_and_core_for_multi` for the 4 `launch_and_watch_pod`-multi
tests that now exercise the rewrite (returns a working `list_node()` fake, but
still `None` for the `BatchV1Api` half, so a regression that reached for it to
touch the Job itself would still fail with `AttributeError`); left unchanged
for `test_cleanup_worker_job_multi_topology_deletes_via_dispatcher`, since
cleanup never needs a node IP.

One extra check beyond minimal TDD: I temporarily reverted the
`spec = _rewrite_endpoints_for_multi(spec)` wiring line and re-ran
`test_launch_and_watch_pod_multi_topology_raises_when_node_lookup_fails` to
confirm it fails for a *different* reason (`FED_KARMADA_CONFIG must be set
...`, from reaching the Karmada dispatcher instead) when the wiring is
missing — proving the test exercises the actual integration, not something
incidental. Restored immediately after.

## Mutation evidence for the agreement tests

For each new config-agreement test, mutated the guarded file, confirmed the
test(s) fail, then restored and re-verified the file is byte-identical to the
original:

- `config/k8s-multi.yaml`'s `minio_nodeport: 30900` → `19999`:
  `test_nodeport_default_layers_agree` and
  `test_multi_config_nodeports_agree_with_the_multi_infra_contract` both
  failed with the expected `19999 == 30900` mismatch.
- `infra.env.multi`'s `FED_NODEPORT_MINIO_API=30900` → `19999`:
  `test_nodeports_match_the_multi_infra_contract` and
  `test_multi_config_nodeports_agree_with_the_multi_infra_contract` both
  failed with the expected `30900 == 19999` mismatch.
- `config/k8s-multi.yaml`'s `mlflow_nodeport: 30500` → `8888`:
  `test_nodeport_default_layers_agree` and
  `test_multi_config_nodeports_agree_with_the_multi_infra_contract` both
  failed with the expected `8888 == 30500` mismatch (the other two nodeport
  tests, not guarding `mlflow_nodeport` from this angle, correctly stayed
  green).

All three mutations were reverted with `cp`/`diff` confirming an identical
restore; full suite re-run green (223/224 passed, ruff clean) after each
restore.

## Test count

206 → 224 (18 new tests: 3 in `test_orchestration_types.py`, 11 in
`test_orchestration_activities.py`, 4 in `test_active_fl_pipeline.py`; plus
edits to 5 pre-existing tests to keep them meaningful — 4 that swapped
`_forbid_k8s_batch_and_core` for the new fake, and `_base_kwargs` in
`test_train_workers_component.py` for the two new required component
parameters).

`ruff check src/ tests/` (the project's actual lint target, per `Makefile`):
clean.

## Concerns

- **`_rewrite_endpoints_for_multi` also rewrites the endpoint the host's own
  Temporal worker uses to poll MinIO for the completion artifact**
  (`_minio_client_for(spec)` reads `spec.minio_endpoint` after the rewrite,
  same as the dispatched Job). This should be harmless — NodePort Services
  are reachable from within the cluster they're defined on, not just
  externally — but it does mean the host pod polls MinIO through an extra
  hop (node IP + NodePort) rather than the in-cluster DNS name it could have
  used directly. I did not special-case this because the task's own
  framing ("Apply it at dispatch time... happens exactly once per worker
  Job") pointed at rewriting the spec once, wholesale, rather than
  threading two different endpoint values through the rest of the function.
  Worth confirming during your live verification that host→NodePort MinIO
  polling behaves identically to the in-cluster-DNS path latency-wise.
- The `ValueError`-vs-`RuntimeError` split for the two failure modes inside
  `_rewrite_endpoints_for_multi` (config misconfiguration = non-retryable,
  node-lookup failure = retryable) was my own judgment call, applying this
  codebase's existing `job_name_for` convention — not explicitly requested
  in the task. Flagging in case a different retry policy is actually wanted
  for a transient-vs-permanent node-lookup failure.
- Did not add a corresponding nonzero/validity check for
  `minio_nodeport`/`mlflow_nodeport` inside `RoundSpec.worker_spec()` itself
  (unlike `member_count`/`member_prefix`, which raise there). The check lives
  only in `activities.py`, at the point of actual use, which is a narrower
  guarantee (a misconfigured `RoundSpec` can still produce a `WorkerSpec`,
  it just fails at dispatch instead of at `worker_spec()` time). This
  matched the task's explicit ask (fail at the node-lookup/rewrite site) but
  is one layer later than the `member_count`/`member_prefix` precedent.
- No live cluster or pipeline was touched, per instructions — everything
  above is unit-level (fakes/monkeypatches for the Kubernetes/MinIO/Karmada
  clients). Live verification (the actual node IP resolving correctly, the
  worker Job's env carrying the rewritten values, a member-cluster worker
  successfully reaching MinIO/MLflow through the NodePort) is yours to run
  against the live 3-cluster environment.
