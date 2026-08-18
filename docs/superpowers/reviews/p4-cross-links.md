# Phase P4, Task 2 — MLflow cross-link tags

Status: DONE_WITH_CONCERNS

## What was built

- `src/tracking/mlflow_logger.py`: new `log_run_context(kfp_run_id,
  temporal_workflow_id, topology, kfp_base_url="http://localhost:8080",
  temporal_base_url="http://localhost:8233")`. Sets up to five tags
  (`kfp_run_id`, `kfp_run_url`, `temporal_workflow_id`,
  `temporal_workflow_url`, `topology`) via `mlflow.set_tags(...)`. Each of
  the three source values is optional — an empty one (and its derived URL)
  is skipped rather than written as an empty tag. The whole body is wrapped
  in try/except with `log.warning(...)` on failure, matching
  `fed-twin/src/core/tracking.py:39-46`'s `log_metrics`.

- `src/pipelines/active_fl_pipeline.py`:
  - `evaluate_global` gained `kfp_run_id: str`, `topology: str`, and
    `worker_report: Input[Artifact]`. It reads `temporal_workflow_id` out of
    `worker_report` (`.get("temporal_workflow_id", "")`, so a malformed/older
    report degrades to a skipped tag, not a `KeyError`) and calls
    `log_run_context(...)` inside the `with mlflow.start_run(...)` block it
    already opens per round.
  - The pipeline-assembly loop now passes `kfp_run_id=run_uid`,
    `topology=topology`, and `worker_report=train_op.outputs["worker_report"]`
    into every `evaluate_global(...)` call.
  - `train_workers` prints `Temporal workflow:
    http://localhost:8233/namespaces/default/workflows/<id>` right after
    starting the workflow, so KFP → Temporal (the most-used direction) is one
    click from the node log.

- `src/orchestration/workflows.py`: `TrainRoundWorkflow.run` calls
  `workflow.upsert_memo({"kfp_run_id": spec.kfp_run_id})` (guarded on
  non-empty) right after computing `parent_id`. `spec` is the workflow's own
  input and `upsert_memo` is a workflow command (not I/O), so this stays
  deterministic/replay-safe. `types.py` was not touched.

- Tests: new `tests/test_mlflow_cross_links.py` (13 tests) covering
  `log_run_context` in isolation and its wiring through
  `evaluate_global.python_func`; two new structural tests in
  `tests/test_active_fl_pipeline.py` proving the compiled IR actually wires
  `worker_report`/`kfp_run_id`/`topology` into `evaluate_global` (not just
  that the Python function accepts them); one new test in
  `tests/test_orchestration_workflows.py` proving the memo is queryable via
  `handle.describe()`.

## Step 3 investigation — was `temporal_workflow_id` already in the report?

**Half right, half wrong, and I did not silently pick a side.**

- `temporal_workflow_id` **was already** written into `worker_report` by
  `train_workers`, on both the success path (`active_fl_pipeline.py`, the
  `payload = {...}` built from `report`) and the quorum-failure path (the
  `except WorkflowFailureError` branch) — confirmed by reading the code and
  by the pre-existing `tests/test_train_workers_component.py`, which already
  exercises both paths.

- **But `evaluate_global` never received that report.** The plan says to
  "read it from the report the aggregator already receives" — the aggregator
  is `score_and_aggregate`, and it does **not** receive `worker_report` at
  all; it reads worker updates directly from MinIO by round number
  (`collect_worker_updates`), and its own output artifact
  (`aggregation_report`, the only thing `evaluate_global` was wired to) never
  carries `temporal_workflow_id`. So the plan's assumed path did not exist.

  I threaded `worker_report` directly from `train_op` into `evaluate_global`
  instead of routing it through `score_and_aggregate` (which has no other use
  for it and would have meant widening its contract for a pass-through
  field). This is a one-hop, not-in-the-plan addition, called out here rather
  than silently invented: `evaluate_global` now takes
  `worker_report: Input[Artifact]` directly, and the pipeline wires
  `train_op.outputs["worker_report"]` into it alongside
  `agg_op.outputs["aggregation_report"]`.

## `kfp_run_id` — a related gap found during implementation

The plan's Step 3 also says to thread `kfp_run_id` "from
`dsl.PIPELINE_JOB_ID_PLACEHOLDER`". This is the exact mechanism the P2 "F4"
gate fix (see `test_compiled_pipeline_has_no_unsubstituted_kfp_placeholder`)
already proved is broken in this KFP deployment — the placeholder is never
substituted and arrives at component code as the literal string
`"{{$.pipeline_job_uuid}}"`. I did not use it. Instead `kfp_run_id` is wired
from the same `run_uid` value already threaded through the whole pipeline
under that name (`RoundSpec.kfp_run_id` / `WorkerSpec.kfp_run_id`, used for
Job/workflow naming) — the design spec
(`docs/superpowers/specs/2026-08-12-active-fed-fed-infra-temporal-design.md`,
§3.6) confirms this is the intended meaning: "the KFP run ID is carried in
the workflow input."

**Concern to flag explicitly:** `run_uid` is generated locally by
`run_pipeline.py` (`uuid.uuid4().hex[:8]`) *before* the run is submitted, and
is not the same value as the KFP-backend-assigned `run.run_id` that
`run_pipeline.py` itself later logs as "View in KFP UI →
.../#/runs/details/{run.run_id}". The two are unrelated identifiers by
construction — the real run ID doesn't exist until after
`create_run_from_pipeline_package()` returns, which is after all pipeline
arguments (including anything threaded into components) are already fixed.
So `kfp_run_url`, built from `run_uid`, will very likely **not** resolve to
the correct run in the KFP UI — the exact "plausible-looking URL that 404s"
failure mode Task 4 Step 3 warns about. Fixing this for real would mean
either making the KFP backend accept a caller-supplied run ID (not offered by
the `kfp` client used here) or having `run_pipeline.py` record a
`run_uid → run.run_id` mapping somewhere queryable from inside the cluster —
both out of scope for Task 2 and not mentioned in the plan. I did not invent
either; I built the tag/URL exactly as specified against the identifier this
codebase already calls `kfp_run_id`, and am flagging the mismatch here for
Task 4 to hit deliberately rather than by surprise.

`temporal_workflow_url`, by contrast, is sound: `handle.id` is the real
Temporal workflow ID used to start the workflow, so the URL it builds does
resolve to the correct workflow.

## Testing

TDD throughout: every new test was run against the pre-fix code first and
confirmed to fail for the intended reason before implementing:
- `tests/test_mlflow_cross_links.py` failed to **collect** at all
  (`ImportError: cannot import name 'log_run_context'`) before the function
  existed.
- The two new pipeline-wiring tests in `test_active_fl_pipeline.py` failed
  with `KeyError: 'worker_report'` / `KeyError: 'kfp_run_id'` against the
  compiled IR before the wiring existed.
- The new workflow-memo test failed with `KeyError: 'Memo does not have a
  value for key kfp_run_id'` before `upsert_memo` was added.

Coverage against the task's minimum bar:
- All five tags set — `test_all_five_tags_set`.
- URLs well-formed and containing the corresponding IDs —
  `test_urls_are_well_formed_and_contain_the_ids` (exact string equality on
  both URLs).
- Empty IDs skipped, not written empty —
  `test_empty_kfp_run_id_is_skipped_not_written_empty`,
  `test_empty_temporal_workflow_id_is_skipped_not_written_empty`,
  `test_empty_topology_is_skipped`,
  `test_all_empty_ids_result_in_no_set_tags_call`.
- MLflow failure caught, not propagated —
  `test_mlflow_failure_is_caught_and_logged_not_raised` (forces
  `mlflow.set_tags` to raise, asserts no exception escapes and a warning is
  logged with the underlying message).
- Both topologies — `test_topology_tag_correct_under_both_topologies` and
  `test_evaluate_global_topology_tag_correct_under_both_topologies`,
  parametrized over `"single"`/`"multi"`.
- Step 3 wiring specifically —
  `test_evaluate_global_threads_temporal_workflow_id_from_worker_report`,
  `test_evaluate_global_does_not_crash_when_worker_report_lacks_workflow_id`,
  and the two compiled-IR structural tests.

## Test count

225 → 241 (+16: 13 in the new `test_mlflow_cross_links.py`, 2 in
`test_active_fl_pipeline.py`, 1 in `test_orchestration_workflows.py`).

## Constraints respected

- `launch_and_watch_pod`, `backoffLimit: 0`, `restartPolicy: Never`: not
  touched.
- `src/orchestration/types.py`: not touched — still stdlib-only.
  `workflow.upsert_memo` was called from `workflows.py` (which already
  imports `temporalio`), not from `types.py`.
- `src/experiment/local_runner.py`, `config/local.yaml`: not touched.
- No real cluster was touched. Task 4 remains the gate, and the `kfp_run_url`
  concern above is specifically something Task 4's "follow the URL and
  confirm it opens the correct run" step should check for the KFP side.
