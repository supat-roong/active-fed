# P3 Task 3 — Adversarial Review

Scope: commit `39ca013` (feat(orchestration): add Karmada job dispatcher
abstraction) and `fbb80f0` (feat(orchestration): round-robin worker->member
cluster assignment) — `src/orchestration/dispatch.py`, `src/orchestration/types.py`,
and their tests (`tests/test_dispatch.py`, `tests/test_orchestration_types.py`).

Method: read both commits and the current files, ran the existing test suite,
then wrote adversarial probe scripts (kept under
`/private/tmp/.../scratchpad/probe*.py`, not committed to the repo) that
imported the real modules and called them with hostile inputs, and one probe
that used Temporal's actual `DataConverter` to exercise the JSON round-trip.

**Tests**: `tests/test_dispatch.py` + `tests/test_orchestration_types.py` = 35
tests, all pass. Full repo suite: 165 passed, 0 failed.

---

## Finding 1 — HIGH — CONFIRMED — deploying fbb80f0 breaks replay of any in-flight `topology="multi"` round

**File**: `src/orchestration/types.py:71-72` (new defaults `member_count: int = 0`,
`member_prefix: str = ""`) and `:115-126` (`_member_cluster_for`'s raises).

**What I did**: `RoundSpec` is a Temporal workflow argument, serialized to JSON
by Temporal's data converter and replayed from history on every worker
restart. I built a `Payload` containing exactly the JSON shape a `RoundSpec`
would have had *before* `fbb80f0` (`topology="multi"`, `member_cluster="active-fed-member1"`,
no `member_count`/`member_prefix` keys — i.e. genuine history from a workflow
started under `39ca013`), decoded it with `temporalio.converter.DataConverter.default`
against the *current* `RoundSpec`, then called `.worker_spec(0)` on the
result — exactly what `TrainRoundWorkflow.run`'s `_one()` does on every replay
(`src/orchestration/workflows.py`, inside `execute_child_workflow(WorkerWorkflow.run, spec.worker_spec(worker_id), ...)`).

**Result**:
```
Decoded old in-flight RoundSpec: RoundSpec(..., topology='multi', member_cluster='active-fed-member1', member_count=0, member_prefix='')
REPLAY BREAKS: worker_spec(0) raised ValueError: topology='multi' requires member_count > 0 to assign a member cluster to worker 0 (round 1); got member_count=0
```

The missing fields deserialize to the dataclass defaults (`0`, `""`), and
those specific defaults are the exact values `_member_cluster_for` treats as
fatal for `topology="multi"`. A round that was running fine under `39ca013`
(single static `member_cluster` passthrough — Task 3's "deliberate stub")
would, on the very next replay after the `fbb80f0` worker binary is deployed,
raise where it used to succeed. In `TrainRoundWorkflow._one`, `spec.worker_spec(worker_id)`
is evaluated as an argument inside the `try:` block wrapping `execute_child_workflow`,
so this gets caught, reported as a spurious worker failure via
`_root_cause_message`, and the round's report is corrupted — with a message
that looks like a config bug on the *current* round, giving no hint that the
real cause is a version-skew replay of stale history.

**Why it matters**: this is precisely the class of bug the task brief calls
out ("An added field that doesn't deserialize breaks replay of in-flight
workflows") — new fields with defaults that are individually reasonable
(`0`/`""` read fine on their own) but become unsafe once combined with
existing `topology="multi"` history. No test in either commit constructs an
old-shaped payload; every test in `test_orchestration_types.py` builds a
`RoundSpec` fresh with all fields set together, so this path was never
exercised. Given `active-fed`'s Karmada multi-cluster rollout is actively in
progress (see `26a8854`, the very next commit after these two), this is a
realistic near-term deploy hazard, not a hypothetical.

**Not fully in this commit's diff to fix** (it's a consequence of adding
fields to a dataclass that already crosses the wire), but it is a direct,
demonstrable defect in what `fbb80f0` shipped, and is exactly the kind of
thing item 5 of the review brief asked to check.

---

## Finding 2 — MEDIUM/HIGH — CONFIRMED — whitespace-only `member_cluster`/`member_prefix` defeats every "critical safety property" guard

**Files**: `src/orchestration/dispatch.py:60` (`if not spec.member_cluster:`),
`src/orchestration/dispatch.py:105` (`if not spec.member_cluster:` in
`dispatcher_for`), `src/orchestration/types.py:121` (`if not self.member_prefix:`).

**What I did**: constructed a `WorkerSpec` with `member_cluster="   "` and
called `build_propagation_policy` and `dispatcher_for` directly; separately
constructed a `RoundSpec` with `topology="multi", member_count=2,
member_prefix="   "` and called `.worker_spec(0)`.

**Result**:
```
build_propagation_policy(member_cluster="   ")  -> NO RAISE, clusterNames = ['   ']
dispatcher_for(member_cluster="   ")             -> NO RAISE, returns KarmadaJobDispatcher
RoundSpec(member_prefix="   ").worker_spec(0)     -> NO RAISE, member_cluster = '   1'
RoundSpec(member_prefix="\t\n").worker_spec(0)    -> NO RAISE, member_cluster = '\t\n1'
```
All three guards use Python truthiness (`if not x`), and a non-empty
whitespace string is truthy, so it sails past every check that exists
specifically to stop this class of input.

**Why it matters**: this does **not** reproduce the literal catastrophic case
the docstrings describe (an empty `clusterNames` list targeting *every*
cluster) — I want to be precise about that, since it's a different failure
mode. A Karmada `clusterAffinity.clusterNames` containing a garbage,
non-matching string (`"   "`, `"\t\n1"`) selects **zero** real clusters (no
joined member cluster has a name with whitespace in it), so the
`PropagationPolicy` is accepted by the Karmada apiserver but nothing ever
schedules the worker Job anywhere. Downstream, `_launch_and_watch_pod_multi`
would then block on `wait_for_worker_artifact` for the full
`POD_WATCH_TIMEOUT_S` (3600s) before failing — a slow, silent, hard-to-diagnose
failure whose root cause (a blank/whitespace config value upstream) is
completely obscured by the time it surfaces. This is exactly the kind of
"safety property that holds for the exact case in the test but not the
adjacent one" gap the review brief asked me to hunt for. Fix would be
`if not spec.member_cluster.strip():` / `if not self.member_prefix.strip():`
(not applied — this is a review, not a fix).

**Confirms**: the literal empty-string (`""`) case *does* raise on every path
I tried — `build_propagation_policy`, `dispatcher_for`, and
`RoundSpec.worker_spec` (via `member_count<=0` and `member_prefix==""`) all
raise `ValueError` for genuinely empty inputs. The gap is specifically
whitespace-only / non-empty-but-meaningless strings, not `None`
(`member_cluster=None` was also tried directly against `build_propagation_policy`
and correctly raises, since `not None` is `True`).

---

## Finding 3 — MEDIUM — CONFIRMED — `test_karmada_job_manifest_reused_unchanged_from_p1` cannot fail for the reason its name claims

**File**: `tests/test_dispatch.py`, the test named
`test_karmada_job_manifest_reused_unchanged_from_p1`.

```python
def test_karmada_job_manifest_reused_unchanged_from_p1():
    ...
    KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)
    manifest = build_job_manifest(spec)          # <- fresh, independent call
    assert manifest["spec"]["backoffLimit"] == 0
    assert manifest["spec"]["template"]["spec"]["restartPolicy"] == "Never"
```

`FakeBatchApi.create_namespaced_job` never records the `body` it was called
with, so the two assertions check a manifest built by a *second, unrelated*
call to `build_job_manifest(spec)` — not what `_ensure_job_with` actually
applied.

**What I did**: monkeypatched `KarmadaJobDispatcher._ensure_job_with` to apply
a deliberately broken manifest (`backoffLimit=3`, `restartPolicy="OnFailure"`,
a different Job name entirely) to an instrumented `FakeBatchApi` that *does*
capture the applied body, then ran the exact assertions the real test uses.

**Result**:
```
Actually applied: backoffLimit=3, restartPolicy=OnFailure, name=totally-different-job-name
Real test's assertions (built from a fresh build_job_manifest(spec) call): PASS anyway
```

The test is green regardless of what the dispatcher actually sent to the
apiserver. Today's implementation is correct (`_ensure_job_with` does call
`build_job_manifest(spec)` and apply it — I read the source), so there's no
live bug — but this is precisely this codebase's recurring "assertion that
can never fail" pattern flagged in the review brief (the bats-assertion
example). A future refactor of `_ensure_job_with` that silently diverges from
`build_job_manifest`'s output would not be caught by this test.

---

## Finding 4 — LOW / informational, out of strict diff scope — CONFIRMED-by-grep — `member_prefix` has no config knob and no layer-agreement test

`src/pipelines/active_fl_pipeline.py`'s `train_workers`/pipeline signature
declares `member_prefix: str = "active-fed-member"`, but:
- `config/k8s.yaml`'s `orchestration:` block has no `member_prefix` key
  (only `topology` and `members`).
- `src/pipelines/run_pipeline.py` never reads or forwards a `member_prefix`
  value — its `arguments = {...}` dict passed to `create_run_from_pipeline_package`
  includes `"topology"` and `"members"` but no `"member_prefix"` (confirmed
  via grep, `run_pipeline.py` has zero occurrences of `member_prefix`).
- `test_topology_and_members_default_layers_agree` (in
  `tests/test_active_fl_pipeline.py`) checks `topology`/`members` agree across
  `config/k8s.yaml`, `run_pipeline.py`'s `DEFAULT_TOPOLOGY`/`DEFAULT_MEMBERS`,
  and the compiled pipeline's dsl defaults — there is no equivalent check for
  `member_prefix`.

This means `member_prefix` is only ever the pipeline's hardcoded default —
there is currently no way to override it per-run/per-config the way
`topology`/`members` can be. It happens to be harmless today because
`infra.env.multi`'s `FED_MEMBER_PREFIX=active-fed-member` matches the
pipeline's hardcoded default, so nothing is actually broken. Flagging this
because `member_prefix` is a field this exact review's scope
(`fbb80f0`/`types.py`) introduced, and this is the same "three layers can
silently disagree" class the codebase has hit before (P2 review Finding 1,
`start_round`) — but the wiring gap itself lives in a different commit
(`2f8f0a6`), outside `39ca013`/`fbb80f0`'s diff, so I'm reporting it as
informational rather than a scored finding against Task 3.

---

## What I checked and found clean

- **Empty-`clusterNames` catastrophic case** (literal `member_cluster=""`):
  raises on every path tried — `build_propagation_policy` directly,
  `dispatcher_for` directly, and `RoundSpec.worker_spec` via
  `member_count<=0` / `member_prefix==""` — including `member_cluster=None`
  passed to a dataclass field typed `str` (still raises, since `not None`
  is `True`). This is the one specific property the task asked to verify
  most carefully, and — modulo Finding 2's whitespace gap — it holds.
- **Round-robin indexing**: confirmed `worker_id` is 0-based at every real
  caller (`range(spec.num_workers)` in `TrainRoundWorkflow.run`'s `_one`,
  `src/orchestration/workflows.py`), and member cluster names are 1-based
  in the actual infra (`vendor/fed-infra/lib/components.sh`: `local i=1; while
  [ "$i" -le "$FED_MEMBER_COUNT" ]; do fed_kind_ensure_cluster
  "${FED_MEMBER_PREFIX}${i}" ...`). `worker_id % member_count + 1` correctly
  bridges the two (0-based worker → 1-based member), verified by probe for
  `member_count=1` (all workers → member1), `member_count=5 > num_workers=3`
  (no wraparound needed, each worker gets a distinct member), and the
  documented wraparound case (3 workers / 2 members → member1, member2,
  member1). Negative `worker_id` doesn't crash (Python's modulo semantics
  keep the result non-negative) but is not reachable from any real caller.
- **Frozen dataclass / Temporal sandbox**: `types.py` imports only
  `dataclasses` (confirmed by AST-parsing the file's imports directly) and
  is loaded into `workflows.py` inside `workflow.unsafe.imports_passed_through()`;
  `dispatch.py` (which does import `kubernetes`, lazily, inside function
  bodies) is never imported at module scope anywhere reachable from the
  workflow sandbox — confirmed by grepping for any top-level `import
  src.orchestration.dispatch` (none found; both call sites in `activities.py`
  import it lazily inside function bodies). Both `WorkerSpec` and `RoundSpec`
  remain `@dataclasses.dataclass(frozen=True)`.
- **Generated `PropagationPolicy` manifest**: `apiVersion:
  policy.karmada.io/v1alpha1`, `kind: PropagationPolicy`, and
  `spec.resourceSelectors[0]` correctly matches `build_job_manifest`'s
  `apiVersion: batch/v1`/`kind: Job`/name (`job_name_for(spec)`)/namespace
  (`spec.namespace`) — the policy would actually select the Job it's paired
  with.
- **Serialization round-trip** (current shape): a `RoundSpec` built with all
  fields set, including `member_count`/`member_prefix`, round-trips through
  `temporalio.converter.DataConverter.default` byte-for-byte equal. The
  *backward-compat* direction is Finding 1, above.
- **`single`-topology regression**: `RoundSpec.worker_spec`'s non-`"multi"`
  branch is textually unchanged from Task 3 (`member_cluster =
  self.member_cluster` passthrough, never touched by `_member_cluster_for`);
  `dispatcher_for("single", ...)` always returns `LocalJobDispatcher`
  regardless of `member_cluster`; `launch_and_watch_pod`'s non-`"multi"`
  branch is untouched code with only a new early-return guard added ahead
  of it. Existing test `test_single_topology_worker_spec_ignores_member_count_and_prefix`
  covers the field-level case; I additionally confirmed no diff exists in the
  actual single-topology code path since before `fbb80f0`.
