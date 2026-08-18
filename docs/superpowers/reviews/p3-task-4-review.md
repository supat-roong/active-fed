# Adversarial review — Phase P3, Task 4

Commit reviewed: `f320d88` "feat(orchestration): wire MinIO completion detection for
multi-cluster workers".

Scope: the `topology='multi'` branch of `src/orchestration/activities.py`
(`launch_and_watch_pod`, `_launch_and_watch_pod_multi`, `wait_for_worker_artifact`,
`classify_job_status`, `build_job_manifest`, `job_name_for`, `cleanup_worker_job`),
its tests (`tests/test_completion_watch.py`,
`tests/test_orchestration_activities.py`), and the `dispatch.py` dependency
`_launch_and_watch_pod_multi` calls into (`src/orchestration/dispatch.py`,
`tests/test_dispatch.py`) since that dependency is exactly the "substitution"
this task is about.

Method: read the diff and current files, ran the full test suite, then wrote
throwaway probe scripts (in `/private/tmp/.../scratchpad/`, not in the repo)
against the actual repo code and the actual installed `minio==7.2.20` package
to check claims that could otherwise be fooled by a mock. No repo files were
modified; no cluster was touched.

---

## Findings, ranked by severity

### 1. HIGH — a failed multi-cluster worker's Temporal retries are inert: every attempt re-attaches to the same dead Job and burns the full timeout again

**CONFIRMED** (executed, two probes).

- `src/orchestration/activities.py:437-476` (`_launch_and_watch_pod_multi`) calls
  `dispatcher.ensure_job(spec)` (line 452) and then waits only on
  `wait_for_worker_artifact` (line 456). It never re-checks Job status once
  the wait starts.
- `dispatcher.ensure_job` for `topology='multi'` resolves to
  `KarmadaJobDispatcher._ensure_job_with`
  (`src/orchestration/dispatch.py:167-188`):
  ```python
  try:
      batch.create_namespaced_job(namespace=spec.namespace, body=manifest)
  except ApiException as e:
      if e.status != 409:
          raise
  ...
  return name
  ```
  On a 409 (Job already exists — exactly what a Temporal *retry* of the same
  deterministically-named Job sees, per `job_name_for`'s own docstring) it just
  swallows the error and returns the existing name. It never calls
  `read_namespaced_job_status`, never classifies the existing Job, and never
  deletes+recreates it.
- Contrast this with the single-topology path's `_ensure_job`
  (`src/orchestration/activities.py:188-235`), which explicitly exists to
  handle this: on 409 it classifies the existing Job via
  `classify_job_status`, and if it is terminally `Failed` it deletes it, waits
  for the delete to complete (`_await_job_deleted`), and recreates it — "or
  simply re-attaching would make the watch loop observe an already-Failed
  object forever" (its own docstring, line 200-204). `KarmadaJobDispatcher`
  never got this treatment.

**Consequence:** if a member-cluster worker's container crashes (bad image,
OOM, config error — anything that makes the Job terminally `Failed`), the
first `launch_and_watch_pod` attempt correctly times out after
`POD_WATCH_TIMEOUT_S` (3600s) and raises `WorkerJobFailed` (retryable). But
Temporal's `RetryPolicy(maximum_attempts=3)` (`src/orchestration/workflows.py:74-86`)
then re-invokes the *same* activity against the *same* dead Job — which
`ensure_job` re-attaches to instead of replacing — so the retry polls MinIO
for an artifact a dead Job can never produce, and burns another full
3600s before failing again. All 3 attempts are wasted: **~3 hours of wall
time to fail a round that could have been failed in one poll interval**, and
zero actual retry benefit for a class of failure retry exists specifically to
recover from.

**Why no test caught it:** `tests/test_dispatch.py`'s
`test_karmada_dispatcher_ensure_job_is_idempotent` (line 206) explicitly
frames this as the retry scenario in its own comment — "Must not raise even
though both the Job and the PropagationPolicy already exist from a previous
(e.g. retried) attempt" — but only asserts that it doesn't raise. Its
`FakeBatchApi` (`tests/test_dispatch.py:131-145`) has no
`read_namespaced_job_status` and no concept of Job status at all, so there is
no way for this test to have caught the missing delete-and-recreate logic.
`tests/test_orchestration_activities.py`'s new multi-topology tests
(`test_launch_and_watch_pod_multi_topology_raises_not_returns_on_missing_artifact`,
line 476) only exercise a single attempt, never a second call against a spec
whose Job already exists and already failed. This is exactly the bug class
named in the task: a failure path no passing test exercises.

**Verification performed:** `/private/tmp/.../scratchpad/probe_retry_reuses_dead_job.py`
drives the real `KarmadaJobDispatcher._ensure_job_with` against a fake
Karmada client that reports the existing Job as terminally `Failed`. Output:

```
Attempt 1 ensure_job -> aflw-abcd1234-r1-w0 | created so far: ['aflw-abcd1234-r1-w0']
Attempt 2 (retry) ensure_job -> aflw-abcd1234-r1-w0 | deleted: [] | created: ['aflw-abcd1234-r1-w0']
CONFIRMED: KarmadaJobDispatcher.ensure_job never deleted the terminally-Failed Job...
```

---

### 2. HIGH — a crashed multi-cluster worker is only ever detected via full timeout, never fast-failed

**CONFIRMED** (executed).

`wait_for_worker_artifact` (`src/orchestration/activities.py:296-351`) has no
way to distinguish "worker crashed 2 seconds after launch" from "worker is
still legitimately training" — it purely polls for the existence of one
MinIO key and loops until `timeout_s` (`POD_WATCH_TIMEOUT_S = 3600`,
`activities.py:24`) elapses. `_karmada_failure_reason`
(`activities.py:479-502`) *does* read the aggregated Job status, but only
**after** `wait_for_worker_artifact` has already given up — it is
diagnostic-message enrichment, never consulted during the wait itself (this
matches the code's own stated intent, "Karmada aggregated API is consulted
only as best-effort failure enrichment," `activities.py:442-447` — but that
intent itself is the bug: nothing in the multi path does what
`classify_job_status` does for the single path, i.e. observe a `Failed`
condition and raise immediately).

**Consequence:** compare directly against the single-topology path, which
detects a `Failed` Job condition within one `POLL_INTERVAL_S` (5s) of it
appearing (`activities.py:414-423`). The multi path takes up to
**3600s (1 hour) per attempt** to report the identical class of failure
(pod crash-loops instantly, `backoffLimit: 0` fails the Job in well under a
minute). This is the "20-minute stall" pattern called out in the task
description, except worse — the ceiling here is an hour, and (per Finding 1)
it repeats on every retry attempt with no mitigation.

**Verification performed:**
`/private/tmp/.../scratchpad/probe_failure_detection.py` runs the real
`launch_and_watch_pod` against a dispatcher whose Job is already Failed the
instant `ensure_job` returns, with `POD_WATCH_TIMEOUT_S` shrunk to 0.5s for a
fast repro. Output:

```
Raised WorkerJobFailed after 0.626s (POD_WATCH_TIMEOUT_S=0.5s used)
This is the ONLY way failure is detected: waiting out the full timeout.
```

Findings 1 and 2 are two faces of one root cause (no active Job-status check
on the multi path, and no repair of a dead Job before a retry), so they are
reported separately because they compound: #2 alone would mean "every failure
costs one hour"; #1 makes it "every failure costs one hour times
`maximum_attempts`, with zero chance of the retry actually helping."

---

### 3. MEDIUM — the MinIO completion key has no run identifier, so a `--bucket`-resumed run can read a stale artifact from an earlier abandoned attempt as "success"

**PLAUSIBLE** (reasoned from code; not exercised against a live resume, since
that requires a real MinIO + two pipeline invocations to demonstrate end to
end).

`wait_for_worker_artifact`'s key (`activities.py:331`) is
`round_{fl_round}/workers/worker_{worker_id}_metrics.json` — it contains the
round and worker id, but **not** `kfp_run_id`/`run_uid`. This exactly matches
the convention `train_worker.py`'s `_push_weights` and
`src/aggregator/collect.py` already use, so it isn't a new convention P3 Task
4 invented — but Task 4 is the first thing that ties a **Temporal retry
decision** to this key. Before this task, `topology='single'` completion was
detected purely from the Kubernetes Job (whose name embeds `kfp_run_id` via
`job_name_for`, guaranteeing uniqueness per run), so a stale MinIO object
could never fool the *completion check itself* — only the aggregator's own
read, which is a separate, pre-existing concern.

`src/pipelines/run_pipeline.py:116-124` documents `--bucket` explicitly as a
resume mechanism ("Reuse an existing MinIO bucket instead of generating a
fresh one"), and `compute_start_round`
(`src/pipelines/run_pipeline.py:31-49`) resumes at the highest round whose
`global.pt` exists — i.e., at a round whose **workers** may already have
uploaded metrics in a prior, abandoned attempt (e.g. the round reached quorum
threshold for workers but the aggregator itself later crashed, or the run was
killed after workers finished but before `global.pt` for that round was
written). Resuming re-launches that round's worker Jobs (fresh Job names if
`kfp_run_id` differs across invocations), but `wait_for_worker_artifact`
checks only the round/worker-scoped key — it will see the leftover
`metrics.json` from the *previous* attempt and report the brand-new Job as
succeeded immediately, before the new pod has necessarily done any work this
run. The aggregator then reads that same (stale) object, silently averaging
in a worker update computed against a different, no-longer-current global
model.

**Why this is worth flagging even though it isn't unique to `topology='multi'`
in principle:** `topology='single'`'s Job-watch was never coupled to this key
for *completion*, so this is a new integrity dependency introduced
specifically by wiring MinIO into the multi-cluster completion check. No test
in `tests/test_completion_watch.py` or `tests/test_orchestration_activities.py`
constructs a `FakeMinioClient`/`FakeMinioClientForActivities` pre-seeded with
an object before the corresponding Job is even created — every test's key
appears only as a *result* of the scripted call sequence, never as leftover
state from a previous run. This blind spot matches the task's specific
prompt #3 (stale-artifact / cross-round contamination) precisely.

**Suggested verification if this needs to move to CONFIRMED:** point two
separate `run_pipeline.py` invocations (or two direct MinIO clients) at the
same bucket, `put_object` a `round_2/workers/worker_0_metrics.json` by hand,
then call `wait_for_worker_artifact` for round 2 / worker 0 with a
dispatcher that never actually launches a Job — it will return `True`
immediately.

---

### 4. LOW/MEDIUM — the "tolerate transient MinIO errors and keep polling" logic only catches `S3Error`, not the connection-level exceptions a real transient MinIO outage actually raises

**CONFIRMED** (executed against the real installed `minio` package).

`wait_for_worker_artifact`'s loop (`activities.py:334-340`) only catches
`except S3Error as e:`, with a comment distinguishing "transient" S3 error
codes (e.g. `InternalError`, `SlowDown` — covered by
`tests/test_completion_watch.py::test_tolerates_a_transient_s3_error_and_keeps_polling`)
from `NoSuchKey`/`NoSuchObject`. But the realistic transient failure for a
member-cluster worker's MinIO endpoint — a brief network blip, the MinIO pod
restarting, DNS hiccup — does not raise `S3Error` at all; it raises a
connection-level exception from `urllib3`, which is **not** caught here and
propagates straight out of the poll loop on the very first occurrence.

Verified directly against the real `minio==7.2.20` package installed in this
repo's `.venv` (not the tests' `FakeMinioClient`, which cannot raise this):

```
>>> Minio('127.0.0.1:1', ...).stat_object('bucket','key')
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='127.0.0.1', port=1):
Max retries exceeded ... Failed to establish a new connection: [Errno 61] Connection refused
```

**Consequence:** this exception is not fatal in the way it looks — it
propagates to `_launch_and_watch_pod_multi`'s broad `except Exception as e:`
(`activities.py:464`), gets wrapped as a retryable `WorkerJobFailed`, and
Temporal retries. Because the underlying Job is very likely still alive and
running (nothing about a MinIO network blip kills the worker pod), the retry
lands in the "still active — re-attach" case that `ensure_job` *does* handle
correctly for a live Job, so this does not compound with Finding 1. Still,
one MinIO hiccup — completely unrelated to worker health — consumes one of
only 3 retry attempts and restarts the wait from zero, which the code's own
docstring ("Heartbeats every failed poll... transient MinIO error polling")
implies it was trying to guard against and only partially does.

---

## Explicitly checked and found clean

- **`launch_and_watch_pod` raises rather than returns on the multi path**:
  CONFIRMED clean. `_launch_and_watch_pod_multi` never returns
  `WorkerResult(succeeded=False, ...)`; any failure (timeout or otherwise)
  from `wait_for_worker_artifact` is wrapped in `WorkerJobFailed` and raised
  (`activities.py:464-472`). This is the exact regression class named in the
  task (the "survived six reviews" bug) and it does not recur here.
- **`S3Error` construction/signature**: CONFIRMED clean. Real signature per
  the installed `minio==7.2.20`:
  `S3Error(response, code, message, resource, request_id, host_id, bucket_name=None, object_name=None)`.
  Both `tests/test_completion_watch.py:31-35` and
  `tests/test_orchestration_activities.py`'s `FakeMinioClientForActivities`
  construct it with the correct positional/keyword args. The
  `"NoSuchObject"` alternative checked alongside `"NoSuchKey"`
  (`activities.py:338`) is dead — grepping the installed package, only
  `"NoSuchKey"` is ever produced for a HEAD/`stat_object` 404 — but it's
  harmless dead code, not a bug (the real code, `"NoSuchKey"`, is present).
- **Partial multi-object artifact sets**: CONFIRMED clean by reading
  `train_worker.py:_push_weights` (`src/agent/train_worker.py:88-118`) —
  weights and delta are uploaded via their own `put_object` calls strictly
  before `metrics.json`'s single `put_object` call, and only the metrics key
  is ever checked. A single `put_object` call is atomic (the task's own
  premise), so there is no way to observe a torn/partial `metrics.json`, and
  checking only that key (never `worker_*_weights.pt`) is verified by
  `test_does_not_treat_the_weights_object_alone_as_completion`
  (`tests/test_completion_watch.py:150-167`).
- **Heartbeats / timeout budget**: CONFIRMED clean. `wait_for_worker_artifact`
  heartbeats every failed poll (`activities.py:342-344`), i.e. every
  `POLL_INTERVAL_S` (5s) — well inside the workflow's
  `heartbeat_timeout=timedelta(seconds=60)` (`workflows.py:73`). No zombie-Job
  heartbeat-timeout-misread-as-failure scenario was reproducible on this
  path (that failure mode lives in the single path's own machinery,
  `_await_job_deleted`, which this commit does not touch and is out of this
  task's scope).
- **`backoffLimit: 0` / `restartPolicy: Never`**: CONFIRMED intact.
  `build_job_manifest` (`activities.py:62-137`) is unchanged and reused
  as-is by `KarmadaJobDispatcher` (`dispatch.py:170`) — Kubernetes never
  retries a multi-cluster worker; Temporal owns retry exclusively, as
  intended.
- **`topology='single'` regression**: CONFIRMED clean. The only change to
  `launch_and_watch_pod`'s single-topology body is the new early-return guard
  (`activities.py:384-385`); the rest of the function is untouched byte-for-byte
  per the diff. All pre-existing single-topology tests
  (`_spec()` defaults to `topology="single"`) pass unchanged.

---

## Test run

```
.venv/bin/python3 -m pytest -q
======================= 165 passed, 2 warnings in 5.62s ========================
```

All tests pass, including the 6 new tests in `test_completion_watch.py` and
the 3 new multi-topology tests in `test_orchestration_activities.py`. As
above, none of them exercise a second activity attempt against an
already-failed multi-cluster Job, which is where Finding 1 lives.

## Probe scripts (not committed, for reference)

- `/private/tmp/claude-501/-Users-supat-workspace-project/27953854-7a31-47e9-8f78-cc064ebf47ea/scratchpad/probe_failure_detection.py`
- `/private/tmp/claude-501/-Users-supat-workspace-project/27953854-7a31-47e9-8f78-cc064ebf47ea/scratchpad/probe_retry_reuses_dead_job.py`

## Fixes applied

Findings 2, 3, and 4 below. **Finding 1 was out of scope** (assigned to a
separate agent working in `src/orchestration/dispatch.py`, which this work
never touched). Three atomic commits on `main`, one per finding; all in
`src/orchestration/activities.py` plus `tests/test_completion_watch.py` /
`tests/test_orchestration_activities.py`. Full suite: 167 -> 184 passed.

### Finding 2 (fast-fail on terminal Karmada status)

`wait_for_worker_artifact` gained an optional `failure_check` parameter: a
zero-arg async callable polled once per iteration alongside the MinIO check.
Returning a diagnostic string means "terminally failed" and raises
`WorkerJobFailed` immediately; returning `None` means "no verdict yet" and
is never treated as failure. `_launch_and_watch_pod_multi` wires this to a
new `_karmada_terminal_failure(spec, job_name)` helper (built on a shared
`_karmada_job_status` that also backs the existing best-effort
`_karmada_failure_reason` enrichment), which returns a reason only when the
Karmada aggregated Job status has a `Failed` condition (via
`classify_job_status`); an absent status (Job not yet propagated), a
still-running status, or an unreachable Karmada apiserver all return `None`
and never short-circuit the wait.

Tests added: `test_raises_immediately_when_failure_check_reports_terminal_failure`,
`test_absent_or_lagging_failure_check_never_short_circuits_the_wait`,
`test_failure_check_is_optional_and_defaults_to_never_firing`
(`tests/test_completion_watch.py`); `test_karmada_terminal_failure_returns_none_when_not_yet_propagated`,
`test_karmada_terminal_failure_returns_none_when_still_running`,
`test_karmada_terminal_failure_returns_none_when_karmada_unreachable`,
`test_karmada_terminal_failure_returns_a_reason_when_terminally_failed`,
`test_launch_and_watch_pod_multi_topology_fast_fails_on_terminal_karmada_status`
(`tests/test_orchestration_activities.py`, the last asserting the raise
happens in well under 1s against a 5s `POD_WATCH_TIMEOUT_S`, not after it).
Verified each failed for the right reason (`TypeError`/`ImportError`, and
the fast-fail test timing out at the full 5s) before the fix.

Commit: `fix(orchestration): fast-fail multi-cluster worker wait on terminal
Karmada status`.

### Finding 3 (ignore stale artifacts predating this attempt)

`wait_for_worker_artifact` gained an optional `not_before: datetime | None`
parameter and a pure `_is_fresh(last_modified, not_before)` helper (tolerance
`_CLOCK_SKEW_TOLERANCE_S = 5.0`s). `_launch_and_watch_pod_multi` captures
`not_before = datetime.now(timezone.utc)` **before** calling
`dispatcher.ensure_job(spec)` — ordinary wall-clock time is fine here
because this is activity code (never replayed by Temporal); only *workflow*
code must be deterministic. Both sides of the comparison are timezone-aware
UTC (`last_modified` comes from `minio.time.from_http_header`, which always
attaches `tzinfo`), so this cannot raise the naive/aware `TypeError` the
review flagged — but only because every caller is required to pass an aware
`not_before`.

5s tolerance: `last_modified` is stamped by the **MinIO server's** clock,
not the worker pod's, so what matters is skew between this process and
MinIO, not between this process and the member-cluster worker. NTP-synced
hosts in the same/nearby infra typically drift well under a second; 5s is
generous headroom for that jitter while staying tiny next to how old a
genuinely stale object would be (at least one full training round —
minutes, not seconds).

The reviewer's repro turned into two real tests: pre-seeding the object
before `not_before`/the Job exists returns `TimeoutError`, not instant
success. Tests added: `test_ignores_a_stale_artifact_pre_seeded_before_this_attempt`,
`test_accepts_a_fresh_artifact_at_or_after_not_before`,
`test_accepts_an_artifact_within_clock_skew_tolerance`,
`test_not_before_is_optional_and_defaults_to_no_freshness_check`
(`tests/test_completion_watch.py`); `test_launch_and_watch_pod_multi_topology_ignores_a_preexisting_stale_artifact`
(`tests/test_orchestration_activities.py`, the reviewer's exact repro
end-to-end through `launch_and_watch_pod`). Confirmed the end-to-end test
fails for the right reason by monkeypatching `_is_fresh` to always report
"fresh" (simulating pre-fix behavior): it correctly produced "DID NOT RAISE
WorkerJobFailed".

Commit: `fix(orchestration): ignore stale MinIO artifacts predating the
current attempt`.

### Finding 4 (tolerate connection-level MinIO errors)

Verified against the installed `minio==7.2.20` (no wrapping — `Minio._url_open`
calls `self._http.urlopen(...)` directly): a refused connection raises
`urllib3.exceptions.MaxRetryError`, not `S3Error`. The poll loop in
`wait_for_worker_artifact` now also catches `urllib3.exceptions.MaxRetryError`
and `urllib3.exceptions.ProtocolError` (a connection dropped mid-response —
the other realistic "network blip" shape), logging and continuing to poll
exactly like the existing transient-`S3Error` branch. Deliberately **not**
`except Exception` — a dedicated test (`test_does_not_swallow_a_genuine_bug_in_the_poll_loop`)
asserts an `AttributeError` still propagates unhandled. The overall
`timeout_s` bound is untouched (`test_connection_level_errors_still_respect_the_overall_timeout`
confirms a poll loop that hits the connection error on every single call
still raises `TimeoutError` at the timeout, not later).

Tests added: `test_tolerates_a_connection_level_error_and_keeps_polling`,
`test_tolerates_a_urllib3_protocol_error_and_keeps_polling`,
`test_does_not_swallow_a_genuine_bug_in_the_poll_loop`,
`test_connection_level_errors_still_respect_the_overall_timeout`
(`tests/test_completion_watch.py`). Verified the first two failed for the
right reason (the real `urllib3` exception propagating unhandled) before the
fix; the guardrail test already passed pre-fix (bare exceptions always
propagated) and continues to pass post-fix.

Commit: `fix(orchestration): tolerate connection-level MinIO errors during
polling`.

### Verified unchanged

- `topology='single'`'s `launch_and_watch_pod` body (from the function
  start through the `if spec.topology == "multi":` branch split) is
  byte-for-byte identical before/after all three commits (diffed directly).
- `src/orchestration/dispatch.py` was never touched (0 lines changed across
  all three commits) — Finding 1 remains for the other agent.
- `launch_and_watch_pod` still raises (never returns a failed
  `WorkerResult`) on every multi-topology failure path, including the new
  fast-fail branch, which raises through the same
  `except Exception as e: ... raise WorkerJobFailed(message) from e` wrapper
  already in `_launch_and_watch_pod_multi`.
- `backoffLimit: 0` / `restartPolicy: Never` in `build_job_manifest`
  untouched.
