"""
P3 Task 4: MinIO-based completion detection for topology='multi' workers.

The host cannot reliably watch a pod Karmada has propagated to a member
cluster (see src/orchestration/dispatch.py), but train_worker.py already
uploads weights, then a delta, then a metrics JSON per worker per round
(_push_weights) -- and the aggregator (src/aggregator/collect.py) already
treats the metrics object's presence as "this worker is done". These tests
cover wait_for_worker_artifact, the function that turns that same signal into
launch_and_watch_pod's multi-cluster completion check.

The subtlety that matters: weights land *before* metrics, so only the
metrics key may ever be read as completion -- see
test_does_not_treat_the_weights_object_alone_as_completion below.
"""

from types import SimpleNamespace

import pytest
from minio.error import S3Error

from src.orchestration.activities import wait_for_worker_artifact

FL_ROUND = 4
WORKER_ID = 1
BUCKET = "bucket"
METRICS_KEY = f"round_{FL_ROUND}/workers/worker_{WORKER_ID}_metrics.json"
WEIGHTS_KEY = f"round_{FL_ROUND}/workers/worker_{WORKER_ID}_weights.pt"


def _s3_error(code: str) -> S3Error:
    return S3Error(
        response=None, code=code, message="test", resource=f"/{BUCKET}/x",
        request_id="req-1", host_id="host-1",
    )


class FakeMinioClient:
    """Scripted stand-in for minio.Minio's stat_object, per test.

    - `keys`: object keys considered "present", but only once the call count
      reaches `present_from_call` (default: present from the very first
      call) -- lets a test simulate an object landing after N polls.
    - `error_calls`: {call_number: S3Error code} to raise instead of the
      normal present/absent check on that specific call, for scripting
      transient (non-"not found") errors.
    """

    def __init__(self, keys=frozenset(), present_from_call=1, error_calls=None):
        self.keys = set(keys)
        self.present_from_call = present_from_call
        self.error_calls = dict(error_calls or {})
        self.calls = 0
        self.queried_keys: list[str] = []

    def stat_object(self, bucket, key):
        self.calls += 1
        self.queried_keys.append(key)
        if self.calls in self.error_calls:
            raise _s3_error(self.error_calls[self.calls])
        if key in self.keys and self.calls >= self.present_from_call:
            return SimpleNamespace(object_name=key, bucket_name=bucket)
        raise _s3_error("NoSuchKey")


def _patch_heartbeat(monkeypatch):
    """wait_for_worker_artifact calls activity.heartbeat() every failed poll.
    These tests call it directly (not through a live Temporal activity), so
    without this, any test that polls more than once would hit
    `RuntimeError: Not in activity context`. Returns the list of recorded
    calls so tests can assert heartbeat cadence."""
    import src.orchestration.activities as activities_module

    calls: list[tuple] = []
    monkeypatch.setattr(
        activities_module.activity, "heartbeat", lambda *a, **k: calls.append((a, k))
    )
    return calls


async def test_returns_true_as_soon_as_the_metrics_object_appears(monkeypatch):
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={METRICS_KEY})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01
    )

    assert result is True
    assert client.calls == 1  # succeeded on the very first poll -- no waiting


async def test_polls_the_exact_metrics_key_train_worker_uploads(monkeypatch):
    # Regression guard for the key format itself, matching train_worker.py's
    # _push_weights and collect.py's collect_worker_updates.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={METRICS_KEY})

    await wait_for_worker_artifact(client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01)

    assert client.queried_keys == [METRICS_KEY]


async def test_raises_timeout_error_instead_of_returning_false(monkeypatch):
    # THE constraint: Temporal only retries an activity on a *raised*
    # exception (see WorkerJobFailed's docstring in activities.py). A MinIO-
    # completion timeout that returned False instead of raising would
    # silently turn RetryPolicy(maximum_attempts=3) into a single attempt for
    # every multi-cluster worker whose artifact never lands -- the exact bug
    # that shipped once already for the single-topology Job-watch path.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys=set())  # metrics object never shows up

    with pytest.raises(TimeoutError):
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=0.03, poll_s=0.01
        )


async def test_heartbeats_on_every_failed_poll(monkeypatch):
    calls = _patch_heartbeat(monkeypatch)
    # Present only from the 4th call onward: 3 failed polls (each must
    # heartbeat) before the 4th call succeeds without an additional one.
    client = FakeMinioClient(keys={METRICS_KEY}, present_from_call=4)

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.001
    )

    assert result is True
    assert client.calls == 4
    assert len(calls) == 3


async def test_tolerates_a_transient_s3_error_and_keeps_polling(monkeypatch):
    _patch_heartbeat(monkeypatch)
    # Calls 1-2 raise a transient (non-"not found") S3Error; call 3 finds the
    # object. A loop that aborted on the first non-NoSuchKey S3Error would
    # never reach that successful 3rd call.
    client = FakeMinioClient(keys={METRICS_KEY}, error_calls={1: "InternalError", 2: "SlowDown"})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.001
    )

    assert result is True
    assert client.calls == 3


async def test_does_not_treat_the_weights_object_alone_as_completion(monkeypatch):
    # THE subtlety: train_worker.py's _push_weights uploads weights, then the
    # delta, then the metrics JSON, in that order -- so only the metrics
    # object's appearance means "done". Treating worker_i_weights.pt as
    # completion would read a partial (weights-uploaded-but-not-finished)
    # worker as a success.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={WEIGHTS_KEY})  # weights present, metrics never uploaded

    with pytest.raises(TimeoutError):
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=0.03, poll_s=0.01
        )

    # It must be the metrics key specifically being polled for -- never
    # satisfied merely because the weights key happens to already exist.
    assert client.queried_keys
    assert all(k == METRICS_KEY for k in client.queried_keys)
