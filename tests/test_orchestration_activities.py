from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

from src.orchestration.activities import (
    _await_job_deleted,
    _ensure_job,
    build_job_manifest,
    classify_job_status,
    cleanup_worker_job,
    job_name_for,
    launch_and_watch_pod,
)
from src.orchestration.types import WorkerSpec


def _spec(**overrides) -> WorkerSpec:
    base = dict(
        fl_round=3, worker_id=2, num_workers=4, local_episodes=25,
        namespace="active-fed", worker_image="active-fed-worker:v1",
        minio_endpoint="minio-service:9000", minio_access_key="ak",
        minio_secret_key="sk", minio_bucket="bucket",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234567890",
    )
    base.update(overrides)
    return WorkerSpec(**base)


def test_job_name_is_deterministic_for_the_same_inputs():
    assert job_name_for(_spec()) == job_name_for(_spec())


def test_job_name_distinguishes_worker_and_round():
    assert job_name_for(_spec(worker_id=1)) != job_name_for(_spec(worker_id=2))
    assert job_name_for(_spec(fl_round=1)) != job_name_for(_spec(fl_round=2))


def test_job_name_is_a_valid_kubernetes_name():
    import re
    name = job_name_for(_spec())
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name), name
    assert len(name) <= 63


# F4 (gate fix): the KFP deployment in this environment does not substitute
# dsl.PIPELINE_JOB_ID_PLACEHOLDER before component code runs, so a kfp_run_id
# of the literal, unsubstituted string "{{$.pipeline_job_uuid}}" previously
# reached job_name_for verbatim and produced "aflw-{{$.pipe-r0-w0", which the
# Kubernetes API rejected with a 422 (not a lowercase RFC 1123 subdomain).
# job_name_for must now reject any run id containing characters that can
# never appear in a valid Kubernetes name, at the point the name is built,
# rather than letting a malformed value reach the API and fail there. This is
# what makes this specific class of bug impossible to reship even if some
# future caller passes an unsubstituted placeholder or similar garbage again.
def test_job_name_for_rejects_the_unsubstituted_kfp_placeholder():
    bad = _spec(kfp_run_id="{{$.pipeline_job_uuid}}")
    with pytest.raises(ValueError):
        job_name_for(bad)


@pytest.mark.parametrize("bad_char", ["{", "}", "$", "."])
def test_job_name_for_rejects_each_disallowed_special_character(bad_char):
    bad = _spec(kfp_run_id=f"ab{bad_char}c1234")
    with pytest.raises(ValueError):
        job_name_for(bad)


def test_job_name_for_rejects_uppercase():
    bad = _spec(kfp_run_id="ABCDEF12")
    with pytest.raises(ValueError):
        job_name_for(bad)


def test_job_name_for_accepts_a_valid_lowercase_alphanumeric_run_id():
    # Regression guard for the positive path: the rejection above must not
    # be so strict it also rejects the kind of value run_pipeline.py actually
    # generates (uuid4().hex[:8] -- lowercase hex).
    name = job_name_for(_spec(kfp_run_id="a1b2c3d4"))
    assert name == "aflw-a1b2c3d4-r3-w2"


def test_manifest_sets_rank_to_the_worker_id():
    env = {e["name"]: e["value"] for e in
           build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["RANK"] == "2"


def test_manifest_passes_round_and_episodes_as_args():
    args = build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--fl-round" in args and args[args.index("--fl-round") + 1] == "3"
    assert "--local-episodes" in args and args[args.index("--local-episodes") + 1] == "25"


def test_manifest_carries_minio_and_mlflow_env():
    env = {e["name"]: e["value"] for e in
           build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["MINIO_ENDPOINT"] == "minio-service:9000"
    assert env["MINIO_BUCKET"] == "bucket"
    assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


def test_manifest_uses_never_restart_policy():
    # F5 (gate fix): Temporal now owns retry entirely (backoffLimit=0 below),
    # so an in-place container restart under restartPolicy=OnFailure would be
    # exactly the per-worker-attribution masking that fix removes -- a
    # crashed container would come back to life inside the *same* Pod and
    # Job, silently, before the activity's polling loop ever saw a failure.
    # With restartPolicy=Never, a crashed container fails the Pod outright,
    # which (combined with backoffLimit=0 below) fails the Job outright,
    # which the poll loop below *does* observe as classify_job_status ==
    # "failed". The earlier P0 concern this replaces (a crashed worker
    # vanishing and the aggregator silently proceeding with N-1 clients) is
    # now handled one layer up, by Temporal's own retry/quorum logic instead
    # of Kubernetes'.
    assert build_job_manifest(_spec())["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def test_manifest_sets_backoff_limit_to_zero():
    # F5: a non-zero backoffLimit lets the Job controller silently replace a
    # failed/deleted pod under the Job's own umbrella before Temporal's poll
    # loop ever notices -- exactly the masking bug the gate found. 0 means
    # any single pod failure fails the Job immediately.
    assert build_job_manifest(_spec())["spec"]["backoffLimit"] == 0


def test_manifest_labels_identify_round_and_worker():
    labels = build_job_manifest(_spec())["spec"]["template"]["metadata"]["labels"]
    assert labels["app"] == "active-fl-worker"
    assert labels["fl-round"] == "3"
    assert labels["worker-id"] == "2"


def _condition(type_: str, status: str) -> SimpleNamespace:
    return SimpleNamespace(type=type_, status=status)


def _job_status(conditions=None, succeeded=None, failed=None, active=None) -> SimpleNamespace:
    return SimpleNamespace(conditions=conditions, succeeded=succeeded, failed=failed, active=active)


def test_classify_job_status_succeeded_on_complete_condition():
    status = _job_status(conditions=[_condition("Complete", "True")])
    assert classify_job_status(status, backoff_limit=2) == "succeeded"


def test_classify_job_status_succeeded_via_status_succeeded_without_condition():
    status = _job_status(conditions=[], succeeded=1)
    assert classify_job_status(status, backoff_limit=2) == "succeeded"


def test_classify_job_status_failed_on_failed_condition():
    status = _job_status(conditions=[_condition("Failed", "True")])
    assert classify_job_status(status, backoff_limit=2) == "failed"


def test_classify_job_status_running_when_neither_condition_present():
    status = _job_status(conditions=[])
    assert classify_job_status(status, backoff_limit=2) == "running"


def test_classify_job_status_falls_back_to_failed_count_over_backoff_limit():
    status = _job_status(conditions=[], failed=3)
    assert classify_job_status(status, backoff_limit=2) == "failed"


def test_classify_job_status_failed_condition_wins_even_when_failed_count_is_low():
    # Regression guard: on a live cluster with restartPolicy OnFailure and
    # backoffLimit=2, status.failed was observed to settle at 1 and never
    # exceed backoffLimit, so a decision that only compares
    # status.failed > backoff_limit never fires and a genuine failure gets
    # misreported as a 1-hour timeout. The Failed condition must be
    # authoritative regardless of the failed-pod count.
    status = _job_status(conditions=[_condition("Failed", "True")], failed=1)
    assert classify_job_status(status, backoff_limit=2) == "failed"


# ---------------------------------------------------------------------------
# F5: _ensure_job / _await_job_deleted — the activity-retry-finds-an-existing-
# Job branch. Job names are deterministic (job_name_for), so a Temporal
# activity retry after a real k8s-level failure (backoffLimit=0 now means any
# pod failure fails the whole Job -- see above) hits `create_namespaced_job`
# for a name that already exists (409). Fake, in-memory stand-in for
# kubernetes.client.BatchV1Api -- mirrors the SimpleNamespace-based status
# fakes above (_job_status/_condition), just stateful enough to script a
# create -> [409] -> read -> (delete -> poll-until-gone) -> create sequence
# without a real API server.
# ---------------------------------------------------------------------------
class FakeBatchApi:
    def __init__(self, initial_status=None, reads_until_deleted=0):
        # initial_status=None means "no Job exists yet". A SimpleNamespace
        # (as built by _job_status above) means one already exists in that
        # state, simulating a prior activity attempt's Job still being there.
        self.status = initial_status
        self.reads_until_deleted = reads_until_deleted
        self.calls: list[str] = []

    def create_namespaced_job(self, namespace, body):
        self.calls.append("create")
        if self.status is not None:
            raise ApiException(status=409)
        self.status = _job_status()  # freshly created: running, no conditions yet

    def read_namespaced_job_status(self, name, namespace):
        self.calls.append("read")
        if self.status is None:
            raise ApiException(status=404)
        return SimpleNamespace(status=self.status)

    def delete_namespaced_job(self, name, namespace, propagation_policy):
        self.calls.append(f"delete:{propagation_policy}")
        # Deletion is async in real Kubernetes -- the object lingers through
        # `reads_until_deleted` more reads before it actually disappears, so
        # tests can prove the wait loop actually polls instead of assuming
        # instant deletion.
        self._pending = self.reads_until_deleted

    def _tick_pending_delete(self):
        if self.status is not None and hasattr(self, "_pending"):
            if self._pending <= 0:
                self.status = None
            else:
                self._pending -= 1


class PollingFakeBatchApi(FakeBatchApi):
    """Like FakeBatchApi, but read_namespaced_job_status ticks the pending
    deletion countdown on every read, so a deleted Job actually disappears
    only after `reads_until_deleted` subsequent reads."""

    def read_namespaced_job_status(self, name, namespace):
        self._tick_pending_delete()
        return super().read_namespaced_job_status(name, namespace)


def _spec_for_ensure_job() -> WorkerSpec:
    return _spec()


def _manifest_for_ensure_job() -> dict:
    return build_job_manifest(_spec_for_ensure_job())


async def test_ensure_job_creates_when_absent():
    batch = FakeBatchApi(initial_status=None)
    await _ensure_job(batch, _spec_for_ensure_job(), _manifest_for_ensure_job())
    assert batch.calls == ["create"]
    assert batch.status is not None


async def test_ensure_job_reattaches_to_a_still_active_existing_job():
    running = _job_status(conditions=[])
    batch = FakeBatchApi(initial_status=running)
    await _ensure_job(batch, _spec_for_ensure_job(), _manifest_for_ensure_job())
    # Re-attach: exactly one create attempt (which 409s), one status read to
    # classify it, and critically no delete -- the whole point of this
    # branch is to leave a genuinely still-running Job alone.
    assert batch.calls == ["create", "read"]
    assert batch.status is running


async def test_ensure_job_deletes_and_recreates_a_terminally_failed_job():
    # reads_until_deleted=0: the deletion is reported gone on the very first
    # poll, so this test exercises the create->read->delete->await->create
    # sequencing without sleeping on _ensure_job's real (multi-second)
    # production poll interval. The wait loop's actual multi-poll behavior
    # (proving it doesn't assume instant deletion) is covered directly by
    # test_await_job_deleted_returns_once_the_job_404s below, with a
    # deliberately tiny poll_s.
    failed = _job_status(conditions=[_condition("Failed", "True")])
    batch = PollingFakeBatchApi(initial_status=failed, reads_until_deleted=0)
    await _ensure_job(batch, _spec_for_ensure_job(), _manifest_for_ensure_job())
    # create (409) -> read (classify as failed) -> delete -> poll reads until
    # gone -> create again. The deletion must fully complete (status becomes
    # None) strictly before the second create, or this would race a
    # half-deleted Job.
    assert batch.calls[0] == "create"
    assert batch.calls[1] == "read"
    assert batch.calls[2] == "delete:Background"
    assert batch.calls.count("create") == 2
    first_delete_idx = batch.calls.index("delete:Background")
    second_create_idx = len(batch.calls) - 1 - batch.calls[::-1].index("create")
    assert second_create_idx > first_delete_idx
    # At least one poll read happened between delete and the recreate.
    assert batch.calls[first_delete_idx + 1 : second_create_idx].count("read") >= 1


async def test_await_job_deleted_returns_once_the_job_404s():
    batch = PollingFakeBatchApi(initial_status=_job_status(), reads_until_deleted=1)
    batch.delete_namespaced_job(name="x", namespace="ns", propagation_policy="Background")
    await _await_job_deleted(batch, "ns", "x", timeout_s=5, poll_s=0.01)
    assert batch.status is None


async def test_await_job_deleted_raises_on_timeout_instead_of_hanging():
    batch = PollingFakeBatchApi(initial_status=_job_status(), reads_until_deleted=10_000)
    batch.delete_namespaced_job(name="x", namespace="ns", propagation_policy="Background")
    with pytest.raises(TimeoutError):
        await _await_job_deleted(batch, "ns", "x", timeout_s=0.05, poll_s=0.01)


# ---------------------------------------------------------------------------
# C1/I1: launch_and_watch_pod itself. Previously untested at any level (I5) --
# these drive the activity end-to-end against fake Batch/Core clients,
# monkeypatching _k8s_batch_and_core (the one seam that needs a real
# kubeconfig) rather than a real cluster.
# ---------------------------------------------------------------------------
class FakeBatchApiTerminal:
    """Job creation succeeds immediately and every subsequent status read
    reports the given terminal status right away -- exercises
    launch_and_watch_pod's succeeded/failed branches on the very first poll,
    without re-exercising _ensure_job's create/409/delete/recreate sequencing
    (already covered directly by the _ensure_job tests above)."""

    def __init__(self, status):
        self._status = status
        self.calls: list[str] = []

    def create_namespaced_job(self, namespace, body):
        self.calls.append("create")

    def read_namespaced_job_status(self, name, namespace):
        self.calls.append("read")
        return SimpleNamespace(status=self._status)


def _terminated_pod(name: str, reason: str, exit_code: int, restart_count: int = 0):
    container_status = SimpleNamespace(
        restart_count=restart_count,
        state=SimpleNamespace(terminated=SimpleNamespace(reason=reason, exit_code=exit_code)),
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        status=SimpleNamespace(container_statuses=[container_status]),
    )


class FakeCoreApi:
    def __init__(self, pods=None, logs=None):
        self.pods = pods or []
        self.logs = logs or {}

    def list_namespaced_pod(self, namespace, label_selector):
        return SimpleNamespace(items=self.pods)

    def read_namespaced_pod_log(self, name, namespace, tail_lines):
        return self.logs.get(name, "")


async def test_launch_and_watch_pod_raises_with_reason_and_log_tail_on_job_failure(monkeypatch):
    # C1: the pod's log is the only evidence of *why* a worker died, and it is
    # captured nowhere on the failure path today -- it must be read here,
    # before the activity returns/raises, since WorkerWorkflow's cleanup
    # deletes the Job (and cascades to the pod) within seconds of that.
    # I1: a Job failure must raise, not return WorkerResult(succeeded=False),
    # because Temporal only retries activities on raised exceptions.
    import src.orchestration.activities as activities_module
    from src.orchestration.activities import WorkerJobFailed

    failed_status = _job_status(conditions=[_condition("Failed", "True")])
    batch = FakeBatchApiTerminal(failed_status)
    core = FakeCoreApi(
        pods=[_terminated_pod("aflw-abcdef12-r3-w2-abc12", "OOMKilled", 137)],
        logs={"aflw-abcdef12-r3-w2-abc12": "Traceback (most recent call last):\nMemoryError\n"},
    )
    monkeypatch.setattr(activities_module, "_k8s_batch_and_core", lambda: (batch, core))

    with pytest.raises(WorkerJobFailed) as exc_info:
        await launch_and_watch_pod(_spec())

    message = str(exc_info.value)
    assert "OOMKilled" in message
    assert "MemoryError" in message


async def test_launch_and_watch_pod_returns_normally_on_success(monkeypatch):
    import src.orchestration.activities as activities_module

    succeeded_status = _job_status(conditions=[_condition("Complete", "True")])
    batch = FakeBatchApiTerminal(succeeded_status)
    core = FakeCoreApi(
        pods=[_terminated_pod("aflw-abcdef12-r3-w2-xyz99", "Completed", 0)],
        logs={"aflw-abcdef12-r3-w2-xyz99": "training complete\n"},
    )
    monkeypatch.setattr(activities_module, "_k8s_batch_and_core", lambda: (batch, core))

    result = await launch_and_watch_pod(_spec())

    assert result.succeeded is True
    assert result.worker_id == 2
    assert result.failure_reason == ""
    assert result.attempts == 1


# ---------------------------------------------------------------------------
# P3 Task 4: topology='multi' wiring. launch_and_watch_pod must dispatch via
# dispatcher_for(spec) and wait on the MinIO completion artifact instead of
# watching the pod directly; cleanup_worker_job must route through the same
# dispatcher so a member-cluster worker's PropagationPolicy gets cleaned up
# too. topology='single' specs (all tests above) must still take the exact
# P1 code path -- the _forbid_k8s_batch_and_core guard below is a deliberate
# regression check that the multi branch never touches the local-cluster
# client, and doubles as a safety net against ever reaching a real
# kubeconfig if the branch selection regresses.
# ---------------------------------------------------------------------------


def _multi_spec(**overrides) -> WorkerSpec:
    return _spec(topology="multi", member_cluster="active-fed-member1", **overrides)


class FakeDispatcher:
    def __init__(self):
        self.ensure_calls: list[int] = []
        self.delete_calls: list[int] = []

    def ensure_job(self, spec):
        self.ensure_calls.append(spec.worker_id)
        return f"fake-job-w{spec.worker_id}"

    def delete_job(self, spec):
        self.delete_calls.append(spec.worker_id)


class FakeMinioClientForActivities:
    """Presence-scripted stand-in for the multi-topology minio client seam."""

    def __init__(self, present: bool):
        self.present = present
        self.calls = 0

    def stat_object(self, bucket, key):
        from minio.error import S3Error

        self.calls += 1
        if self.present:
            return SimpleNamespace(object_name=key)
        raise S3Error(
            response=None, code="NoSuchKey", message="nope", resource=f"/{bucket}/{key}",
            request_id="r", host_id="h",
        )


def _forbid_k8s_batch_and_core(monkeypatch):
    """topology='multi' must never touch _k8s_batch_and_core -- that seam is
    the local-cluster client; dispatching to a member cluster goes through
    dispatcher_for/Karmada instead."""
    import src.orchestration.activities as activities_module

    def _boom():
        raise AssertionError("_k8s_batch_and_core must not be called for topology='multi'")

    monkeypatch.setattr(activities_module, "_k8s_batch_and_core", _boom)


async def test_launch_and_watch_pod_multi_topology_dispatches_and_waits_for_artifact(monkeypatch):
    import src.orchestration.activities as activities_module
    import src.orchestration.dispatch as dispatch_module

    _forbid_k8s_batch_and_core(monkeypatch)
    fake_dispatcher = FakeDispatcher()
    monkeypatch.setattr(dispatch_module, "dispatcher_for", lambda spec: fake_dispatcher)
    fake_minio = FakeMinioClientForActivities(present=True)
    monkeypatch.setattr(activities_module, "_minio_client_for", lambda spec: fake_minio)

    result = await launch_and_watch_pod(_multi_spec())

    assert result.succeeded is True
    assert result.job_name == "fake-job-w2"
    assert result.failure_reason == ""
    assert fake_dispatcher.ensure_calls == [2]
    assert fake_minio.calls == 1


async def test_launch_and_watch_pod_multi_topology_raises_not_returns_on_missing_artifact(
    monkeypatch,
):
    # THE constraint of this task: a multi-cluster worker whose completion
    # artifact never appears must raise -- not return
    # WorkerResult(succeeded=False, ...) -- or Temporal's
    # RetryPolicy(maximum_attempts=3) silently becomes a single attempt,
    # exactly the bug WorkerJobFailed already exists to prevent for the
    # single-topology Job-watch path above.
    import src.orchestration.activities as activities_module
    import src.orchestration.dispatch as dispatch_module
    from src.orchestration.activities import WorkerJobFailed

    _forbid_k8s_batch_and_core(monkeypatch)
    monkeypatch.setattr(activities_module, "POD_WATCH_TIMEOUT_S", 0.03)
    monkeypatch.setattr(activities_module, "POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(activities_module.activity, "heartbeat", lambda *a, **k: None)

    fake_dispatcher = FakeDispatcher()
    monkeypatch.setattr(dispatch_module, "dispatcher_for", lambda spec: fake_dispatcher)
    monkeypatch.setattr(
        activities_module, "_minio_client_for",
        lambda spec: FakeMinioClientForActivities(present=False),
    )
    # Karmada diagnostics are best-effort enrichment only -- make the lookup
    # itself blow up too, and confirm that still doesn't mask the real
    # failure (it must never turn this into a reported success).
    monkeypatch.setattr(
        dispatch_module, "_karmada_clients",
        lambda: (_ for _ in ()).throw(RuntimeError("karmada apiserver unreachable")),
    )

    with pytest.raises(WorkerJobFailed) as exc_info:
        await launch_and_watch_pod(_multi_spec())

    assert "active-fed-member1" in str(exc_info.value)
    assert fake_dispatcher.ensure_calls == [2]


async def test_cleanup_worker_job_multi_topology_deletes_via_dispatcher(monkeypatch):
    import src.orchestration.dispatch as dispatch_module

    _forbid_k8s_batch_and_core(monkeypatch)
    fake_dispatcher = FakeDispatcher()
    monkeypatch.setattr(dispatch_module, "dispatcher_for", lambda spec: fake_dispatcher)

    await cleanup_worker_job(_multi_spec())

    assert fake_dispatcher.delete_calls == [2]


# ---------------------------------------------------------------------------
# Finding 2 (p3-task-4-review.md): a crashed multi-cluster worker was only
# detected via wait_for_worker_artifact's full timeout_s, because nothing
# actively consulted the Karmada aggregated Job status *during* the wait --
# only after it gave up, as best-effort diagnostics (_karmada_failure_reason).
# _karmada_terminal_failure is the fast-fail check now wired into the poll
# loop via wait_for_worker_artifact's failure_check parameter: these tests
# cover its own absent/lagging/failed/unreachable contract directly, plus one
# end-to-end test proving the wiring actually fires through
# launch_and_watch_pod.
# ---------------------------------------------------------------------------


class FakeKarmadaBatchApi404:
    """Simulates a Job Karmada hasn't propagated to the member cluster yet --
    read_namespaced_job_status 404s, exactly like a Job that doesn't exist."""

    def read_namespaced_job_status(self, name, namespace):
        raise ApiException(status=404)


class FakeKarmadaBatchApiStatus:
    """Reports whatever status object it's given, unconditionally."""

    def __init__(self, status):
        self.status = status

    def read_namespaced_job_status(self, name, namespace):
        return SimpleNamespace(status=self.status)


async def test_karmada_terminal_failure_returns_none_when_not_yet_propagated(monkeypatch):
    # THE safety property: absent status (Karmada hasn't propagated the Job
    # yet -- true for every worker in the first few seconds after dispatch)
    # must never be misread as failure.
    import src.orchestration.dispatch as dispatch_module
    from src.orchestration.activities import _karmada_terminal_failure

    monkeypatch.setattr(
        dispatch_module, "_karmada_clients", lambda: (FakeKarmadaBatchApi404(), None)
    )

    result = await _karmada_terminal_failure(_multi_spec(), "fake-job-w2")

    assert result is None


async def test_karmada_terminal_failure_returns_none_when_still_running(monkeypatch):
    # Lagging/in-progress status (no terminal condition yet) must not fail
    # the wait either -- only a definite Failed condition may.
    import src.orchestration.dispatch as dispatch_module
    from src.orchestration.activities import _karmada_terminal_failure

    running = _job_status(conditions=[])
    monkeypatch.setattr(
        dispatch_module, "_karmada_clients", lambda: (FakeKarmadaBatchApiStatus(running), None)
    )

    result = await _karmada_terminal_failure(_multi_spec(), "fake-job-w2")

    assert result is None


async def test_karmada_terminal_failure_returns_none_when_karmada_unreachable(monkeypatch):
    # Best-effort: an unreachable Karmada apiserver (or unset
    # FED_KARMADA_CONFIG) is "unknown", not "failed" -- MinIO polling remains
    # the ground truth and must not be short-circuited by a diagnostics-only
    # lookup failing.
    import src.orchestration.dispatch as dispatch_module
    from src.orchestration.activities import _karmada_terminal_failure

    monkeypatch.setattr(
        dispatch_module, "_karmada_clients",
        lambda: (_ for _ in ()).throw(RuntimeError("karmada apiserver unreachable")),
    )

    result = await _karmada_terminal_failure(_multi_spec(), "fake-job-w2")

    assert result is None


async def test_karmada_terminal_failure_returns_a_reason_when_terminally_failed(monkeypatch):
    import src.orchestration.dispatch as dispatch_module
    from src.orchestration.activities import _karmada_terminal_failure

    failed = _job_status(conditions=[_condition("Failed", "True")])
    monkeypatch.setattr(
        dispatch_module, "_karmada_clients", lambda: (FakeKarmadaBatchApiStatus(failed), None)
    )

    result = await _karmada_terminal_failure(_multi_spec(), "fake-job-w2")

    assert result is not None
    assert "failed" in result.lower()


async def test_launch_and_watch_pod_multi_topology_fast_fails_on_terminal_karmada_status(
    monkeypatch,
):
    # End-to-end: the wiring from launch_and_watch_pod through
    # wait_for_worker_artifact's failure_check must actually fire, raising
    # long before POD_WATCH_TIMEOUT_S (kept comparatively large here) elapses.
    import time

    import src.orchestration.activities as activities_module
    import src.orchestration.dispatch as dispatch_module
    from src.orchestration.activities import WorkerJobFailed

    _forbid_k8s_batch_and_core(monkeypatch)
    monkeypatch.setattr(activities_module, "POD_WATCH_TIMEOUT_S", 5)
    monkeypatch.setattr(activities_module, "POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(activities_module.activity, "heartbeat", lambda *a, **k: None)

    fake_dispatcher = FakeDispatcher()
    monkeypatch.setattr(dispatch_module, "dispatcher_for", lambda spec: fake_dispatcher)
    monkeypatch.setattr(
        activities_module, "_minio_client_for",
        lambda spec: FakeMinioClientForActivities(present=False),
    )
    failed = _job_status(conditions=[_condition("Failed", "True")])
    monkeypatch.setattr(
        dispatch_module, "_karmada_clients", lambda: (FakeKarmadaBatchApiStatus(failed), None)
    )

    start = time.monotonic()
    with pytest.raises(WorkerJobFailed):
        await launch_and_watch_pod(_multi_spec())
    elapsed = time.monotonic() - start

    # The whole point of Finding 2: nowhere near the 5s POD_WATCH_TIMEOUT_S,
    # let alone the real 3600s default.
    assert elapsed < 1.0
