"""
I2: when a round fails quorum, TrainRoundWorkflow's per-worker report must
still reach the KFP artifact and stdout. Before this fix, `await
handle.result()` (active_fl_pipeline.py:100) raised WorkflowFailureError and
everything downstream -- the print and the worker_report artifact write --
was skipped, so the one scenario this phase exists to illuminate produced no
artifact at all.

These tests drive `train_workers.python_func` (the KFP component's raw
function, unwrapped) directly, faking the Temporal client so no real
Temporal server is needed.
"""

import json
from types import SimpleNamespace

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError

from src.orchestration.types import WorkerStatus
from src.pipelines.active_fl_pipeline import train_workers


class _FakeHandle:
    def __init__(self, statuses, cause, report=None):
        self.id = "train-abcdef12-r0"
        self._statuses = statuses
        self._cause = cause
        self._report = report

    async def result(self):
        if self._cause is not None:
            raise WorkflowFailureError(cause=self._cause)
        return self._report

    async def query(self, query_fn, *args, **kwargs):
        return self._statuses


class _FakeClient:
    def __init__(self, handle):
        self._handle = handle

    async def start_workflow(self, *args, **kwargs):
        return self._handle


def _base_kwargs(worker_report) -> dict:
    return dict(
        fl_round=0, num_workers=2, min_workers=2, local_episodes=5,
        namespace="ns", worker_launcher="temporal", temporal_address="temporal:7233",
        kfp_run_id="abcdef1234", mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="exp", minio_endpoint="minio:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        worker_image="img:v1", worker_report=worker_report,
    )


def test_worker_report_artifact_written_when_round_fails_quorum(tmp_path, monkeypatch):
    statuses = {
        0: WorkerStatus(worker_id=0, phase="Succeeded", attempt=1, message=""),
        1: WorkerStatus(worker_id=1, phase="Failed", attempt=1,
                        message="worker 1 job aflw-abcdef12-r0-w1 failed: OOMKilled (exit 137)"),
    }
    cause = ApplicationError(
        "round 0: only 1 of 2 workers succeeded, need 2", non_retryable=True
    )
    handle = _FakeHandle(statuses, cause)

    async def fake_connect(target_host, **kwargs):
        return _FakeClient(handle)

    import temporalio.client
    monkeypatch.setattr(temporalio.client.Client, "connect", fake_connect)

    report_path = tmp_path / "worker_report.json"
    worker_report = SimpleNamespace(path=str(report_path))

    # The component must still fail the KFP task -- the fix must not swallow
    # the error just to make the artifact appear.
    with pytest.raises(Exception):  # noqa: B017 - WorkflowFailureError re-raised as-is
        train_workers.python_func(**_base_kwargs(worker_report))

    assert report_path.exists(), (
        "worker_report artifact must be written even when the round fails "
        "quorum -- this is exactly the scenario per-worker attribution "
        "exists to illuminate"
    )
    payload = json.loads(report_path.read_text())
    assert payload["succeeded"] == [0]
    assert payload["failed"] == [1]
    assert "OOMKilled" in json.dumps(payload)


def test_worker_report_artifact_still_written_on_success(tmp_path, monkeypatch):
    # Regression guard: the happy path must keep working unchanged.
    from src.orchestration.types import RoundReport, WorkerResult

    report = RoundReport(
        fl_round=0,
        results=[
            WorkerResult(worker_id=0, succeeded=True, attempts=1, failure_reason="", job_name="j0"),
            WorkerResult(worker_id=1, succeeded=True, attempts=1, failure_reason="", job_name="j1"),
        ],
    )
    handle = _FakeHandle(statuses={}, cause=None, report=report)

    async def fake_connect(target_host, **kwargs):
        return _FakeClient(handle)

    import temporalio.client
    monkeypatch.setattr(temporalio.client.Client, "connect", fake_connect)

    report_path = tmp_path / "worker_report.json"
    worker_report = SimpleNamespace(path=str(report_path))

    train_workers.python_func(**_base_kwargs(worker_report))

    payload = json.loads(report_path.read_text())
    assert payload["succeeded"] == [0, 1]
    assert payload["failed"] == []
