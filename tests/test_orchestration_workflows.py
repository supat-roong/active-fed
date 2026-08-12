import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from src.orchestration.types import RoundSpec, WorkerResult, WorkerSpec
from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow, WorkerWorkflow


def _round_spec(num_workers=3, min_workers=2) -> RoundSpec:
    return RoundSpec(
        fl_round=0, num_workers=num_workers, min_workers=min_workers, local_episodes=5,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234",
    )


def _worker_spec(worker_id=0) -> WorkerSpec:
    return WorkerSpec(
        fl_round=0, worker_id=worker_id, num_workers=1, local_episodes=5,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234",
    )


def _ok(spec: WorkerSpec) -> WorkerResult:
    return WorkerResult(worker_id=spec.worker_id, succeeded=True, attempts=1,
                        failure_reason="", job_name=f"j{spec.worker_id}")


async def _run(env: WorkflowEnvironment, acts, spec: RoundSpec):
    async with Worker(
        env.client, task_queue=TASK_QUEUE,
        workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=acts,
    ):
        return await env.client.execute_workflow(
            TrainRoundWorkflow.run, spec,
            id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
        )


async def _run_worker(env: WorkflowEnvironment, acts, spec: WorkerSpec):
    async with Worker(
        env.client, task_queue=TASK_QUEUE,
        workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=acts,
    ):
        return await env.client.execute_workflow(
            WorkerWorkflow.run, spec,
            id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
        )


@pytest.mark.asyncio
async def test_all_workers_succeed():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec())
    assert report.succeeded_ids == [0, 1, 2]
    assert report.failed_ids == []


@pytest.mark.asyncio
async def test_partial_failure_above_quorum_still_returns_survivors():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 1:
            raise RuntimeError("pod OOMKilled")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec(min_workers=2))
    assert report.succeeded_ids == [0, 2]
    assert report.failed_ids == [1]
    assert report.meets_quorum(2) is True


@pytest.mark.asyncio
async def test_below_quorum_fails_the_round():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 0:
            return _ok(spec)
        raise RuntimeError("boom")

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    from temporalio.client import WorkflowFailureError

    async with await WorkflowEnvironment.start_time_skipping() as env:
        with pytest.raises(WorkflowFailureError):
            await _run(env, [launch, cleanup], _round_spec(min_workers=2))


@pytest.mark.asyncio
async def test_worker_retries_then_succeeds():
    calls: dict[int, int] = {}

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        calls[spec.worker_id] = calls.get(spec.worker_id, 0) + 1
        if spec.worker_id == 1 and calls[spec.worker_id] == 1:
            raise RuntimeError("transient")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec(min_workers=3))
    assert report.succeeded_ids == [0, 1, 2]
    assert calls[1] == 2  # proves it actually retried rather than passing first time


@pytest.mark.asyncio
async def test_cleanup_runs_for_every_worker():
    cleaned: list[int] = []

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        cleaned.append(spec.worker_id)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        await _run(env, [launch, cleanup], _round_spec())
    assert sorted(cleaned) == [0, 1, 2]


@pytest.mark.asyncio
async def test_cleanup_still_runs_when_worker_activity_raises():
    from temporalio.client import WorkflowFailureError

    cleaned: list[int] = []

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        raise RuntimeError("boom")

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        cleaned.append(spec.worker_id)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        with pytest.raises(WorkflowFailureError):
            await _run_worker(env, [launch, cleanup], _worker_spec(worker_id=7))
    assert cleaned == [7]


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_worker_failure_reason():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return WorkerResult(worker_id=spec.worker_id, succeeded=False, attempts=1,
                            failure_reason="OOMKilled (exit 137)", job_name=f"j{spec.worker_id}")

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        raise RuntimeError("cleanup exploded")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run_worker(env, [launch, cleanup], _worker_spec(worker_id=3))
    assert result.succeeded is False
    assert result.failure_reason == "OOMKilled (exit 137)"


@pytest.mark.asyncio
async def test_round_status_query_reports_populated_map():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client, task_queue=TASK_QUEUE,
            workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=[launch, cleanup],
        ):
            handle = await env.client.start_workflow(
                TrainRoundWorkflow.run, _round_spec(),
                id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )
            await handle.result()
            statuses = await handle.query(TrainRoundWorkflow.status)
    assert set(statuses.keys()) == {0, 1, 2}
    assert all(s.phase == "Succeeded" for s in statuses.values())


@pytest.mark.asyncio
async def test_gather_exception_reports_root_cause_not_generic_wrapper():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 1:
            raise RuntimeError("pod OOMKilled")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec(min_workers=2))

    failed = next(r for r in report.results if r.worker_id == 1)
    assert "OOMKilled" in failed.failure_reason
    assert failed.failure_reason != "Child Workflow execution failed"
