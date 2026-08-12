"""
Temporal workflows owning the worker fleet within one federated round.

Scope boundary: KFP sequences rounds and owns the DAG and artifact lineage;
these workflows own the fleet inside a round. The two never overlap, so there
is no two-schedulers conflict.

Workflow code is replayed by Temporal and must stay deterministic — no I/O, no
clocks, no randomness. Everything with a side effect is an activity.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from src.orchestration.types import (
        RoundReport,
        RoundSpec,
        WorkerResult,
        WorkerSpec,
        WorkerStatus,
    )

TASK_QUEUE = "active-fed"


@workflow.defn
class WorkerWorkflow:
    """One durable entity per (round, worker).

    Gives per-pod retry, a per-pod failure reason in Temporal history, and a
    queryable live status the dashboard and the KFP component both read.
    """

    def __init__(self) -> None:
        self._status = WorkerStatus(worker_id=-1)

    @workflow.run
    async def run(self, spec: WorkerSpec) -> WorkerResult:
        self._status = WorkerStatus(worker_id=spec.worker_id, phase="Pending")
        try:
            result: WorkerResult = await workflow.execute_activity(
                "launch_and_watch_pod",
                spec,
                result_type=WorkerResult,
                start_to_close_timeout=timedelta(seconds=3900),
                heartbeat_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=10),
                ),
            )
            self._status.phase = "Succeeded" if result.succeeded else "Failed"
            self._status.message = result.failure_reason
            return result
        finally:
            # Runs on success, failure and cancellation. Without it a failed
            # round leaves orphaned Jobs that collide with the next attempt's
            # deterministic names.
            await workflow.execute_activity(
                "cleanup_worker_job",
                spec,
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

    @workflow.query
    def status(self) -> WorkerStatus:
        return self._status


@workflow.defn
class TrainRoundWorkflow:
    """Fans out one WorkerWorkflow child per worker and gathers the outcomes."""

    def __init__(self) -> None:
        self._statuses: dict[int, WorkerStatus] = {}

    @workflow.run
    async def run(self, spec: RoundSpec) -> RoundReport:
        parent_id = workflow.info().workflow_id

        async def _one(worker_id: int) -> WorkerResult:
            return await workflow.execute_child_workflow(
                WorkerWorkflow.run,
                spec.worker_spec(worker_id),
                id=f"{parent_id}-w{worker_id}",
                task_queue=TASK_QUEUE,
            )

        # return_exceptions=True so one dead worker does not abort the fleet;
        # the quorum check below decides whether the round can still proceed.
        raw = await asyncio.gather(
            *[_one(i) for i in range(spec.num_workers)], return_exceptions=True
        )

        results: list[WorkerResult] = []
        for worker_id, item in enumerate(raw):
            if isinstance(item, BaseException):
                results.append(
                    WorkerResult(
                        worker_id=worker_id, succeeded=False, attempts=0,
                        failure_reason=str(item), job_name="",
                    )
                )
            else:
                results.append(item)

        report = RoundReport(fl_round=spec.fl_round, results=results)
        if not report.meets_quorum(spec.min_workers):
            raise ApplicationError(
                f"round {spec.fl_round}: only {len(report.succeeded_ids)} of "
                f"{spec.num_workers} workers succeeded, need {spec.min_workers}",
                non_retryable=True,
            )
        return report

    @workflow.query
    def status(self) -> dict[int, WorkerStatus]:
        return self._statuses
