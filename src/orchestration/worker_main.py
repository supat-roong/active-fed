"""
Temporal worker process: registers the workflows and activities and polls.

Runs as a Deployment in the consumer namespace. Its ServiceAccount needs
create/delete on jobs and get/list/watch on pods plus pods/log, because
`launch_and_watch_pod` does exactly those things.
"""

from __future__ import annotations

import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from src.orchestration.activities import cleanup_worker_job, launch_and_watch_pod
from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow, WorkerWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


async def main() -> None:
    address = os.environ.get("TEMPORAL_ADDRESS", "temporal-frontend:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    log.info(f"connecting to Temporal at {address} (namespace={namespace})")

    client = await Client.connect(address, namespace=namespace)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TrainRoundWorkflow, WorkerWorkflow],
        activities=[launch_and_watch_pod, cleanup_worker_job],
    )
    log.info(f"worker started on task queue '{TASK_QUEUE}'")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
