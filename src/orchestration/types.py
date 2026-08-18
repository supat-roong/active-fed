"""
Plain data types shared by orchestration workflows and activities.

Deliberately free of any import beyond the standard library: Temporal's
workflow sandbox restricts what workflow code may import, and both workflow
and activity code depend on these types.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    """Everything one worker needs. Frozen so Temporal replay is deterministic."""

    fl_round: int
    worker_id: int
    num_workers: int
    local_episodes: int
    namespace: str
    worker_image: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    kfp_run_id: str
    # P3: which deployment topology this worker's Job dispatches into.
    # "single" (default) is today's local-cluster behaviour; "multi" means
    # the Job is propagated by Karmada to member_cluster. See dispatch.py.
    topology: str = "single"
    # The Karmada member cluster this worker is pinned to. Only meaningful
    # when topology == "multi" -- and required to be non-empty in that case:
    # an empty clusterNames list in a Karmada PropagationPolicy targets ALL
    # clusters, not none, so dispatch.py's dispatcher_for/build_propagation_
    # policy raise rather than let this default silently fan a worker out to
    # every member.
    member_cluster: str = ""


@dataclasses.dataclass(frozen=True)
class RoundSpec:
    """Everything one federated round needs."""

    fl_round: int
    num_workers: int
    min_workers: int
    local_episodes: int
    namespace: str
    worker_image: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    kfp_run_id: str
    topology: str = "single"
    member_cluster: str = ""

    def worker_spec(self, worker_id: int) -> WorkerSpec:
        return WorkerSpec(
            fl_round=self.fl_round,
            worker_id=worker_id,
            num_workers=self.num_workers,
            local_episodes=self.local_episodes,
            namespace=self.namespace,
            worker_image=self.worker_image,
            minio_endpoint=self.minio_endpoint,
            minio_access_key=self.minio_access_key,
            minio_secret_key=self.minio_secret_key,
            minio_bucket=self.minio_bucket,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            mlflow_experiment_name=self.mlflow_experiment_name,
            kfp_run_id=self.kfp_run_id,
            topology=self.topology,
            member_cluster=self.member_cluster,
        )


@dataclasses.dataclass
class WorkerStatus:
    """Live state of one worker, returned by the workflow query."""

    worker_id: int
    phase: str = "Pending"
    attempt: int = 0
    episodes_done: int = 0
    message: str = ""


@dataclasses.dataclass
class WorkerResult:
    """Terminal outcome of one worker."""

    worker_id: int
    succeeded: bool
    attempts: int
    failure_reason: str
    job_name: str


@dataclasses.dataclass
class RoundReport:
    """Aggregate outcome of one round's fleet."""

    fl_round: int
    results: list[WorkerResult]

    @property
    def succeeded_ids(self) -> list[int]:
        return sorted(r.worker_id for r in self.results if r.succeeded)

    @property
    def failed_ids(self) -> list[int]:
        return sorted(r.worker_id for r in self.results if not r.succeeded)

    def meets_quorum(self, min_workers: int) -> bool:
        return len(self.succeeded_ids) >= min_workers
