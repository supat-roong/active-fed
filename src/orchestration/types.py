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
    # How many Karmada member clusters worker_spec() round-robins workers
    # across, and the name prefix used to build each member's cluster name
    # (f"{member_prefix}{worker_id % member_count + 1}"). Only meaningful
    # when topology == "multi" -- see worker_spec() below. Task 3 added
    # member_cluster (above) but deliberately left it a single static
    # passthrough value shared by every worker in the round; these two
    # fields are what let worker_spec() compute a *distinct* member per
    # worker instead.
    member_count: int = 0
    member_prefix: str = ""

    def worker_spec(self, worker_id: int) -> WorkerSpec:
        member_cluster = self.member_cluster
        if self.topology == "multi":
            member_cluster = self._member_cluster_for(worker_id)
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
            member_cluster=member_cluster,
        )

    def _member_cluster_for(self, worker_id: int) -> str:
        """Round-robin worker -> member assignment for topology='multi'.

        f"{member_prefix}{worker_id % member_count + 1}" pins consecutive
        worker_ids to distinct member clusters, wrapping once member_count is
        exceeded (3 workers over 2 members -> member1, member2, member1).
        physics_seed derives from worker_id, so one worker per member
        cluster is what gives each member a distinct physical variation --
        the geo-distributed fleet story the multi topology exists to
        demonstrate.

        Raises instead of returning an empty/degenerate name when
        member_count <= 0 or member_prefix is blank -- the same critical
        safety property dispatch.py enforces one layer downstream (an empty
        clusterNames list in a Karmada PropagationPolicy targets ALL
        clusters, not none). Catching it here too means a bad multi spec
        can never even produce a WorkerSpec with an unsafe member_cluster,
        regardless of which entry point a future caller uses.
        """
        if self.member_count <= 0:
            raise ValueError(
                f"topology='multi' requires member_count > 0 to assign a member "
                f"cluster to worker {worker_id} (round {self.fl_round}); got "
                f"member_count={self.member_count!r}"
            )
        if not self.member_prefix:
            raise ValueError(
                f"topology='multi' requires a non-empty member_prefix to assign "
                f"a member cluster to worker {worker_id} (round {self.fl_round}); "
                f"got member_prefix={self.member_prefix!r}"
            )
        return f"{self.member_prefix}{worker_id % self.member_count + 1}"


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
