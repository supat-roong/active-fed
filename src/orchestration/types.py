"""
Plain data types shared by orchestration workflows and activities.

Deliberately free of any import beyond the standard library: Temporal's
workflow sandbox restricts what workflow code may import, and both workflow
and activity code depend on these types.
"""

from __future__ import annotations

import dataclasses
import re

# RFC 1123 label prefix: must start with a lowercase alphanumeric and contain
# only lowercase alphanumerics/'-' after that. Not the full RFC 1123 label
# rule (which also forbids ending in '-') because member_prefix is only ever
# used as a *prefix*: worker_spec() always appends a digit
# (f"{member_prefix}{worker_id % member_count + 1}"), so the concatenated
# member_cluster name always ends in a digit regardless of what member_prefix
# itself ends with. dispatch.py separately re-validates the full concatenated
# name against the complete RFC 1123 rule before it ever reaches a
# PropagationPolicy -- this is defense-in-depth one layer earlier, not the
# only check.
#
# Duplicated here (not imported) from dispatch.py's own copy of the same
# character-class rule: this module must stay import-stdlib-only (Temporal's
# workflow sandbox restricts what workflow code may import), and dispatch.py
# imports `kubernetes` at module scope, so it can never be imported from
# workflow-reachable code. `re` itself is stdlib, so it's safe here.
_VALID_MEMBER_PREFIX = re.compile(r"[a-z0-9][-a-z0-9]*")


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
    # NodePorts exposing the host cluster's MinIO/MLflow Services, only
    # meaningful when topology == "multi": a member-cluster pod cannot resolve
    # minio_endpoint/mlflow_tracking_uri's in-cluster DNS names (they belong to
    # the host cluster's own internal DNS, unreachable from a separate
    # cluster), but every kind cluster in this project's multi-cluster
    # environment shares one Docker bridge network, so a member pod can reach
    # the host's NodePorts via the host's node IP. activities.py's dispatch-
    # time rewrite (_rewrite_endpoints_for_multi) reads these two fields off
    # the WorkerSpec it is about to dispatch and rewrites minio_endpoint/
    # mlflow_tracking_uri to "<host-node-ip>:<nodeport>" before
    # build_job_manifest ever sees the spec. Left as plain ints with a 0
    # default (not validated here, unlike member_count/member_prefix above):
    # topology="single" never reads them at all, and the rewrite itself -- the
    # only code that does read them for topology="multi" -- is exactly where
    # a misconfigured (zero) value is caught and raised on, next to the value.
    minio_nodeport: int = 0
    mlflow_nodeport: int = 0


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
    # NodePorts exposing the host cluster's MinIO/MLflow Services to Karmada
    # member clusters. See WorkerSpec.minio_nodeport/mlflow_nodeport above for
    # why these exist and where they're actually consumed; worker_spec() below
    # just threads them through unchanged, the same as every other
    # RoundSpec-driving field.
    minio_nodeport: int = 0
    mlflow_nodeport: int = 0

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
            minio_nodeport=self.minio_nodeport,
            mlflow_nodeport=self.mlflow_nodeport,
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

        p3-task-3-review.md Finding 2: a bare `if not self.member_prefix:`
        used Python truthiness, so a whitespace-only prefix ("   ", "\\t\\n")
        sailed straight through -- non-empty strings are truthy regardless of
        content. That doesn't hit the catastrophic empty-clusterNames case
        (a whitespace-garbage member_cluster selects zero real clusters, not
        all), but it silently produces a Job that Karmada schedules nowhere,
        stalling the round with the root cause nowhere near the error.
        Rejecting outright (not stripping-and-accepting) is the deliberate
        choice: silently trimming a value that had leading/trailing
        whitespace would just as silently paper over whatever upstream bug
        produced it, instead of surfacing it here where the offending value
        can be quoted directly. The same regex also catches non-blank-but-
        still-invalid prefixes (uppercase, embedded spaces, a leading '-')
        that a bare `.strip()` truthiness check would still let through --
        exactly the "adjacent case" gap the review asked about.
        """
        if self.member_count <= 0:
            raise ValueError(
                f"topology='multi' requires member_count > 0 to assign a member "
                f"cluster to worker {worker_id} (round {self.fl_round}); got "
                f"member_count={self.member_count!r}"
            )
        if not self.member_prefix or not _VALID_MEMBER_PREFIX.fullmatch(self.member_prefix):
            raise ValueError(
                f"topology='multi' requires member_prefix to be a non-empty, "
                f"RFC-1123-style name fragment (lowercase alphanumerics and '-', "
                f"starting with a lowercase alphanumeric) to assign a member "
                f"cluster to worker {worker_id} (round {self.fl_round}); got "
                f"member_prefix={self.member_prefix!r}"
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
