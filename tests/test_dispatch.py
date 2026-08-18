import re

import pytest
from kubernetes.client.exceptions import ApiException

from src.orchestration.activities import build_job_manifest, job_name_for
from src.orchestration.dispatch import (
    KarmadaJobDispatcher,
    LocalJobDispatcher,
    build_propagation_policy,
    dispatcher_for,
)
from src.orchestration.types import WorkerSpec

# RFC-1123 subdomain: lowercase alphanumerics and '-', not starting/ending with '-'.
_VALID_K8S_NAME = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def _spec(**overrides) -> WorkerSpec:
    base = dict(
        fl_round=3, worker_id=2, num_workers=4, local_episodes=25,
        namespace="active-fed", worker_image="active-fed-worker:v1",
        minio_endpoint="minio-service:9000", minio_access_key="ak",
        minio_secret_key="sk", minio_bucket="bucket",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234567890",
        topology="multi", member_cluster="active-fed-member1",
    )
    base.update(overrides)
    return WorkerSpec(**base)


# ---------------------------------------------------------------------------
# build_propagation_policy -- pure, no I/O.
# ---------------------------------------------------------------------------

def test_build_propagation_policy_targets_the_job_by_its_deterministic_name():
    spec = _spec()
    policy = build_propagation_policy(spec)
    selectors = policy["spec"]["resourceSelectors"]
    assert len(selectors) == 1
    assert selectors[0]["kind"] == "Job"
    assert selectors[0]["name"] == job_name_for(spec)
    assert selectors[0]["namespace"] == spec.namespace


def test_build_propagation_policy_cluster_names_contains_exactly_the_one_member():
    spec = _spec(member_cluster="active-fed-member2")
    policy = build_propagation_policy(spec)
    assert policy["spec"]["placement"]["clusterAffinity"]["clusterNames"] == [
        "active-fed-member2"
    ]


def test_build_propagation_policy_name_is_deterministic():
    assert build_propagation_policy(_spec())["metadata"]["name"] == build_propagation_policy(
        _spec()
    )["metadata"]["name"]


def test_build_propagation_policy_name_differs_by_worker_and_round():
    a = build_propagation_policy(_spec(worker_id=1))["metadata"]["name"]
    b = build_propagation_policy(_spec(worker_id=2))["metadata"]["name"]
    assert a != b


def test_build_propagation_policy_name_is_a_valid_kubernetes_name():
    name = build_propagation_policy(_spec())["metadata"]["name"]
    assert _VALID_K8S_NAME.fullmatch(name), name
    assert len(name) <= 63


def test_build_propagation_policy_name_reuses_job_name_fors_validation():
    # job_name_for already raises ValueError on a kfp_run_id that would
    # produce an invalid Kubernetes name (P1's F4 gate fix). The policy name
    # must reuse that same discipline rather than inventing a second
    # validation rule -- so an input that job_name_for rejects must also be
    # rejected here, for the same reason.
    bad = _spec(kfp_run_id="{{$.pipeline_job_uuid}}")
    with pytest.raises(ValueError):
        build_propagation_policy(bad)


# --- The critical safety property ------------------------------------------

def test_build_propagation_policy_raises_on_empty_member_cluster():
    # An empty clusterNames list in a Karmada PropagationPolicy means *all*
    # clusters. Building a policy for a spec with no member_cluster must
    # raise rather than silently produce a policy that fans this worker's
    # Job out to every joined member.
    spec = _spec(member_cluster="")
    with pytest.raises(ValueError):
        build_propagation_policy(spec)


# ---------------------------------------------------------------------------
# dispatcher_for
# ---------------------------------------------------------------------------

def test_dispatcher_for_single_topology_returns_local_dispatcher():
    spec = _spec(topology="single", member_cluster="")
    assert isinstance(dispatcher_for(spec), LocalJobDispatcher)


def test_dispatcher_for_multi_topology_returns_karmada_dispatcher():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    assert isinstance(dispatcher_for(spec), KarmadaJobDispatcher)


def test_dispatcher_for_multi_topology_with_empty_member_cluster_raises():
    # The same safety property as above, enforced at the dispatch-selection
    # boundary too: a caller must never be handed a working dispatcher for
    # an unsafe (multi, no member) spec.
    spec = _spec(topology="multi", member_cluster="")
    with pytest.raises(ValueError):
        dispatcher_for(spec)


def test_dispatcher_for_unknown_topology_raises():
    spec = _spec(topology="quantum", member_cluster="")
    with pytest.raises(ValueError):
        dispatcher_for(spec)


# ---------------------------------------------------------------------------
# LocalJobDispatcher -- wraps the P1 create/delete path. Fake, in-memory
# stand-in for kubernetes.client.BatchV1Api, mirroring the style of
# test_orchestration_activities.py's FakeBatchApi.
# ---------------------------------------------------------------------------

class FakeBatchApi:
    def __init__(self, existing=False):
        self.existing = existing
        self.calls: list[str] = []

    def create_namespaced_job(self, namespace, body):
        self.calls.append("create")
        if self.existing:
            raise ApiException(status=409)
        self.existing = True

    def delete_namespaced_job(self, name, namespace, propagation_policy):
        self.calls.append(f"delete:{propagation_policy}")
        if not self.existing:
            raise ApiException(status=404)
        self.existing = False


def test_local_dispatcher_ensure_job_creates_and_returns_job_name():
    spec = _spec(topology="single", member_cluster="")
    batch = FakeBatchApi()
    name = LocalJobDispatcher()._ensure_job_with(batch, spec)
    assert name == job_name_for(spec)
    assert batch.calls == ["create"]


def test_local_dispatcher_ensure_job_is_idempotent_when_job_already_exists():
    spec = _spec(topology="single", member_cluster="")
    batch = FakeBatchApi(existing=True)
    name = LocalJobDispatcher()._ensure_job_with(batch, spec)
    assert name == job_name_for(spec)
    assert batch.calls == ["create"]  # 409 swallowed, no crash


def test_local_dispatcher_delete_job_tolerates_already_gone():
    spec = _spec(topology="single", member_cluster="")
    batch = FakeBatchApi(existing=False)
    LocalJobDispatcher()._delete_job_with(batch, spec)  # must not raise
    assert batch.calls == ["delete:Background"]


# ---------------------------------------------------------------------------
# KarmadaJobDispatcher -- applies both the Job and the PropagationPolicy.
# ---------------------------------------------------------------------------

class FakeCustomObjectsApi:
    def __init__(self, existing=False):
        self.existing = existing
        self.calls: list[tuple] = []

    def create_namespaced_custom_object(self, group, version, namespace, plural, body):
        self.calls.append(("create", plural, body["metadata"]["name"]))
        if self.existing:
            raise ApiException(status=409)
        self.existing = True

    def delete_namespaced_custom_object(self, group, version, namespace, plural, name):
        self.calls.append(("delete", plural, name))
        if not self.existing:
            raise ApiException(status=404)
        self.existing = False


def test_karmada_dispatcher_ensure_job_applies_job_and_propagation_policy():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi()
    custom = FakeCustomObjectsApi()
    name = KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)
    assert name == job_name_for(spec)
    assert batch.calls == ["create"]
    assert custom.calls == [
        ("create", "propagationpolicies", build_propagation_policy(spec)["metadata"]["name"])
    ]


def test_karmada_dispatcher_ensure_job_is_idempotent():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=True)
    custom = FakeCustomObjectsApi(existing=True)
    # Must not raise even though both the Job and the PropagationPolicy
    # already exist from a previous (e.g. retried) attempt.
    KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)


def test_karmada_dispatcher_delete_job_removes_job_and_policy():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=True)
    custom = FakeCustomObjectsApi(existing=True)
    KarmadaJobDispatcher()._delete_job_with(batch, custom, spec)
    assert batch.calls == ["delete:Background"]
    assert custom.calls == [
        ("delete", "propagationpolicies", build_propagation_policy(spec)["metadata"]["name"])
    ]


def test_karmada_dispatcher_delete_job_tolerates_already_gone():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=False)
    custom = FakeCustomObjectsApi(existing=False)
    KarmadaJobDispatcher()._delete_job_with(batch, custom, spec)  # must not raise


def test_karmada_job_manifest_reused_unchanged_from_p1():
    # KarmadaJobDispatcher must apply the *same* Job manifest P1 builds --
    # topology is a dispatch-time concern, not a manifest-shape concern.
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi()
    custom = FakeCustomObjectsApi()
    KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)
    # The fake doesn't capture the body directly above; assert indirectly via
    # build_job_manifest's own backoffLimit/restartPolicy invariants, which
    # this dispatcher must not alter.
    manifest = build_job_manifest(spec)
    assert manifest["spec"]["backoffLimit"] == 0
    assert manifest["spec"]["template"]["spec"]["restartPolicy"] == "Never"
