"""
Dispatch abstraction: create/delete a worker Job either in the local cluster
(P1's behaviour) or, via Karmada, pinned to exactly one member cluster.

Split out of activities.py rather than folded into it: `build_propagation_policy`
must stay pure and importable without a kubeconfig, exactly like
`build_job_manifest`/`job_name_for` in activities.py, so it can be unit-tested
directly. Anything that talks to a Kubernetes/Karmada apiserver is imported
lazily inside a function, following the same discipline as
`activities._k8s_batch_and_core`.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from src.orchestration.activities import build_job_manifest, job_name_for
from src.orchestration.types import WorkerSpec

log = logging.getLogger(__name__)

# RFC 1123 label: lowercase alphanumerics and '-', not starting/ending with
# '-'. Kubernetes/Karmada cluster names must satisfy this -- matches
# job_name_for's own validation discipline (activities.py's
# _VALID_RUN_ID_FRAGMENT) applied to the other name this module builds.
_VALID_CLUSTER_NAME = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def _require_valid_cluster_name(member_cluster: str, context: str) -> None:
    """Raise ValueError unless member_cluster is a non-blank, RFC-1123-valid
    Kubernetes/Karmada cluster name.

    p3-task-3-review.md Finding 2: both call sites below used to check only
    `if not spec.member_cluster:`, Python truthiness -- so "   " / "\t\n"
    sailed through (a non-empty whitespace string is truthy). That doesn't
    reach the catastrophic empty-`clusterNames` case (an empty string still
    raises; a whitespace-garbage name is merely non-empty), but a
    PropagationPolicy with a `clusterNames` entry that matches no real
    joined member is accepted by the Karmada apiserver and silently selects
    *zero* clusters -- the Job schedules nowhere, and the round stalls for
    the full watch timeout with nothing pointing at a blank/garbage config
    value as the cause.

    Rejects outright rather than stripping-and-accepting: a caller that
    silently trims a value with leading/trailing whitespace would just as
    silently launder whatever upstream bug produced it (a stray newline from
    a shell substitution, a YAML block scalar, ...) instead of surfacing it
    here, next to the offending value. Checking the full RFC 1123 label
    format -- not just blankness -- also catches the same class of
    non-blank-but-still-broken input a bare `.strip()` truthiness check
    would still miss (uppercase, embedded spaces, a leading/trailing '-'):
    any of these selects zero clusters exactly like whitespace does, so they
    get the same treatment.
    """
    if not member_cluster or not _VALID_CLUSTER_NAME.fullmatch(member_cluster):
        raise ValueError(
            f"{context} requires member_cluster to be a valid Kubernetes/Karmada "
            f"cluster name (RFC 1123 label: lowercase alphanumerics and '-', not "
            f"starting/ending with '-'); got {member_cluster!r}"
        )


class JobDispatcher(Protocol):
    def ensure_job(self, spec: WorkerSpec) -> str:
        """Create the worker Job if absent. Idempotent. Returns the Job name."""
        ...

    def delete_job(self, spec: WorkerSpec) -> None:
        """Delete the worker Job (and anything dispatch created alongside it).

        Safe to call when already gone.
        """
        ...


def propagation_policy_name_for(spec: WorkerSpec) -> str:
    """Deterministic PropagationPolicy name for one worker's Job.

    Derived from job_name_for(spec) rather than validated independently:
    job_name_for already raises ValueError on a kfp_run_id fragment that
    would produce an invalid Kubernetes name (P1's F4 gate fix). Reusing it
    here means the policy name inherits that validation instead of a second,
    parallel naming rule.
    """
    return f"{job_name_for(spec)}-pp"


def build_propagation_policy(spec: WorkerSpec) -> dict:
    """Render the Karmada PropagationPolicy that pins one worker's Job to one
    member cluster. Pure -- no I/O, directly unit-testable.

    CRITICAL SAFETY PROPERTY: an empty `clusterNames` list in a Karmada
    PropagationPolicy targets *all* clusters, not none. A spec with an empty
    member_cluster must never reach as far as a rendered policy silently --
    it would propagate this worker's Job to every joined member, running N
    times the intended work on the wrong physics seeds. Raise instead of
    defaulting clusterNames to empty.
    """
    _require_valid_cluster_name(
        spec.member_cluster,
        f"build_propagation_policy (worker {spec.worker_id}, round {spec.fl_round})",
    )

    return {
        "apiVersion": "policy.karmada.io/v1alpha1",
        "kind": "PropagationPolicy",
        "metadata": {
            "name": propagation_policy_name_for(spec),
            "namespace": spec.namespace,
        },
        "spec": {
            "resourceSelectors": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": job_name_for(spec),
                    "namespace": spec.namespace,
                }
            ],
            "placement": {
                "clusterAffinity": {
                    "clusterNames": [spec.member_cluster],
                }
            },
        },
    }


def dispatcher_for(spec: WorkerSpec) -> JobDispatcher:
    """Pick the dispatcher matching spec.topology.

    Refuses (raises ValueError) rather than guessing for a "multi" spec with
    no member_cluster, and for any topology value that isn't recognised --
    the same critical safety property build_propagation_policy enforces,
    caught here too so a caller never even gets a working dispatcher for an
    unsafe spec.
    """
    if spec.topology == "single":
        return LocalJobDispatcher()
    if spec.topology == "multi":
        _require_valid_cluster_name(
            spec.member_cluster,
            f"topology='multi' (worker {spec.worker_id}, round {spec.fl_round})",
        )
        return KarmadaJobDispatcher()
    raise ValueError(f"unknown topology {spec.topology!r}; expected 'single' or 'multi'")


class LocalJobDispatcher:
    """Today's (P1) behaviour: create/delete a batch/v1 Job in the local cluster."""

    def ensure_job(self, spec: WorkerSpec) -> str:
        batch, _ = _k8s_batch_and_core()
        return self._ensure_job_with(batch, spec)

    def delete_job(self, spec: WorkerSpec) -> None:
        batch, _ = _k8s_batch_and_core()
        self._delete_job_with(batch, spec)

    def _ensure_job_with(self, batch, spec: WorkerSpec) -> str:
        from kubernetes.client.exceptions import ApiException

        manifest = build_job_manifest(spec)
        name = manifest["metadata"]["name"]
        try:
            batch.create_namespaced_job(namespace=spec.namespace, body=manifest)
        except ApiException as e:
            if e.status != 409:  # 409 = already exists; re-attach
                raise
        return name

    def _delete_job_with(self, batch, spec: WorkerSpec) -> None:
        from kubernetes.client.exceptions import ApiException

        try:
            batch.delete_namespaced_job(
                name=job_name_for(spec), namespace=spec.namespace,
                propagation_policy="Background",
            )
        except ApiException as e:
            if e.status != 404:
                raise


class KarmadaJobDispatcher:
    """Applies the worker Job and its PropagationPolicy to the Karmada apiserver.

    Uses the same Job manifest (build_job_manifest) P1 builds for the local
    cluster -- topology is a dispatch-time concern, not a manifest-shape one.
    """

    def ensure_job(self, spec: WorkerSpec) -> str:
        batch, custom = _karmada_clients()
        return self._ensure_job_with(batch, custom, spec)

    def delete_job(self, spec: WorkerSpec) -> None:
        batch, custom = _karmada_clients()
        self._delete_job_with(batch, custom, spec)

    def _ensure_job_with(self, batch, custom, spec: WorkerSpec) -> str:
        from kubernetes.client.exceptions import ApiException

        manifest = build_job_manifest(spec)
        name = manifest["metadata"]["name"]
        try:
            batch.create_namespaced_job(namespace=spec.namespace, body=manifest)
        except ApiException as e:
            if e.status != 409:
                raise

        policy = build_propagation_policy(spec)
        group, version = policy["apiVersion"].split("/")
        try:
            custom.create_namespaced_custom_object(
                group=group, version=version, namespace=spec.namespace,
                plural="propagationpolicies", body=policy,
            )
        except ApiException as e:
            if e.status != 409:
                raise
        return name

    def _delete_job_with(self, batch, custom, spec: WorkerSpec) -> None:
        from kubernetes.client.exceptions import ApiException

        try:
            batch.delete_namespaced_job(
                name=job_name_for(spec), namespace=spec.namespace,
                propagation_policy="Background",
            )
        except ApiException as e:
            if e.status != 404:
                raise

        group, version = "policy.karmada.io", "v1alpha1"
        try:
            custom.delete_namespaced_custom_object(
                group=group, version=version, namespace=spec.namespace,
                plural="propagationpolicies", name=propagation_policy_name_for(spec),
            )
        except ApiException as e:
            if e.status != 404:
                raise


def _k8s_batch_and_core():
    """Lazily build a BatchV1Api/CoreV1Api pointed at the local cluster.

    Deliberately duplicated (not imported) from activities.py: importing
    activities._k8s_batch_and_core here would be equally correct, but keeping
    this module's only dependency on activities.py limited to the two pure
    functions (build_job_manifest, job_name_for) keeps the "who talks to
    Kubernetes" boundary in one place per topology.
    """
    from kubernetes import client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.BatchV1Api(), client.CoreV1Api()


def _karmada_clients():
    """Lazily build clients pointed at the Karmada apiserver.

    The kubeconfig path comes from FED_KARMADA_CONFIG, mounted into the
    Temporal worker pod (fed-infra Task 5) -- never the in-cluster/local
    config, since the Karmada apiserver is a distinct control plane the
    worker pod reaches over its own kubeconfig secret.
    """
    import os

    from kubernetes import client
    from kubernetes import config as k8s_config

    kubeconfig = os.environ.get("FED_KARMADA_CONFIG")
    if not kubeconfig:
        raise RuntimeError(
            "FED_KARMADA_CONFIG must be set to the Karmada apiserver kubeconfig "
            "path mounted into the Temporal worker pod for topology='multi'"
        )
    api_client = k8s_config.new_client_from_config(kubeconfig)
    return client.BatchV1Api(api_client), client.CustomObjectsApi(api_client)
