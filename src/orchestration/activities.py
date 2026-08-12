"""
Temporal activities: the only orchestration code that touches Kubernetes.

Workflow code must be deterministic because Temporal replays it, so every
side effect lives here. `build_job_manifest` and `job_name_for` are pure and
separately testable; the activities themselves are thin wrappers around the
Kubernetes API plus heartbeating.
"""

from __future__ import annotations

import asyncio
import logging

from temporalio import activity

from src.orchestration.types import WorkerResult, WorkerSpec

log = logging.getLogger(__name__)

# How long a worker pod may run before we give up on it.
POD_WATCH_TIMEOUT_S = 3600
# How often to poll pod state and emit a heartbeat.
POLL_INTERVAL_S = 5


def job_name_for(spec: WorkerSpec) -> str:
    """Deterministic Job name.

    Deterministic on purpose: a retried activity must re-attach to the existing
    Job rather than launch a second fleet. The previous implementation embedded
    uuid4() and raced 2N workers onto the same MinIO keys on retry.
    """
    return f"aflw-{spec.kfp_run_id[:8]}-r{spec.fl_round}-w{spec.worker_id}"


def build_job_manifest(spec: WorkerSpec) -> dict:
    """Render the batch/v1 Job for one worker. Pure — no I/O."""
    name = job_name_for(spec)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": spec.namespace,
            "labels": {
                "app": "active-fl-worker",
                "fl-round": str(spec.fl_round),
                "worker-id": str(spec.worker_id),
            },
        },
        "spec": {
            "backoffLimit": 2,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "active-fl-worker",
                        "fl-round": str(spec.fl_round),
                        "worker-id": str(spec.worker_id),
                    }
                },
                "spec": {
                    "restartPolicy": "OnFailure",
                    "containers": [
                        {
                            "name": "worker",
                            "image": spec.worker_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["uv", "run", "python", "-m", "src.agent.train_worker"],
                            "args": [
                                "--fl-round", str(spec.fl_round),
                                "--local-episodes", str(spec.local_episodes),
                                "--device", "cpu",
                            ],
                            "env": [
                                # train_worker.py reads RANK first, falling back to
                                # --worker-id, so no change is needed there.
                                {"name": "RANK", "value": str(spec.worker_id)},
                                {"name": "MINIO_ENDPOINT", "value": spec.minio_endpoint},
                                {"name": "MINIO_ACCESS_KEY", "value": spec.minio_access_key},
                                {"name": "MINIO_SECRET_KEY", "value": spec.minio_secret_key},
                                {"name": "MINIO_BUCKET", "value": spec.minio_bucket},
                                {"name": "MLFLOW_TRACKING_URI", "value": spec.mlflow_tracking_uri},
                                {
                                    "name": "MLFLOW_EXPERIMENT_NAME",
                                    "value": spec.mlflow_experiment_name,
                                },
                            ],
                        }
                    ],
                },
            },
        },
    }


def _k8s_batch_and_core():
    """Import and configure the Kubernetes client lazily.

    Kept out of module scope so the pure functions above remain importable
    (and unit-testable) without a kubeconfig present.
    """
    from kubernetes import client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.BatchV1Api(), client.CoreV1Api()


@activity.defn
async def launch_and_watch_pod(spec: WorkerSpec) -> WorkerResult:
    """Create the worker Job if absent, then watch it to completion.

    Heartbeats every poll so Temporal can distinguish a slow worker from a
    wedged one — this is what replaces the old fixed 20-minute ceiling.
    """
    from kubernetes.client.exceptions import ApiException

    batch, core = _k8s_batch_and_core()
    name = job_name_for(spec)
    manifest = build_job_manifest(spec)

    try:
        batch.create_namespaced_job(namespace=spec.namespace, body=manifest)
        activity.logger.info(f"created Job {name}")
    except ApiException as e:
        if e.status != 409:  # 409 = already exists; retry re-attaches
            raise
        activity.logger.info(f"Job {name} already exists, re-attaching")

    waited = 0
    while waited < POD_WATCH_TIMEOUT_S:
        job = batch.read_namespaced_job_status(name=name, namespace=spec.namespace)
        status = job.status
        if status.succeeded:
            await _log_tail(core, spec, name)
            return WorkerResult(
                worker_id=spec.worker_id, succeeded=True,
                attempts=int(status.failed or 0) + 1, failure_reason="", job_name=name,
            )
        if status.failed and status.failed > manifest["spec"]["backoffLimit"]:
            reason = await _failure_reason(core, spec, name)
            return WorkerResult(
                worker_id=spec.worker_id, succeeded=False,
                attempts=int(status.failed), failure_reason=reason, job_name=name,
            )

        activity.heartbeat(
            {"worker_id": spec.worker_id, "active": int(status.active or 0), "waited_s": waited}
        )
        await asyncio.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S

    return WorkerResult(
        worker_id=spec.worker_id, succeeded=False, attempts=0,
        failure_reason=f"timed out after {POD_WATCH_TIMEOUT_S}s", job_name=name,
    )


async def _failure_reason(core, spec: WorkerSpec, job_name: str) -> str:
    """Best-effort human-readable cause from the most recent pod."""
    try:
        pods = core.list_namespaced_pod(
            namespace=spec.namespace, label_selector=f"job-name={job_name}"
        )
        for pod in pods.items:
            for cs in pod.status.container_statuses or []:
                term = cs.state.terminated
                if term is not None and term.reason:
                    return f"{term.reason} (exit {term.exit_code})"
        return "unknown"
    except Exception as e:  # diagnostics must never mask the real failure
        return f"unavailable: {e}"


async def _log_tail(core, spec: WorkerSpec, job_name: str, lines: int = 20) -> None:
    try:
        pods = core.list_namespaced_pod(
            namespace=spec.namespace, label_selector=f"job-name={job_name}"
        )
        for pod in pods.items:
            text = core.read_namespaced_pod_log(
                name=pod.metadata.name, namespace=spec.namespace, tail_lines=lines
            )
            activity.logger.info(f"[{pod.metadata.name}] {text}")
    except Exception as e:
        activity.logger.warning(f"could not read logs for {job_name}: {e}")


@activity.defn
async def cleanup_worker_job(spec: WorkerSpec) -> None:
    """Delete the Job and its pods. Safe to call when already gone."""
    from kubernetes.client.exceptions import ApiException

    batch, _ = _k8s_batch_and_core()
    try:
        batch.delete_namespaced_job(
            name=job_name_for(spec), namespace=spec.namespace, propagation_policy="Background"
        )
    except ApiException as e:
        if e.status != 404:
            raise
