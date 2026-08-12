
from src.orchestration.activities import build_job_manifest, job_name_for
from src.orchestration.types import WorkerSpec


def _spec(**overrides) -> WorkerSpec:
    base = dict(
        fl_round=3, worker_id=2, num_workers=4, local_episodes=25,
        namespace="active-fed", worker_image="active-fed-worker:v1",
        minio_endpoint="minio-service:9000", minio_access_key="ak",
        minio_secret_key="sk", minio_bucket="bucket",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234567890",
    )
    base.update(overrides)
    return WorkerSpec(**base)


def test_job_name_is_deterministic_for_the_same_inputs():
    assert job_name_for(_spec()) == job_name_for(_spec())


def test_job_name_distinguishes_worker_and_round():
    assert job_name_for(_spec(worker_id=1)) != job_name_for(_spec(worker_id=2))
    assert job_name_for(_spec(fl_round=1)) != job_name_for(_spec(fl_round=2))


def test_job_name_is_a_valid_kubernetes_name():
    import re
    name = job_name_for(_spec())
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name), name
    assert len(name) <= 63


def test_manifest_sets_rank_to_the_worker_id():
    env = {e["name"]: e["value"] for e in
           build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["RANK"] == "2"


def test_manifest_passes_round_and_episodes_as_args():
    args = build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--fl-round" in args and args[args.index("--fl-round") + 1] == "3"
    assert "--local-episodes" in args and args[args.index("--local-episodes") + 1] == "25"


def test_manifest_carries_minio_and_mlflow_env():
    env = {e["name"]: e["value"] for e in
           build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["MINIO_ENDPOINT"] == "minio-service:9000"
    assert env["MINIO_BUCKET"] == "bucket"
    assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


def test_manifest_uses_onfailure_restart_policy():
    # P0 review: restartPolicy Never meant a crashed worker vanished and the
    # aggregator silently proceeded with N-1 clients.
    assert build_job_manifest(_spec())["spec"]["template"]["spec"]["restartPolicy"] == "OnFailure"


def test_manifest_labels_identify_round_and_worker():
    labels = build_job_manifest(_spec())["spec"]["template"]["metadata"]["labels"]
    assert labels["app"] == "active-fl-worker"
    assert labels["fl-round"] == "3"
    assert labels["worker-id"] == "2"
