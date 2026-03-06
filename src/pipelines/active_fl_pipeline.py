"""
Kubeflow Pipeline for Active Federated Learning on CartPole

Pipeline DAG per FL round:
  train_workers → collect_and_score_aggregate → evaluate_global

Components run inside the K8s cluster with access to MinIO and MLflow.
Workers are launched as a PyTorchJob (managed by Training Operator).
"""

import os

import kfp
from kfp import dsl
from kfp.dsl import Artifact, Input, Output, component


@component(
    base_image="python:3.9-slim",
    packages_to_install=["jinja2", "pyyaml", "requests"],
)
def train_workers(
    fl_round: int,
    num_workers: int,
    local_episodes: int,
    namespace: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    worker_image: str,
    job_status: Output[Artifact],
) -> None:
    """
    Render a PyTorchJob manifest from the Jinja2 template and submit it via kubectl.
    Waits for all workers to complete before returning.
    """
    import json
    import os
    import subprocess
    import time
    import urllib.request
    import uuid

    # ---- Install kubectl ----
    kubectl_url = "https://dl.k8s.io/release/v1.29.0/bin/linux/amd64/kubectl"
    kubectl_path = "/usr/local/bin/kubectl"
    if not os.path.exists(kubectl_path):
        urllib.request.urlretrieve(kubectl_url, kubectl_path)
        os.chmod(kubectl_path, 0o755)

    # ---- Render PyTorchJob YAML ----
    from jinja2 import Template

    job_name = f"active-fl-round-{fl_round}-{uuid.uuid4().hex[:6]}"
    template_str = """
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: {{ job_name }}
  namespace: {{ namespace }}
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        spec:
          serviceAccountName: pipeline-runner
          containers:
            - name: pytorch
              image: {{ worker_image }}
              imagePullPolicy: IfNotPresent
              command:
                - uv
                - run
                - python
                - -m
                - src.agent.train_worker
                - --fl-round
                - "{{ fl_round }}"
                - --local-episodes
                - "{{ local_episodes }}"
                - --device
                - "cpu"
              env:
                - name: RANK
                  value: "0"
                - name: MLFLOW_TRACKING_URI
                  value: "{{ mlflow_tracking_uri }}"
                - name: MLFLOW_EXPERIMENT_NAME
                  value: "{{ mlflow_experiment_name }}"
                - name: MINIO_ENDPOINT
                  value: "{{ minio_endpoint }}"
                - name: MINIO_ACCESS_KEY
                  value: "{{ minio_access_key }}"
                - name: MINIO_SECRET_KEY
                  value: "{{ minio_secret_key }}"
                - name: MINIO_BUCKET
                  value: "{{ minio_bucket }}"
    Worker:
      replicas: {{ num_workers - 1 }}
      restartPolicy: Never
      template:
        spec:
          serviceAccountName: pipeline-runner
          containers:
            - name: pytorch
              image: {{ worker_image }}
              imagePullPolicy: IfNotPresent
              command:
                - uv
                - run
                - python
                - -m
                - src.agent.train_worker
                - --fl-round
                - "{{ fl_round }}"
                - --local-episodes
                - "{{ local_episodes }}"
                - --device
                - "cpu"
              env:
                - name: MLFLOW_TRACKING_URI
                  value: "{{ mlflow_tracking_uri }}"
                - name: MLFLOW_EXPERIMENT_NAME
                  value: "{{ mlflow_experiment_name }}"
                - name: MINIO_ENDPOINT
                  value: "{{ minio_endpoint }}"
                - name: MINIO_ACCESS_KEY
                  value: "{{ minio_access_key }}"
                - name: MINIO_SECRET_KEY
                  value: "{{ minio_secret_key }}"
                - name: MINIO_BUCKET
                  value: "{{ minio_bucket }}"
                - name: MLFLOW_S3_ENDPOINT_URL
                  value: "http://{{ minio_endpoint }}"
                - name: AWS_ACCESS_KEY_ID
                  value: "{{ minio_access_key }}"
                - name: AWS_SECRET_ACCESS_KEY
                  value: "{{ minio_secret_key }}"
                - name: MLFLOW_S3_IGNORE_TLS
                  value: "true"
"""
    manifest = Template(template_str).render(
        job_name=job_name,
        fl_round=fl_round,
        num_workers=num_workers,
        local_episodes=local_episodes,
        namespace=namespace,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
        worker_image=worker_image,
    )
    with open("/tmp/job.yaml", "w") as f:
        f.write(manifest)

    # ---- Submit job ----
    subprocess.run([kubectl_path, "apply", "-f", "/tmp/job.yaml", "-n", namespace], check=True)

    # ---- Wait for completion ----
    print(f"Waiting for PyTorchJob {job_name} to complete...")
    for attempt in range(120):  # 20 minute timeout
        time.sleep(10)
        result = subprocess.run(
            [
                kubectl_path,
                "get",
                "pytorchjob",
                job_name,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.conditions[-1].type}",
            ],
            capture_output=True,
            text=True,
        )
        status = result.stdout.strip()
        print(f"[{attempt * 10}s] PyTorchJob status: {status}")
        if status == "Succeeded":
            print("PyTorchJob completed successfully")
            break
        if status == "Failed":
            raise RuntimeError(f"PyTorchJob {job_name} failed")
    else:
        raise TimeoutError(f"PyTorchJob {job_name} timed out after 20 minutes")

    with open(job_status.path, "w") as f:
        json.dump({"job_name": job_name, "fl_round": fl_round, "status": "Succeeded"}, f)


# ---------------------------------------------------------------------------
# Component: Score and Aggregate
# ---------------------------------------------------------------------------
@component(
    base_image="active-fed-aggregator:v1",
    packages_to_install=[],
)
def score_and_aggregate(
    fl_round: int,
    num_workers: int,
    score_threshold: float,
    score_temperature: float,
    eval_episodes: int,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    active_data_mode: str,  # "none" | "bc"
    active_data_threshold: float,  # min improvement for trajectory inclusion
    active_data_steps: int,  # gradient epochs for fine-tuning
    weight_mode: str,  # "active" | "fedavg" | "data_only"
    aggregation_report: Output[Artifact],
) -> None:
    """
    1. Collect worker weights from MinIO
    2. Run target-env evaluation probes (parallel), capturing trajectory data
    3. Score all clients (4-factor)
    4. Active Weight: weighted FedAvg over accepted clients
    5. Active Data: BC fine-tune on high-value trajectories (if mode != none)
    6. Push new global model to MinIO
    """
    import sys

    sys.path.insert(0, "/app")

    import json
    import logging

    import numpy as np
    import torch
    from minio import Minio

    from src.aggregator.aggregator import aggregate
    from src.aggregator.collect import collect_worker_updates, push_global_weights
    from src.aggregator.evaluator import evaluate_all_candidates
    from src.aggregator.scorer import score_clients

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    minio_client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

    # Load current global weights
    import io

    try:
        response = minio_client.get_object(minio_bucket, f"round_{fl_round}/global.pt")
        global_weights = torch.load(io.BytesIO(response.read()), weights_only=True)
        log.info(f"Loaded global weights for round {fl_round}")
    except Exception:
        log.info("No global weights found, initialising fresh model")
        from src.agent.model import ActorCritic

        global_weights = {k: v.clone() for k, v in ActorCritic().state_dict().items()}

    # Collect worker updates
    clients = collect_worker_updates(minio_client, minio_bucket, fl_round, num_workers)
    if not clients:
        raise RuntimeError("No worker updates found! Cannot aggregate.")

    # Evaluate candidates on target env (collect_data=True for active data mode)
    collect_data = active_data_mode != "none"
    weight_deltas = {c.worker_id: c.weight_delta for c in clients}
    eval_results = evaluate_all_candidates(
        global_weights=global_weights,
        weight_deltas=weight_deltas,
        n_eval_episodes=eval_episodes,
        max_workers=min(num_workers, 4),
        collect_data=collect_data,
    )

    # Score clients
    scored = score_clients(
        clients=clients,
        eval_results=eval_results,
        score_threshold=score_threshold,
    )

    # Aggregate: Active Weight + Active Data
    result = aggregate(
        global_weights=global_weights,
        clients=clients,
        scored_clients=scored,
        eval_results=eval_results,
        temperature=score_temperature,
        weight_mode=weight_mode,
        active_data_mode=active_data_mode,
        active_data_threshold=active_data_threshold,
        active_data_steps=active_data_steps,
    )

    # Push new global model
    push_global_weights(minio_client, minio_bucket, fl_round, result.global_weights)

    # Build a quick worker_id -> avg_reward lookup from clients list
    clients_own_env = {c.worker_id: c.avg_reward for c in clients}

    # Write report
    report = {
        "fl_round": fl_round,
        "round_summary": result.round_summary,
        "active_data_applied": result.active_data_applied,
        "active_data_n_steps": result.active_data_n_steps,
        "active_data_source_workers": result.active_data_source_workers,
        "scored_clients": [
            {
                "worker_id": sc.worker_id,
                "score": sc.score,
                "improvement": sc.improvement,
                "accepted": sc.accepted,
                # Target env rewards — directly from evaluate_all_candidates EvalResult
                "target_env_reward_mean": float(
                    np.mean(eval_results[sc.worker_id].raw_episode_rewards)
                )
                if sc.worker_id in eval_results and eval_results[sc.worker_id].raw_episode_rewards
                else 0.0,
                "target_env_reward_std": float(
                    np.std(eval_results[sc.worker_id].raw_episode_rewards)
                )
                if sc.worker_id in eval_results and eval_results[sc.worker_id].raw_episode_rewards
                else 0.0,
                "raw_eval_rewards": eval_results[sc.worker_id].raw_episode_rewards
                if sc.worker_id in eval_results
                else [],
                "raw_eval_steps": eval_results[sc.worker_id].raw_episode_steps
                if sc.worker_id in eval_results
                else [],
                # Own env reward — from worker training metrics
                "own_env_reward_mean": clients_own_env.get(sc.worker_id, 0.0),
            }
            for sc in result.scored_clients
        ],
    }
    with open(aggregation_report.path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(
        f"Aggregation complete for round {fl_round} | "
        f"active_data_applied={result.active_data_applied} ({active_data_mode})"
    )


# ---------------------------------------------------------------------------
# Component: Evaluate Global Model
# ---------------------------------------------------------------------------
@component(
    base_image="active-fed-aggregator:v1",
    packages_to_install=[],
)
def evaluate_global(
    fl_round: int,
    n_eval_episodes: int,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    aggregation_report: Input[Artifact],
    eval_result: Output[Artifact],
) -> None:
    """
    Evaluate the new global model on CartPole and log all metrics to MLflow.
    """
    import sys

    sys.path.insert(0, "/app")

    import io
    import json
    import logging
    import os

    import mlflow
    import numpy as np
    import torch
    from minio import Minio

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    from src.aggregator.evaluator import _rollout

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    minio_client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

    # Load new global weights (written by score_and_aggregate)
    response = minio_client.get_object(minio_bucket, f"round_{fl_round + 1}/global.pt")
    global_weights = torch.load(io.BytesIO(response.read()), weights_only=True)

    # Evaluate
    mean_reward, rewards, steps_list = _rollout(
        global_weights, "CartPole-v1", n_eval_episodes, 42, 42
    )
    std_reward = float(np.std(rewards))
    solved = mean_reward >= 195.0

    log.info(
        f"Round {fl_round} global eval: mean={mean_reward:.2f} ± {std_reward:.2f} | solved={solved}"
    )

    # Load aggregation report for per-client metrics
    with open(aggregation_report.path) as f:
        report = json.load(f)

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    with mlflow.start_run(run_name=f"round_{fl_round}", nested=True):
        mlflow.log_metrics(
            {
                "global_eval_reward_mean": mean_reward,
                "global_eval_reward_std": std_reward,
                "solved": float(solved),
                "clients_accepted": float(report["round_summary"].get("clients_accepted", 0)),
                "clients_rejected": float(report["round_summary"].get("clients_rejected", 0)),
                "effective_weight_norm": float(
                    report["round_summary"].get("effective_weight_norm", 0)
                ),
                "active_data_applied": float(report.get("active_data_applied", False)),
                "active_data_n_steps": float(report.get("active_data_n_steps", 0)),
                "num_active_data_sources": float(len(report.get("active_data_source_workers", []))),
            },
            step=fl_round,
        )
        for sc_info in report["scored_clients"]:
            wid = sc_info["worker_id"]
            mlflow.log_metrics(
                {
                    f"client_{wid}_score": sc_info["score"],
                    f"client_{wid}_improvement": sc_info["improvement"],
                    f"client_{wid}_accepted": float(sc_info["accepted"]),
                    f"client_{wid}_target_env_reward_mean": sc_info.get(
                        "target_env_reward_mean", 0.0
                    ),
                    f"client_{wid}_target_env_reward_std": sc_info.get(
                        "target_env_reward_std", 0.0
                    ),
                    f"client_{wid}_own_env_reward_mean": sc_info.get("own_env_reward_mean", 0.0),
                },
                step=fl_round,
            )

        # Save global model checkpoint
        buf = io.BytesIO()
        torch.save(global_weights, buf)
        buf.seek(0)
        with open("/tmp/global_model.pt", "wb") as f:
            f.write(buf.read())
        mlflow.log_artifact("/tmp/global_model.pt", artifact_path=f"global_models/round_{fl_round}")

        # Also log the full aggregation report as an artifact in MLflow to make it
        # easy to download directly
        report_path = "/tmp/aggregation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path=f"reports/round_{fl_round}")

    result_data = {
        "fl_round": fl_round,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "solved": solved,
        "raw_global_rewards": rewards,
        "raw_global_steps": steps_list,
    }
    with open(eval_result.path, "w") as f:
        json.dump(result_data, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline Definition
# ---------------------------------------------------------------------------
@dsl.pipeline(
    name="active-fl-cartpole",
    description="Active Federated Learning on CartPole-v1",
)
def active_fl_pipeline(
    num_workers: int = 4,
    local_episodes: int = 200,
    eval_episodes: int = 10,
    score_threshold: float = 0.0,
    score_temperature: float = 1.0,
    # Active Data params
    active_data_mode: str = "bc",  # "none" | "bc"
    active_data_threshold: float = 0.0,  # min improvement to include trajectory
    active_data_steps: int = 3,  # fine-tuning epochs per round
    # Weight aggregation mode
    weight_mode: str = "active",  # "active" | "fedavg" | "data_only"
    # Infrastructure
    namespace: str = "active-fed",
    mlflow_tracking_uri: str = "http://mlflow-service.active-fed.svc.cluster.local:5000",
    mlflow_experiment_name: str = "active-fl-cartpole",
    minio_endpoint: str = "minio-service.active-fed.svc.cluster.local:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    minio_bucket: str = "active-fed",
    worker_image: str = "active-fed-worker:v1",
) -> None:
    """Full active-FL pipeline (Active Weight + Active Data) over fl_rounds sequential rounds."""

    # KFP doesn't have native loops with dynamic round count,
    # so we chain rounds explicitly. For production with many rounds,
    # use the pipeline as a single-round step and submit it repeatedly.
    # Here we unroll for clarity (up to fl_rounds via recursive chaining).

    import yaml

    try:
        cfg_path = os.environ.get("CONFIG_PATH", "config/k8s.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        compile_time_rounds = cfg.get("training", {}).get("fl_rounds", 10)
    except Exception:
        compile_time_rounds = int(os.environ.get("FL_ROUNDS", "5"))

    prev_op = None
    for round_idx in range(compile_time_rounds):
        train_op = train_workers(
            fl_round=round_idx,
            num_workers=num_workers,
            local_episodes=local_episodes,
            namespace=namespace,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            worker_image=worker_image,
        )
        if prev_op is not None:
            train_op.after(prev_op)

        agg_op = score_and_aggregate(
            fl_round=round_idx,
            num_workers=num_workers,
            score_threshold=score_threshold,
            score_temperature=score_temperature,
            eval_episodes=eval_episodes,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            active_data_mode=active_data_mode,
            active_data_threshold=active_data_threshold,
            active_data_steps=active_data_steps,
            weight_mode=weight_mode,
        ).after(train_op)

        eval_op = evaluate_global(
            fl_round=round_idx,
            n_eval_episodes=20,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            aggregation_report=agg_op.outputs["aggregation_report"],
        ).after(agg_op)

        prev_op = eval_op


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="active_fl_pipeline.yaml")
    parser.add_argument(
        "--rounds", type=int, default=None, help="Force override of compile-time fl_rounds"
    )
    args = parser.parse_args()

    if args.rounds is not None:
        os.environ["FL_ROUNDS"] = str(args.rounds)

    kfp.compiler.Compiler().compile(
        pipeline_func=active_fl_pipeline,
        package_path=args.output,
    )
    print(f"Pipeline compiled → {args.output}")
