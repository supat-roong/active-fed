"""
Kubeflow Pipeline for Active Federated Learning on CartPole

Pipeline DAG per FL round:
  train_workers → collect_and_score_aggregate → evaluate_global

Components run inside the K8s cluster with access to MinIO and MLflow.
`train_workers` is a thin Temporal client: it starts a `TrainRoundWorkflow`
that fans out one durable `WorkerWorkflow` per worker, each of which launches
and watches a Kubernetes Job. KFP still owns the round-level DAG and artifact
lineage; Temporal owns the worker fleet within a round. The `worker_launcher`
migration flag retains a `pytorchjob` fallback path (removed in Phase P2) so a
Temporal outage cannot block experiments.
"""

import os

import kfp
from kfp import dsl
from kfp.dsl import Artifact, Input, Output, component


@component(base_image="active-fed-aggregator:v1", packages_to_install=[])
def train_workers(
    fl_round: int,
    num_workers: int,
    min_workers: int,
    local_episodes: int,
    namespace: str,
    worker_launcher: str,
    temporal_address: str,
    kfp_run_id: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    worker_image: str,
    worker_report: Output[Artifact],
) -> None:
    """Run one round's worker fleet.

    With worker_launcher='temporal' this starts a TrainRoundWorkflow and blocks
    on it, streaming per-worker status into this node's logs. The workflow ID is
    deterministic, so a retried component re-attaches to the running fleet
    instead of launching a second one.
    """
    import asyncio
    import json
    import sys

    sys.path.insert(0, "/app")

    if worker_launcher == "pytorchjob":
        # Migration fallback, removed in P2.
        raise NotImplementedError(
            "the pytorchjob launcher path is retained only for rollback; "
            "restore it from git history if you need it"
        )

    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import WorkflowIDConflictPolicy

    from src.orchestration.types import RoundSpec
    from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow

    spec = RoundSpec(
        fl_round=fl_round,
        num_workers=num_workers,
        min_workers=min_workers,
        local_episodes=local_episodes,
        namespace=namespace,
        worker_image=worker_image,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        kfp_run_id=kfp_run_id,
    )

    async def _run() -> dict:
        client = await Client.connect(temporal_address)
        handle = await client.start_workflow(
            TrainRoundWorkflow.run,
            spec,
            id=f"train-{kfp_run_id[:8]}-r{fl_round}",
            task_queue=TASK_QUEUE,
            # A retried component re-runs this same call with the same
            # deterministic id. USE_EXISTING is what makes that reattach to a
            # still-running round's workflow (and hence its result()) instead
            # of failing with WorkflowAlreadyStartedError -- the default
            # UNSPECIFIED policy fails fast on a running duplicate ID and
            # would defeat the whole point of the deterministic id.
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )
        print(f"started Temporal workflow {handle.id}")
        try:
            report = await handle.result()
        except WorkflowFailureError as e:
            # I2 (final review): quorum wasn't met, so TrainRoundWorkflow raised
            # instead of returning its RoundReport -- the full report was built
            # and then discarded. Queries still work against a closed workflow,
            # so the per-worker status the workflow already tracked survives
            # here even though the return value didn't. Recovering it is the
            # difference between "one exception line and no artifact" and the
            # per-worker attribution this whole component exists to provide.
            # The round must still fail -- re-raise after writing the artifact.
            statuses = await handle.query(TrainRoundWorkflow.status)
            payload = {
                "fl_round": fl_round,
                "succeeded": sorted(
                    wid for wid, s in statuses.items() if s.phase == "Succeeded"
                ),
                "failed": sorted(wid for wid, s in statuses.items() if s.phase != "Succeeded"),
                "results": [vars(statuses[wid]) for wid in sorted(statuses)],
                "temporal_workflow_id": handle.id,
                "error": str(e),
            }
            print(json.dumps(payload, indent=2))
            with open(worker_report.path, "w") as f:
                json.dump(payload, f, indent=2)
            raise
        return {
            "fl_round": report.fl_round,
            "succeeded": report.succeeded_ids,
            "failed": report.failed_ids,
            "results": [vars(r) for r in report.results],
            "temporal_workflow_id": handle.id,
        }

    payload = asyncio.run(_run())
    print(json.dumps(payload, indent=2))
    with open(worker_report.path, "w") as f:
        json.dump(payload, f, indent=2)


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
    # Worker orchestration (Temporal)
    min_workers: int = 2,
    worker_launcher: str = "temporal",  # "temporal" | "pytorchjob" (removed in P2)
    temporal_address: str = "temporal-frontend.active-fed.svc.cluster.local:7233",
    # Per-run identifier (F4 gate fix): generated by run_pipeline.py and passed
    # in as a normal argument. NOT dsl.PIPELINE_JOB_ID_PLACEHOLDER -- this KFP
    # deployment never substitutes that placeholder before component code
    # runs, so it previously arrived at train_workers verbatim as the literal
    # string "{{$.pipeline_job_uuid}}", which fails Kubernetes' RFC-1123 name
    # validation for every Job built from it. The default below is only used
    # when compiling/running this pipeline directly (e.g. `make
    # compile-pipeline`) without going through run_pipeline.py.
    run_uid: str = "localdev",
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
            min_workers=min_workers,
            local_episodes=local_episodes,
            namespace=namespace,
            worker_launcher=worker_launcher,
            temporal_address=temporal_address,
            kfp_run_id=run_uid,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            worker_image=worker_image,
        ).set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
        if prev_op is not None:
            train_op.after(prev_op)

        agg_op = (
            score_and_aggregate(
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
            )
            .after(train_op)
            .set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
        )

        eval_op = (
            evaluate_global(
                fl_round=round_idx,
                n_eval_episodes=20,
                minio_endpoint=minio_endpoint,
                minio_access_key=minio_access_key,
                minio_secret_key=minio_secret_key,
                minio_bucket=minio_bucket,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_experiment_name=mlflow_experiment_name,
                aggregation_report=agg_op.outputs["aggregation_report"],
            )
            .after(agg_op)
            .set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
        )

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
