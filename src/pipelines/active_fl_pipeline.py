"""
Kubeflow Pipeline for Active Federated Learning on CartPole

Pipeline DAG per FL round:
  train_workers → collect_and_score_aggregate → evaluate_global

Components run inside the K8s cluster with access to MinIO and MLflow.
`train_workers` is a thin Temporal client: it starts a `TrainRoundWorkflow`
that fans out one durable `WorkerWorkflow` per worker, each of which launches
and watches a Kubernetes Job. KFP still owns the round-level DAG and artifact
lineage; Temporal owns the worker fleet within a round. Temporal is the only
worker-launch path (the `worker_launcher` migration flag and its `pytorchjob`
fallback were removed in Phase P2).

`init_global_model` runs once, ahead of every round, and seeds
`round_0/global.pt` so round 0 aggregates updates to one shared model instead
of averaging N independently-random networks. It is idempotent, so re-running
it against a bucket that already has round 0 (e.g. a resumed pipeline) is a
no-op.
"""

import os

import kfp
from kfp import dsl
from kfp.dsl import Artifact, Input, Output, component

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
# This has to run here, before `active_fl_pipeline` is defined below, and NOT
# in the "Compile" block at the bottom of this file where a `--rounds`/
# `--start-round` override would look like it belongs. `@dsl.pipeline` (used
# on `active_fl_pipeline` below) traces that function's body immediately when
# it decorates it -- i.e. the moment Python executes that `def` statement --
# not later when `Compiler().compile()` is called (confirmed against the
# installed kfp: `GraphComponent.__init__` calls `pipeline_func(*args_list)`
# synchronously in its own constructor; `Compiler.compile()` only reads back
# the already-built `pipeline_spec`, it never re-invokes the function). Env
# vars set after that definition -- e.g. in a single `if __name__` block at
# the bottom of the file, as this used to be structured -- are silently too
# late: the DAG shape is already frozen, so neither flag would have any
# effect on the compiled output despite compiling without error.
if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--output", default="active_fl_pipeline.yaml")
    _parser.add_argument(
        "--rounds", type=int, default=None, help="Force override of compile-time fl_rounds"
    )
    _parser.add_argument(
        "--start-round",
        type=int,
        default=None,
        help="Force override of compile-time start_round (resume point)",
    )
    args = _parser.parse_args()

    if args.rounds is not None:
        os.environ["FL_ROUNDS"] = str(args.rounds)
    if args.start_round is not None:
        os.environ["START_ROUND"] = str(args.start_round)


# ---------------------------------------------------------------------------
# Component: Init Global Model
# ---------------------------------------------------------------------------
@component(base_image="active-fed-aggregator:v1", packages_to_install=[])
def init_global_model(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    seed: int,
) -> None:
    """Ensure a single seeded round-0 global model exists before any worker runs."""
    import sys

    sys.path.insert(0, "/app")

    import logging

    from minio import Minio

    from src.aggregator.collect import write_initial_global_weights

    logging.basicConfig(level=logging.INFO)
    client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )
    if not client.bucket_exists(minio_bucket):
        client.make_bucket(minio_bucket)
    write_initial_global_weights(client, minio_bucket, seed=seed)


@component(base_image="active-fed-aggregator:v1", packages_to_install=[])
def train_workers(
    fl_round: int,
    num_workers: int,
    min_workers: int,
    local_episodes: int,
    namespace: str,
    temporal_address: str,
    kfp_run_id: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    worker_image: str,
    # P3: deployment topology ("single" default | "multi") and, when multi,
    # how many Karmada member clusters (members) and what prefix
    # (member_prefix) RoundSpec.worker_spec round-robins workers across --
    # see types.py for the assignment formula and its non-empty-result
    # guarantee. Required (no default) here, matching every other
    # RoundSpec-driving parameter above: active_fl_pipeline (below) is the
    # only caller and always passes explicit values.
    topology: str,
    members: int,
    member_prefix: str,
    # P3 multi-endpoints fix: NodePorts exposing the host cluster's MinIO/
    # MLflow Services, only meaningful when topology == "multi" -- see
    # RoundSpec/WorkerSpec.minio_nodeport/mlflow_nodeport in types.py and
    # activities.py's _rewrite_endpoints_for_multi, which is what actually
    # consumes them. Required (no default), matching every other
    # RoundSpec-driving parameter above.
    minio_nodeport: int,
    mlflow_nodeport: int,
    worker_report: Output[Artifact],
) -> None:
    """Run one round's worker fleet.

    Starts a TrainRoundWorkflow and blocks on it, streaming per-worker status
    into this node's logs. The workflow ID is deterministic, so a retried
    component re-attaches to the running fleet instead of launching a second
    one.
    """
    import asyncio
    import json
    import sys

    sys.path.insert(0, "/app")

    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import WorkflowIDConflictPolicy

    from src.orchestration.types import RoundSpec
    from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow
    from src.tracking.mlflow_logger import kfp_run_id_from_artifact_uri

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
        topology=topology,
        member_count=members,
        member_prefix=member_prefix,
        minio_nodeport=minio_nodeport,
        mlflow_nodeport=mlflow_nodeport,
        # KFP stamps its own run id into the URIs of the artifacts it mints,
        # and worker_report is one of ours, so the id KFP's UI resolves is
        # recoverable right here. Used only for the workflow memo (the
        # Temporal -> KFP reverse link) -- never for naming, which stays on
        # run_uid because a full UUID would exceed the 63-char Job name limit.
        kfp_backend_run_id=kfp_run_id_from_artifact_uri(
            getattr(worker_report, "uri", "")
        ),
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
        # P4 Task 2, Step 4: KFP -> Temporal is the single most-used
        # cross-link direction (a round looks wrong in MLflow, go see the
        # fleet), so print a clickable line into this node's KFP logs. Base
        # URL matches the Temporal UI's documented local port-forward
        # (README.md, `make temporal-ui`).
        print(f"Temporal workflow: http://localhost:8233/namespaces/default/workflows/{handle.id}")
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
    # P4 Task 2: cross-link tags. kfp_run_id is the same identifier already
    # threaded through the pipeline as train_workers' kfp_run_id (RoundSpec/
    # WorkerSpec.kfp_run_id) -- NOT dsl.PIPELINE_JOB_ID_PLACEHOLDER, which
    # this KFP deployment never substitutes (see the F4 gate-fix comments on
    # run_uid below in active_fl_pipeline). temporal_workflow_id is read from
    # worker_report below, not passed as a parameter -- it doesn't exist
    # until train_workers starts the workflow, and evaluate_global already
    # has an artifact-level path to that report via train_op's output.
    kfp_run_id: str,
    topology: str,
    aggregation_report: Input[Artifact],
    worker_report: Input[Artifact],
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
    from src.tracking.mlflow_logger import kfp_run_id_from_artifact_uri, log_run_context

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

    # P4 Task 2, Step 3: the Temporal workflow id train_workers writes into
    # worker_report -- verified present in both the success and
    # quorum-failure payloads (active_fl_pipeline.py's train_workers, and
    # tests/test_train_workers_component.py). Falls back to "" (not a
    # KeyError) so a malformed/legacy report degrades to a skipped tag
    # (log_run_context) rather than failing the round -- tracking must never
    # fail a training run.
    with open(worker_report.path) as f:
        worker_payload = json.load(f)
    temporal_workflow_id = worker_payload.get("temporal_workflow_id", "")

    # The kfp_run_id parameter carries run_uid, which names the MinIO bucket
    # and the worker Jobs but is not the id KFP's UI resolves in
    # /#/runs/details/<id>. KFP does stamp its own run id into every artifact
    # URI it mints, and worker_report is one, so recover it from there.
    # Falls back to "" -- log_run_context then skips the tag rather than
    # writing a URL that goes nowhere.
    kfp_backend_run_id = kfp_run_id_from_artifact_uri(getattr(worker_report, "uri", ""))

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    with mlflow.start_run(run_name=f"round_{fl_round}", nested=True):
        log_run_context(
            kfp_run_id=kfp_backend_run_id,
            temporal_workflow_id=temporal_workflow_id,
            topology=topology,
        )
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
    temporal_address: str = "temporal-frontend.active-fed.svc.cluster.local:7233",
    # P3: deployment topology. "single" (default) is today's local-cluster
    # behaviour, unchanged. "multi" means workers are propagated by Karmada
    # to `members` member clusters, round-robinning worker_id % members
    # (RoundSpec.worker_spec, types.py) with names built from
    # member_prefix. Sourced from config/k8s.yaml's orchestration.topology /
    # orchestration.members by run_pipeline.py; the defaults below apply
    # only to direct compilation (`make compile-pipeline`) and MUST agree
    # with run_pipeline.py's own DEFAULT_TOPOLOGY/DEFAULT_MEMBERS -- see
    # test_topology_and_members_default_layers_agree.
    topology: str = "single",
    members: int = 0,
    member_prefix: str = "active-fed-member",
    # P3 multi-endpoints fix: NodePorts exposing the host cluster's MinIO/
    # MLflow Services to Karmada member clusters (see the train_workers
    # component above and activities.py's _rewrite_endpoints_for_multi).
    # Only meaningful when topology == "multi". Sourced from
    # config/k8s-multi.yaml's orchestration.minio_nodeport/mlflow_nodeport by
    # run_pipeline.py; the defaults below apply only to direct compilation
    # (`make compile-pipeline`) and MUST agree with run_pipeline.py's own
    # DEFAULT_MINIO_NODEPORT/DEFAULT_MLFLOW_NODEPORT -- see
    # test_nodeport_default_layers_agree.
    minio_nodeport: int = 30900,
    mlflow_nodeport: int = 30500,
    # Per-run identifier (F4 gate fix): generated by run_pipeline.py and passed
    # in as a normal argument. NOT dsl.PIPELINE_JOB_ID_PLACEHOLDER -- this KFP
    # deployment never substitutes that placeholder before component code
    # runs, so it previously arrived at train_workers verbatim as the literal
    # string "{{$.pipeline_job_uuid}}", which fails Kubernetes' RFC-1123 name
    # validation for every Job built from it. The default below is only used
    # when compiling/running this pipeline directly (e.g. `make
    # compile-pipeline`) without going through run_pipeline.py.
    run_uid: str = "localdev",
    # Seed for the round-0 global model init_global_model writes. Sourced from
    # config/k8s.yaml's experiment.seed by run_pipeline.py; the default here
    # only applies to direct compilation (`make compile-pipeline`).
    seed: int = 42,
) -> None:
    """Full active-FL pipeline (Active Weight + Active Data) over fl_rounds sequential rounds."""

    # KFP doesn't have native loops with dynamic round count,
    # so we chain rounds explicitly. For production with many rounds,
    # use the pipeline as a single-round step and submit it repeatedly.
    # Here we unroll for clarity (up to fl_rounds via recursive chaining).
    #
    # fl_rounds and start_round are deliberately NOT dsl parameters. KFP v2
    # traces this function exactly once, substituting a
    # PipelineParameterChannel placeholder for every declared dsl parameter --
    # never the literal default or a submitted value (confirmed empirically:
    # using such a parameter as a `range()` bound raises "TypeError:
    # 'PipelineParameterChannel' object cannot be interpreted as an integer").
    # A dsl parameter therefore cannot change how many train_workers tasks
    # exist; that shape is frozen at trace time. This is exactly the trap a
    # `start_round` dsl parameter used to set (P2 review Finding 1): it
    # showed up in the KFP UI/API as if it worked, but silently had zero
    # effect on which rounds execute. Both round counts below are instead
    # resolved from the environment ahead of the trace -- see the first
    # `if __name__ == "__main__":` block near the top of this file for why
    # that must happen *before* this function is defined, not merely before
    # `Compiler().compile()` is called.

    import yaml

    try:
        cfg_path = os.environ.get("CONFIG_PATH", "config/k8s.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        compile_time_rounds = cfg.get("training", {}).get("fl_rounds", 10)
    except Exception:
        compile_time_rounds = int(os.environ.get("FL_ROUNDS", "5"))

    # Resolved from the START_ROUND env var: run_pipeline.py computes this
    # from `compute_start_round` (which round to resume from, if any) and
    # passes it as a --start-round CLI flag to this module's __main__ block.
    compile_time_start_round = int(os.environ.get("START_ROUND", "0"))

    # init_global_model is idempotent -- it leaves an existing round_0/global.pt
    # untouched -- so it always runs first, even on a resumed pipeline.
    init_op = init_global_model(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
        seed=seed,
    ).set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)

    prev_op = init_op
    for round_idx in range(compile_time_start_round, compile_time_rounds):
        train_op = train_workers(
            fl_round=round_idx,
            num_workers=num_workers,
            min_workers=min_workers,
            local_episodes=local_episodes,
            namespace=namespace,
            temporal_address=temporal_address,
            kfp_run_id=run_uid,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            worker_image=worker_image,
            topology=topology,
            members=members,
            member_prefix=member_prefix,
            minio_nodeport=minio_nodeport,
            mlflow_nodeport=mlflow_nodeport,
        ).set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
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
                # P4 Task 2: same kfp_run_id/topology values train_workers
                # already receives above, plus train_op's own worker_report
                # output -- the only place temporal_workflow_id is known.
                kfp_run_id=run_uid,
                topology=topology,
                aggregation_report=agg_op.outputs["aggregation_report"],
                worker_report=train_op.outputs["worker_report"],
            )
            .after(agg_op)
            .set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
        )

        prev_op = eval_op


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
# `args` (and the FL_ROUNDS/START_ROUND env vars) were already set by the
# `if __name__ == "__main__":` block at the top of this file, before
# `active_fl_pipeline` was defined -- see the comment there.
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=active_fl_pipeline,
        package_path=args.output,
    )
    print(f"Pipeline compiled → {args.output}")
