"""
Pipeline Orchestrator

Reads config/k8s.yaml and submits the Active-FL pipeline to Kubeflow.
"""

import argparse
import logging
import os
import yaml
from kfp import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Submit Active-FL pipeline")
    parser.add_argument("--config", default="config/k8s.yaml", help="Path to config YAML file")
    parser.add_argument(
        "--kfp-host", default="http://localhost:8080", help="Kubeflow Pipelines host"
    )
    parser.add_argument(
        "--output", default="/tmp/active_fl_pipeline.yaml", help="Output compiled YAML"
    )
    parser.add_argument("--wait", action="store_true", help="Wait for pipeline runs to complete")
    parser.add_argument("--timeout", type=int, default=3600, help="Wait timeout in seconds")
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Auto-download results from MLflow after waiting (requires --wait)",
    )
    args = parser.parse_args()

    if args.auto_download and not args.wait:
        log.warning("--auto-download was passed without --wait. Forcing --wait to true.")
        args.wait = True

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Export CONFIG_PATH so the compiler knows where to get fl_rounds
    os.environ["CONFIG_PATH"] = args.config

    t = cfg.get("training", {})
    agg = cfg.get("aggregation", {})
    exp = cfg.get("experiment", {})

    fl_rounds = t.get("fl_rounds", 10)
    workers = t.get("num_workers", 4)
    episodes = t.get("local_episodes", 200)
    score_threshold = agg.get("score_threshold", 0.0)
    eval_episodes = t.get("eval_episodes", 10)
    score_temperature = agg.get("score_temperature", 1.0)
    active_data_threshold = agg.get("active_data_threshold", 0.0)
    active_data_steps = agg.get("active_data_steps", 3)

    combos = cfg.get("combinations", [{"weight_mode": "active", "active_data_mode": "bc"}])

    experiment_name = exp.get("name", "active-fl-cartpole")

    log.info("=================================================")
    log.info(f"  Active-FL Pipeline Run")
    log.info(f"  rounds={fl_rounds} | workers={workers} | episodes={episodes}")
    log.info(f"  combinations to run: {len(combos)}")
    log.info(f"  threshold={score_threshold} | kfp_host={args.kfp_host}")
    log.info("=================================================")

    import subprocess
    from minio import Minio

    log.info("[1/3] Setting up MinIO for MLflow...")
    minio_client = Minio(
        "localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False
    )
    if not minio_client.bucket_exists("mlflow-artifacts"):
        minio_client.make_bucket("mlflow-artifacts")
        log.info("      Created mlflow-artifacts bucket")

    log.info("[2/3] Compiling pipeline...")
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "src/pipelines/active_fl_pipeline.py",
            "--output",
            args.output,
            "--rounds",
            str(fl_rounds),
        ],
        check=True,
    )
    log.info(f"      Compiled → {args.output}")

    log.info("[3/3] Submitting pipeline runs...")
    client = Client(host=args.kfp_host)

    try:
        experiment = client.create_experiment(name=experiment_name)
    except Exception:
        experiment = client.get_experiment(experiment_name=experiment_name)

    import time

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    submitted_runs = []

    for i, combo in enumerate(combos):
        weight_mode = combo.get("weight_mode", "active")
        active_data_mode = combo.get("active_data_mode", "bc")

        log.info(
            f"  [{i + 1}/{len(combos)}] Submitting: weight_mode={weight_mode}, active_data_mode={active_data_mode}"
        )

        run_name = (
            f"{experiment_name}-{weight_mode}-{active_data_mode}-r{fl_rounds}-{run_timestamp}"
        )
        import uuid

        bucket_name = f"fed-{uuid.uuid4().hex[:12]}"

        arguments = {
            "num_workers": workers,
            "local_episodes": episodes,
            "eval_episodes": eval_episodes,
            "score_threshold": score_threshold,
            "score_temperature": score_temperature,
            "active_data_mode": active_data_mode,
            "active_data_threshold": active_data_threshold,
            "active_data_steps": active_data_steps,
            "weight_mode": weight_mode,
            "mlflow_experiment_name": run_name,
            "minio_bucket": bucket_name,
        }

        run = client.create_run_from_pipeline_package(
            pipeline_file=args.output,
            arguments=arguments,
            experiment_id=experiment.experiment_id,
            run_name=run_name,
        )
        submitted_runs.append((run.run_id, run_name))

        log.info(f"      ✅ Run ID: {run.run_id}")
        log.info(f"      View in KFP UI → {args.kfp_host}/#/runs/details/{run.run_id}")

    if args.wait:
        log.info("=================================================")
        log.info(f"Waiting for {len(submitted_runs)} runs to complete...")
        for run_id, run_name in submitted_runs:
            try:
                log.info(f"Waiting on {run_name} (ID: {run_id})...")
                res = client.wait_for_run_completion(run_id=run_id, timeout=args.timeout)
                log.info(f"✅ {run_name} completed with status: {res.state}")
            except Exception as e:
                log.error(f"❌ {run_name} failed or timed out: {e}")

        if args.auto_download:
            log.info("=================================================")
            log.info("Auto-downloading results from MLflow...")
            for _, run_name in submitted_runs:
                log.info(f"Fetching results for {run_name}...")
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "python",
                        "analysis/fetch_k8s_runs.py",
                        "--experiment-prefix",
                        run_name,
                    ],
                    check=False,
                )
            log.info("✨ Done! Generating plots...")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "analysis/compare_runs.py",
                    "--input-dir",
                    "results/k8s_results",
                    "--output-dir",
                    "results/k8s_plots",
                ],
                check=False,
            )
            log.info("Plots generated in results/k8s_plots/")


if __name__ == "__main__":
    main()
