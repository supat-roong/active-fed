"""
Pipeline Orchestrator

Reads config/k8s.yaml and submits the Active-FL pipeline to Kubeflow.
"""

import argparse
import logging
import os
import subprocess

import yaml
from kfp import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def compute_start_round(minio_client, bucket: str) -> int:
    """Highest round N for which round_N/global.pt exists, else 0.

    Deliberately does not rely on KFP caching, which is left at its default and
    depends on input hashing we do not control. Reading the checkpoint that
    actually exists is explicit and inspectable.
    """
    highest = 0
    for obj in minio_client.list_objects(bucket, prefix="round_", recursive=True):
        name = obj.object_name
        if not name.endswith("/global.pt"):
            continue
        head = name.split("/", 1)[0]
        if not head.startswith("round_"):
            continue
        try:
            highest = max(highest, int(head[len("round_") :]))
        except ValueError:
            continue
    return highest


def run_step(argv: list[str], description: str) -> None:
    """Run a post-processing step, failing loudly.

    These calls previously used check=False, so a failed fetch or plot was
    discarded and the sweep reported success while producing nothing.
    """
    log.info(f"{description}: {' '.join(argv)}")
    subprocess.run(argv, check=True)


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
    parser.add_argument(
        "--bucket",
        default=None,
        help=(
            "Reuse an existing MinIO bucket instead of generating a fresh "
            "fed-<uuid> one per combination. Required for resume: "
            "compute_start_round only finds a checkpoint to resume from when "
            "the bucket already has one. Only safe with a single combination "
            "in config.combinations -- every combination submitted in this "
            "invocation would otherwise write into the same bucket."
        ),
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
    orch = cfg.get("orchestration", {})

    fl_rounds = t.get("fl_rounds", 10)
    workers = t.get("num_workers", 4)
    min_workers = t.get("min_workers", 2)
    episodes = t.get("local_episodes", 200)
    score_threshold = agg.get("score_threshold", 0.0)
    eval_episodes = t.get("eval_episodes", 10)
    score_temperature = agg.get("score_temperature", 1.0)
    active_data_threshold = agg.get("active_data_threshold", 0.0)
    active_data_steps = agg.get("active_data_steps", 3)
    temporal_address = orch.get(
        "temporal_address", "temporal-frontend.active-fed.svc.cluster.local:7233"
    )
    seed = exp.get("seed", 42)

    combos = cfg.get("combinations", [{"weight_mode": "active", "active_data_mode": "bc"}])

    if args.bucket and len(combos) > 1:
        log.warning(
            f"--bucket was passed with {len(combos)} combinations configured; all of them "
            "will read and write round_N/global.pt in the SAME bucket and will corrupt each "
            "other's checkpoints. --bucket is only safe with a single combination."
        )

    experiment_name = exp.get("name", "active-fl-cartpole")

    log.info("=================================================")
    log.info("  Active-FL Pipeline Run")
    log.info(f"  rounds={fl_rounds} | workers={workers} | episodes={episodes}")
    log.info(f"  combinations to run: {len(combos)}")
    log.info(f"  threshold={score_threshold} | kfp_host={args.kfp_host}")
    log.info("=================================================")

    from minio import Minio

    log.info("[1/3] Setting up MinIO for MLflow...")
    minio_client = Minio(
        "localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False
    )
    if not minio_client.bucket_exists("mlflow-artifacts"):
        minio_client.make_bucket("mlflow-artifacts")
        log.info("      Created mlflow-artifacts bucket")

    # Resume only means something when --bucket points at a bucket a previous
    # attempt already wrote into: the default per-combination bucket_name
    # (fed-<uuid>, computed below in the per-combination loop) is always
    # empty, so compute_start_round against it would always be 0. Without
    # --bucket there is nothing to resume from, and start_round stays 0.
    start_round = 0
    if args.bucket:
        if minio_client.bucket_exists(args.bucket):
            start_round = compute_start_round(minio_client, args.bucket)
        if start_round:
            log.info(
                f"resuming from round {start_round} (checkpoint found in bucket {args.bucket})"
            )
        else:
            log.info(f"--bucket {args.bucket} has no round checkpoints yet; starting at round 0")

    log.info("[2/3] Compiling pipeline...")
    run_step(
        [
            "uv",
            "run",
            "python",
            "src/pipelines/active_fl_pipeline.py",
            "--output",
            args.output,
            "--rounds",
            str(fl_rounds),
            "--start-round",
            str(start_round),
        ],
        "compile pipeline",
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
            f"  [{i + 1}/{len(combos)}] Submitting: weight_mode={weight_mode}, "
            f"active_data_mode={active_data_mode}"
        )

        run_name = (
            f"{experiment_name}-{weight_mode}-{active_data_mode}-r{fl_rounds}-{run_timestamp}"
        )
        import uuid

        # --bucket reuses an existing bucket (the resume path); otherwise a
        # fresh bucket is generated per combination as before, which always
        # starts at round 0 -- see the --bucket help text above.
        bucket_name = args.bucket if args.bucket else f"fed-{uuid.uuid4().hex[:12]}"
        # F4 (gate fix): generated here instead of relying on KFP's
        # dsl.PIPELINE_JOB_ID_PLACEHOLDER, which this KFP deployment never
        # substitutes before component code runs (it arrives literally as
        # "{{$.pipeline_job_uuid}}", which is not a valid Kubernetes name
        # fragment). uuid4().hex is already lowercase hex, so this is
        # RFC-1123-safe by construction and stable for the lifetime of this
        # run, same as bucket_name above.
        run_uid = uuid.uuid4().hex[:8]

        arguments = {
            "num_workers": workers,
            "min_workers": min_workers,
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
            "temporal_address": temporal_address,
            "run_uid": run_uid,
            "seed": seed,
            # Informational only: the compiled pipeline's DAG shape already
            # baked in this start_round (see active_fl_pipeline.py's
            # START_ROUND handling), so this cannot change behavior after the
            # fact -- it is recorded here so the KFP run's arguments show what
            # was actually compiled in.
            "start_round": start_round,
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
            failures: list[str] = []
            for _, run_name in submitted_runs:
                try:
                    run_step(
                        [
                            "uv",
                            "run",
                            "python",
                            "analysis/fetch_k8s_runs.py",
                            "--experiment-prefix",
                            run_name,
                        ],
                        f"fetch results for {run_name}",
                    )
                except Exception as e:
                    failures.append(f"fetch results for {run_name}: {e}")

            log.info("✨ Done! Generating plots...")
            try:
                run_step(
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
                    "generate plots",
                )
                log.info("Plots generated in results/k8s_plots/")
            except Exception as e:
                failures.append(f"generate plots: {e}")

            if failures:
                log.error("post-processing failed for %d step(s):", len(failures))
                for f in failures:
                    log.error("  %s", f)
                raise SystemExit(1)


if __name__ == "__main__":
    main()
