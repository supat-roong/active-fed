"""
Fetch K8s Runs

Fetches Active-FL K8s pipeline run metrics from MLflow and saves them
as JSON files in the same format as local experiments. This allows
using `compare_runs.py` directly on the K8s runs.

Usage:
  python analysis/fetch_k8s_runs.py
  python analysis/fetch_k8s_runs.py --tracking-uri http://localhost:5050 --output-dir results/k8s_results/
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def fetch_experiments(tracking_uri: str, output_dir: str, experiment_prefix: str = None):
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # K8s run format: {experiment_name}-{weight_mode}-{active_data_mode}-r{rounds}-{timestamp}
    # We look for experiments starting with "active-fl-cartpole" (or anything ending with timestamp if we wanted)
    experiments = client.search_experiments()
    k8s_exps = [e for e in experiments if e.name != "Default"]

    if experiment_prefix:
        k8s_exps = [e for e in k8s_exps if e.name.startswith(experiment_prefix)]

    if not k8s_exps:
        log.warning(
            f"No custom MLflow experiments found matching prefix '{experiment_prefix}'. Have you run the K8s pipeline?"
        )
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for exp in k8s_exps:
        log.info(f"Processing experiment: {exp.name}")

        # Parse modes from name
        # Format usually: active-fl-cartpole-{weight_mode}-{active_data_mode}-r{fl_rounds}-{timestamp}
        # Since weight_mode might be 'active', 'fedavg', 'data_only' and active_data_mode 'none', 'bc',
        # we can just try to extract them if they exist.
        name_parts = exp.name.split("-")

        weight_mode = "unknown"
        active_data_mode = "unknown"

        for wm in ["active", "fedavg", "data_only"]:
            if wm in name_parts:
                weight_mode = wm
                break

        for adm in ["bc", "none"]:
            if adm in name_parts:
                active_data_mode = adm
                break

        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        if not runs:
            log.info(f"  No runs found in {exp.name}, skipping.")
            continue

        # Group runs by round. K8s logs runs with name "round_X"
        round_runs = []
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "")
            m = re.match(r"^round_(\d+)$", run_name)
            if m:
                round_runs.append((int(m.group(1)), run))

        if not round_runs:
            log.warning(f"  No valid round runs found in {exp.name}")
            continue

        round_runs.sort(key=lambda x: x[0])

        # Build the result dictionary matching the local json output
        rounds_data = []
        solved_at = None

        for fl_round, run in round_runs:
            metrics = run.data.metrics

            # Extract client-specific metrics
            client_scores = {}
            client_improvements = {}
            client_accepted = {}
            client_own_env_rewards = {}
            client_target_env_rewards = {}

            for key, val in metrics.items():
                m_score = re.match(r"^client_(.*)_score$", key)
                if m_score:
                    client_scores[m_score.group(1)] = val

                m_imp = re.match(r"^client_(.*)_improvement$", key)
                if m_imp:
                    client_improvements[m_imp.group(1)] = val

                m_acc = re.match(r"^client_(.*)_accepted$", key)
                if m_acc:
                    client_accepted[m_acc.group(1)] = bool(val)

                m_own = re.match(r"^client_(.*)_own_env_reward_mean$", key)
                if m_own:
                    client_own_env_rewards[m_own.group(1)] = val

                m_tgt = re.match(r"^client_(.*)_target_env_reward_mean$", key)
                if m_tgt:
                    client_target_env_rewards[m_tgt.group(1)] = val

            solved = bool(metrics.get("solved", 0.0))
            if solved and solved_at is None:
                solved_at = fl_round

            rounds_data.append(
                {
                    "fl_round": fl_round,
                    "global_reward_mean": metrics.get("global_eval_reward_mean", 0.0),
                    "global_reward_std": metrics.get("global_eval_reward_std", 0.0),
                    "clients_accepted": int(metrics.get("clients_accepted", 0)),
                    "clients_rejected": int(metrics.get("clients_rejected", 0)),
                    "client_scores": client_scores,
                    "client_improvements": client_improvements,
                    "client_accepted": client_accepted,
                    "client_own_env_rewards": client_own_env_rewards,
                    "client_target_env_rewards": client_target_env_rewards,
                    "effective_weight_norm": metrics.get("effective_weight_norm", 0.0),
                    "active_data_applied": bool(metrics.get("active_data_applied", 0.0)),
                    "active_data_n_steps": int(metrics.get("active_data_n_steps", 0)),
                    "wall_time_s": 0.0,  # Not tracked in MLflow currently
                    "solved": solved,
                }
            )

        if not rounds_data:
            continue

        final_round = rounds_data[-1]

        result_dict = {
            "run_name": exp.name,
            "weight_mode": weight_mode,
            "active_data_mode": active_data_mode,
            "final_reward_mean": final_round["global_reward_mean"],
            "final_reward_std": final_round["global_reward_std"],
            "total_wall_time_s": 0.0,
            "solved_at_round": solved_at,
            "metadata": {"source": "mlflow", "tracking_uri": tracking_uri},
            "rounds": rounds_data,
        }

        out_path = os.path.join(output_dir, f"{exp.name}.json")
        with open(out_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        log.info(f"  Saved {len(rounds_data)} rounds to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch K8s pipeline runs from MLflow to JSON")
    parser.add_argument(
        "--tracking-uri", default="http://localhost:5050", help="MLflow Tracking URI"
    )
    parser.add_argument(
        "--output-dir", default="results/k8s_results", help="Output directory for JSONs"
    )
    parser.add_argument(
        "--experiment-prefix", default=None, help="Prefix to filter MLflow experiments"
    )
    args = parser.parse_args()

    fetch_experiments(args.tracking_uri, args.output_dir, args.experiment_prefix)


if __name__ == "__main__":
    main()
