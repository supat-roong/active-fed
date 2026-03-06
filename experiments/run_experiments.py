"""
Experiment Orchestrator

Reads config/local.yaml and runs Active-FL experiments locally.
Combinations to run are defined as a plain list under config.combinations.
Add or remove entries there to control which experiments run.

To run a one-off combination without editing the config:
  python experiments/run_experiments.py --mode single \
    --weight-mode data_only --active-data-mode bc

Examples:
  python experiments/run_experiments.py                         # runs config list
  python experiments/run_experiments.py --rounds 2 --workers 2  # override training params
  python experiments/run_experiments.py --mode single --weight-mode active --active-data-mode bc
  python experiments/run_experiments.py --config config/local.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment.local_runner import RunConfig, RunResult, run_experiment


# ---------------------------------------------------------------------------
# Subprocess worker (must be module-level to be picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _run_one(cfg: dict, wm: str, adm: str, device: str, output_dir: str) -> tuple[RunResult, str]:
    """Build config, run experiment, save result. Executes in a subprocess.

    Returns (result, log_output) so the parent process can flush subprocess
    logs — ProcessPoolExecutor workers don't forward their output automatically.
    """
    import io as _io
    import logging as _logging

    # Redirect all subprocess logging to a string buffer so the parent can print it.
    _log_buf = _io.StringIO()
    _handler = _logging.StreamHandler(_log_buf)
    _handler.setFormatter(
        _logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    )
    _root = _logging.getLogger()
    _root.handlers.clear()
    _root.addHandler(_handler)
    _root.setLevel(_logging.INFO)

    # MPS is not safe in spawned subprocesses — the Metal allocator context is
    # not inherited across process boundaries, causing "Placeholder storage has
    # not been allocated on MPS device". Fall back to CPU inside subprocesses;
    # the overhead is acceptable since each subprocess runs independently.
    if device == "mps":
        device = "cpu"

    run_cfg = build_run_config(cfg, wm, adm, device)
    result = run_experiment(run_cfg)
    save_result(result, output_dir)
    return result, _log_buf.getvalue()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_run_config(
    cfg: dict, weight_mode: str, active_data_mode: str, device: str = "cpu"
) -> RunConfig:
    """Build a RunConfig from the YAML config dict and a specific mode combination."""
    t = cfg.get("training", {})
    agg = cfg.get("aggregation", {})
    exp = cfg.get("experiment", {})
    return RunConfig(
        weight_mode=weight_mode,
        active_data_mode=active_data_mode,
        fl_rounds=t.get("fl_rounds", 5),
        num_workers=t.get("num_workers", 4),
        local_episodes=t.get("local_episodes", 100),
        eval_episodes=t.get("eval_episodes", 10),
        env_id=t.get("env_id", "CartPole-v1"),
        score_threshold=agg.get("score_threshold", 0.0),
        score_temperature=agg.get("score_temperature", 1.0),
        active_data_threshold=agg.get("active_data_threshold", 0.0),
        active_data_steps=agg.get("active_data_steps", 3),
        active_data_lr=agg.get("active_data_lr", 1e-4),
        seed=exp.get("seed", 42),
        device=device,
    )


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def result_to_dict(result: RunResult) -> dict:
    """Convert RunResult to a JSON-serializable dict."""
    rounds = []
    for r in result.rounds:
        rounds.append(
            {
                "fl_round": r.fl_round,
                "global_reward_mean": r.global_reward_mean,
                "global_reward_std": r.global_reward_std,
                "clients_accepted": r.clients_accepted,
                "clients_rejected": r.clients_rejected,
                "client_scores": r.client_scores,
                "client_improvements": r.client_improvements,
                "client_accepted": r.client_accepted,
                "client_target_env_rewards": r.client_target_env_rewards,
                "client_own_env_rewards": r.client_own_env_rewards,
                "effective_weight_norm": r.effective_weight_norm,
                "active_data_applied": r.active_data_applied,
                "active_data_n_steps": r.active_data_n_steps,
                "wall_time_s": r.wall_time_s,
                "solved": r.solved,
            }
        )
    return {
        "run_name": result.run_name,
        "weight_mode": result.weight_mode,
        "active_data_mode": result.active_data_mode,
        "final_reward_mean": result.final_reward_mean,
        "final_reward_std": result.final_reward_std,
        "total_wall_time_s": result.total_wall_time_s,
        "solved_at_round": result.solved_at_round,
        "metadata": result.metadata,
        "rounds": rounds,
    }


def save_result(result: RunResult, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"{result.run_name}.json")
    with open(path, "w") as f:
        json.dump(result_to_dict(result), f, indent=2)
    log.info(f"Saved result: {path}")
    return path


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------


def get_combinations(cfg: dict) -> list[tuple[str, str]]:
    """Return list of (weight_mode, active_data_mode) pairs from the config list."""
    combos = cfg.get("combinations", [])
    return [(c["weight_mode"], c["active_data_mode"]) for c in combos]


# ---------------------------------------------------------------------------
# Summary / progress printing
# ---------------------------------------------------------------------------


def print_summary(results: list[RunResult]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Run Name':<25} {'Final Reward':>14} {'Solved At':>10} {'Time (s)':>10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: -x.final_reward_mean):
        solved = str(r.solved_at_round) if r.solved_at_round is not None else "—"
        print(
            f"{r.run_name:<25} "
            f"{r.final_reward_mean:>8.1f} ± {r.final_reward_std:>5.1f} "
            f"{solved:>10} "
            f"{r.total_wall_time_s:>9.1f}s"
        )
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Active-FL experiments")
    parser.add_argument(
        "--config",
        default="config/local.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["single"],
        help="'single': run one combo via --weight-mode / --active-data-mode",
    )
    parser.add_argument("--weight-mode", default=None, help="Weight mode for --mode single")
    parser.add_argument(
        "--active-data-mode", default=None, help="Active data mode for --mode single"
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--rounds", type=int, default=None, help="Override fl_rounds")
    parser.add_argument("--workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--episodes", type=int, default=None, help="Override local_episodes")
    parser.add_argument("--no-viz", action="store_true", help="Skip auto-visualization after runs")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Max number of experiment combinations to run in parallel (default: all at once)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg.get("experiment", {})
    output_dir = args.output_dir or exp_cfg.get("output_dir", "results")

    # Allow CLI overrides on training params
    if args.rounds is not None:
        cfg.setdefault("training", {})["fl_rounds"] = args.rounds
    if args.workers is not None:
        cfg.setdefault("training", {})["num_workers"] = args.workers
    if args.episodes is not None:
        cfg.setdefault("training", {})["local_episodes"] = args.episodes

    # Determine combinations
    if args.mode == "single":
        if not args.weight_mode or not args.active_data_mode:
            parser.error("--mode single requires --weight-mode and --active-data-mode")
        combos = [(args.weight_mode, args.active_data_mode)]
    else:
        combos = get_combinations(cfg)

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info(f"Using device: {device}")

    log.info(f"Running {len(combos)} combination(s): {combos}")
    total_start = time.time()
    results: list[RunResult] = []

    if args.mode == "single" or len(combos) == 1:
        # Single combo: run directly in the main process (no subprocess overhead)
        wm, adm = combos[0]
        log.info(f"[1/1] weight_mode={wm} | active_data_mode={adm}")
        try:
            res, log_out = _run_one(cfg, wm, adm, device, output_dir)
            if log_out:
                print(log_out, end="", flush=True)
            results.append(res)
        except Exception as e:
            log.error(f"Run failed ({wm}+{adm}): {e}", exc_info=True)
    else:
        # Multiple combos: run in parallel subprocesses
        max_workers = args.jobs or len(combos)
        log.info(f"Launching {len(combos)} experiment(s) in parallel (max_workers={max_workers})")
        future_to_combo: dict = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for wm, adm in combos:
                fut = pool.submit(_run_one, cfg, wm, adm, device, output_dir)
                future_to_combo[fut] = (wm, adm)
            for fut in as_completed(future_to_combo):
                wm, adm = future_to_combo[fut]
                try:
                    result, subprocess_log = fut.result()
                    # Flush subprocess output: it's invisible otherwise
                    if subprocess_log:
                        print(subprocess_log, end="", flush=True)
                    results.append(result)
                    log.info(f"  Finished: {wm}+{adm}")
                except Exception as e:
                    log.error(f"Run failed ({wm}+{adm}): {e}", exc_info=True)

    total_time = time.time() - total_start
    log.info(f"\nAll {len(results)}/{len(combos)} runs completed in {total_time:.1f}s")
    print_summary(results)

    # Auto-run visualization
    if not args.no_viz and results:
        log.info("Generating comparison plots...")
        viz_cfg = cfg.get("visualization", {})
        plots_dir = (
            args.output_dir
            and os.path.join(args.output_dir, "plots")
            or viz_cfg.get("output_dir", "results/plots")
        )
        try:
            from analysis.compare_runs import generate_all_plots

            generate_all_plots(output_dir, plots_dir, viz_cfg)
            log.info(f"Plots saved to: {plots_dir}")
        except Exception as e:
            log.warning(
                f"Visualization failed: {e}. Run `python analysis/compare_runs.py` manually."
            )


if __name__ == "__main__":
    main()
