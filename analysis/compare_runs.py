"""
Comparison Visualization

Loads experiment results from JSON files in the results/ directory and
generates comparison plots:
  1. Learning curves    — reward vs fl_round, one line per combination
  2. Final reward bar   — bar chart ranked by final reward
  3. Heatmap            — weight_mode × active_data_mode final reward grid
  4. Acceptance rate    — % clients accepted per round, per combination
  5. Active data usage  — rounds where active data was applied, per combo

Usage:
  python analysis/compare_runs.py                         # use results/ dir
  python analysis/compare_runs.py --input-dir results/
  python analysis/compare_runs.py --input-dir results/ --output-dir results/plots/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

LINE_STYLES = {
    "none": "-",
    "bc": "--",
}
MARKERS = {
    "none": "o",
    "bc": "s",
}
DEFAULT_COLORS = {
    "active": "#4C72B0",
    "fedavg": "#DD8452",
    "data_only": "#55A868",
}
SOLVED_LINE_COLOR = "#C44E52"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(input_dir: str) -> list[dict]:
    """Load all JSON result files from input_dir."""
    results = []
    for path in sorted(Path(input_dir).glob("*.json")):
        with open(path) as f:
            results.append(json.load(f))
    log.info(f"Loaded {len(results)} results from {input_dir}")
    return results


def _label(r: dict) -> str:
    return r.get("run_name", f"{r['weight_mode']}+{r['active_data_mode']}")


# ---------------------------------------------------------------------------
# Plot 1: Learning Curves
# ---------------------------------------------------------------------------


def _rolling_mean(vals: list[float], w: int) -> list[float]:
    """Compute expanding-then-fixed rolling mean."""
    out = []
    for i in range(len(vals)):
        start = max(0, i - w + 1)
        out.append(sum(vals[start : i + 1]) / (i - start + 1))
    return out


def plot_learning_curves(
    results: list[dict],
    output_path: str,
    solved_threshold: float = 195.0,
    show_std: bool = True,
    colors: dict | None = None,
    dpi: int = 150,
    smooth_window: int = 10,
) -> None:
    colors = colors or DEFAULT_COLORS
    fig, ax = plt.subplots(figsize=(11, 6))

    for r in results:
        wm = r["weight_mode"]
        adm = r["active_data_mode"]
        rounds_data = r["rounds"]
        if not rounds_data:
            continue

        xs = [rd["fl_round"] + 1 for rd in rounds_data]
        means = [rd["global_reward_mean"] for rd in rounds_data]
        stds = [rd["global_reward_std"] for rd in rounds_data]
        smoothed = _rolling_mean(means, smooth_window)
        color = colors.get(wm, "#888888")
        ls = LINE_STYLES.get(adm, "-")
        label = _label(r)

        # Raw reward — faint background
        ax.plot(xs, means, color=color, linestyle=ls, linewidth=0.6, alpha=0.2, zorder=1)
        # Smoothed trend — bold foreground
        ax.plot(xs, smoothed, color=color, linestyle=ls, linewidth=2.2, label=label, zorder=2)
        if show_std:
            sm_np = np.array(smoothed)
            stds_np = np.array(stds)
            ax.fill_between(xs, sm_np - stds_np, sm_np + stds_np, color=color, alpha=0.08)

    # Solved threshold
    ax.axhline(
        solved_threshold,
        color=SOLVED_LINE_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=f"Solved ({solved_threshold:.0f})",
    )

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Global Eval Reward (mean)", fontsize=12)
    ax.set_title(
        f"Active-FL: Learning Curves by Mode Combination (rolling avg, window={smooth_window})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    # Legend for line styles (active data mode encoding)
    style_patches = [
        mpatches.Patch(color="grey", label="solid  = active_data: none"),
        mpatches.Patch(color="grey", linestyle="--", label="dashed = active_data: bc"),
    ]
    fig.legend(
        handles=style_patches,
        loc="lower center",
        ncol=2,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Final Reward Bar Chart
# ---------------------------------------------------------------------------


def plot_final_reward_bar(
    results: list[dict],
    output_path: str,
    solved_threshold: float = 195.0,
    colors: dict | None = None,
    dpi: int = 150,
) -> None:
    colors = colors or DEFAULT_COLORS
    sorted_results = sorted(results, key=lambda r: -r["final_reward_mean"])

    labels = [_label(r) for r in sorted_results]
    means = [r["final_reward_mean"] for r in sorted_results]
    stds = [r["final_reward_std"] for r in sorted_results]
    bar_colors = [colors.get(r["weight_mode"], "#888888") for r in sorted_results]
    hatch_map = {"none": "", "bc": "//"}
    hatches = [hatch_map.get(r["active_data_mode"], "") for r in sorted_results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=bar_colors,
        hatch=hatches,
        capsize=5,
        width=0.6,
        edgecolor="white",
        linewidth=1.2,
    )

    # Solved threshold line
    ax.axhline(
        solved_threshold,
        color=SOLVED_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Solved ({solved_threshold:.0f})",
    )

    # Value annotations
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 1,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Final Global Eval Reward", fontsize=12)
    ax.set_title("Final Reward by Mode Combination (ranked)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Color legend (weight_mode)
    color_patches = [mpatches.Patch(facecolor=c, label=wm) for wm, c in colors.items()]
    ax.legend(
        handles=color_patches
        + [mpatches.Patch(facecolor="grey", label="Solved line", linestyle="--")],
        fontsize=8,
        loc="upper right",
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Heatmap — weight_mode × active_data_mode
# ---------------------------------------------------------------------------


def plot_heatmap(
    results: list[dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    wm_order = ["active", "fedavg", "data_only"]
    adm_order = ["none", "bc"]

    # Build matrix
    matrix = np.full((len(wm_order), len(adm_order)), np.nan)
    for r in results:
        wm = r["weight_mode"]
        adm = r["active_data_mode"]
        if wm in wm_order and adm in adm_order:
            wi, ai = wm_order.index(wm), adm_order.index(adm)
            matrix[wi, ai] = r["final_reward_mean"]

    fig, ax = plt.subplots(figsize=(7, 5))
    # Use a masked array so NaN cells show as blank
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="YlGn", vmin=0, aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Final Reward", fontsize=11)

    ax.set_xticks(range(len(adm_order)))
    ax.set_xticklabels(adm_order, fontsize=11)
    ax.set_yticks(range(len(wm_order)))
    ax.set_yticklabels(wm_order, fontsize=11)
    ax.set_xlabel("Active Data Mode", fontsize=12)
    ax.set_ylabel("Weight Mode", fontsize=12)
    ax.set_title(
        "Final Reward Heatmap\n(weight_mode × active_data_mode)", fontsize=13, fontweight="bold"
    )

    # Annotate cells
    for (i, j), val in np.ndenumerate(matrix):
        if not np.isnan(val):
            color = "white" if val < (np.nanmax(matrix) * 0.6) else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
            )
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#aaaaaa")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: Client Acceptance Rate
# ---------------------------------------------------------------------------


def plot_acceptance_rate(
    results: list[dict],
    output_path: str,
    colors: dict | None = None,
    dpi: int = 150,
    smooth_window: int = 10,
) -> None:
    colors = colors or DEFAULT_COLORS
    fig, ax = plt.subplots(figsize=(11, 5))

    for r in results:
        rounds_data = r["rounds"]
        if not rounds_data:
            continue

        # Skip data_only (no weight aggregation, acceptance is trivial)
        if r["weight_mode"] == "data_only":
            continue

        xs = [rd["fl_round"] + 1 for rd in rounds_data]
        total = [rd["clients_accepted"] + rd["clients_rejected"] for rd in rounds_data]
        accepted = [rd["clients_accepted"] for rd in rounds_data]

        # FedAvg accepts all clients by definition — override any stored data
        if r["weight_mode"] == "fedavg":
            rates = [100.0] * len(xs)
            smoothed = rates
        else:
            rates = [a / t * 100 if t > 0 else 0 for a, t in zip(accepted, total)]
            smoothed = _rolling_mean(rates, smooth_window)

        color = colors.get(r["weight_mode"], "#888888")
        ls = LINE_STYLES.get(r["active_data_mode"], "-")
        label = _label(r)

        # Raw data — faint background
        ax.plot(xs, rates, color=color, linestyle=ls, linewidth=0.6, alpha=0.2, zorder=1)
        # Smoothed trend — bold foreground
        ax.plot(xs, smoothed, color=color, linestyle=ls, linewidth=2.2, label=label, zorder=2)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Client Acceptance Rate (%)", fontsize=12)
    ax.set_title(
        f"Client Acceptance Rate per Round (rolling avg, window={smooth_window})\n"
        "(weight_mode: active / fedavg only)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 108)
    ax.axhline(100, color="#aaaaaa", linestyle=":", linewidth=1)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 5: Active Data Application
# ---------------------------------------------------------------------------


def plot_active_data_usage(
    results: list[dict],
    output_path: str,
    colors: dict | None = None,
    dpi: int = 150,
) -> None:
    """Show which rounds had active data applied and how many steps."""
    colors = colors or DEFAULT_COLORS
    # Only show results that actually use active data
    ad_results = [r for r in results if r["active_data_mode"] != "none"]
    if not ad_results:
        log.info("No active-data runs found, skipping active data usage plot")
        return

    fig, axes = plt.subplots(
        1,
        len(ad_results),
        figsize=(max(6, 2.5 * len(ad_results)), 4),
        sharey=True,
    )
    if len(ad_results) == 1:
        axes = [axes]

    for ax, r in zip(axes, ad_results):
        rounds_data = r["rounds"]
        xs = [rd["fl_round"] + 1 for rd in rounds_data]
        steps = [
            rd["active_data_n_steps"] if rd["active_data_applied"] else 0 for rd in rounds_data
        ]
        color = colors.get(r["weight_mode"], "#888888")
        import matplotlib.colors as mcolors
        dark_color = tuple(c * 0.6 for c in mcolors.to_rgb(color))
        
        ax.bar(xs, steps, color=dark_color, alpha=1.0, edgecolor="none", width=1.0)
        ax.set_title(_label(r), fontsize=9)
        ax.set_xlabel("Round")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Active Data Steps Applied", fontsize=11)
    fig.suptitle("Active Data Usage per Round", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 6: Improvement per client over time
# ---------------------------------------------------------------------------


def plot_client_improvements(
    results: list[dict],
    output_path: str,
    colors: dict | None = None,
    dpi: int = 150,
    smooth_window: int = 10,
) -> None:
    colors = colors or DEFAULT_COLORS
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8), squeeze=False)

    client_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (r, ax) in enumerate(zip(results, axes.flat)):
        rounds_data = r["rounds"]
        client_ids = sorted({k for rd in rounds_data for k in rd["client_improvements"]})
        for ci, cid in enumerate(client_ids):
            xs, ys = [], []
            for rd in rounds_data:
                if str(cid) in rd["client_improvements"]:
                    xs.append(rd["fl_round"] + 1)
                    ys.append(rd["client_improvements"][str(cid)])
            if not ys:
                continue
            smoothed = _rolling_mean(ys, smooth_window)
            color = client_colors[ci % len(client_colors)]
            # Raw — faint background
            ax.plot(xs, ys, color=color, linewidth=0.5, alpha=0.2, zorder=1)
            # Smoothed trend — bold foreground
            ax.plot(xs, smoothed, color=color, linewidth=1.8, label=f"w{cid}", zorder=2)

        ax.axhline(0, color="#aaaaaa", linestyle=":", linewidth=1)
        ax.set_title(_label(r), fontsize=9)
        ax.set_xlabel("Round", fontsize=8)
        ax.set_ylabel("Target-Env Improvement", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, nrows * ncols):
        fig.axes[j].set_visible(False)

    fig.suptitle(
        f"Per-Client Target-Env Improvement per Round (rolling avg, window={smooth_window})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 7: Per-Worker Learning Curves
# ---------------------------------------------------------------------------


def plot_worker_metric_curves(
    results: list[dict],
    output_path: str,
    metric_key: str,
    title_prefix: str,
    solved_threshold: float = 195.0,
    colors: dict | None = None,
    dpi: int = 150,
    smooth_window: int = 10,
) -> None:
    """
    For each run combination, plot each worker's specific reward metric over rounds.
    The global model reward is also shown as a bold background reference.
    """
    colors = colors or DEFAULT_COLORS
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8), squeeze=False)

    client_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (r, ax) in enumerate(zip(results, axes.flat)):
        rounds_data = r["rounds"]
        if not rounds_data:
            ax.set_visible(False)
            continue

        xs = [rd["fl_round"] + 1 for rd in rounds_data]
        global_means = [rd["global_reward_mean"] for rd in rounds_data]
        global_smoothed = _rolling_mean(global_means, smooth_window)
        run_color = colors.get(r["weight_mode"], "#888888")

        # Global model reward (reference)
        ax.plot(xs, global_means, color=run_color, linewidth=0.6, alpha=0.2, zorder=1)
        ax.plot(
            xs,
            global_smoothed,
            color=run_color,
            linewidth=2.5,
            linestyle="-",
            label="global",
            zorder=2,
            alpha=0.9,
        )

        # Per-worker estimated reward
        client_ids = set()
        for rd in rounds_data:
            client_ids.update(rd.get(metric_key, {}).keys())
            if not rd.get(metric_key) and metric_key == "client_target_env_rewards":
                client_ids.update(rd.get("client_improvements", {}).keys())
        client_ids = sorted(list(client_ids))

        for ci, cid in enumerate(client_ids):
            worker_xs, worker_ys = [], []
            for rd in rounds_data:
                val = rd.get(metric_key, {}).get(str(cid))
                if val is None and metric_key == "client_target_env_rewards":
                    imp = rd.get("client_improvements", {}).get(str(cid))
                    if imp is not None:
                        val = rd["global_reward_mean"] + imp
                if val is not None:
                    worker_xs.append(rd["fl_round"] + 1)
                    worker_ys.append(val)
            if not worker_ys:
                continue
            smoothed = _rolling_mean(worker_ys, smooth_window)
            wcolor = client_colors[ci % len(client_colors)]
            # Raw — faint
            ax.plot(worker_xs, worker_ys, color=wcolor, linewidth=0.5, alpha=0.15, zorder=1)
            # Smoothed — bold
            ax.plot(
                worker_xs,
                smoothed,
                color=wcolor,
                linewidth=1.6,
                linestyle="--",
                label=f"w{cid}",
                zorder=3,
            )

        ax.axhline(solved_threshold, color="#C44E52", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.set_title(_label(r), fontsize=9, fontweight="bold")
        ax.set_xlabel("FL Round", fontsize=8)
        ax.set_ylabel(f"{title_prefix} Reward", fontsize=8)
        ax.legend(fontsize=7, ncol=3, framealpha=0.85)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, nrows * ncols):
        fig.axes[j].set_visible(False)

    fig.suptitle(
        f"Worker {title_prefix} Learning Curves (rolling avg, window={smooth_window})\n"
        f"Bold = global model | Dashed = per-worker {title_prefix.lower()}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------


def generate_all_plots(
    input_dir: str,
    output_dir: str,
    viz_cfg: dict | None = None,
) -> None:
    cfg = viz_cfg or {}
    results = load_results(input_dir)
    if not results:
        log.warning(f"No results found in {input_dir}")
        return

    colors = cfg.get("color_palette", DEFAULT_COLORS)
    solved = cfg.get("solved_threshold", 195.0)
    show_std = cfg.get("show_std", True)
    dpi = cfg.get("figure_dpi", 150)

    plot_learning_curves(
        results,
        os.path.join(output_dir, "learning_curves.png"),
        solved_threshold=solved,
        show_std=show_std,
        colors=colors,
        dpi=dpi,
    )
    plot_final_reward_bar(
        results,
        os.path.join(output_dir, "final_reward_bar.png"),
        solved_threshold=solved,
        colors=colors,
        dpi=dpi,
    )
    plot_heatmap(
        results,
        os.path.join(output_dir, "heatmap.png"),
        dpi=dpi,
    )
    plot_acceptance_rate(
        results,
        os.path.join(output_dir, "acceptance_rate.png"),
        colors=colors,
        dpi=dpi,
    )
    plot_active_data_usage(
        results,
        os.path.join(output_dir, "active_data_usage.png"),
        colors=colors,
        dpi=dpi,
    )
    plot_client_improvements(
        results,
        os.path.join(output_dir, "client_improvements.png"),
        colors=colors,
        dpi=dpi,
    )
    plot_worker_metric_curves(
        results,
        os.path.join(output_dir, "worker_own_env_curves.png"),
        metric_key="client_own_env_rewards",
        title_prefix="Own-Env",
        solved_threshold=solved,
        colors=colors,
        dpi=dpi,
    )
    plot_worker_metric_curves(
        results,
        os.path.join(output_dir, "worker_target_env_curves.png"),
        metric_key="client_target_env_rewards",
        title_prefix="Target-Env",
        solved_threshold=solved,
        colors=colors,
        dpi=dpi,
    )
    log.info(f"All plots saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Compare Active-FL experiment results")
    parser.add_argument("--input-dir", default="results", help="Directory with run json files")
    parser.add_argument("--output-dir", default="results/plots", help="Directory to save plots")
    parser.add_argument(
        "--config", default="config/local.yaml", help="Config YAML for visualization settings"
    )
    args = parser.parse_args()

    viz_cfg: dict = {}
    if os.path.exists(args.config):
        import yaml

        with open(args.config) as f:
            viz_cfg = yaml.safe_load(f).get("visualization", {})

    generate_all_plots(args.input_dir, args.output_dir, viz_cfg)
    print(f"\nPlots saved to: {args.output_dir}/")
    for name in [
        "learning_curves.png",
        "final_reward_bar.png",
        "heatmap.png",
        "acceptance_rate.png",
        "active_data_usage.png",
        "client_improvements.png",
        "worker_own_env_curves.png",
        "worker_target_env_curves.png",
    ]:
        path = os.path.join(args.output_dir, name)
        if os.path.exists(path):
            print(f"  {name}")


if __name__ == "__main__":
    main()
