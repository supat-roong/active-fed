"""
Active Federated Averaging Aggregator

Supports three weight aggregation modes and one active data mode:

  Weight mode (--weight-mode):
    "active"    — scored softmax FedAvg  (importance-weighted, default)
    "fedavg"    — plain equal-weight FedAvg (standard FL, no scoring)
    "data_only" — skip weight aggregation entirely; only Active Data updates global

  Active Data mode (--active-data-mode):
    "none" — no trajectory fine-tuning (default)
    "bc"   — behavioral cloning on high-value eval trajectories

Modes can be mixed freely, e.g.:
  weight_mode="data_only" + active_data_mode="bc"  → pure active data (no FL weight avg)
  weight_mode="active"   + active_data_mode="bc"  → both active paths combined
  weight_mode="fedavg"   + active_data_mode="none" → vanilla FedAvg baseline
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import numpy as np
import torch

from src.aggregator.scorer import ClientUpdate, ScoredClient
from src.aggregator.active_data import (
    ActiveDataMode,
    ActiveDataUpdater,
    ActiveDataset,
    build_active_dataset,
)
from src.aggregator.evaluator import EvalResult

log = logging.getLogger(__name__)

WeightMode = Literal["active", "fedavg", "data_only"]


@dataclasses.dataclass
class AggregationResult:
    global_weights: dict[str, torch.Tensor]  # final W_global (after both active paths)
    scored_clients: list[ScoredClient]
    accepted_ids: list[int]
    rejected_ids: list[int]
    aggregation_weights: dict[int, float]  # per-accepted-client weight (or equal for fedavg)
    round_summary: dict  # for MLflow logging
    weight_mode: str = "active"  # which weight aggregation was used
    # Active data metadata
    active_data_applied: bool = False
    active_data_n_steps: int = 0
    active_data_source_workers: list[int] = dataclasses.field(default_factory=list)
    active_data_metrics: dict = dataclasses.field(default_factory=dict)


def aggregate(
    global_weights: dict[str, torch.Tensor],
    clients: list[ClientUpdate],
    scored_clients: list[ScoredClient],
    eval_results: dict[int, EvalResult] | None = None,
    temperature: float = 1.0,
    # Weight aggregation mode
    weight_mode: WeightMode = "active",
    # Active data options
    active_data_mode: ActiveDataMode = "none",
    active_data_threshold: float = 0.0,
    active_data_steps: int = 3,
    active_data_lr: float = 1e-4,
    # Hardware
    device: str = "cpu",
) -> AggregationResult:
    """
    Aggregation with three independent weight modes + optional active data fine-tune.

    weight_mode:
      "active"    — scored softmax FedAvg (uses scored_clients scores)
      "fedavg"    — plain equal-weight FedAvg (all accepted clients equally)
      "data_only" — skip weight averaging; start from W_global, apply active data only

    active_data_mode:
      "none" — no fine-tuning
      "bc"   — behavioral cloning on high-value eval trajectories

    Modes combine freely. "data_only" + "bc" = pure active data (no FL weight averaging).
    """
    client_map = {c.worker_id: c for c in clients}

    # FedAvg is a no-scoring baseline — all clients are unconditionally accepted.
    # For active / data_only, respect the scored accept/reject decisions.
    if weight_mode == "fedavg":
        accepted = list(scored_clients)  # all clients accepted
        rejected: list[ScoredClient] = []
    else:
        accepted = [s for s in scored_clients if s.accepted]
        rejected = [s for s in scored_clients if not s.accepted]

    accepted_ids = [s.worker_id for s in accepted]
    rejected_ids = [s.worker_id for s in rejected]

    # ---- Path 1: Weight Aggregation ----
    agg_weight_map: dict[int, float] = {}
    effective_norm = 0.0

    if weight_mode == "data_only":
        # Skip weight aggregation — use global model as-is
        new_global = {k: v.clone() for k, v in global_weights.items()}
        log.info("[WeightMode:data_only] Skipping FL weight aggregation")

    elif weight_mode == "fedavg":
        # Plain equal-weight FedAvg (all accepted clients, no scoring)
        if not accepted:
            log.warning("[WeightMode:fedavg] All clients rejected, keeping global weights")
            new_global = global_weights
        else:
            equal_w = 1.0 / len(accepted)
            agg_weight_map = {s.worker_id: equal_w for s in accepted}
            new_global = {}
            for key in global_weights:
                avg = sum(client_map[s.worker_id].weights[key].float() for s in accepted) / len(
                    accepted
                )
                new_global[key] = avg.to(global_weights[key].dtype)
            effective_norm = float(
                sum(
                    torch.norm(new_global[k].float() - global_weights[k].float()) ** 2
                    for k in global_weights
                )
                ** 0.5
            )
            log.info(f"[WeightMode:fedavg] Equal-weight avg of {len(accepted)} clients")

    else:  # weight_mode == "active"
        # Scored softmax FedAvg
        if not accepted:
            log.warning("[WeightMode:active] All clients rejected, keeping global weights")
            new_global = global_weights
        else:
            scores = np.array([s.score for s in accepted])
            exp_scores = np.exp((scores - scores.max()) / temperature)
            softmax_weights = exp_scores / exp_scores.sum()
            agg_weight_map = {s.worker_id: float(w) for s, w in zip(accepted, softmax_weights)}

            log.info(
                f"[WeightMode:active] {len(accepted)} clients | "
                f"weights={{{', '.join(f'{wid}:{w:.3f}' for wid, w in agg_weight_map.items())}}}"
            )

            new_global = {}
            for key in global_weights:
                weighted_sum = sum(
                    agg_weight_map[s.worker_id] * client_map[s.worker_id].weights[key].float()
                    for s in accepted
                )
                new_global[key] = weighted_sum.to(global_weights[key].dtype)

            effective_norm = float(
                sum(
                    torch.norm(new_global[k].float() - global_weights[k].float()) ** 2
                    for k in global_weights
                )
                ** 0.5
            )

    # ---- Path 2: Active Data (trajectory fine-tuning) ----
    active_data_applied = False
    active_data_n_steps = 0
    active_data_source_workers: list[int] = []
    active_data_metrics: dict = {}

    if active_data_mode != "none" and eval_results is not None:
        # Collect all trajectories from eval results
        all_trajectories = []
        for result in eval_results.values():
            all_trajectories.extend(result.trajectories)

        if all_trajectories:
            dataset = build_active_dataset(
                trajectories=all_trajectories,
                data_threshold=active_data_threshold,
            )
            if dataset is not None:
                updater = ActiveDataUpdater(
                    mode=active_data_mode,
                    lr=active_data_lr,
                    n_epochs=active_data_steps,
                    device=device,
                )
                new_global, ad_metrics = updater.update(new_global, dataset)
                active_data_applied = True
                active_data_n_steps = dataset.n_steps
                active_data_source_workers = dataset.source_workers
                active_data_metrics = ad_metrics
                log.info(
                    f"[ActiveData:{active_data_mode}] Applied to global model | "
                    f"steps={active_data_n_steps} | "
                    f"workers={active_data_source_workers}"
                )
        else:
            log.info("[ActiveData] No trajectory data found (collect_data=False?), skipping")
    elif active_data_mode != "none":
        log.warning("[ActiveData] eval_results not provided — cannot collect trajectory data")

    round_summary = {
        "weight_mode": weight_mode,
        "clients_accepted": len(accepted_ids),
        "clients_rejected": len(rejected_ids),
        "effective_weight_norm": effective_norm,
        "active_data_mode": active_data_mode,
        "active_data_applied": active_data_applied,
        "active_data_n_steps": active_data_n_steps,
        "active_data_source_workers": active_data_source_workers,
        **active_data_metrics,
    }
    if weight_mode != "data_only" and accepted:
        round_summary["top_client"] = accepted[0].worker_id
        round_summary["top_client_improvement"] = accepted[0].improvement
        round_summary["aggregation_weights"] = agg_weight_map

    log.info(
        f"Aggregation done | weight_mode={weight_mode} | "
        f"accepted={len(accepted_ids)} | "
        f"active_data={active_data_applied} ({active_data_mode})"
    )

    return AggregationResult(
        global_weights=new_global,
        scored_clients=scored_clients,
        accepted_ids=accepted_ids,
        rejected_ids=rejected_ids,
        aggregation_weights=agg_weight_map,
        round_summary=round_summary,
        weight_mode=weight_mode,
        active_data_applied=active_data_applied,
        active_data_n_steps=active_data_n_steps,
        active_data_source_workers=active_data_source_workers,
        active_data_metrics=active_data_metrics,
    )
