"""
MLflow logging helpers for Active-FL pipeline.

Centralises all MLflow calls to keep pipeline components clean.
"""

from __future__ import annotations

import io
import logging

import mlflow
import torch

from src.aggregator.aggregator import AggregationResult

log = logging.getLogger(__name__)


def log_round_metrics(
    fl_round: int,
    agg_result: AggregationResult,
    global_eval_reward: float,
    global_eval_std: float,
    run_name: str = "aggregator",
) -> None:
    """Log per-round aggregation and evaluation metrics to MLflow."""
    summary = agg_result.round_summary

    flat_metrics: dict[str, float] = {
        "global_eval_reward_mean": global_eval_reward,
        "global_eval_reward_std": global_eval_std,
        "clients_accepted": float(summary.get("clients_accepted", 0)),
        "clients_rejected": float(summary.get("clients_rejected", 0)),
        "effective_weight_norm": float(summary.get("effective_weight_norm", 0.0)),
    }

    # Per-client scores and acceptance
    for sc in agg_result.scored_clients:
        flat_metrics[f"client_{sc.worker_id}_score"] = sc.score
        flat_metrics[f"client_{sc.worker_id}_improvement"] = sc.improvement
        flat_metrics[f"client_{sc.worker_id}_accepted"] = float(sc.accepted)

    mlflow.log_metrics(flat_metrics, step=fl_round)
    log.info(f"[MLflow] Round {fl_round} | global_reward={global_eval_reward:.2f}")


def log_global_model(
    global_weights: dict[str, torch.Tensor],
    fl_round: int,
    artifact_path: str = "global_models",
) -> None:
    """Save global model checkpoint as MLflow artifact."""
    buf = io.BytesIO()
    torch.save(global_weights, buf)
    buf.seek(0)

    with mlflow.start_run(nested=True):
        mlflow.log_artifact(buf, artifact_path=f"{artifact_path}/round_{fl_round}")
