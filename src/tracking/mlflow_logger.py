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


def log_run_context(
    kfp_run_id: str,
    temporal_workflow_id: str,
    topology: str,
    kfp_base_url: str = "http://localhost:8080",
    temporal_base_url: str = "http://localhost:8233",
) -> None:
    """Tag the active MLflow run with cross-links to its KFP round and its
    Temporal worker fleet, so the four observability surfaces (KFP, Temporal,
    MLflow, Kubernetes Dashboard) can be navigated between instead of
    correlated by hand across three UIs by timestamp.

    Sets up to five tags: kfp_run_id, kfp_run_url, temporal_workflow_id,
    temporal_workflow_url, topology. Tracking must never fail a training run
    that has been going for minutes, so the whole body is wrapped in
    try/except with a warning on failure -- matching the defensive style of
    fed-twin/src/core/tracking.py's log_metrics. Empty ids are skipped rather
    than written as empty tags: an empty kfp_run_id tag is worse than none,
    because it looks like data.
    """
    try:
        tags: dict[str, str] = {}
        if kfp_run_id:
            tags["kfp_run_id"] = kfp_run_id
            tags["kfp_run_url"] = f"{kfp_base_url}/#/runs/details/{kfp_run_id}"
        if temporal_workflow_id:
            tags["temporal_workflow_id"] = temporal_workflow_id
            tags["temporal_workflow_url"] = (
                f"{temporal_base_url}/namespaces/default/workflows/{temporal_workflow_id}"
            )
        if topology:
            tags["topology"] = topology
        if tags:
            mlflow.set_tags(tags)
    except Exception as e:
        log.warning(f"Failed to log MLflow run-context tags: {e}")


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
