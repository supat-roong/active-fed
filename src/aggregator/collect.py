"""
MinIO Weight Collector

Downloads all worker weight files from MinIO for a given FL round,
assembles them into ClientUpdate dataclass instances ready for scoring.
"""

from __future__ import annotations

import io
import json
import logging

import torch
from minio import Minio

from src.aggregator.scorer import ClientUpdate

log = logging.getLogger(__name__)


def collect_worker_updates(
    minio_client: Minio,
    bucket: str,
    fl_round: int,
    num_workers: int,
) -> list[ClientUpdate]:
    """
    Download weights, deltas, and metrics for all workers in a given round.

    Expects MinIO keys in the format:
      round_{fl_round}/workers/worker_{id}_weights.pt
      round_{fl_round}/workers/worker_{id}_delta.pt
      round_{fl_round}/workers/worker_{id}_metrics.json

    Args:
        minio_client: authenticated MinIO client
        bucket: bucket name (e.g. "active-fed")
        fl_round: current FL round index
        num_workers: expected number of workers

    Returns:
        list of ClientUpdate (one per worker that uploaded successfully)
    """
    updates: list[ClientUpdate] = []

    for worker_id in range(num_workers):
        weights_key = f"round_{fl_round}/workers/worker_{worker_id}_weights.pt"
        delta_key = f"round_{fl_round}/workers/worker_{worker_id}_delta.pt"
        metrics_key = f"round_{fl_round}/workers/worker_{worker_id}_metrics.json"

        try:
            weights = _load_tensor(minio_client, bucket, weights_key)
            delta = _load_tensor(minio_client, bucket, delta_key)
            metrics = _load_json(minio_client, bucket, metrics_key)

            weight_norm = float(sum(torch.norm(v.float()) ** 2 for v in delta.values()) ** 0.5)

            updates.append(
                ClientUpdate(
                    worker_id=worker_id,
                    weights=weights,
                    weight_delta=delta,
                    avg_reward=float(metrics.get("avg_reward", 0.0)),
                    avg_td_error=float(metrics.get("avg_td_error", 0.0)),
                    weight_norm=weight_norm,
                )
            )
            log.info(
                f"Loaded worker {worker_id} | "
                f"avg_reward={metrics.get('avg_reward', 0):.2f} | "
                f"weight_norm={weight_norm:.4f}"
            )
        except Exception as e:
            log.warning(f"Failed to load worker {worker_id} from round {fl_round}: {e}")

    log.info(f"Collected {len(updates)}/{num_workers} worker updates for round {fl_round}")
    return updates


def push_global_weights(
    minio_client: Minio,
    bucket: str,
    fl_round: int,
    global_weights: dict[str, torch.Tensor],
) -> str:
    """Upload new global model to MinIO. Returns the MinIO key."""
    key = f"round_{fl_round + 1}/global.pt"
    buf = io.BytesIO()
    torch.save(global_weights, buf)
    buf.seek(0)
    minio_client.put_object(bucket, key, buf, length=buf.getbuffer().nbytes)
    log.info(f"Pushed global weights → MinIO: {key}")
    return key


def _load_tensor(client: Minio, bucket: str, key: str) -> dict[str, torch.Tensor]:
    response = client.get_object(bucket, key)
    data = response.read()
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _load_json(client: Minio, bucket: str, key: str) -> dict:
    response = client.get_object(bucket, key)
    return json.loads(response.read().decode())
