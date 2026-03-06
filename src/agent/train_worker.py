"""
Worker entrypoint for Kubeflow PyTorchJob pods.

Each worker:
1. Fetches the current global model from MinIO
2. Trains PPO for --local-episodes episodes
3. Evaluates trained model on own physics-randomized env (EVAL_EPISODES)
4. Pushes weight delta + metrics JSON back to MinIO
5. Logs to MLflow: train_reward_mean, own_env_eval_reward_mean,
   and the target-env eval reward is logged separately in evaluate_global.

Environment variables expected:
  MLFLOW_TRACKING_URI   - e.g. http://mlflow-service:5000
  MINIO_ENDPOINT        - e.g. minio-service:9000
  MINIO_ACCESS_KEY
  MINIO_SECRET_KEY
  MINIO_BUCKET          - default: active-fed
  WORKER_ID             - injected by PyTorchJob as RANK
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys

import gymnasium as gym
import mlflow
import numpy as np
import torch
from minio import Minio
from minio.error import S3Error

from src.agent.ppo_agent import PPOAgent
from src.experiment.env_wrapper import RandomizeCartPolePhysics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

EVAL_EPISODES = 20  # post-training eval episodes on own env


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------


def _get_minio_client() -> Minio:
    return Minio(
        endpoint=os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False,
    )


def _fetch_global_weights(
    client: Minio, bucket: str, fl_round: int
) -> dict[str, torch.Tensor] | None:
    """Download global model weights for the current round (or None if round 0)."""
    key = f"round_{fl_round}/global.pt"
    try:
        response = client.get_object(bucket, key)
        data = response.read()
        buf = io.BytesIO(data)
        weights = torch.load(buf, map_location="cpu", weights_only=True)
        log.info(f"Fetched global weights from MinIO: {key}")
        return weights
    except S3Error as e:
        if e.code == "NoSuchKey":
            log.info(f"No global weights found for round {fl_round}, starting fresh.")
            return None
        raise


def _push_weights(
    client: Minio,
    bucket: str,
    fl_round: int,
    worker_id: int,
    weights: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    metrics: dict,
) -> None:
    """Upload local weights, weight delta, and metrics to MinIO."""
    _ensure_bucket(client, bucket)

    # Upload local weights
    weights_buf = io.BytesIO()
    torch.save(weights, weights_buf)
    weights_buf.seek(0)
    weights_key = f"round_{fl_round}/workers/worker_{worker_id}_weights.pt"
    client.put_object(bucket, weights_key, weights_buf, length=weights_buf.getbuffer().nbytes)

    # Upload weight delta
    delta_buf = io.BytesIO()
    torch.save(delta, delta_buf)
    delta_buf.seek(0)
    delta_key = f"round_{fl_round}/workers/worker_{worker_id}_delta.pt"
    client.put_object(bucket, delta_key, delta_buf, length=delta_buf.getbuffer().nbytes)

    # Upload metrics JSON
    metrics_bytes = json.dumps(metrics).encode()
    metrics_buf = io.BytesIO(metrics_bytes)
    metrics_key = f"round_{fl_round}/workers/worker_{worker_id}_metrics.json"
    client.put_object(bucket, metrics_key, metrics_buf, length=len(metrics_bytes))

    log.info(f"Uploaded weights + delta + metrics for worker {worker_id}, round {fl_round}")


def _ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        log.info(f"Created MinIO bucket: {bucket}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    worker_id = int(os.environ.get("RANK", args.worker_id))
    bucket = os.environ.get("MINIO_BUCKET", "active-fed")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", args.mlflow_tracking_uri)

    log.info(
        f"Worker {worker_id} | FL round {args.fl_round} | "
        f"episodes={args.local_episodes} | device={args.device}"
    )

    # -- MLflow setup --
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "active-fl-cartpole")
    mlflow.set_experiment(mlflow_experiment)
    run_name = f"worker_{worker_id}_round_{args.fl_round}"

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(
            {
                "worker_id": worker_id,
                "fl_round": args.fl_round,
                "local_episodes": args.local_episodes,
                "device": args.device,
            }
        )

        # -- Agent setup --
        agent = PPOAgent(device=args.device)

        # -- Fetch global weights (if available) --
        if not args.dry_run:
            minio_client = _get_minio_client()
            _ensure_bucket(minio_client, bucket)
            global_weights = _fetch_global_weights(minio_client, bucket, args.fl_round)
            if global_weights is not None:
                agent.set_weights(global_weights)
        else:
            log.info("DRY RUN: skipping MinIO, using fresh model")
            global_weights = None

        # -- Snapshot weights before training --
        weights_before = agent.get_weights()

        # -- Train --
        physics_seed = worker_id
        episode_seed = worker_id + (args.fl_round * 100)
        env = RandomizeCartPolePhysics(gym.make("CartPole-v1"), seed=physics_seed)
        # Pin env RNG per round/worker logic using worker_id and fl_round
        env.reset(seed=episode_seed)
        metrics = agent.train_episodes(env, num_episodes=args.local_episodes)
        env.close()

        # -- Compute weight delta --
        weights_after = agent.get_weights()
        weight_delta = agent.compute_weight_delta(weights_before)
        weight_norm = float(
            sum(torch.norm(v.float()).item() ** 2 for v in weight_delta.values()) ** 0.5
        )

        # -- Post-training own-env evaluation --
        eval_env = RandomizeCartPolePhysics(gym.make("CartPole-v1"), seed=physics_seed)
        own_env_rewards: list[float] = []
        agent.model.eval()
        with torch.no_grad():
            for ep in range(EVAL_EPISODES):
                obs, _ = eval_env.reset(seed=episode_seed + args.local_episodes + ep)
                ep_reward = 0.0
                done = False
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(args.device)
                    action, _ = agent.model.forward(obs_t)
                    act = action.sample().item()
                    obs, reward, terminated, truncated, _ = eval_env.step(act)
                    ep_reward += float(reward)
                    done = terminated or truncated
                own_env_rewards.append(ep_reward)
        eval_env.close()
        agent.model.train()

        own_env_eval_mean = float(np.mean(own_env_rewards))
        own_env_eval_std = float(np.std(own_env_rewards))

        metrics_dict = {
            "worker_id": worker_id,
            "fl_round": args.fl_round,
            "avg_reward": metrics.avg_reward,
            "avg_td_error": metrics.avg_td_error,
            "total_episodes": metrics.total_episodes,
            "policy_loss": metrics.policy_loss,
            "value_loss": metrics.value_loss,
            "entropy": metrics.entropy,
            "weight_norm": weight_norm,
            "own_env_eval_reward_mean": own_env_eval_mean,
            "own_env_eval_reward_std": own_env_eval_std,
            "raw_own_env_rewards": own_env_rewards,
            "raw_episode_rewards": metrics.raw_episode_rewards,
            "raw_episode_steps": metrics.raw_episode_steps,
        }

        train_std = (
            float(np.std(metrics.raw_episode_rewards)) if metrics.raw_episode_rewards else 0.0
        )
        mlflow.log_metrics(
            {
                "train_reward_mean": metrics.avg_reward,
                "train_reward_std": train_std,
                "own_env_eval_reward_mean": own_env_eval_mean,
                "own_env_eval_reward_std": own_env_eval_std,
                "avg_td_error": metrics.avg_td_error,
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "entropy": metrics.entropy,
                "weight_norm": weight_norm,
                "total_episodes": metrics.total_episodes,
            },
            step=args.fl_round,
        )

        log.info(
            f"Worker {worker_id} done | "
            f"train_reward={metrics.avg_reward:.2f} | "
            f"own_env_eval={own_env_eval_mean:.2f} +/- {own_env_eval_std:.2f} | "
            f"weight_norm={weight_norm:.4f}"
        )

        # -- Push to MinIO --
        if not args.dry_run:
            _push_weights(
                minio_client,
                bucket,
                args.fl_round,
                worker_id,
                weights_after,
                weight_delta,
                metrics_dict,
            )
        else:
            log.info(f"DRY RUN metrics: {metrics_dict}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO worker for Active-FL on CartPole")
    p.add_argument("--fl-round", type=int, required=True, help="Current federated round index")
    p.add_argument("--local-episodes", type=int, default=200, help="Episodes to train locally")
    p.add_argument(
        "--worker-id", type=int, default=0, help="Worker ID (fallback, overridden by RANK env)"
    )
    p.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    p.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000")
    p.add_argument("--dry-run", action="store_true", help="Skip MinIO; useful for local testing")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
