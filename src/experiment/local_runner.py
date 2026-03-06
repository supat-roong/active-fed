"""
Local FL Runner

Runs a complete Active-FL experiment loop in-process (no Kubernetes, MinIO, or
Kubeflow required). Ideal for fast local iteration and running all combinations.

Each round:
  1. Spawn `num_workers` PPO agents in threads, each trains `local_episodes`
  2. Evaluate candidate weights on CartPole (evaluation probe)
  3. Score clients with 4-factor scorer
  4. Aggregate (Active Weight / FedAvg / data_only) + optional Active Data fine-tune
  5. Evaluate new global model -> record metrics

Results are returned as a list of per-round metric dicts.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mlflow
import numpy as np
import torch

from src.experiment.env_wrapper import RandomizeCartPolePhysics

from src.agent.model import ActorCritic
from src.agent.ppo_agent import PPOAgent
from src.aggregator.aggregator import aggregate, WeightMode
from src.aggregator.active_data import ActiveDataMode
from src.aggregator.collect import ClientUpdate
from src.aggregator.evaluator import evaluate_all_candidates
from src.aggregator.scorer import score_clients
from src.aggregator.scorer import ClientUpdate  # re-export for type reference

log = logging.getLogger(__name__)

EVAL_EPISODES = 20  # post-training eval episodes on own env (matching train_worker.py)


@dataclass
class RunConfig:
    """All parameters for one experiment run."""

    # Identity
    weight_mode: WeightMode = "active"
    active_data_mode: ActiveDataMode = "none"
    run_name: str = ""  # auto-generated if empty

    # FL params
    fl_rounds: int = 5
    num_workers: int = 4
    local_episodes: int = 100
    eval_episodes: int = 10
    env_id: str = "CartPole-v1"

    # Scoring / aggregation
    score_threshold: float = 0.0
    score_temperature: float = 1.0
    active_data_threshold: float = 0.0
    active_data_steps: int = 3
    active_data_lr: float = 1e-4

    # Evaluation
    final_eval_episodes: int = 20
    solved_threshold: float = 195.0

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = "cpu"  # "cuda", "mps", or "cpu" — set by run_experiments.py

    # MLflow tracking (optional)
    # Set to None to use local ./mlruns (no server required).
    # Set to e.g. "http://localhost:5000" to use a running MLflow server.
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "active-fl-local"

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = f"{self.weight_mode}+{self.active_data_mode}"


@dataclass
class RoundMetrics:
    """Per-round metrics recorded during a run."""

    fl_round: int
    global_reward_mean: float
    global_reward_std: float
    clients_accepted: int
    clients_rejected: int
    client_scores: dict[int, float]
    client_improvements: dict[int, float]
    client_accepted: dict[int, bool]
    client_target_env_rewards: dict[int, float]  # mean target-env reward per worker
    client_own_env_rewards: dict[int, float]  # mean own-env training reward per worker
    effective_weight_norm: float
    active_data_applied: bool
    active_data_n_steps: int
    wall_time_s: float
    solved: bool


@dataclass
class RunResult:
    """Complete result of one experiment run."""

    run_name: str
    weight_mode: str
    active_data_mode: str
    config: RunConfig
    rounds: list[RoundMetrics] = field(default_factory=list)
    final_reward_mean: float = 0.0
    final_reward_std: float = 0.0
    total_wall_time_s: float = 0.0
    solved_at_round: int | None = None  # first round where solved
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _run_worker(
    worker_id: int,
    global_weights: dict[str, torch.Tensor],
    local_episodes: int,
    env_id: str,
    physics_seed: int,
    episode_seed: int,
    device: str = "cpu",
) -> tuple[ClientUpdate, float, float]:
    """Train one PPO worker, run a post-training eval, return (ClientUpdate, eval_mean, eval_std)."""
    # Seed per-worker for reproducibility
    torch.manual_seed(episode_seed)
    np.random.seed(episode_seed % (2**31))
    agent = PPOAgent(device=device)
    agent.set_weights(deepcopy(global_weights))
    w_before = agent.get_weights()

    env = RandomizeCartPolePhysics(gym.make(env_id), seed=physics_seed, variance=0.2)
    env.reset(seed=episode_seed)
    metrics = agent.train_episodes(env, num_episodes=local_episodes)
    env.close()

    w_after = agent.get_weights()
    delta = {k: w_after[k] - w_before[k] for k in w_after}
    norm = float(sum(torch.norm(v) ** 2 for v in delta.values()) ** 0.5)

    w_after_cpu = {k: v.cpu() for k, v in w_after.items()}
    delta_cpu = {k: v.cpu() for k, v in delta.items()}

    # -- Post-training own-env eval rollout (greedy, no gradient) --
    eval_env = RandomizeCartPolePhysics(gym.make(env_id), seed=physics_seed, variance=0.2)
    own_env_rewards: list[float] = []
    agent.model.eval()
    dev = torch.device(device)
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            obs, _ = eval_env.reset(seed=episode_seed + local_episodes + ep)
            ep_reward = 0.0
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(dev)
                action, _ = agent.model.forward(obs_t)
                act = action.sample().item()
                obs, reward, terminated, truncated, _ = eval_env.step(act)
                ep_reward += float(reward)
                done = terminated or truncated
            own_env_rewards.append(ep_reward)
    eval_env.close()

    own_eval_mean = float(np.mean(own_env_rewards))
    own_eval_std = float(np.std(own_env_rewards))

    return (
        ClientUpdate(
            worker_id=worker_id,
            weights=w_after_cpu,
            weight_delta=delta_cpu,
            avg_reward=metrics.avg_reward,
            avg_td_error=metrics.avg_td_error,
            weight_norm=norm,
        ),
        own_eval_mean,
        own_eval_std,
    )


# ---------------------------------------------------------------------------
# Global evaluation
# ---------------------------------------------------------------------------


def _eval_global(
    weights: dict[str, torch.Tensor],
    env_id: str,
    n_episodes: int,
    physics_seed: int,
    episode_seed: int,
    device: str = "cpu",
) -> tuple[float, float]:
    """Evaluate global model on target env. Returns (mean_reward, std_reward)."""
    dev = torch.device(device)
    model = ActorCritic().to(dev)
    # Always move through CPU first — avoids MPS placeholder errors when weights
    # were produced in a different process or device context.
    model.load_state_dict({k: v.cpu().to(dev) for k, v in weights.items()})
    model.eval()
    rewards = []
    env = RandomizeCartPolePhysics(gym.make(env_id), seed=physics_seed, variance=0.2)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed + ep)
        ep_r = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(dev)
            with torch.no_grad():
                dist, _ = model.forward(obs_t)
                act = dist.sample().item()
            obs, r, terminated, truncated, _ = env.step(act)
            ep_r += float(r)
            done = terminated or truncated
        rewards.append(ep_r)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_experiment(config: RunConfig) -> RunResult:
    """
    Run a complete Active-FL experiment locally.

    Args:
        config: all parameters for this run

    Returns:
        RunResult with per-round metrics
    """
    log.info(
        f"\n{'=' * 60}\n"
        f"  Starting: {config.run_name}\n"
        f"  weight_mode={config.weight_mode} | active_data_mode={config.active_data_mode}\n"
        f"  rounds={config.fl_rounds} | workers={config.num_workers} | episodes={config.local_episodes}\n"
        f"{'=' * 60}"
    )

    # --- MLflow setup ---
    # Uses local ./mlruns file store by default (no server needed).
    # Override with config.mlflow_tracking_uri to point at a remote server.
    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    result = RunResult(
        run_name=config.run_name,
        weight_mode=config.weight_mode,
        active_data_mode=config.active_data_mode,
        config=config,
    )

    # Initialize global model
    global_model = ActorCritic()
    torch.manual_seed(config.seed)
    global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}
    total_start = time.time()

    collect_data = config.active_data_mode != "none"

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(
            {
                "weight_mode": config.weight_mode,
                "active_data_mode": config.active_data_mode,
                "fl_rounds": config.fl_rounds,
                "num_workers": config.num_workers,
                "local_episodes": config.local_episodes,
                "eval_episodes": config.eval_episodes,
                "score_threshold": config.score_threshold,
                "score_temperature": config.score_temperature,
                "active_data_threshold": config.active_data_threshold,
                "active_data_steps": config.active_data_steps,
                "active_data_lr": config.active_data_lr,
                "seed": config.seed,
                "device": config.device,
            }
        )

        for fl_round in range(config.fl_rounds):
            round_start = time.time()
            log.info(f"\n--- Round {fl_round + 1}/{config.fl_rounds} [{config.run_name}] ---")

            # Step 1: Train workers in parallel
            clients: list[ClientUpdate] = []
            own_env_evals: dict[int, tuple[float, float]] = {}  # wid -> (mean, std)
            with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
                futures = {
                    executor.submit(
                        _run_worker,
                        wid,
                        deepcopy(global_weights),
                        config.local_episodes,
                        config.env_id,
                        config.seed + wid,
                        config.seed + wid + fl_round * 100,
                        config.device,
                    ): wid
                    for wid in range(config.num_workers)
                }
                for future in as_completed(futures):
                    wid = futures[future]
                    try:
                        client, own_mean, own_std = future.result()
                        clients.append(client)
                        own_env_evals[wid] = (own_mean, own_std)
                        log.debug(
                            f"  Worker {wid} done: train_reward={client.avg_reward:.1f} | own_eval={own_mean:.1f}"
                        )
                    except Exception as e:
                        log.error(f"  Worker {wid} failed: {e}")

            if not clients:
                log.error("All workers failed! Skipping round.")
                continue

            # Step 2: Evaluate candidates on target env
            weight_deltas = {c.worker_id: c.weight_delta for c in clients}
            eval_results = evaluate_all_candidates(
                global_weights=deepcopy(global_weights),
                weight_deltas=weight_deltas,
                env_id=config.env_id,
                n_eval_episodes=config.eval_episodes,
                physics_seed=config.seed + 9999,  # eval physics target
                episode_seed=config.seed + fl_round * 100,  # eval episodes (varying per round)
                max_workers=min(config.num_workers, 4),
                collect_data=collect_data,
            )

            # Step 3: Score clients
            scored = score_clients(
                clients=clients,
                eval_results=eval_results,
                score_threshold=config.score_threshold,
            )

            # Step 4: Aggregate (Active Weight + optional Active Data)
            agg_result = aggregate(
                global_weights=deepcopy(global_weights),
                clients=clients,
                scored_clients=scored,
                eval_results=eval_results,
                temperature=config.score_temperature,
                weight_mode=config.weight_mode,
                active_data_mode=config.active_data_mode,
                active_data_threshold=config.active_data_threshold,
                active_data_steps=config.active_data_steps,
                active_data_lr=config.active_data_lr,
                device=config.device,
            )
            # Move to CPU: MPS tensors cannot safely cross subprocess boundaries
            # or be passed back as pickled results from ProcessPoolExecutor.
            global_weights = {k: v.cpu() for k, v in agg_result.global_weights.items()}

            # Step 5: Evaluate new global model
            mean_r, std_r = _eval_global(
                global_weights,
                config.env_id,
                config.final_eval_episodes,
                config.seed + 9999,  # eval physics target
                config.seed + fl_round * 1000,  # eval episodes (varying per round)
                config.device,
            )
            solved = mean_r >= config.solved_threshold

            if solved and result.solved_at_round is None:
                result.solved_at_round = fl_round

            wall = time.time() - round_start

            # Per-worker reward dicts from evaluate_all_candidates results
            client_target_env_rewards: dict[int, float] = {}
            client_own_env_rewards: dict[int, float] = {}
            for c in clients:
                wid = c.worker_id
                # Use dedicated post-training eval (not training average)
                client_own_env_rewards[wid] = (
                    own_env_evals[wid][0] if wid in own_env_evals else c.avg_reward
                )
                if wid in eval_results:
                    client_target_env_rewards[wid] = eval_results[wid].candidate_reward

            metrics = RoundMetrics(
                fl_round=fl_round,
                global_reward_mean=mean_r,
                global_reward_std=std_r,
                clients_accepted=len(agg_result.accepted_ids),
                clients_rejected=len(agg_result.rejected_ids),
                client_scores={s.worker_id: s.score for s in scored},
                client_improvements={s.worker_id: s.improvement for s in scored},
                client_accepted={s.worker_id: s.accepted for s in scored},
                client_target_env_rewards=client_target_env_rewards,
                client_own_env_rewards=client_own_env_rewards,
                effective_weight_norm=agg_result.round_summary.get("effective_weight_norm", 0.0),
                active_data_applied=agg_result.active_data_applied,
                active_data_n_steps=agg_result.active_data_n_steps,
                wall_time_s=wall,
                solved=solved,
            )
            result.rounds.append(metrics)

            # --- Log per-round metrics to MLflow ---
            round_mlflow_metrics: dict[str, float] = {
                "global_reward_mean": mean_r,
                "global_reward_std": std_r,
                "clients_accepted": float(metrics.clients_accepted),
                "clients_rejected": float(metrics.clients_rejected),
                "effective_weight_norm": metrics.effective_weight_norm,
                "active_data_applied": float(metrics.active_data_applied),
                "active_data_n_steps": float(metrics.active_data_n_steps),
                "solved": float(solved),
                "wall_time_s": wall,
            }
            for sc in scored:
                wid = sc.worker_id
                round_mlflow_metrics[f"client_{wid}_score"] = sc.score
                round_mlflow_metrics[f"client_{wid}_improvement"] = sc.improvement
                round_mlflow_metrics[f"client_{wid}_accepted"] = float(sc.accepted)
                if wid in eval_results:
                    er = eval_results[wid]
                    rews = er.raw_episode_rewards
                    round_mlflow_metrics[f"client_{wid}_target_env_reward_mean"] = (
                        er.candidate_reward
                    )
                    round_mlflow_metrics[f"client_{wid}_target_env_reward_std"] = (
                        float(np.std(rews)) if rews else 0.0
                    )
                if wid in own_env_evals:
                    own_mean, own_std = own_env_evals[wid]
                    round_mlflow_metrics[f"client_{wid}_own_env_reward_mean"] = own_mean
                    round_mlflow_metrics[f"client_{wid}_own_env_reward_std"] = own_std
            mlflow.log_metrics(round_mlflow_metrics, step=fl_round)

            log.info(
                f"  Round {fl_round + 1}: reward={mean_r:.1f}±{std_r:.1f} | "
                f"accepted={len(agg_result.accepted_ids)}/{len(clients)} | "
                f"active_data={agg_result.active_data_applied} | "
                f"solved={solved} | {wall:.1f}s"
            )

        # Final evaluation
        result.final_reward_mean, result.final_reward_std = _eval_global(
            global_weights,
            config.env_id,
            config.final_eval_episodes * 2,
            config.seed + 9999,  # eval physics target
            config.seed + 99999,  # final eval episodes
            config.device,
        )
        result.total_wall_time_s = time.time() - total_start
        result.metadata = {
            "total_rounds": config.fl_rounds,
            "total_workers": config.num_workers,
            "local_episodes": config.local_episodes,
        }

        # --- Log final summary metrics to MLflow ---
        mlflow.log_metrics(
            {
                "final_reward_mean": result.final_reward_mean,
                "final_reward_std": result.final_reward_std,
                "total_wall_time_s": result.total_wall_time_s,
                "solved_at_round": float(result.solved_at_round)
                if result.solved_at_round is not None
                else -1.0,
            }
        )

        log.info(
            f"\nCompleted: {config.run_name} | "
            f"final_reward={result.final_reward_mean:.1f} | "
            f"solved_at={result.solved_at_round} | "
            f"total_time={result.total_wall_time_s:.1f}s"
        )
    return result
