"""
Target-Environment Weight Evaluation Probe

Core active learning component: before final aggregation, we test each
client's candidate weights directly on CartPole to measure actual
improvement vs. the current global model.

Supports two collection modes:
- Scalar only (default):  returns mean reward improvement
- With trajectory data:   additionally captures (obs, actions, log_probs,
                          values, rewards) for active data fine-tuning
"""

from __future__ import annotations

import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch

from src.agent.model import ActorCritic
from src.experiment.env_wrapper import RandomizeCartPolePhysics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvalTrajectory:
    """
    Full trajectory data collected during an evaluation probe episode.

    Used by active data module to fine-tune the global model on
    provably-good target-environment experiences.
    """

    worker_id: int
    improvement: float  # improvement score for this client
    obs: np.ndarray  # (T, obs_dim) float32
    actions: np.ndarray  # (T,) int64
    log_probs: np.ndarray  # (T,) float32 — from candidate model
    values: np.ndarray  # (T,) float32 — from candidate critic
    rewards: np.ndarray  # (T,) float32
    dones: np.ndarray  # (T,) bool


@dataclasses.dataclass
class EvalResult:
    worker_id: int
    candidate_reward: float  # reward with (W_global + ΔW_i)
    baseline_reward: float  # reward with W_global
    improvement: float  # candidate_reward - baseline_reward
    trajectories: list[EvalTrajectory] = dataclasses.field(default_factory=list)
    raw_episode_rewards: list[float] = dataclasses.field(default_factory=list)
    raw_episode_steps: list[int] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def _rollout(
    weights: dict[str, torch.Tensor],
    env_id: str,
    n_episodes: int,
    physics_seed: int,
    episode_seed: int,
) -> tuple[float, list[float], list[int]]:
    """Run `n_episodes` in a fresh env with the given weights. Returns (mean_reward, rewards, steps)."""
    model = ActorCritic()
    model.load_state_dict(weights)
    model.eval()
    torch.manual_seed(episode_seed)

    rewards = []
    steps = []
    env = RandomizeCartPolePhysics(gym.make(env_id), seed=physics_seed)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed + ep)
        ep_reward = 0.0
        ep_step = 0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = model.forward(obs_t)
                act = action.sample().item()
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_reward += float(reward)
            ep_step += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        steps.append(ep_step)
    env.close()
    return float(np.mean(rewards)), rewards, steps


def _rollout_with_data(
    weights: dict[str, torch.Tensor],
    env_id: str,
    n_episodes: int,
    physics_seed: int,
    episode_seed: int,
    worker_id: int,
    improvement: float,
) -> tuple[float, list[EvalTrajectory]]:
    """
    Run `n_episodes` and capture full trajectory data for active data fine-tuning.

    Returns:
        (mean_reward, list[EvalTrajectory]) — one EvalTrajectory per episode
    """
    model = ActorCritic()
    model.load_state_dict(weights)
    model.eval()
    torch.manual_seed(episode_seed)

    all_rewards = []
    trajectories: list[EvalTrajectory] = []
    env = RandomizeCartPolePhysics(gym.make(env_id), seed=physics_seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed + ep)
        ep_obs, ep_acts, ep_logps, ep_vals, ep_rews, ep_dones = [], [], [], [], [], []
        ep_reward = 0.0
        done = False

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, value = model.forward(obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            ep_obs.append(obs.copy())
            ep_acts.append(int(action.item()))
            ep_logps.append(float(log_prob.item()))
            ep_vals.append(float(value.item()))

            obs, reward, terminated, truncated, _ = env.step(action.item())
            ep_rews.append(float(reward))
            done = terminated or truncated
            ep_dones.append(done)
            ep_reward += float(reward)

        all_rewards.append(ep_reward)
        trajectories.append(
            EvalTrajectory(
                worker_id=worker_id,
                improvement=improvement,
                obs=np.array(ep_obs, dtype=np.float32),
                actions=np.array(ep_acts, dtype=np.int64),
                log_probs=np.array(ep_logps, dtype=np.float32),
                values=np.array(ep_vals, dtype=np.float32),
                rewards=np.array(ep_rews, dtype=np.float32),
                dones=np.array(ep_dones, dtype=bool),
            )
        )

    env.close()
    return float(np.mean(all_rewards)), trajectories


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate(
    global_weights: dict[str, torch.Tensor],
    weight_delta: dict[str, torch.Tensor],
    worker_id: int,
    env_id: str = "CartPole-v1",
    n_eval_episodes: int = 10,
    baseline_reward: float | None = None,
    physics_seed: int = 42,
    episode_seed: int = 42,
    collect_data: bool = False,
) -> EvalResult:
    """
    Tentatively apply weight delta to global model and evaluate on target env.

    Args:
        collect_data: if True, capture full trajectory data for active data fine-tuning
    """
    candidate_weights = {k: global_weights[k] + weight_delta[k] for k in global_weights}

    if baseline_reward is None:
        baseline_reward, _, _ = _rollout(
            global_weights, env_id, n_eval_episodes, physics_seed, episode_seed
        )
        log.debug(f"Baseline reward: {baseline_reward:.2f}")

    trajectories: list[EvalTrajectory] = []
    raw_rews: list[float] = []
    raw_steps: list[int] = []

    if collect_data:
        # We don't know improvement yet at this stage (need baseline first).
        # We'll patch improvement into trajectories after computing it.
        candidate_reward, raw_trajectories = _rollout_with_data(
            candidate_weights,
            env_id,
            n_eval_episodes,
            physics_seed,
            episode_seed,
            worker_id=worker_id,
            improvement=0.0,  # patched below
        )
        improvement = candidate_reward - baseline_reward
        # Patch improvement score into all episodes
        for traj in raw_trajectories:
            traj.improvement = improvement
            raw_rews.append(float(np.sum(traj.rewards)))
            raw_steps.append(len(traj.rewards))
        trajectories = raw_trajectories
    else:
        candidate_reward, raw_rews, raw_steps = _rollout(
            candidate_weights, env_id, n_eval_episodes, physics_seed, episode_seed
        )
        improvement = candidate_reward - baseline_reward

    log.info(
        f"Worker {worker_id}: candidate={candidate_reward:.2f}, "
        f"baseline={baseline_reward:.2f}, improvement={improvement:+.2f}"
        + (f" | {len(trajectories)} trajectories captured" if collect_data else "")
    )

    return EvalResult(
        worker_id=worker_id,
        candidate_reward=candidate_reward,
        baseline_reward=baseline_reward,
        improvement=improvement,
        trajectories=trajectories,
        raw_episode_rewards=raw_rews,
        raw_episode_steps=raw_steps,
    )


def evaluate_all_candidates(
    global_weights: dict[str, torch.Tensor],
    weight_deltas: dict[int, dict[str, torch.Tensor]],
    env_id: str = "CartPole-v1",
    n_eval_episodes: int = 10,
    physics_seed: int = 42,
    episode_seed: int = 42,
    max_workers: int = 4,
    collect_data: bool = False,
) -> dict[int, EvalResult]:
    """
    Parallel evaluation of all client candidates.

    Args:
        collect_data: if True, capture trajectory data for active data fine-tuning
    """
    log.info(
        f"Evaluating {len(weight_deltas)} client candidates "
        f"(workers={max_workers}, collect_data={collect_data})"
    )

    baseline_reward, _, _ = _rollout(
        global_weights, env_id, n_eval_episodes, physics_seed, episode_seed
    )
    log.info(f"Global baseline reward: {baseline_reward:.2f}")

    results: dict[int, EvalResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_candidate,
                deepcopy(global_weights),
                delta,
                worker_id,
                env_id,
                n_eval_episodes,
                baseline_reward,
                physics_seed,
                episode_seed,
                collect_data,
            ): worker_id
            for worker_id, delta in weight_deltas.items()
        }
        for future in as_completed(futures):
            wid = futures[future]
            try:
                results[wid] = future.result()
            except Exception as e:
                log.error(f"Evaluation failed for worker {wid}: {e}")

    return results
