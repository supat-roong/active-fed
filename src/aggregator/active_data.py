"""
Active Data Fine-Tuning Module

Implements the "Active Data" mode of active learning:
instead of only selecting/weighting model parameters (Active Weight),
we also collect the actual experience trajectories from high-improvement
eval probes and use them to directly fine-tune the global model.

Fine-tuning strategy:
  "bc"  — Behavioral Cloning: maximize log-likelihood of high-value actions

The key insight: trajectories captured during eval probes are:
  1. On-task (CartPole target env, not local training env)
  2. From provably-improved policies (improvement > threshold)
  3. Already labeled with log_probs and values — no extra env calls needed
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.aggregator.evaluator import EvalTrajectory
from src.agent.model import ActorCritic

log = logging.getLogger(__name__)

ActiveDataMode = Literal["none", "bc"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ActiveDataset:
    """
    Combined experience dataset built from high-improvement eval trajectories.
    Trajectories are weighted by their client's improvement score.
    """

    obs: torch.Tensor  # (N, obs_dim)
    actions: torch.Tensor  # (N,) int64
    log_probs_old: torch.Tensor  # (N,) from candidate model at collection time
    advantages: torch.Tensor  # (N,) GAE-estimated from rewards/values
    returns: torch.Tensor  # (N,) discounted returns
    weights: torch.Tensor  # (N,) per-step importance weight (from improvement score)
    n_steps: int
    source_workers: list[int]


def build_active_dataset(
    trajectories: list[EvalTrajectory],
    data_threshold: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> ActiveDataset | None:
    """
    Build a combined active dataset from collected evaluation trajectories.

    Args:
        trajectories: all captured EvalTrajectory objects (from all clients)
        data_threshold: minimum improvement score to include a trajectory
        gamma: discount factor for advantage computation
        gae_lambda: GAE lambda

    Returns:
        ActiveDataset or None if no trajectories pass the threshold
    """
    # Filter by improvement threshold
    accepted = [t for t in trajectories if t.improvement > data_threshold]
    if not accepted:
        log.warning(
            f"No trajectories passed data_threshold={data_threshold:.2f}, skipping active data"
        )
        return None

    improvements = np.array([t.improvement for t in accepted])
    # Softmax-normalize improvement scores for per-trajectory weighting
    exp_imp = np.exp(improvements - improvements.max())
    traj_weights = exp_imp / exp_imp.sum()

    all_obs, all_acts, all_logps, all_advs, all_rets, all_weights = [], [], [], [], [], []
    source_workers = []

    for traj, traj_weight in zip(accepted, traj_weights):
        T = len(traj.rewards)
        advantages = _compute_gae(traj.rewards, traj.values, traj.dones, gamma, gae_lambda)
        returns = [adv + val for adv, val in zip(advantages, traj.values)]

        # Per-step weight = trajectory-level importance weight (broadcast over steps)
        step_weights = np.full(T, traj_weight, dtype=np.float32)

        all_obs.append(traj.obs)
        all_acts.append(traj.actions)
        all_logps.append(traj.log_probs)
        all_advs.append(np.array(advantages, dtype=np.float32))
        all_rets.append(np.array(returns, dtype=np.float32))
        all_weights.append(step_weights)
        source_workers.append(int(traj.worker_id))

    obs_t = torch.tensor(np.concatenate(all_obs), dtype=torch.float32)
    acts_t = torch.tensor(np.concatenate(all_acts), dtype=torch.long)
    logps_t = torch.tensor(np.concatenate(all_logps), dtype=torch.float32)
    advs_t = torch.tensor(np.concatenate(all_advs), dtype=torch.float32)
    rets_t = torch.tensor(np.concatenate(all_rets), dtype=torch.float32)
    weights_t = torch.tensor(np.concatenate(all_weights), dtype=torch.float32)

    # Normalize advantages
    advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

    log.info(
        f"ActiveDataset: {len(accepted)} trajectories from worker(s) {sorted(set(source_workers))} "
        f"| {obs_t.shape[0]} steps | threshold={data_threshold:.2f}"
    )

    return ActiveDataset(
        obs=obs_t,
        actions=acts_t,
        log_probs_old=logps_t,
        advantages=advs_t,
        returns=rets_t,
        weights=weights_t,
        n_steps=obs_t.shape[0],
        source_workers=sorted(set(source_workers)),
    )


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> list[float]:
    """GAE advantage estimation for a single trajectory."""
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < T else 0.0)
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
    return advantages


# ---------------------------------------------------------------------------
# Fine-tuning strategies
# ---------------------------------------------------------------------------


class ActiveDataUpdater:
    """
    Fine-tunes the global model using active trajectory data.

    Supports one fine-tuning mode:
      "bc"  — Behavioral Cloning (fast, stable)
    """

    def __init__(
        self,
        mode: ActiveDataMode = "bc",
        lr: float = 1e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 3,
        minibatch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        if mode not in ("none", "bc"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'none' or 'bc'.")
        self.mode = mode
        self.lr = lr
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

    def update(
        self,
        global_weights: dict[str, torch.Tensor],
        dataset: ActiveDataset,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """
        Fine-tune global_weights on the active dataset.

        Args:
            global_weights: current aggregated global model state_dict
            dataset: ActiveDataset from build_active_dataset()

        Returns:
            (updated_weights, training_metrics)
        """
        if self.mode == "none":
            return global_weights, {"skipped": True}

        model = ActorCritic().to(self.device)
        model.load_state_dict({k: v.to(self.device) for k, v in global_weights.items()})
        optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-5)

        if self.mode == "bc":
            metrics = self._bc_update(model, optimizer, dataset)

        updated_weights = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

        log.info(
            f"[ActiveData:{self.mode}] {self.n_epochs} epochs | "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not isinstance(v, bool))
        )
        return updated_weights, metrics

    def _bc_update(
        self,
        model: ActorCritic,
        optimizer: optim.Adam,
        dataset: ActiveDataset,
    ) -> dict:
        """
        Behavioral Cloning: maximize weighted log-likelihood of collected actions.

        Loss: -Σ w_i · log π_θ(a_i | s_i)

        The importance weight w_i comes from the client's improvement score,
        so actions from better-performing clients are emphasized.
        """
        model.train()
        total_loss = 0.0
        n_updates = 0

        obs = dataset.obs.to(self.device)
        acts = dataset.actions.to(self.device)
        weights = dataset.weights.to(self.device)

        for _ in range(self.n_epochs):
            idx = torch.randperm(dataset.n_steps)
            for start in range(0, dataset.n_steps, self.minibatch_size):
                mb_idx = idx[start : start + self.minibatch_size]
                mb_obs = obs[mb_idx]
                mb_acts = acts[mb_idx]
                mb_weights = weights[mb_idx]

                dist, _ = model.forward(mb_obs)
                log_probs = dist.log_prob(mb_acts)

                # Weighted negative log-likelihood
                loss = -(mb_weights * log_probs).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return {"bc_loss": total_loss / max(n_updates, 1)}
