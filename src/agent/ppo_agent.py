"""
PPO Agent for CartPole-v1

Implements:
- Rollout buffer collection (obs, actions, rewards, values, log_probs, dones)
- GAE advantage estimation
- Clipped PPO surrogate loss (ε=0.2)
- Value function loss (MSE)
- Entropy bonus for exploration
- Per-step TD-error tracking (for active scoring)
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.model import ActorCritic


class RolloutBatch(NamedTuple):
    obs: torch.Tensor  # (T, obs_dim)
    actions: torch.Tensor  # (T,)
    log_probs: torch.Tensor  # (T,)
    advantages: torch.Tensor  # (T,)
    returns: torch.Tensor  # (T,)
    values: torch.Tensor  # (T,)


@dataclasses.dataclass
class TrainingMetrics:
    avg_reward: float
    avg_td_error: float
    total_episodes: int
    policy_loss: float
    value_loss: float
    entropy: float
    raw_episode_rewards: list[float] = dataclasses.field(default_factory=list)
    raw_episode_steps: list[int] = dataclasses.field(default_factory=list)


class PPOAgent:
    """
    PPO agent with GAE advantage estimation.

    Key design choices for federated use:
    - All state lives in self.model (a single ActorCritic)
    - Training produces weight_delta = W_after - W_before
    - td_errors are accumulated per step for active scoring
    """

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.device = torch.device(device)

        self.model = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    # Weight management (for federated communication)
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return a deepcopy of model weights (detached from grad graph)."""
        return {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load weights from a state_dict (e.g. global model from server)."""
        self.model.load_state_dict({k: v.to(self.device) for k, v in weights.items()})

    def compute_weight_delta(
        self, weights_before: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """ΔW = W_after - W_before (used by aggregator scorer)."""
        weights_after = self.get_weights()
        return {k: weights_after[k] - weights_before[k].cpu() for k in weights_after}

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        env,
        num_steps: int = 2048,
    ) -> tuple[RolloutBatch, list[float], list[float]]:
        """
        Collect a rollout of `num_steps` transitions.

        Returns:
            batch: RolloutBatch with advantages computed via GAE
            episode_rewards: list of per-episode total rewards
            episode_steps: list of per-episode step counts
            td_errors: list of per-step |V(s) - (r + γV(s'))|
        """
        obs_buf = []
        act_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []
        td_errors: list[float] = []
        episode_rewards: list[float] = []
        episode_steps: list[int] = []
        ep_reward = 0.0
        ep_steps = 0

        obs, _ = env.reset()
        done = False

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, log_prob, _, value = self.model.get_action_and_value(obs_t)

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                obs_buf.append(obs)
                act_buf.append(action.item())
                logp_buf.append(log_prob.item())
                val_buf.append(value.item())
                rew_buf.append(float(reward))
                done_buf.append(float(done))
                ep_reward += float(reward)
                ep_steps += 1

                # TD-error tracking (for active scoring)
                if not done:
                    next_obs_t = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    _, next_value = self.model.forward(next_obs_t)
                    td_err = abs(value.item() - (reward + self.gamma * next_value.item()))
                else:
                    td_err = abs(value.item() - reward)
                td_errors.append(td_err)

                obs = next_obs
                if done:
                    episode_rewards.append(ep_reward)
                    episode_steps.append(ep_steps)
                    ep_reward = 0.0
                    ep_steps = 0
                    obs, _ = env.reset()

        # Bootstrap value for last state
        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_value = self.model.forward(last_obs_t)
            last_value = last_value.item()

        # GAE advantage computation
        advantages = self._compute_gae(rew_buf, val_buf, done_buf, last_value)
        returns = [adv + val for adv, val in zip(advantages, val_buf)]

        batch = RolloutBatch(
            obs=torch.tensor(np.array(obs_buf), dtype=torch.float32),
            actions=torch.tensor(act_buf, dtype=torch.long),
            log_probs=torch.tensor(logp_buf, dtype=torch.float32),
            advantages=torch.tensor(advantages, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
            values=torch.tensor(val_buf, dtype=torch.float32),
        )
        return batch, episode_rewards, episode_steps, td_errors

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[float],
        last_value: float,
    ) -> list[float]:
        """Generalized Advantage Estimation (Schulman et al., 2016)."""
        advantages = [0.0] * len(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        return advantages

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, batch: RolloutBatch) -> tuple[float, float, float]:
        """
        Run PPO update epochs over collected batch.
        Returns: (mean_policy_loss, mean_value_loss, mean_entropy)
        """
        self.model.train()
        T = batch.obs.shape[0]
        minibatch_size = T // self.num_minibatches

        # Normalise advantages
        adv = batch.advantages.to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            idx = torch.randperm(T)
            for start in range(0, T, minibatch_size):
                mb_idx = idx[start : start + minibatch_size]
                mb_obs = batch.obs[mb_idx].to(self.device)
                mb_act = batch.actions[mb_idx].to(self.device)
                mb_logp_old = batch.log_probs[mb_idx].to(self.device)
                mb_adv = adv[mb_idx]
                mb_ret = batch.returns[mb_idx].to(self.device)

                _, mb_logp, mb_ent, mb_val = self.model.get_action_and_value(mb_obs, mb_act)

                # PPO clipped surrogate loss
                ratio = torch.exp(mb_logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = nn.functional.mse_loss(mb_val, mb_ret)

                # Entropy bonus
                entropy = mb_ent.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                num_updates += 1

        n = max(num_updates, 1)
        return total_pl / n, total_vl / n, total_ent / n

    # ------------------------------------------------------------------
    # Convenience: train for N episodes (used in worker entrypoint)
    # ------------------------------------------------------------------

    def train_episodes(
        self,
        env,
        num_episodes: int,
        steps_per_update: int = 2048,
    ) -> TrainingMetrics:
        """
        Train until at least `num_episodes` complete episodes have been collected.
        Returns aggregated TrainingMetrics.
        """
        all_rewards: list[float] = []
        all_steps: list[int] = []
        all_td_errors: list[float] = []
        all_pl, all_vl, all_ent = [], [], []

        while len(all_rewards) < num_episodes:
            batch, ep_rews, ep_sts, td_errs = self.collect_rollout(env, steps_per_update)
            pl, vl, ent = self.update(batch)
            all_rewards.extend(ep_rews)
            all_steps.extend(ep_sts)
            all_td_errors.extend(td_errs)
            all_pl.append(pl)
            all_vl.append(vl)
            all_ent.append(ent)

        return TrainingMetrics(
            avg_reward=float(np.mean(all_rewards)),
            avg_td_error=float(np.mean(all_td_errors)),
            total_episodes=len(all_rewards),
            policy_loss=float(np.mean(all_pl)),
            value_loss=float(np.mean(all_vl)),
            entropy=float(np.mean(all_ent)),
            raw_episode_rewards=all_rewards,
            raw_episode_steps=all_steps,
        )
