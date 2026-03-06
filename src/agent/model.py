"""
Shared Actor-Critic MLP for CartPole-v1

Architecture:
  - Shared trunk: obs(4) → 64 → 64 (tanh activations)
  - Actor head: 64 → num_actions(2) → softmax (action probs)
  - Critic head: 64 → 1 (state value V(s))

Both heads are exported together as a single state_dict,
making weight serialization / federated aggregation straightforward.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network for CartPole."""

    OBS_DIM: int = 4
    ACT_DIM: int = 2
    HIDDEN: int = 64

    def __init__(self) -> None:
        super().__init__()
        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.OBS_DIM, self.HIDDEN),
            nn.Tanh(),
            nn.Linear(self.HIDDEN, self.HIDDEN),
            nn.Tanh(),
        )
        # Policy head: outputs logits over actions
        self.actor_head = nn.Linear(self.HIDDEN, self.ACT_DIM)
        # Value head: outputs scalar state-value estimate
        self.critic_head = nn.Linear(self.HIDDEN, 1)

        # Orthogonal initialisation (standard for PPO)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        # Actor head gets a smaller gain for stable early exploration
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Args:
            obs: (batch, OBS_DIM) float tensor

        Returns:
            (action_distribution, state_value)
        """
        features = self.trunk(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method used in PPO rollout + update steps.
        Returns: (action, log_prob, entropy, value)
        """
        dist, value = self.forward(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
