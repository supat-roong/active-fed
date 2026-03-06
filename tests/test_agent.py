"""
Smoke tests for PPO agent: can it run without error on CartPole?
"""

from __future__ import annotations

import gymnasium as gym
import torch

from src.agent.model import ActorCritic
from src.agent.ppo_agent import PPOAgent, TrainingMetrics


class TestActorCritic:
    def test_forward_returns_dist_and_value(self):
        model = ActorCritic()
        obs = torch.randn(4, 4)  # batch of 4 CartPole observations
        dist, value = model.forward(obs)
        assert dist.batch_shape == (4,)
        assert value.shape == (4,)

    def test_action_is_in_valid_range(self):
        model = ActorCritic()
        obs = torch.randn(1, 4)
        dist, _ = model.forward(obs)
        action = dist.sample()
        assert action.item() in [0, 1]

    def test_state_dict_has_expected_keys(self):
        model = ActorCritic()
        keys = set(model.state_dict().keys())
        assert any("trunk" in k for k in keys)
        assert any("actor_head" in k for k in keys)
        assert any("critic_head" in k for k in keys)


class TestPPOAgent:
    def test_get_set_weights_roundtrip(self):
        agent = PPOAgent()
        weights = agent.get_weights()
        # Modify and reload
        modified = {k: v + 1.0 for k, v in weights.items()}
        agent.set_weights(modified)
        reloaded = agent.get_weights()
        for key in weights:
            assert torch.allclose(reloaded[key].float(), modified[key].float(), atol=1e-5)

    def test_weight_delta_is_difference(self):
        agent = PPOAgent()
        w_before = agent.get_weights()
        # Artificially shift weights
        perturbed = {k: v + 0.5 for k, v in w_before.items()}
        agent.set_weights(perturbed)
        delta = agent.compute_weight_delta(w_before)
        for key in delta:
            assert torch.allclose(delta[key].float(), torch.full_like(delta[key], 0.5), atol=1e-5)

    def test_short_training_smoke(self):
        """Run 3 complete episodes without error."""
        agent = PPOAgent()
        env = gym.make("CartPole-v1")
        metrics = agent.train_episodes(env, num_episodes=3, steps_per_update=256)
        env.close()

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.total_episodes >= 3
        assert metrics.avg_reward > 0
        assert metrics.avg_td_error >= 0

    def test_zero_delta_after_zero_update_steps(self):
        """If we don't call update, weights should not change."""
        agent = PPOAgent()
        before = agent.get_weights()
        after = agent.get_weights()
        for k in before:
            assert torch.allclose(before[k], after[k])
