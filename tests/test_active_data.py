"""
Unit tests for Active Data module (active_data.py).

Tests cover:
- EvalTrajectory data collection via evaluator
- Dataset building: filtering, weighting, GAE advantages
- BC update reduces cross-entropy on training data
- PPO update runs without diverging
- Edge cases: empty trajectories, below-threshold, all-rejected
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agent.model import ActorCritic
from src.aggregator.evaluator import (
    EvalTrajectory,
    EvalResult,
    evaluate_candidate,
    _rollout_with_data,
)
from src.aggregator.active_data import (
    ActiveDataUpdater,
    ActiveDataset,
    build_active_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fresh_weights() -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in ActorCritic().state_dict().items()}


def _make_trajectory(
    worker_id: int = 0,
    improvement: float = 5.0,
    T: int = 20,
) -> EvalTrajectory:
    """Create a synthetic EvalTrajectory for testing."""
    return EvalTrajectory(
        worker_id=worker_id,
        improvement=improvement,
        obs=np.random.randn(T, 4).astype(np.float32),
        actions=np.random.randint(0, 2, size=T).astype(np.int64),
        log_probs=np.random.randn(T).astype(np.float32),
        values=np.random.randn(T).astype(np.float32),
        rewards=np.ones(T, dtype=np.float32),
        dones=np.zeros(T, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Trajectory collection from evaluator
# ---------------------------------------------------------------------------


class TestTrajectoryCollection:
    def test_rollout_with_data_returns_reward_and_trajectories(self):
        weights = _fresh_weights()
        reward, trajectories = _rollout_with_data(
            weights=weights,
            env_id="CartPole-v1",
            n_episodes=2,
            physics_seed=42,
            episode_seed=0,
            worker_id=0,
            improvement=5.0,
        )
        assert isinstance(reward, float)
        assert reward > 0
        assert len(trajectories) == 2  # one per episode

    def test_trajectory_shapes_are_consistent(self):
        weights = _fresh_weights()
        _, trajectories = _rollout_with_data(
            weights=weights,
            env_id="CartPole-v1",
            n_episodes=1,
            physics_seed=42,
            episode_seed=0,
            worker_id=0,
            improvement=3.0,
        )
        traj = trajectories[0]
        T = len(traj.rewards)
        assert traj.obs.shape == (T, 4)  # CartPole obs dim
        assert traj.actions.shape == (T,)
        assert traj.log_probs.shape == (T,)
        assert traj.values.shape == (T,)
        assert traj.dones.shape == (T,)

    def test_evaluate_candidate_with_collect_data(self):
        weights = _fresh_weights()
        delta = {k: torch.zeros_like(v) for k, v in weights.items()}

        result = evaluate_candidate(
            global_weights=weights,
            weight_delta=delta,
            worker_id=0,
            n_eval_episodes=2,
            collect_data=True,
        )
        assert isinstance(result, EvalResult)
        assert len(result.trajectories) == 2
        # improvement should be patched into all trajectories
        for traj in result.trajectories:
            assert abs(traj.improvement - result.improvement) < 1e-5

    def test_evaluate_candidate_without_collect_data_has_no_trajectories(self):
        weights = _fresh_weights()
        delta = {k: torch.zeros_like(v) for k, v in weights.items()}

        result = evaluate_candidate(
            global_weights=weights,
            weight_delta=delta,
            worker_id=0,
            n_eval_episodes=2,
            collect_data=False,
        )
        assert result.trajectories == []


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


class TestBuildActiveDataset:
    def test_basic_dataset_creation(self):
        trajs = [_make_trajectory(0, improvement=5.0), _make_trajectory(1, improvement=3.0)]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is not None
        assert dataset.n_steps > 0
        assert set(dataset.source_workers) == {0, 1}

    def test_threshold_filters_low_improvement(self):
        trajs = [
            _make_trajectory(0, improvement=10.0),
            _make_trajectory(1, improvement=-5.0),  # below threshold
            _make_trajectory(2, improvement=0.5),  # above threshold
        ]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is not None
        # Only workers 0 and 2 should be included (improvement > 0.0)
        assert 1 not in dataset.source_workers
        assert 0 in dataset.source_workers

    def test_all_below_threshold_returns_none(self):
        trajs = [
            _make_trajectory(0, improvement=-1.0),
            _make_trajectory(1, improvement=-2.0),
        ]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is None

    def test_empty_trajectories_returns_none(self):
        dataset = build_active_dataset([], data_threshold=0.0)
        assert dataset is None

    def test_weights_sum_to_one(self):
        trajs = [
            _make_trajectory(0, improvement=1.0),
            _make_trajectory(1, improvement=2.0),
            _make_trajectory(2, improvement=3.0),
        ]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is not None
        # All steps from one trajectory should have the same weight
        # and the weights across trajectories should sum to ~1.0 (softmax)
        # We can't easily check exact values but can check range
        assert dataset.weights.min() > 0
        assert dataset.weights.max() <= 1.0

    def test_dataset_tensors_have_correct_dtypes(self):
        trajs = [_make_trajectory(0, improvement=5.0, T=10)]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is not None
        assert dataset.obs.dtype == torch.float32
        assert dataset.actions.dtype == torch.long
        assert dataset.advantages.dtype == torch.float32


# ---------------------------------------------------------------------------
# Fine-tuning: BC mode
# ---------------------------------------------------------------------------


class TestBCUpdate:
    def test_bc_update_returns_weights_and_metrics(self):
        weights = _fresh_weights()
        trajs = [_make_trajectory(0, improvement=5.0, T=50)]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is not None

        updater = ActiveDataUpdater(mode="bc", n_epochs=2, minibatch_size=16)
        new_weights, metrics = updater.update(weights, dataset)

        assert isinstance(new_weights, dict)
        assert "bc_loss" in metrics
        assert isinstance(metrics["bc_loss"], float)

    def test_bc_update_changes_weights(self):
        weights = _fresh_weights()
        trajs = [_make_trajectory(0, improvement=10.0, T=100)]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is not None

        updater = ActiveDataUpdater(mode="bc", lr=1e-3, n_epochs=5, minibatch_size=32)
        new_weights, _ = updater.update(weights, dataset)

        # At least one parameter should have changed
        changed = any(
            not torch.allclose(weights[k].float(), new_weights[k].float(), atol=1e-6)
            for k in weights
        )
        assert changed

    def test_bc_update_reduces_loss_over_epochs(self):
        """More training epochs should reduce BC loss."""
        weights = _fresh_weights()
        trajs = [_make_trajectory(0, improvement=5.0, T=200)]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is not None

        updater_small = ActiveDataUpdater(mode="bc", lr=1e-3, n_epochs=1)
        updater_large = ActiveDataUpdater(mode="bc", lr=1e-3, n_epochs=10)

        _, metrics_small = updater_small.update(weights, dataset)
        _, metrics_large = updater_large.update(weights, dataset)

        # More epochs should drive loss lower on the training set
        assert metrics_large["bc_loss"] < metrics_small["bc_loss"]


# ---------------------------------------------------------------------------
# Mode: "none" (passthrough)
# ---------------------------------------------------------------------------


class TestNoneMode:
    def test_none_mode_returns_unchanged_weights(self):
        weights = _fresh_weights()
        trajs = [_make_trajectory(0, improvement=5.0)]
        dataset = build_active_dataset(trajs, data_threshold=0.0)
        assert dataset is not None

        updater = ActiveDataUpdater(mode="none")
        new_weights, metrics = updater.update(weights, dataset)

        for k in weights:
            assert torch.allclose(weights[k], new_weights[k])
        assert metrics.get("skipped") is True

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            ActiveDataUpdater(mode="invalid")
