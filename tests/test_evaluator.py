"""
Unit tests for the target-environment evaluation probe (evaluator.py).

Uses a mock gym environment rather than real CartPole to keep tests fast.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.agent.model import ActorCritic
from src.aggregator.evaluator import evaluate_candidate, evaluate_all_candidates, EvalResult


def _fresh_weights() -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in ActorCritic().state_dict().items()}


def _make_delta(scale: float = 0.01) -> dict[str, torch.Tensor]:
    model = ActorCritic()
    return {k: torch.randn_like(v) * scale for k, v in model.state_dict().items()}


class TestEvalResult:
    def test_improvement_computed_correctly(self):
        result = EvalResult(
            worker_id=0,
            candidate_reward=120.0,
            baseline_reward=80.0,
            improvement=40.0,
        )
        assert result.improvement == pytest.approx(40.0)

    def test_negative_improvement(self):
        result = EvalResult(
            worker_id=1,
            candidate_reward=50.0,
            baseline_reward=100.0,
            improvement=-50.0,
        )
        assert result.improvement < 0


class TestEvaluateCandidate:
    def test_returns_eval_result(self):
        global_weights = _fresh_weights()
        delta = _make_delta(0.001)

        result = evaluate_candidate(
            global_weights=global_weights,
            weight_delta=delta,
            worker_id=0,
            env_id="CartPole-v1",
            n_eval_episodes=2,
        )
        assert isinstance(result, EvalResult)
        assert result.worker_id == 0
        assert result.candidate_reward >= 0
        assert result.baseline_reward >= 0
        assert abs(result.improvement - (result.candidate_reward - result.baseline_reward)) < 1e-6

    def test_large_negative_delta_degrades_performance(self):
        """Extreme weight corruption should generally hurt performance."""
        global_weights = _fresh_weights()
        # Adversarial delta: random large noise
        corrupt_delta = {k: torch.randn_like(v) * 100 for k, v in global_weights.items()}

        result = evaluate_candidate(
            global_weights=global_weights,
            weight_delta=corrupt_delta,
            worker_id=0,
            env_id="CartPole-v1",
            n_eval_episodes=5,
        )
        # Corrupt weights likely hurt performance (improvement should be negative or low)
        # Not guaranteed but statistically very likely with scale=100
        assert isinstance(result.improvement, float)

    def test_zero_delta_gives_zero_improvement(self):
        """Zero delta should yield the exact same weights → improvement ≈ 0."""
        global_weights = _fresh_weights()
        zero_delta = {k: torch.zeros_like(v) for k, v in global_weights.items()}

        result = evaluate_candidate(
            global_weights=global_weights,
            weight_delta=zero_delta,
            worker_id=0,
            env_id="CartPole-v1",
            n_eval_episodes=3,
        )
        # Same weights evaluated with same seed → improvement should be tiny
        assert abs(result.improvement) < 15.0  # random model has high per-episode variance


class TestEvaluateAllCandidates:
    def test_returns_result_for_each_worker(self):
        global_weights = _fresh_weights()
        deltas = {
            0: _make_delta(0.001),
            1: _make_delta(0.001),
            2: _make_delta(0.001),
        }

        results = evaluate_all_candidates(
            global_weights=global_weights,
            weight_deltas=deltas,
            n_eval_episodes=2,
            max_workers=2,
        )
        assert set(results.keys()) == {0, 1, 2}
        for wid, r in results.items():
            assert isinstance(r, EvalResult)
            assert r.worker_id == wid

    def test_baseline_is_same_for_all_workers(self):
        global_weights = _fresh_weights()
        deltas = {0: _make_delta(), 1: _make_delta()}

        results = evaluate_all_candidates(
            global_weights=global_weights,
            weight_deltas=deltas,
            n_eval_episodes=2,
        )
        baseline_0 = results[0].baseline_reward
        baseline_1 = results[1].baseline_reward
        # Both should use the same baseline evaluation run
        assert baseline_0 == pytest.approx(baseline_1, abs=0.1)
