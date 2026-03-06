"""
Unit tests for the multi-factor scorer (scorer.py).
"""

from __future__ import annotations

import torch

from src.agent.model import ActorCritic
from src.aggregator.evaluator import EvalResult
from src.aggregator.scorer import ClientUpdate, ScoredClient, score_clients


def _make_client(
    worker_id: int, avg_reward: float, avg_td_error: float, delta_scale: float = 0.01
) -> ClientUpdate:
    model = ActorCritic()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    delta = {k: torch.randn_like(v) * delta_scale for k, v in model.state_dict().items()}
    norm = float(sum(torch.norm(v.float()) ** 2 for v in delta.values()) ** 0.5)
    return ClientUpdate(
        worker_id=worker_id,
        weights=weights,
        weight_delta=delta,
        avg_reward=avg_reward,
        avg_td_error=avg_td_error,
        weight_norm=norm,
    )


def _make_eval(worker_id: int, improvement: float) -> EvalResult:
    return EvalResult(
        worker_id=worker_id,
        candidate_reward=100.0 + improvement,
        baseline_reward=100.0,
        improvement=improvement,
    )


class TestScoreClients:
    def test_returns_scored_clients_for_all(self):
        clients = [_make_client(i, avg_reward=100.0 + i * 10, avg_td_error=0.1) for i in range(3)]
        eval_results = {i: _make_eval(i, improvement=float(i * 5)) for i in range(3)}

        scored = score_clients(clients, eval_results, score_threshold=0.0)
        assert len(scored) == 3
        assert all(isinstance(s, ScoredClient) for s in scored)

    def test_sorted_by_score_descending(self):
        clients = [_make_client(i, avg_reward=100.0, avg_td_error=0.1) for i in range(4)]
        # Client 3 has largest improvement → should rank highest
        eval_results = {
            0: _make_eval(0, improvement=1.0),
            1: _make_eval(1, improvement=5.0),
            2: _make_eval(2, improvement=10.0),
            3: _make_eval(3, improvement=20.0),
        }

        scored = score_clients(clients, eval_results, score_threshold=0.0)
        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_rejects_negative_improvement(self):
        clients = [_make_client(i, avg_reward=100.0, avg_td_error=0.1) for i in range(3)]
        eval_results = {
            0: _make_eval(0, improvement=10.0),
            1: _make_eval(1, improvement=-5.0),  # should be rejected
            2: _make_eval(2, improvement=3.0),
        }

        scored = score_clients(clients, eval_results, score_threshold=0.0)
        accepted = [s for s in scored if s.accepted]
        rejected = [s for s in scored if not s.accepted]

        assert len(accepted) == 2
        assert len(rejected) == 1
        assert rejected[0].worker_id == 1

    def test_all_equal_clients_get_identical_scores(self):
        """When all clients are identical, score normalization yields 0.5 per factor."""
        delta = {k: torch.ones_like(v) * 0.01 for k, v in ActorCritic().state_dict().items()}
        weights = {k: v.clone() for k, v in ActorCritic().state_dict().items()}
        norm = float(sum(torch.norm(v.float()) ** 2 for v in delta.values()) ** 0.5)
        clients = [
            ClientUpdate(
                worker_id=i,
                weights=weights,
                weight_delta=delta,
                avg_reward=100.0,
                avg_td_error=0.1,
                weight_norm=norm,
            )
            for i in range(3)
        ]
        eval_results = {i: _make_eval(i, improvement=5.0) for i in range(3)}

        scored = score_clients(clients, eval_results)
        scores = [s.score for s in scored]
        # All scores should be equal (within floating point)
        assert max(scores) - min(scores) < 1e-6

    def test_empty_clients_returns_empty(self):
        result = score_clients([], {})
        assert result == []
