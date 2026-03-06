"""
Unit tests for weighted aggregation (aggregator.py).
"""

from __future__ import annotations

import torch

from src.agent.model import ActorCritic
from src.aggregator.aggregator import AggregationResult, aggregate
from src.aggregator.scorer import ClientUpdate, ScoredClient


def _weights(scale: float = 1.0) -> dict[str, torch.Tensor]:
    return {k: v.clone() * scale for k, v in ActorCritic().state_dict().items()}


def _client(worker_id: int, scale: float = 1.0) -> ClientUpdate:
    w = _weights(scale)
    delta = {k: torch.zeros_like(v) for k, v in w.items()}
    return ClientUpdate(
        worker_id=worker_id,
        weights=w,
        weight_delta=delta,
        avg_reward=100.0,
        avg_td_error=0.1,
        weight_norm=0.0,
    )


def _scored(worker_id: int, score: float, improvement: float, accepted: bool) -> ScoredClient:
    return ScoredClient(
        worker_id=worker_id,
        score=score,
        improvement=improvement,
        norm_score=0.5,
        diversity_score=0.5,
        td_penalty=0.1,
        accepted=accepted,
    )


# ---------------------------------------------------------------------------
# Existing Active Weight tests
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_equal_scores_yields_equal_weights(self):
        """Two clients with equal scores should yield equal blending -> their average."""
        w_global = _weights(0.0)  # zero global
        clients = [_client(0, scale=1.0), _client(1, scale=3.0)]
        scored = [
            _scored(0, score=1.0, improvement=1.0, accepted=True),
            _scored(1, score=1.0, improvement=1.0, accepted=True),
        ]

        result = aggregate(w_global, clients, scored, temperature=1.0)
        assert isinstance(result, AggregationResult)
        assert set(result.accepted_ids) == {0, 1}

        # With equal softmax weights (~0.5 each), new global ~ mean(scale=1.0, scale=3.0) = 2.0
        for key in w_global:
            expected = (clients[0].weights[key] + clients[1].weights[key]) / 2
            assert torch.allclose(result.global_weights[key].float(), expected.float(), atol=1e-5)

    def test_all_rejected_keeps_global_unchanged(self):
        w_global = _weights(2.0)
        clients = [_client(0)]
        scored = [_scored(0, score=0.5, improvement=-10.0, accepted=False)]

        result = aggregate(w_global, clients, scored)
        assert result.accepted_ids == []
        assert result.rejected_ids == [0]
        for key in w_global:
            assert torch.allclose(result.global_weights[key], w_global[key])

    def test_single_accepted_client_copies_its_weights(self):
        w_global = _weights(0.0)
        client = _client(0, scale=5.0)
        scored = [_scored(0, score=1.0, improvement=10.0, accepted=True)]

        result = aggregate(w_global, [client], scored)
        assert result.accepted_ids == [0]
        for key in w_global:
            assert torch.allclose(
                result.global_weights[key].float(), client.weights[key].float(), atol=1e-5
            )

    def test_higher_score_gets_more_weight(self):
        """High-score client should dominate aggregation at low temperature."""
        w_global = _weights(0.0)
        client_low = _client(0, scale=0.0)  # weights = 0
        client_high = _client(1, scale=10.0)  # weights = 10

        scored = [
            _scored(0, score=0.01, improvement=1.0, accepted=True),  # very low score
            _scored(1, score=10.0, improvement=9.0, accepted=True),  # very high score
        ]

        result = aggregate(w_global, [client_low, client_high], scored, temperature=0.01)
        # At very low temperature, softmax -> winner take all -> result ~ client_high weights
        for key in w_global:
            assert torch.allclose(
                result.global_weights[key].float(),
                client_high.weights[key].float(),
                atol=0.5,
            )

    def test_round_summary_populated(self):
        w_global = _weights(1.0)
        clients = [_client(0, scale=1.5)]
        scored = [_scored(0, score=1.0, improvement=5.0, accepted=True)]

        result = aggregate(w_global, clients, scored)
        assert "clients_accepted" in result.round_summary
        assert "effective_weight_norm" in result.round_summary
        assert result.round_summary["clients_accepted"] == 1


# ---------------------------------------------------------------------------
# Weight mode tests
# ---------------------------------------------------------------------------


class TestWeightMode:
    def test_fedavg_mode_averages_equally(self):
        """fedavg must ignore scores and average all clients equally."""
        w_global = _weights(0.0)
        clients = [_client(0, scale=0.0), _client(1, scale=4.0)]
        # dramatically different scores -- should NOT affect fedavg
        scored = [
            _scored(0, score=0.01, improvement=1.0, accepted=True),
            _scored(1, score=99.0, improvement=9.0, accepted=True),
        ]

        result = aggregate(w_global, clients, scored, weight_mode="fedavg")
        assert result.weight_mode == "fedavg"
        # Equal weights -> mean of scale=0 and scale=4 -> scale=2
        for key in w_global:
            expected = (clients[0].weights[key] + clients[1].weights[key]) / 2
            assert torch.allclose(result.global_weights[key].float(), expected.float(), atol=1e-5)

    def test_data_only_mode_keeps_global_unchanged(self):
        """data_only must NOT touch global weights via FL weight averaging."""
        w_global = _weights(3.0)
        clients = [_client(0, scale=10.0), _client(1, scale=20.0)]
        scored = [
            _scored(0, score=1.0, improvement=5.0, accepted=True),
            _scored(1, score=1.0, improvement=5.0, accepted=True),
        ]

        result = aggregate(w_global, clients, scored, weight_mode="data_only")
        assert result.weight_mode == "data_only"
        # Global weights must remain unchanged (active data not triggered since 
        # mode=none by default)
        for key in w_global:
            assert torch.allclose(
                result.global_weights[key].float(), w_global[key].float(), atol=1e-5
            )

    def test_weight_mode_recorded_in_result_and_summary(self):
        w_global = _weights(1.0)
        clients = [_client(0)]
        scored = [_scored(0, score=1.0, improvement=1.0, accepted=True)]

        for mode in ["active", "fedavg", "data_only"]:
            result = aggregate(w_global, clients, scored, weight_mode=mode)
            assert result.weight_mode == mode
            assert result.round_summary["weight_mode"] == mode

    def test_active_mode_is_default(self):
        w_global = _weights(1.0)
        clients = [_client(0)]
        scored = [_scored(0, score=1.0, improvement=1.0, accepted=True)]

        result = aggregate(w_global, clients, scored)
        assert result.weight_mode == "active"

    def test_active_is_closer_to_high_score_client_than_fedavg(self):
        """active mode should pull result toward high-score client; fedavg stays at midpoint."""
        w_global = _weights(0.0)
        clients = [_client(0, scale=0.0), _client(1, scale=10.0)]
        scored = [
            _scored(0, score=0.001, improvement=1.0, accepted=True),  # near-zero
            _scored(1, score=1000.0, improvement=9.0, accepted=True),  # dominant
        ]

        result_active = aggregate(
            w_global, clients, scored, weight_mode="active", temperature=0.001
        )
        result_fedavg = aggregate(w_global, clients, scored, weight_mode="fedavg")

        # active (T=0.001) --> nearly winner-take-all -> close to client_1 (scale=10)
        # fedavg           --> exactly halfway (scale=5)
        # Verify: sum of distances to client_1 is smaller for active than fedavg
        def total_dist(result, ref_client):
            return sum(
                (result.global_weights[k] - ref_client.weights[k]).abs().sum().item()
                for k in w_global
            )

        dist_active = total_dist(result_active, clients[1])
        dist_fedavg = total_dist(result_fedavg, clients[1])
        assert dist_active < dist_fedavg
