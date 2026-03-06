"""
Multi-factor Importance Scorer for Federated RL Clients

Combines four signals into a single score per client:
  1. Target-env improvement  (primary: does this weight help the global model?)
  2. Gradient norm           (filter clients that barely moved)
  3. Weight diversity        (penalize redundant / identical updates)
  4. TD-error penalty        (prefer stable value estimates)

All factors are normalized to [0, 1] per round before combining.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import torch

from src.aggregator.evaluator import EvalResult

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ClientUpdate:
    worker_id: int
    weights: dict[str, torch.Tensor]  # W_local (full weights)
    weight_delta: dict[str, torch.Tensor]  # ΔW_i = W_local - W_global
    avg_reward: float  # local training reward (informational only)
    avg_td_error: float  # mean per-step TD error from local training
    weight_norm: float  # ‖ΔW_i‖₂


@dataclasses.dataclass
class ScoredClient:
    worker_id: int
    score: float
    improvement: float  # from evaluator
    norm_score: float  # gradient norm contribution
    diversity_score: float  # uniqueness contribution
    td_penalty: float  # td_error penalty
    accepted: bool  # True if score >= threshold


def _l2_norm(delta: dict[str, torch.Tensor]) -> float:
    """Compute ‖ΔW‖₂ across all parameter tensors."""
    total = sum(torch.norm(v.float()) ** 2 for v in delta.values())
    return float(total**0.5)


def _cosine_similarity_to_mean(
    delta: dict[str, torch.Tensor],
    mean_delta: dict[str, torch.Tensor],
) -> float:
    """Cosine similarity between a client's delta and the mean delta."""
    flat_d = torch.cat([v.float().flatten() for v in delta.values()])
    flat_m = torch.cat([v.float().flatten() for v in mean_delta.values()])
    cos = torch.nn.functional.cosine_similarity(flat_d.unsqueeze(0), flat_m.unsqueeze(0))
    return float(cos.item())


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. All-equal → 0.5."""
    arr = np.array(values, dtype=np.float64)
    rng = arr.max() - arr.min()
    if rng < 1e-8:
        return [0.5] * len(values)
    return list((arr - arr.min()) / rng)


def score_clients(
    clients: list[ClientUpdate],
    eval_results: dict[int, EvalResult],
    score_threshold: float = 0.0,
    alpha: float = 0.5,  # weight for target-env improvement
    beta: float = 0.2,  # weight for gradient norm
    gamma: float = 0.2,  # weight for diversity
    delta: float = 0.1,  # weight for TD-error penalty
) -> list[ScoredClient]:
    """
    Score all clients and determine which are accepted for aggregation.

    Args:
        clients: list of ClientUpdate from this round
        eval_results: per-worker EvalResult from evaluator.py
        score_threshold: min improvement required to be accepted
        alpha, beta, gamma, delta: factor weights

    Returns:
        list of ScoredClient sorted by score (descending)
    """
    if not clients:
        return []

    # ---- Compute raw factor values ----
    improvements = [eval_results[c.worker_id].improvement for c in clients]
    norms = [_l2_norm(c.weight_delta) for c in clients]
    td_errors = [c.avg_td_error for c in clients]

    # Mean delta for diversity computation
    mean_delta: dict[str, torch.Tensor] = {}
    for key in clients[0].weight_delta:
        stacked = torch.stack([c.weight_delta[key].float() for c in clients])
        mean_delta[key] = stacked.mean(dim=0)

    cos_sims = [_cosine_similarity_to_mean(c.weight_delta, mean_delta) for c in clients]
    diversities = [1.0 - cs for cs in cos_sims]  # high diversity = low similarity to mean

    # ---- Normalize all factors to [0, 1] ----
    norm_improvements = _normalize(improvements)
    norm_norms = _normalize(norms)
    norm_diversities = _normalize(diversities)
    norm_td_errors = _normalize(td_errors)

    # ---- Combine ----
    scored: list[ScoredClient] = []
    for i, client in enumerate(clients):
        score = (
            alpha * norm_improvements[i]
            + beta * norm_norms[i]
            + gamma * norm_diversities[i]
            - delta * norm_td_errors[i]
        )
        accepted = improvements[i] >= score_threshold

        scored.append(
            ScoredClient(
                worker_id=client.worker_id,
                score=score,
                improvement=improvements[i],
                norm_score=norm_norms[i],
                diversity_score=norm_diversities[i],
                td_penalty=norm_td_errors[i],
                accepted=accepted,
            )
        )
        log.info(
            f"Worker {client.worker_id}: score={score:.4f} | "
            f"improvement={improvements[i]:+.2f} | "
            f"norm={norm_norms[i]:.3f} | diversity={norm_diversities[i]:.3f} | "
            f"td_penalty={norm_td_errors[i]:.3f} | accepted={accepted}"
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    n_accepted = sum(1 for s in scored if s.accepted)
    log.info(f"Accepted {n_accepted}/{len(scored)} clients for aggregation")
    return scored
