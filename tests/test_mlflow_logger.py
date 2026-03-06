"""
Unit tests for the mlflow_logger.py module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tracking.mlflow_logger import log_round_metrics, log_global_model


@patch("src.tracking.mlflow_logger.mlflow")
def test_log_round_metrics(mock_mlflow):
    mock_agg_result = MagicMock()
    mock_agg_result.round_summary = {
        "clients_accepted": 2,
        "clients_rejected": 1,
        "effective_weight_norm": 1.5,
    }

    # Mock scored client
    mock_sc = MagicMock()
    mock_sc.worker_id = 1
    mock_sc.score = 0.8
    mock_sc.improvement = 5.0
    mock_sc.accepted = True

    mock_agg_result.scored_clients = [mock_sc]

    log_round_metrics(
        fl_round=5,
        agg_result=mock_agg_result,
        global_eval_reward=150.0,
        global_eval_std=10.0,
    )

    mock_mlflow.log_metrics.assert_called_once()
    calls = mock_mlflow.log_metrics.mock_calls
    metrics_dict = calls[0].args[0]

    # Check that key metrics were logged
    assert metrics_dict["global_eval_reward_mean"] == 150.0
    assert metrics_dict["clients_accepted"] == 2.0
    assert metrics_dict["client_1_score"] == 0.8


@patch("src.tracking.mlflow_logger.torch.save")
@patch("src.tracking.mlflow_logger.mlflow")
def test_log_global_model(mock_mlflow, mock_save):
    # Mock nested start_run
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    dummy_weights = {"layer": MagicMock()}
    log_global_model(dummy_weights, fl_round=2)

    mock_save.assert_called_once()
    mock_mlflow.log_artifact.assert_called_once()
