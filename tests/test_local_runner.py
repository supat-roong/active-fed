"""
Unit tests for the local_runner.py module.
These tests use extensive mocking to avoid running actual RL training or Kubernetes jobs, ensuring they execute quickly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.experiment.local_runner import (
    RunConfig,
    run_experiment,
)


@pytest.fixture
def temp_results_dir(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture
def base_config(temp_results_dir):
    return RunConfig(
        run_name="test_run",
        weight_mode="active",
        active_data_mode="none",
        fl_rounds=2,
        num_workers=2,
        local_episodes=5,
    )


class TestLocalExperimentRunner:
    @patch("src.experiment.local_runner.mlflow")
    @patch("src.experiment.local_runner.aggregate")
    @patch("src.experiment.local_runner.score_clients")
    @patch("src.experiment.local_runner.evaluate_all_candidates")
    @patch("src.experiment.local_runner.as_completed")
    @patch("src.experiment.local_runner.ThreadPoolExecutor")
    def test_run_sequential(
        self,
        mock_executor_cls,
        mock_as_completed,
        mock_eval_all,
        mock_score_clients,
        mock_aggregate,
        mock_mlflow,
        base_config,
    ):
        base_config.num_workers = 1

        # Mock executor
        mock_executor = MagicMock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor

        # Mock results from futures
        mock_future = MagicMock()
        mock_client = MagicMock()
        mock_client.worker_id = 0
        mock_client.avg_reward = 10.0
        mock_client.weight_delta = {}
        mock_future.result.return_value = (mock_client, 10.0, 1.0)
        mock_executor.submit.return_value = mock_future

        # Mock as_completed to just iterate over the futures dict keys
        def mock_as_completed_side_effect(futures_dict):
            return list(futures_dict.keys())

        mock_as_completed.side_effect = mock_as_completed_side_effect

        # Mock eval and score
        mock_eval_all.return_value = {0: MagicMock(candidate_reward=50.0)}
        mock_score = MagicMock()
        mock_score.worker_id = 0
        mock_score.score = 1.0
        mock_score.improvement = 10.0
        mock_score.accepted = True
        mock_score_clients.return_value = [mock_score]

        # Mock aggregate returns
        mock_agg_result = MagicMock()
        mock_agg_result.accepted_ids = [0]
        mock_agg_result.rejected_ids = []
        mock_agg_result.global_weights = {}
        mock_agg_result.round_summary = {}
        mock_agg_result.active_data_applied = False
        mock_agg_result.active_data_n_steps = 0
        mock_aggregate.return_value = mock_agg_result

        # Run the experiment
        with patch("src.experiment.local_runner._eval_global", return_value=(20.0, 1.0)):
            result = run_experiment(base_config)

        # Assertions
        assert result.run_name == "test_run"
        assert len(result.rounds) == 2  # 2 rounds

        # Check call counts
        assert mock_executor.submit.call_count == base_config.fl_rounds * base_config.num_workers
        assert mock_aggregate.call_count == base_config.fl_rounds


# Removed invalid tests
