"""
Unit tests for train_worker.py
Uses mocking to avoid real RL training.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.train_worker import train


@pytest.fixture
def temp_global_weights(tmp_path):
    weights_path = tmp_path / "global_weights.pt"
    # Create dummy file
    weights_path.write_text("dummy")
    return str(weights_path)


@pytest.fixture
def temp_output_dir(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return str(out_dir)


@patch("src.agent.train_worker.PPOAgent")
@patch("src.agent.train_worker.mlflow")
def test_train_worker_main(mock_mlflow, mock_agent_cls, temp_global_weights, temp_output_dir):
    # Setup mock agent
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent

    mock_agent.evaluate.return_value = 100.0  # Initial eval

    # Mock train returns (reward, length)
    mock_agent.train.return_value = (50.0, 100)

    # Final eval returns reward
    mock_agent.evaluate.return_value = 150.0

    # Mock the train metrics
    mock_metrics = MagicMock()
    mock_metrics.avg_reward = 50.0
    mock_metrics.avg_td_error = 1.0
    mock_metrics.total_episodes = 100
    mock_metrics.policy_loss = 0.5
    mock_metrics.value_loss = 0.5
    mock_metrics.entropy = 0.1
    mock_metrics.raw_episode_rewards = [50.0, 50.0]
    mock_metrics.raw_episode_steps = [100, 100]
    mock_agent.train_episodes.return_value = mock_metrics

    # Mock the own-env eval forward pass
    mock_action = MagicMock()
    mock_action.sample.return_value.item.return_value = 0
    mock_agent.model.forward.return_value = (mock_action, MagicMock())

    train(
        MagicMock(
            fl_round=1,
            worker_id=0,
            local_episodes=2,
            device="cpu",
            dry_run=True,
            mlflow_tracking_uri="http://localhost:5000",
        )
    )

    # Assertions
    assert mock_agent.set_weights.call_count == 0  # Dry run, shouldn't load global weights
    mock_agent.train_episodes.assert_called()
