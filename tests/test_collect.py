"""
Unit tests for the collect.py module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.aggregator.collect import collect_worker_updates


@pytest.fixture
def temp_results_dir(tmp_path):
    # Dummy worker test doesn't matter much for Minio, need to mock MinIO client
    pass


@patch("src.aggregator.collect._load_tensor")
@patch("src.aggregator.collect._load_json")
def test_collect_worker_updates(mock_load_json, mock_load_tensor):
    mock_minio = MagicMock()

    # Mock return values for tensor and json
    mock_load_tensor.return_value = {"layer": torch.zeros(1)}
    mock_load_json.return_value = {
        "avg_reward": 10.0,
        "avg_td_error": 1.0,
    }

    results = collect_worker_updates(mock_minio, "test-bucket", fl_round=1, num_workers=3)

    assert len(results) == 3

    assert results[0].avg_reward == 10.0
    assert results[1].avg_reward == 10.0
    assert results[2].avg_reward == 10.0


@patch("src.aggregator.collect._load_tensor", side_effect=Exception("Failed"))
def test_collect_worker_updates_empty(mock_load_tensor):
    mock_minio = MagicMock()
    results = collect_worker_updates(mock_minio, "test-bucket", fl_round=1, num_workers=3)
    assert len(results) == 0
