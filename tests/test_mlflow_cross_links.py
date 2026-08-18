"""
Phase P4 Task 2: MLflow cross-link tags.

Covers `log_run_context` (src/tracking/mlflow_logger.py) in isolation: the
function that tags an active MLflow run with `kfp_run_id`, `kfp_run_url`,
`temporal_workflow_id`, `temporal_workflow_url` and `topology`, so the four
observability surfaces (KFP, Temporal, MLflow, Kubernetes Dashboard) can be
navigated between instead of correlated by hand across three UIs by
timestamp.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.tracking.mlflow_logger import log_run_context

# ---------------------------------------------------------------------------
# log_run_context in isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mlflow(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("src.tracking.mlflow_logger.mlflow", mock)
    return mock


def test_all_five_tags_set(mock_mlflow):
    log_run_context(
        kfp_run_id="run-abc123",
        temporal_workflow_id="train-abcdef12-r0",
        topology="single",
    )
    mock_mlflow.set_tags.assert_called_once()
    tags = mock_mlflow.set_tags.call_args.args[0]
    assert set(tags.keys()) == {
        "kfp_run_id",
        "kfp_run_url",
        "temporal_workflow_id",
        "temporal_workflow_url",
        "topology",
    }
    assert tags["kfp_run_id"] == "run-abc123"
    assert tags["temporal_workflow_id"] == "train-abcdef12-r0"
    assert tags["topology"] == "single"


def test_urls_are_well_formed_and_contain_the_ids(mock_mlflow):
    log_run_context(
        kfp_run_id="run-abc123",
        temporal_workflow_id="train-abcdef12-r0",
        topology="multi",
        kfp_base_url="http://localhost:8080",
        temporal_base_url="http://localhost:8233",
    )
    tags = mock_mlflow.set_tags.call_args.args[0]
    assert tags["kfp_run_url"] == "http://localhost:8080/#/runs/details/run-abc123"
    assert tags["temporal_workflow_url"] == (
        "http://localhost:8233/namespaces/default/workflows/train-abcdef12-r0"
    )


def test_empty_kfp_run_id_is_skipped_not_written_empty(mock_mlflow):
    log_run_context(
        kfp_run_id="",
        temporal_workflow_id="train-abcdef12-r0",
        topology="single",
    )
    tags = mock_mlflow.set_tags.call_args.args[0]
    assert "kfp_run_id" not in tags
    assert "kfp_run_url" not in tags
    assert tags["temporal_workflow_id"] == "train-abcdef12-r0"


def test_empty_temporal_workflow_id_is_skipped_not_written_empty(mock_mlflow):
    log_run_context(
        kfp_run_id="run-abc123",
        temporal_workflow_id="",
        topology="single",
    )
    tags = mock_mlflow.set_tags.call_args.args[0]
    assert "temporal_workflow_id" not in tags
    assert "temporal_workflow_url" not in tags
    assert tags["kfp_run_id"] == "run-abc123"


def test_empty_topology_is_skipped(mock_mlflow):
    log_run_context(kfp_run_id="run-abc123", temporal_workflow_id="wf-1", topology="")
    tags = mock_mlflow.set_tags.call_args.args[0]
    assert "topology" not in tags


def test_all_empty_ids_result_in_no_set_tags_call(mock_mlflow):
    log_run_context(kfp_run_id="", temporal_workflow_id="", topology="")
    mock_mlflow.set_tags.assert_not_called()


def test_mlflow_failure_is_caught_and_logged_not_raised(mock_mlflow, caplog):
    mock_mlflow.set_tags.side_effect = RuntimeError("mlflow unreachable")
    with caplog.at_level("WARNING"):
        log_run_context(
            kfp_run_id="run-abc123", temporal_workflow_id="wf-1", topology="single"
        )  # must not raise -- tracking must never fail a training run
    assert any("mlflow unreachable" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("topology", ["single", "multi"])
def test_topology_tag_correct_under_both_topologies(mock_mlflow, topology):
    log_run_context(kfp_run_id="run-1", temporal_workflow_id="wf-1", topology=topology)
    tags = mock_mlflow.set_tags.call_args.args[0]
    assert tags["topology"] == topology
