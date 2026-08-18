"""
Phase P4 Task 2: MLflow cross-link tags.

Covers `log_run_context` (src/tracking/mlflow_logger.py) in isolation, and its
wiring into `evaluate_global` (src/pipelines/active_fl_pipeline.py), which
reads the `temporal_workflow_id` that `train_workers` writes into the
`worker_report` artifact -- investigated and confirmed present in both the
success and quorum-failure payloads (see train_workers in
active_fl_pipeline.py and tests/test_train_workers_component.py) -- and opens
one MLflow run per round that these tags attach to.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

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


# ---------------------------------------------------------------------------
# Wiring into evaluate_global: it must read temporal_workflow_id out of the
# worker_report artifact (written by train_workers) and pass it, together
# with kfp_run_id and topology, into log_run_context -- inside the MLflow run
# it already opens per round.
# ---------------------------------------------------------------------------


class _FakeMinioResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeMinio:
    """Stands in for the real Minio client `evaluate_global` constructs.

    `evaluate_global` only ever calls get_object for the new global weights
    (round_{fl_round + 1}); the actual tensor contents don't matter because
    `_rollout` is monkeypatched below and never really reads them.
    """

    def __init__(self, endpoint=None, access_key=None, secret_key=None, secure=None):
        pass

    def get_object(self, bucket: str, key: str) -> _FakeMinioResponse:
        buf = io.BytesIO()
        torch.save({"w": torch.zeros(1)}, buf)
        return _FakeMinioResponse(buf.getvalue())


@pytest.fixture(autouse=True)
def _patch_heavy_deps(monkeypatch):
    monkeypatch.setattr("minio.Minio", _FakeMinio)
    monkeypatch.setattr(
        "src.aggregator.evaluator._rollout",
        lambda weights, env_id, n_episodes, physics_seed, episode_seed: (
            150.0,
            [150.0] * n_episodes,
            [200] * n_episodes,
        ),
    )
    mlflow_mock = MagicMock()
    mlflow_mock.start_run.return_value.__enter__.return_value = MagicMock()
    mlflow_mock.start_run.return_value.__exit__.return_value = False
    monkeypatch.setattr("mlflow.set_tracking_uri", mlflow_mock.set_tracking_uri)
    monkeypatch.setattr("mlflow.set_experiment", mlflow_mock.set_experiment)
    monkeypatch.setattr("mlflow.start_run", mlflow_mock.start_run)
    monkeypatch.setattr("mlflow.log_metrics", mlflow_mock.log_metrics)
    monkeypatch.setattr("mlflow.log_artifact", mlflow_mock.log_artifact)


def _write_json(path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f)


def _aggregation_report_payload() -> dict:
    return {
        "round_summary": {
            "clients_accepted": 2,
            "clients_rejected": 0,
            "effective_weight_norm": 1.0,
        },
        "active_data_applied": False,
        "active_data_n_steps": 0,
        "active_data_source_workers": [],
        "scored_clients": [],
    }


def _run_evaluate_global(tmp_path, monkeypatch, *, kfp_run_id, topology, temporal_workflow_id,
    artifact_uri=(
        "minio://mlpipeline/v2/artifacts/active-fl-cartpole/"
        "51debb42-f3fa-4c95-946a-63dbd0e70ece/train-workers/uuid/worker_report"
    ),
):
    from src.pipelines.active_fl_pipeline import evaluate_global

    logged: dict = {}
    monkeypatch.setattr(
        "src.tracking.mlflow_logger.log_run_context", lambda **kw: logged.update(kw)
    )

    agg_report_path = tmp_path / "aggregation_report.json"
    _write_json(agg_report_path, _aggregation_report_payload())

    worker_report_path = tmp_path / "worker_report.json"
    worker_payload = {"fl_round": 0, "succeeded": [0, 1], "failed": []}
    if temporal_workflow_id:
        worker_payload["temporal_workflow_id"] = temporal_workflow_id
    _write_json(worker_report_path, worker_payload)

    eval_result_path = tmp_path / "eval_result.json"

    evaluate_global.python_func(
        fl_round=0,
        n_eval_episodes=1,
        minio_endpoint="minio:9000",
        minio_access_key="a",
        minio_secret_key="b",
        minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="exp",
        kfp_run_id=kfp_run_id,
        topology=topology,
        aggregation_report=SimpleNamespace(path=str(agg_report_path)),
        worker_report=SimpleNamespace(
            path=str(worker_report_path),
            # KFP stamps its own run id into the artifact URI; that is
            # where the kfp_run_id tag comes from, not the run_uid
            # parameter (which names the bucket and Jobs and does not
            # resolve in the KFP UI).
            uri=artifact_uri,
        ),
        eval_result=SimpleNamespace(path=str(eval_result_path)),
    )
    return logged


def test_evaluate_global_threads_temporal_workflow_id_from_worker_report(tmp_path, monkeypatch):
    logged = _run_evaluate_global(
        tmp_path,
        monkeypatch,
        kfp_run_id="abcdef1234",
        topology="single",
        temporal_workflow_id="train-abcdef12-r0",
    )
    assert logged["kfp_run_id"] == "51debb42-f3fa-4c95-946a-63dbd0e70ece", (
        "the tag must carry KFP's own run id, recovered from the artifact URI, "
        "not the locally-generated run_uid -- a kfp_run_url built from run_uid "
        "resolves to nothing in the KFP UI"
    )
    assert logged["temporal_workflow_id"] == "train-abcdef12-r0"
    assert logged["topology"] == "single"


@pytest.mark.parametrize("topology", ["single", "multi"])
def test_evaluate_global_topology_tag_correct_under_both_topologies(
    tmp_path, monkeypatch, topology
):
    logged = _run_evaluate_global(
        tmp_path,
        monkeypatch,
        kfp_run_id="abcdef1234",
        topology=topology,
        temporal_workflow_id="train-abcdef12-r0",
    )
    assert logged["topology"] == topology


def test_evaluate_global_does_not_crash_when_worker_report_lacks_workflow_id(
    tmp_path, monkeypatch
):
    # Regression guard for the Step 3 investigation: if a future change ever
    # drops temporal_workflow_id from worker_report, evaluate_global must
    # degrade to an empty string (which log_run_context then skips) rather
    # than raising a KeyError that would fail the whole round.
    logged = _run_evaluate_global(
        tmp_path,
        monkeypatch,
        kfp_run_id="abcdef1234",
        topology="single",
        temporal_workflow_id="",
    )
    assert logged["temporal_workflow_id"] == ""


class TestKfpRunIdFromArtifactUri:
    """The KFP run id has to come from somewhere that actually resolves.

    `run_uid` is a locally generated fragment used to name the MinIO bucket and
    the worker Jobs; it is NOT the id KFP's UI uses in
    `/#/runs/details/<id>`, so tagging it produced a kfp_run_url that silently
    goes nowhere -- worse than no tag, because a broken link looks like data.

    KFP embeds the real run id in every artifact URI it mints, e.g.
      minio://mlpipeline/v2/artifacts/<pipeline>/<run-id>/<task>/<uuid>/<name>
    and evaluate_global already receives such an artifact, so the id can be
    recovered where it is needed without a new API call or placeholder.
    """

    def test_extracts_the_run_id_from_a_real_kfp_artifact_uri(self):
        from src.tracking.mlflow_logger import kfp_run_id_from_artifact_uri

        # Captured from a live gate run: run_pipeline.py reported this exact
        # Run ID, and this is the artifact URI KFP minted for it.
        uri = (
            "minio://mlpipeline/v2/artifacts/active-fl-cartpole/"
            "51debb42-f3fa-4c95-946a-63dbd0e70ece/train-workers/"
            "0c443c88-d036-4e29-b990-bf939498c5aa/worker_report"
        )
        assert (
            kfp_run_id_from_artifact_uri(uri) == "51debb42-f3fa-4c95-946a-63dbd0e70ece"
        )

    def test_returns_empty_for_an_unrecognised_uri_rather_than_guessing(self):
        from src.tracking.mlflow_logger import kfp_run_id_from_artifact_uri

        for uri in [
            "",
            "minio://mlpipeline/something/else",
            "/local/path/worker_report",
            "minio://mlpipeline/v2/artifacts",
        ]:
            assert kfp_run_id_from_artifact_uri(uri) == "", uri

    def test_does_not_mistake_the_pipeline_name_for_the_run_id(self):
        from src.tracking.mlflow_logger import kfp_run_id_from_artifact_uri

        uri = (
            "minio://mlpipeline/v2/artifacts/my-pipeline/"
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/task/uuid/artifact"
        )
        assert kfp_run_id_from_artifact_uri(uri) == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_never_raises_on_malformed_input(self):
        from src.tracking.mlflow_logger import kfp_run_id_from_artifact_uri

        for bad in [None, 123, object()]:
            assert kfp_run_id_from_artifact_uri(bad) == ""


def test_evaluate_global_skips_the_kfp_tag_when_the_uri_is_unrecognised(tmp_path, monkeypatch):
    """No tag beats a broken one.

    If the artifact URI is not in KFP's expected layout, the run id cannot be
    recovered, and writing kfp_run_url from a guess would produce a link that
    404s while looking like real data.
    """
    logged = _run_evaluate_global(
        tmp_path,
        monkeypatch,
        kfp_run_id="abcdef1234",
        topology="single",
        temporal_workflow_id="wf-1",
        artifact_uri="/local/path/worker_report",
    )
    assert logged["kfp_run_id"] == ""
