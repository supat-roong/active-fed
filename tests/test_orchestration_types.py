import dataclasses

import pytest

from src.orchestration.types import RoundReport, RoundSpec, WorkerResult, WorkerSpec


def _round_spec(**overrides):
    base = dict(
        fl_round=0,
        num_workers=3,
        min_workers=2,
        local_episodes=10,
        namespace="ns",
        worker_image="img:v1",
        minio_endpoint="minio:9000",
        minio_access_key="a",
        minio_secret_key="b",
        minio_bucket="bucket",
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="exp",
        kfp_run_id="run-1",
    )
    base.update(overrides)
    return RoundSpec(**base)


def _result(worker_id: int, succeeded: bool) -> WorkerResult:
    return WorkerResult(
        worker_id=worker_id,
        succeeded=succeeded,
        attempts=1,
        failure_reason="" if succeeded else "OOMKilled",
        job_name=f"job-{worker_id}",
    )


def test_worker_spec_inherits_round_fields_and_sets_id():
    spec = _round_spec().worker_spec(2)
    assert spec.worker_id == 2
    assert spec.fl_round == 0
    assert spec.num_workers == 3
    assert spec.minio_bucket == "bucket"


def test_round_spec_is_immutable():
    spec = _round_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.fl_round = 5  # type: ignore[misc]


def test_worker_spec_defaults_topology_to_single_with_no_member_cluster():
    spec = WorkerSpec(
        fl_round=0, worker_id=0, num_workers=1, local_episodes=1,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="run-1",
    )
    assert spec.topology == "single"
    assert spec.member_cluster == ""


def test_worker_spec_topology_and_member_cluster_are_frozen():
    spec = WorkerSpec(
        fl_round=0, worker_id=0, num_workers=1, local_episodes=1,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="run-1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.member_cluster = "member1"  # type: ignore[misc]


def test_round_spec_defaults_topology_to_single_with_no_member_cluster():
    spec = _round_spec()
    assert spec.topology == "single"
    assert spec.member_cluster == ""


def test_worker_spec_inherits_round_topology_and_member_cluster():
    spec = _round_spec(topology="multi", member_cluster="active-fed-member1").worker_spec(1)
    assert spec.topology == "multi"
    assert spec.member_cluster == "active-fed-member1"


def test_report_partitions_succeeded_and_failed():
    report = RoundReport(
        fl_round=0,
        results=[_result(0, True), _result(1, False), _result(2, True)],
    )
    assert report.succeeded_ids == [0, 2]
    assert report.failed_ids == [1]


def test_quorum_met_when_enough_workers_succeed():
    report = RoundReport(
        fl_round=0,
        results=[_result(0, True), _result(1, True), _result(2, False)],
    )
    assert report.meets_quorum(2) is True


def test_quorum_not_met_below_threshold():
    report = RoundReport(
        fl_round=0,
        results=[_result(0, True), _result(1, False), _result(2, False)],
    )
    assert report.meets_quorum(2) is False


def test_quorum_with_zero_successes_is_never_met():
    report = RoundReport(fl_round=0, results=[_result(0, False)])
    assert report.meets_quorum(1) is False
