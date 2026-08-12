import subprocess
from types import SimpleNamespace

import pytest

from src.pipelines.run_pipeline import report_failures_and_exit, run_step, wait_for_runs


def test_successful_step_returns_quietly():
    run_step(["true"], "noop")


def test_failing_step_raises_naming_the_step():
    with pytest.raises(subprocess.CalledProcessError):
        run_step(["false"], "fetch results")


def test_missing_binary_raises():
    with pytest.raises((FileNotFoundError, subprocess.CalledProcessError)):
        run_step(["definitely-not-a-real-binary-xyz"], "plot")


# ---------------------------------------------------------------------------
# F3: the --wait loop must not swallow a failed/timed-out run. A failure must
# be collected (not just logged) and must not stop the loop from waiting on
# the remaining runs, and the collected failures must ultimately fail the
# process (non-zero exit) rather than let main() report success.
# ---------------------------------------------------------------------------
class _FakeClient:
    def __init__(self, failing_ids):
        self._failing_ids = set(failing_ids)
        self.calls: list[str] = []

    def wait_for_run_completion(self, run_id, timeout):
        self.calls.append(run_id)
        if run_id in self._failing_ids:
            raise TimeoutError(f"run {run_id} did not complete within {timeout}s")
        return SimpleNamespace(state="SUCCEEDED")


def test_wait_for_runs_still_waits_on_remaining_runs_after_a_failure():
    client = _FakeClient(failing_ids={"run-bad"})
    submitted = [("run-bad", "bad-name"), ("run-good", "good-name")]

    failures = wait_for_runs(client, submitted, timeout=30)

    assert client.calls == ["run-bad", "run-good"], (
        "a failed/timed-out run must not stop the loop from waiting on the "
        "remaining submitted runs"
    )
    assert failures, "a failed run must be collected, not merely logged"
    assert any("bad-name" in f for f in failures)


def test_wait_for_runs_returns_empty_when_everything_succeeds():
    client = _FakeClient(failing_ids=set())
    submitted = [("run-a", "a-name"), ("run-b", "b-name")]

    failures = wait_for_runs(client, submitted, timeout=30)

    assert client.calls == ["run-a", "run-b"]
    assert failures == []


def test_report_failures_and_exit_raises_systemexit_when_any_failure():
    # This is the half of F3 that a "logged but not collected" failure
    # cannot satisfy: a caller scripting around run_pipeline.py's exit code
    # (CI, cron) must see non-zero, not a silent success.
    with pytest.raises(SystemExit) as exc_info:
        report_failures_and_exit(["run x failed or timed out"])
    assert exc_info.value.code != 0


def test_report_failures_and_exit_is_quiet_when_no_failures():
    report_failures_and_exit([])  # must not raise
