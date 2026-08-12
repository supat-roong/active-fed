import subprocess

import pytest

from src.pipelines.run_pipeline import run_step


def test_successful_step_returns_quietly():
    run_step(["true"], "noop")


def test_failing_step_raises_naming_the_step():
    with pytest.raises(subprocess.CalledProcessError):
        run_step(["false"], "fetch results")


def test_missing_binary_raises():
    with pytest.raises((FileNotFoundError, subprocess.CalledProcessError)):
        run_step(["definitely-not-a-real-binary-xyz"], "plot")
