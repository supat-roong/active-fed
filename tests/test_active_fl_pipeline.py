"""
Structural checks on the compiled Active-FL Kubeflow pipeline IR.

F4 (gate fix): this KFP deployment never substitutes
dsl.PIPELINE_JOB_ID_PLACEHOLDER before component code runs. Task 6 already
proved -- purely structurally, from the compiled IR -- that the placeholder
compiles to the expected `{{$.pipeline_job_uuid}}` string; only a real
cluster run (task-7-report.md, Step 5) could show that string then arrives at
`train_workers` *verbatim*, producing Job names like
"aflw-{{$.pipe-r0-w0" that Kubernetes' API rejects outright (422). The fix is
to stop depending on KFP placeholder substitution entirely: `run_uid` is now
a normal pipeline argument that `run_pipeline.py` fills in with an explicit,
generated value. These tests guard both sides of that:
  - the compiled IR never contains a KFP runtime placeholder anywhere, and
  - the pipeline's own default for direct compilation is itself a valid,
    job_name_for-safe value (lowercase alphanumeric).
"""

import re

import kfp
import yaml

from src.pipelines.active_fl_pipeline import active_fl_pipeline

_VALID_RUN_ID_FRAGMENT = re.compile(r"[a-z0-9]+")


def _compile(tmp_path):
    output = tmp_path / "active_fl_pipeline.yaml"
    kfp.compiler.Compiler().compile(pipeline_func=active_fl_pipeline, package_path=str(output))
    return output.read_text()


def test_compiled_pipeline_has_no_unsubstituted_kfp_placeholder(tmp_path):
    text = _compile(tmp_path)
    assert "{{$." not in text, (
        "compiled IR contains a KFP runtime placeholder (e.g. "
        "'{{$.pipeline_job_uuid}}'); this KFP deployment does not substitute "
        "these before component code runs, so any value built from one "
        "(e.g. a Kubernetes Job name) is broken at runtime even though it "
        "compiles fine"
    )


def test_compiled_pipeline_threads_run_uid_into_train_workers(tmp_path):
    text = _compile(tmp_path)
    spec = yaml.safe_load(text)
    root_params = spec["root"]["inputDefinitions"]["parameters"]
    assert "run_uid" in root_params, "run_uid must be a top-level pipeline parameter"


def test_run_uid_default_is_a_valid_job_name_for_fragment(tmp_path):
    # Guards the direct-compilation path (`make compile-pipeline`, no
    # run_pipeline.py involved): the pipeline's own default for run_uid must
    # independently satisfy job_name_for's validation, since nothing else
    # supplies a value in that path.
    text = _compile(tmp_path)
    spec = yaml.safe_load(text)
    default = spec["root"]["inputDefinitions"]["parameters"]["run_uid"]["defaultValue"]
    assert isinstance(default, str)
    assert _VALID_RUN_ID_FRAGMENT.fullmatch(default[:8]), default
