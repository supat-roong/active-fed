"""
Structural checks on the compiled Active-FL Kubeflow pipeline IR.

P2 review fix wave, Finding 1: `start_round` used to be declared as a genuine
`@dsl.pipeline` parameter that the function body never read -- the unroll
loop was driven entirely by the START_ROUND env var resolved before
`active_fl_pipeline` is defined. A KFP UI/API caller overriding the declared
parameter got no effect and no error. The parameter is now removed entirely;
`start_round` is compile-time only (mirroring how `fl_rounds` already
works), resolved by `run_pipeline.py` (or a direct `--start-round` CLI flag)
before compiling. See the two `if __name__ == "__main__":` blocks in
active_fl_pipeline.py for why that env var must be set before the pipeline
function is defined, not merely before `Compiler().compile()` is called.

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
import subprocess
import sys
from pathlib import Path

import kfp
import yaml

from src.pipelines.active_fl_pipeline import active_fl_pipeline

_REPO_ROOT = Path(__file__).resolve().parents[1]

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


# ---------------------------------------------------------------------------
# P2 task 2: seeded init head + start_round resume + worker_launcher removal
# ---------------------------------------------------------------------------
def test_start_round_parameter_is_removed(tmp_path):
    # Finding 1: start_round used to be a declared dsl parameter the function
    # body never read. A KFP UI/API caller overriding it got no effect and no
    # error -- a silent trap. It is removed entirely; start_round is
    # compile-time only (see test_start_round_env_var_shapes_the_compiled_dag
    # below for proof the env var it's replaced by actually works).
    text = _compile(tmp_path)
    spec = yaml.safe_load(text)
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert "start_round" not in params, sorted(params)


def test_seed_is_a_pipeline_parameter(tmp_path):
    text = _compile(tmp_path)
    spec = yaml.safe_load(text)
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert "seed" in params, sorted(params)


def test_start_round_env_var_shapes_the_compiled_dag(tmp_path):
    """The round count and the fl_round baked into each task must reflect
    the START_ROUND env var actually used at compile time.

    Exercised through the real `--start-round` CLI flag on
    active_fl_pipeline.py's own __main__, as a subprocess with no env vars
    pre-set -- exactly how run_pipeline.py invokes it. This is also a
    regression guard for a bug found while fixing Finding 1: `@dsl.pipeline`
    traces `active_fl_pipeline` immediately when it decorates it, at
    module-definition time, which is *before* a bottom-of-file
    `if __name__ == "__main__":` block runs. Setting FL_ROUNDS/START_ROUND
    only there (as this file used to be structured) is silently too late --
    every compile ignored --rounds/--start-round entirely and always
    unrolled from round 0, regardless of what was passed on the command
    line. The fix splits argument parsing into an earlier `if __name__`
    block, before active_fl_pipeline is defined.
    """
    with open(_REPO_ROOT / "config" / "k8s.yaml") as f:
        cfg = yaml.safe_load(f)
    fl_rounds = cfg["training"]["fl_rounds"]
    assert fl_rounds >= 2, "test needs at least 2 configured rounds to prove a resume point"
    start_round = fl_rounds - 1

    output = tmp_path / "resumed.yaml"
    subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "src" / "pipelines" / "active_fl_pipeline.py"),
            "--output",
            str(output),
            "--start-round",
            str(start_round),
        ],
        check=True,
        cwd=_REPO_ROOT,
    )
    spec = yaml.safe_load(output.read_text())
    tasks = spec["root"]["dag"]["tasks"]
    train_tasks = [k for k in tasks if k.startswith("train-workers")]
    assert len(train_tasks) == fl_rounds - start_round, (
        f"expected {fl_rounds - start_round} train-workers task(s) for "
        f"--start-round {start_round} with fl_rounds={fl_rounds}, found {train_tasks}"
    )
    baked_in_round = tasks["train-workers"]["inputs"]["parameters"]["fl_round"]["runtimeValue"][
        "constant"
    ]
    assert baked_in_round == float(start_round), (
        f"the first train-workers task must run fl_round={start_round}, not {baked_in_round}"
    )


def test_worker_launcher_parameter_is_gone(tmp_path):
    # The Temporal path is now the only path; the pytorchjob migration
    # fallback and its flag were always meant to be temporary.
    text = _compile(tmp_path)
    spec = yaml.safe_load(text)
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert "worker_launcher" not in params, "migration flag still present"


def test_init_global_model_precedes_the_first_train_workers(tmp_path):
    # Until init_global_model runs, round 0 has no global model, every worker
    # keeps its own independently-random ActorCritic, and the aggregator
    # averages N unrelated networks. The first train_workers task must not be
    # able to start before the seeded round_0/global.pt is written.
    text = _compile(tmp_path)
    spec = yaml.safe_load(text)
    tasks = spec["root"]["dag"]["tasks"]
    init_tasks = [k for k in tasks if k.startswith("init-global-model")]
    assert len(init_tasks) == 1, f"expected exactly one init task, found {init_tasks}"
    init_task = init_tasks[0]

    first_train = tasks["train-workers"]
    assert init_task in first_train.get("dependentTasks", []), (
        f"train-workers must depend on {init_task}, "
        f"got dependentTasks={first_train.get('dependentTasks')}"
    )


# ---------------------------------------------------------------------------
# P3 Task 5: topology/members threaded into the pipeline. RoundSpec.worker_spec
# (types.py) does the actual round-robin assignment; this only guards that the
# pipeline actually carries the two parameters that drive it, and that the
# three layers which each hold a default for them (config/k8s.yaml,
# run_pipeline.py's own fallback, and this pipeline's own dsl parameter
# default) agree -- the same "silently decorative parameter" trap Finding 1
# (start_round, above) already burned this file once.
# ---------------------------------------------------------------------------


def test_compiled_pipeline_carries_topology_and_members(tmp_path):
    text = _compile(tmp_path)
    spec = yaml.safe_load(text)
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert "topology" in params, sorted(params)
    assert "members" in params, sorted(params)


def test_topology_and_members_default_layers_agree(tmp_path):
    from src.pipelines.run_pipeline import DEFAULT_MEMBERS, DEFAULT_TOPOLOGY

    # Layer 1: config/k8s.yaml -- the value actually authored there today.
    with open(_REPO_ROOT / "config" / "k8s.yaml") as f:
        cfg = yaml.safe_load(f)
    orch = cfg.get("orchestration", {})
    assert orch.get("topology", DEFAULT_TOPOLOGY) == DEFAULT_TOPOLOGY
    assert orch.get("members", DEFAULT_MEMBERS) == DEFAULT_MEMBERS

    # Layer 3: active_fl_pipeline's own dsl parameter default, read from the
    # compiled IR (same technique as test_run_uid_default_is_a_valid_job_name_
    # for_fragment above -- @dsl.pipeline wraps the function into a
    # GraphComponent, so inspect.signature no longer exposes the declared
    # parameter defaults; the compiled pipeline_spec is the ground truth).
    # This is what applies when run_pipeline.py isn't the caller (e.g. `make
    # compile-pipeline`).
    text = _compile(tmp_path)
    spec = yaml.safe_load(text)
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert params["topology"]["defaultValue"] == DEFAULT_TOPOLOGY, (
        "active_fl_pipeline's topology default disagrees with run_pipeline.py's "
        "DEFAULT_TOPOLOGY -- exactly the layer-disagreement Finding 1 warned about"
    )
    assert params["members"]["defaultValue"] == DEFAULT_MEMBERS, (
        "active_fl_pipeline's members default disagrees with run_pipeline.py's "
        "DEFAULT_MEMBERS -- exactly the layer-disagreement Finding 1 warned about"
    )


def test_member_prefix_is_threaded_from_config_not_hardcoded(tmp_path):
    """member_prefix must agree across all three layers, like topology/members.

    The pipeline builds each worker's target cluster as
    f"{member_prefix}{worker_id % members + 1}", and that name has to match
    the kind clusters fed-infra actually creates from FED_MEMBER_PREFIX in
    infra.env.multi. If member_prefix is a hardcoded dsl default with no
    config source, changing FED_MEMBER_PREFIX alone silently desynchronises
    the two: the PropagationPolicy names a cluster that does not exist,
    Karmada matches nothing, the Job never lands anywhere, and the round
    hangs until the activity times out rather than failing.
    """
    from src.pipelines.run_pipeline import DEFAULT_MEMBER_PREFIX

    with open(_REPO_ROOT / "config" / "k8s.yaml") as f:
        cfg = yaml.safe_load(f)
    orch = cfg.get("orchestration", {})
    assert "member_prefix" in orch, (
        "config/k8s.yaml has no orchestration.member_prefix, so the value is "
        "unreachable from the config layer that infra.env.multi is tuned against"
    )
    assert orch["member_prefix"] == DEFAULT_MEMBER_PREFIX

    spec = yaml.safe_load(_compile(tmp_path))
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert params["member_prefix"]["defaultValue"] == DEFAULT_MEMBER_PREFIX


def test_member_prefix_matches_the_multi_infra_contract():
    """The pipeline default and FED_MEMBER_PREFIX in infra.env.multi must match.

    These are the two ends of the same coupling: fed-infra names the member
    kind clusters, the pipeline addresses them.
    """
    from src.pipelines.run_pipeline import DEFAULT_MEMBER_PREFIX

    env = {}
    with open(_REPO_ROOT / "infra.env.multi") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    assert env["FED_MEMBER_PREFIX"] == DEFAULT_MEMBER_PREFIX


def test_multi_config_agrees_with_the_multi_infra_contract():
    """config/k8s-multi.yaml and infra.env.multi describe the same clusters.

    infra.env.multi tells fed-infra how many member clusters to create and
    what to name them; config/k8s-multi.yaml tells the pipeline which ones to
    address. A mismatch is silent: Karmada accepts a PropagationPolicy naming
    a cluster that does not exist and simply matches nothing, so the workers
    never land anywhere and the round stalls until it times out.
    """
    with open(_REPO_ROOT / "config" / "k8s-multi.yaml") as f:
        cfg = yaml.safe_load(f)
    orch = cfg["orchestration"]
    assert orch["topology"] == "multi"

    env = {}
    with open(_REPO_ROOT / "infra.env.multi") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v

    assert orch["members"] == int(env["FED_MEMBER_COUNT"]), (
        "config/k8s-multi.yaml's members disagrees with infra.env.multi's "
        "FED_MEMBER_COUNT, so the pipeline would address a different number "
        "of member clusters than fed-infra creates"
    )
    assert orch["member_prefix"] == env["FED_MEMBER_PREFIX"]
