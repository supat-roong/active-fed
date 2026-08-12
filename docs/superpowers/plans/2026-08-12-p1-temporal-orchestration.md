# Temporal Worker-Fleet Orchestration Implementation Plan (Phase P1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `active-fed`'s opaque `train_workers` KFP component — which submits one PyTorchJob and busy-polls kubectl with a hard 20-minute ceiling — with a Temporal-managed fleet where every worker is a durable entity with its own retry, failure reason, and live progress.

**Architecture:** KFP keeps the round DAG and artifact lineage; it still executes. Inside a round, the `train_workers` component becomes a thin Temporal client: it starts a `TrainRoundWorkflow`, which fans out one `WorkerWorkflow` child per worker, each launching and watching a Kubernetes `Job`. The component blocks on the result while streaming per-worker status into its own node logs. Scopes stay disjoint — KFP sequences rounds, Temporal manages a fleet within a round — so there is no two-schedulers conflict.

**Tech Stack:** Python 3.10, `temporalio` SDK, Temporal server + PostgreSQL deployed in-cluster by `fed-infra` via Helm, Kubernetes Python client, existing pytest suite.

## Global Constraints

- **Prerequisite:** Phase P0 is complete. `fed-infra` is at `329e578f526c6d15056c2c7f03e931939a6a61d1`; both consumers vendor it at `vendor/fed-infra`.
- **`fed-infra` MUST NOT contain the strings `active-fed` or `fed-twin`.** Enforced by `tests/agnostic.bats`. Temporal is a `fed-infra` *component*; everything active-fed-specific (the Temporal worker Deployment, the workflows) lives in `active-fed`.
- **`lib/*.sh` are sourced and MUST NOT set shell options**; `bin/*` and `tests/stubs/*` MUST. Enforced by a regression test.
- Dry-run guards use `if [ "${FED_DRY_RUN:-0}" = "1" ]; then …; return 0; fi` — never `[ … ] && return 0`. Enforced by `tests/agnostic.bats`.
- `fed_` function prefix, `FED_` variable prefix, `fed-` resource names in `fed-infra`. Bash 3.2 target. `shellcheck -x` clean.
- **`fl_rounds` / `fl_round` are active-fed's own domain terms** — never rename them.
- Python: ruff line-length 100, `select = ["E","F","I","UP"]`, mypy `python_version = 3.10`. Match the existing style in `src/`.
- **Do not change** `src/experiment/local_runner.py`, `config/local.yaml`, or any RL/ML algorithm code. The local runner must stay dependency-free — it is the fast iteration path.
- Commit messages: Conventional Commits, no trailers.
- Work on `main` in all repos; **no `git push`, no `gh` commands** unless the human partner says otherwise.
- Environment: macOS/Apple Silicon, colima 4 CPU / ~9.7 GiB. Temporal + PostgreSQL add roughly 1.5 GiB on top of KFP.

## Prior-phase findings that bind this work

From `docs/superpowers/reviews/2026-08-12-P0-deferred-findings.md`:

- `FED_KFP_VERSION` is defaulted and exported but absent from `FED_TEMPLATE_VARS`. If you add a Temporal template referencing a `FED_*` variable, **it must be added to that whitelist or it renders as an empty string silently.**
- Test-harness facts, verified by hand: `STUB_*_FAIL_GLOB` **can** contain spaces (`case` patterns are not word-split) but **cannot** use `|` alternation — after expansion `|` is a literal, so `a*|b*` matches nothing and a test relying on it passes for the wrong reason. To fail two different subcommands, use `*`. `STUB_KUBECTL_FAIL_ONCE_GLOB` / `_COUNT` exist for fail-once-then-succeed.
- `tests/stubs/kind` captures stdin to `$STUB_STDIN_LOG` only for `--config -`.

---

## File Structure

**`fed-infra` — new**

| Path | Responsibility |
|---|---|
| `lib/temporal.sh` | `fed_temporal_install` — Helm repo add, install Temporal + PostgreSQL, wait for frontend |
| `tests/temporal.bats` | Unit tests with the `helm` stub |
| `tests/stubs/helm` | New stub — records argv, honours `STUB_HELM_FAIL_GLOB` |

**`active-fed` — new**

| Path | Responsibility |
|---|---|
| `src/orchestration/__init__.py` | Package marker |
| `src/orchestration/types.py` | `RoundSpec`, `WorkerSpec`, `WorkerStatus`, `WorkerResult`, `RoundReport` — plain dataclasses, no I/O |
| `src/orchestration/activities.py` | `launch_and_watch_pod`, `cleanup_worker_job` — the only code touching Kubernetes |
| `src/orchestration/workflows.py` | `WorkerWorkflow`, `TrainRoundWorkflow` — deterministic, no I/O |
| `src/orchestration/worker_main.py` | Entrypoint registering workflows + activities against the task queue |
| `k8s/temporal-worker.yaml` | Deployment + ServiceAccount for the Temporal worker process |
| `tests/test_orchestration_types.py` | Dataclass/quorum logic |
| `tests/test_orchestration_workflows.py` | Workflow tests via `WorkflowEnvironment` time-skipping, mocked activities |
| `tests/test_orchestration_activities.py` | Activity tests against a faked Kubernetes client |

**`active-fed` — modified**

| Path | Change |
|---|---|
| `src/pipelines/active_fl_pipeline.py` | `train_workers` becomes a Temporal client; `worker_launcher` switch retains the PyTorchJob path during migration |
| `k8s/rbac.yaml` | Extend ClusterRole: `jobs` create/delete, `pods/log` get |
| `config/k8s.yaml` | Add `temporal.*`, `min_workers`, `worker_launcher` |
| `infra.env` | `FED_COMPONENTS` gains `temporal`; add `FED_TEMPORAL_*` |
| `docker/Dockerfile.aggregator` | Add `temporalio` |
| `pyproject.toml` | Add `temporalio`, `kubernetes` |
| `Makefile` | `temporal-ui`, `run-temporal-worker` targets |

**Boundary rationale:** `types.py` has no imports beyond stdlib so both workflows and activities can use it without Temporal's sandbox complaining. `activities.py` is the only module importing the Kubernetes client — workflow code must stay deterministic, and Temporal replays it. Keeping I/O out of `workflows.py` is what makes the workflow tests runnable in milliseconds with no cluster.

---

## Task 1: `fed-infra` — Temporal component

**Files:**
- Create: `fed-infra/lib/temporal.sh`, `fed-infra/tests/temporal.bats`, `fed-infra/tests/stubs/helm`
- Modify: `fed-infra/lib/config.sh`, `fed-infra/lib/components.sh`, `fed-infra/bin/fed-infra-up`, `fed-infra/bin/fed-infra-down`

**Interfaces:**
- Consumes: `fed_log`, `fed_die`, `fed_require_cmd` (`lib/common.sh`); `fed_has_component` (`lib/config.sh`).
- Produces: `fed_temporal_install(namespace, version)`. New config: `FED_TEMPORAL_VERSION` (default `0.62.0`), `FED_TEMPORAL_NAMESPACE` (default `${FED_NAMESPACE}`), `FED_NODEPORT_TEMPORAL_UI` (default `30733`), `FED_HOSTPORT_TEMPORAL_UI` (default `8233`).

- [ ] **Step 1: Write `tests/stubs/helm`**

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "helm $*" >> "$STUB_LOG"
# Unquoted by design: STUB_HELM_FAIL_GLOB is a shell glob pattern.
# shellcheck disable=SC2254
case "$*" in
  ${STUB_HELM_FAIL_GLOB:-__never_matches__}) exit 1 ;;
esac
[ -n "${STUB_HELM_OUT:-}" ] && printf '%s' "$STUB_HELM_OUT"
exit 0
```

Then `chmod +x tests/stubs/helm`.

- [ ] **Step 2: Write the failing test**

Create `tests/temporal.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  source "$FED_INFRA_ROOT/lib/common.sh"
  source "$FED_INFRA_ROOT/lib/config.sh"
  source "$FED_INFRA_ROOT/lib/temporal.sh"
  fed_config_defaults
  export FED_NAMESPACE=demo-ns
}

@test "fed_temporal_install adds the repo and installs the chart" {
  fed_temporal_install demo-ns 0.62.0
  assert_called "helm repo add temporal"
  assert_called "helm upgrade --install temporal"
  assert_called "--namespace demo-ns"
}

@test "fed_temporal_install pins the requested chart version" {
  fed_temporal_install demo-ns 0.62.0
  assert_called "--version 0.62.0"
}

@test "fed_temporal_install disables the bundled elasticsearch and extra services" {
  fed_temporal_install demo-ns 0.62.0
  assert_called "elasticsearch.enabled=false"
  assert_called "prometheus.enabled=false"
  assert_called "grafana.enabled=false"
}

@test "fed_temporal_install waits for the frontend to roll out" {
  fed_temporal_install demo-ns 0.62.0
  assert_called "rollout status deployment/temporal-frontend -n demo-ns"
}

@test "fed_temporal_install does nothing at all under FED_DRY_RUN=1" {
  export FED_DRY_RUN=1
  fed_temporal_install demo-ns 0.62.0
  refute_called "helm"
  refute_called "kubectl"
}

@test "fed_temporal_install fails fast when the chart install fails" {
  export STUB_HELM_FAIL_GLOB="upgrade --install*"
  run fed_temporal_install demo-ns 0.62.0
  [ "$status" -ne 0 ]
  refute_called "rollout status"
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `bats tests/temporal.bats`
Expected: FAIL — `lib/temporal.sh: No such file or directory`

- [ ] **Step 4: Implement `lib/temporal.sh`**

```bash
#!/usr/bin/env bash
# temporal.sh — Temporal server + PostgreSQL via the official Helm chart.
#
# The chart bundles Elasticsearch, Prometheus and Grafana by default. All three
# are disabled here: this is a local single-node kind cluster, and Elasticsearch
# alone would roughly double the memory footprint. Visibility falls back to the
# PostgreSQL store, which is sufficient for workflow listing and history.

fed_temporal_install() {
  local ns=$1 ver=$2
  if [ "${FED_DRY_RUN:-0}" = "1" ]; then
    fed_log "dry-run: would install Temporal ${ver} into ${ns}"
    return 0
  fi
  fed_require_cmd helm

  fed_log "adding the Temporal Helm repo"
  helm repo add temporal https://go.temporal.io/helm-charts || return 1
  helm repo update >/dev/null 2>&1 || return 1

  fed_log "installing Temporal ${ver} into ${ns}"
  helm upgrade --install temporal temporal/temporal \
    --namespace "$ns" --create-namespace \
    --version "$ver" \
    --set server.replicaCount=1 \
    --set cassandra.enabled=false \
    --set postgresql.enabled=true \
    --set server.config.persistence.default.driver=sql \
    --set elasticsearch.enabled=false \
    --set prometheus.enabled=false \
    --set grafana.enabled=false \
    --wait --timeout 15m || return 1

  fed_log "waiting for the Temporal frontend"
  kubectl rollout status deployment/temporal-frontend -n "$ns" --timeout=600s || return 1
}
```

- [ ] **Step 5: Add config defaults**

In `lib/config.sh`, inside `fed_config_defaults`, after the KFP defaults:

```bash
  : "${FED_TEMPORAL_VERSION:=0.62.0}"
  : "${FED_TEMPORAL_NAMESPACE:=${FED_NAMESPACE}}"
  : "${FED_NODEPORT_TEMPORAL_UI:=30733}"
  : "${FED_HOSTPORT_TEMPORAL_UI:=8233}"
```

Add all four to the `export` list in the same function.

- [ ] **Step 6: Wire into dispatch**

In `lib/components.sh`, in `fed_up`, **after** the `training` block and **before** `minio`:

```bash
  if fed_has_component temporal; then
    fed_temporal_install "$FED_TEMPORAL_NAMESPACE" "$FED_TEMPORAL_VERSION"
  fi
```

And in the NodePort section, alongside the others:

```bash
  if fed_has_component temporal; then
    fed_expose_nodeport temporal-web "$FED_TEMPORAL_NAMESPACE" \
      "[{\"port\":8080,\"targetPort\":8080,\"nodePort\":${FED_NODEPORT_TEMPORAL_UI}}]"
  fi
```

Add `temporal` to the module list in **both** `bin/fed-infra-up` and `bin/fed-infra-down`. Add a Temporal line to `fed_up_summary` using the `if fed_has_component …; then … fi` form.

- [ ] **Step 7: Add the UI port to the kind template**

In `kind/single-cluster.yaml.tpl`, append a fifth mapping:

```yaml
      - containerPort: ${FED_NODEPORT_TEMPORAL_UI}
        hostPort: ${FED_HOSTPORT_TEMPORAL_UI}
        protocol: TCP
```

Add `${FED_NODEPORT_TEMPORAL_UI}` and `${FED_HOSTPORT_TEMPORAL_UI}` to `FED_TEMPLATE_VARS` in `lib/render.sh`. **Without this they render as empty strings and kind fails to parse the config.**

- [ ] **Step 8: Regenerate golden files and confirm the expected diff**

The kind template changed, so goldens will move. This is expected — but check the diff is *only* the new port mapping:

```bash
STUB_LOG=/dev/null bin/fed-infra-up --env tests/fixtures/consumer-a.env \
  --dry-run --render-dir tests/golden/consumer-a
STUB_LOG=/dev/null bin/fed-infra-up --env tests/fixtures/consumer-b.env \
  --dry-run --render-dir tests/golden/consumer-b
git diff tests/golden/
```

Expected: no change at all (the kind config is not written to the render dir — only manifests are). If any golden manifest changed, stop and investigate.

- [ ] **Step 9: Run tests and commit**

```bash
make check
git add lib/temporal.sh lib/config.sh lib/components.sh lib/render.sh bin/ kind/ tests/
git commit -m "feat: Temporal server component with postgres persistence"
```

---

## Task 2: `active-fed` — orchestration types

**Files:**
- Create: `active-fed/src/orchestration/__init__.py`, `types.py`, `tests/test_orchestration_types.py`

**Interfaces:**
- Produces (used by every later task):

```python
@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    fl_round: int
    worker_id: int
    num_workers: int
    local_episodes: int
    namespace: str
    worker_image: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    kfp_run_id: str

@dataclasses.dataclass(frozen=True)
class RoundSpec:
    fl_round: int
    num_workers: int
    min_workers: int
    local_episodes: int
    namespace: str
    worker_image: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    kfp_run_id: str
    def worker_spec(self, worker_id: int) -> WorkerSpec: ...

@dataclasses.dataclass
class WorkerStatus:
    worker_id: int
    phase: str          # "Pending" | "Running" | "Succeeded" | "Failed" | "Unknown"
    attempt: int
    episodes_done: int
    message: str

@dataclasses.dataclass
class WorkerResult:
    worker_id: int
    succeeded: bool
    attempts: int
    failure_reason: str
    job_name: str

@dataclasses.dataclass
class RoundReport:
    fl_round: int
    results: list[WorkerResult]
    @property
    def succeeded_ids(self) -> list[int]: ...
    @property
    def failed_ids(self) -> list[int]: ...
    def meets_quorum(self, min_workers: int) -> bool: ...
```

**Why frozen specs:** Temporal serialises workflow arguments and replays them. Immutable inputs make replay deterministic and prevent a whole class of "worked once, failed on replay" bugs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_orchestration_types.py`:

```python
import pytest

from src.orchestration.types import RoundReport, RoundSpec, WorkerResult


def _round_spec(**overrides):
    base = dict(
        fl_round=0, num_workers=3, min_workers=2, local_episodes=10,
        namespace="ns", worker_image="img:v1",
        minio_endpoint="minio:9000", minio_access_key="a", minio_secret_key="b",
        minio_bucket="bucket", mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="exp", kfp_run_id="run-1",
    )
    base.update(overrides)
    return RoundSpec(**base)


def _result(worker_id: int, succeeded: bool) -> WorkerResult:
    return WorkerResult(
        worker_id=worker_id, succeeded=succeeded, attempts=1,
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
    with pytest.raises(dataclasses_FrozenInstanceError()):
        spec.fl_round = 5  # type: ignore[misc]


def dataclasses_FrozenInstanceError():
    import dataclasses
    return dataclasses.FrozenInstanceError


def test_report_partitions_succeeded_and_failed():
    report = RoundReport(fl_round=0, results=[_result(0, True), _result(1, False), _result(2, True)])
    assert report.succeeded_ids == [0, 2]
    assert report.failed_ids == [1]


def test_quorum_met_when_enough_workers_succeed():
    report = RoundReport(fl_round=0, results=[_result(0, True), _result(1, True), _result(2, False)])
    assert report.meets_quorum(2) is True


def test_quorum_not_met_below_threshold():
    report = RoundReport(fl_round=0, results=[_result(0, True), _result(1, False), _result(2, False)])
    assert report.meets_quorum(2) is False


def test_quorum_with_zero_successes_is_never_met():
    report = RoundReport(fl_round=0, results=[_result(0, False)])
    assert report.meets_quorum(1) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_orchestration_types.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestration'`

- [ ] **Step 3: Implement `src/orchestration/types.py`**

```python
"""
Plain data types shared by orchestration workflows and activities.

Deliberately free of any import beyond the standard library: Temporal's
workflow sandbox restricts what workflow code may import, and both workflow
and activity code depend on these types.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    """Everything one worker needs. Frozen so Temporal replay is deterministic."""

    fl_round: int
    worker_id: int
    num_workers: int
    local_episodes: int
    namespace: str
    worker_image: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    kfp_run_id: str


@dataclasses.dataclass(frozen=True)
class RoundSpec:
    """Everything one federated round needs."""

    fl_round: int
    num_workers: int
    min_workers: int
    local_episodes: int
    namespace: str
    worker_image: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    kfp_run_id: str

    def worker_spec(self, worker_id: int) -> WorkerSpec:
        return WorkerSpec(
            fl_round=self.fl_round,
            worker_id=worker_id,
            num_workers=self.num_workers,
            local_episodes=self.local_episodes,
            namespace=self.namespace,
            worker_image=self.worker_image,
            minio_endpoint=self.minio_endpoint,
            minio_access_key=self.minio_access_key,
            minio_secret_key=self.minio_secret_key,
            minio_bucket=self.minio_bucket,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            mlflow_experiment_name=self.mlflow_experiment_name,
            kfp_run_id=self.kfp_run_id,
        )


@dataclasses.dataclass
class WorkerStatus:
    """Live state of one worker, returned by the workflow query."""

    worker_id: int
    phase: str = "Pending"
    attempt: int = 0
    episodes_done: int = 0
    message: str = ""


@dataclasses.dataclass
class WorkerResult:
    """Terminal outcome of one worker."""

    worker_id: int
    succeeded: bool
    attempts: int
    failure_reason: str
    job_name: str


@dataclasses.dataclass
class RoundReport:
    """Aggregate outcome of one round's fleet."""

    fl_round: int
    results: list[WorkerResult]

    @property
    def succeeded_ids(self) -> list[int]:
        return sorted(r.worker_id for r in self.results if r.succeeded)

    @property
    def failed_ids(self) -> list[int]:
        return sorted(r.worker_id for r in self.results if not r.succeeded)

    def meets_quorum(self, min_workers: int) -> bool:
        return len(self.succeeded_ids) >= min_workers
```

Also create an empty `src/orchestration/__init__.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_orchestration_types.py -q`
Expected: PASS — 6 tests

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/orchestration tests/test_orchestration_types.py
uv run ruff format --check src/orchestration
git add src/orchestration tests/test_orchestration_types.py
git commit -m "feat(orchestration): shared dataclasses for worker fleet orchestration"
```

---

## Task 3: `active-fed` — Kubernetes activities

**Files:**
- Create: `active-fed/src/orchestration/activities.py`, `tests/test_orchestration_activities.py`
- Modify: `active-fed/pyproject.toml` (add `temporalio`, `kubernetes`)

**Interfaces:**
- Consumes: `WorkerSpec`, `WorkerStatus` (Task 2).
- Produces:
  - `build_job_manifest(spec: WorkerSpec) -> dict` — pure function, no I/O, so it is directly testable.
  - `async launch_and_watch_pod(spec: WorkerSpec) -> WorkerResult` — Temporal activity.
  - `async cleanup_worker_job(spec: WorkerSpec) -> None` — Temporal activity.
  - Job name is **deterministic**: `f"aflw-{spec.kfp_run_id[:8]}-r{spec.fl_round}-w{spec.worker_id}"`.

**Why deterministic job names:** the P0 review found that `uuid.uuid4()` in the old job name meant a retried component launched a *second* PyTorchJob, racing 2N workers on the same MinIO keys. A deterministic name makes retry idempotent — the activity re-attaches to the existing Job instead of creating another.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml` `dependencies`, add `"temporalio>=1.7.0"` and `"kubernetes>=29.0.0"`. Then `uv sync --extra dev`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_orchestration_activities.py`:

```python
import pytest

from src.orchestration.activities import build_job_manifest, job_name_for
from src.orchestration.types import WorkerSpec


def _spec(**overrides) -> WorkerSpec:
    base = dict(
        fl_round=3, worker_id=2, num_workers=4, local_episodes=25,
        namespace="active-fed", worker_image="active-fed-worker:v1",
        minio_endpoint="minio-service:9000", minio_access_key="ak",
        minio_secret_key="sk", minio_bucket="bucket",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234567890",
    )
    base.update(overrides)
    return WorkerSpec(**base)


def test_job_name_is_deterministic_for_the_same_inputs():
    assert job_name_for(_spec()) == job_name_for(_spec())


def test_job_name_distinguishes_worker_and_round():
    assert job_name_for(_spec(worker_id=1)) != job_name_for(_spec(worker_id=2))
    assert job_name_for(_spec(fl_round=1)) != job_name_for(_spec(fl_round=2))


def test_job_name_is_a_valid_kubernetes_name():
    import re
    name = job_name_for(_spec())
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name), name
    assert len(name) <= 63


def test_manifest_sets_rank_to_the_worker_id():
    env = {e["name"]: e["value"] for e in
           build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["RANK"] == "2"


def test_manifest_passes_round_and_episodes_as_args():
    args = build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--fl-round" in args and args[args.index("--fl-round") + 1] == "3"
    assert "--local-episodes" in args and args[args.index("--local-episodes") + 1] == "25"


def test_manifest_carries_minio_and_mlflow_env():
    env = {e["name"]: e["value"] for e in
           build_job_manifest(_spec())["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["MINIO_ENDPOINT"] == "minio-service:9000"
    assert env["MINIO_BUCKET"] == "bucket"
    assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


def test_manifest_uses_onfailure_restart_policy():
    # P0 review: restartPolicy Never meant a crashed worker vanished and the
    # aggregator silently proceeded with N-1 clients.
    assert build_job_manifest(_spec())["spec"]["template"]["spec"]["restartPolicy"] == "OnFailure"


def test_manifest_labels_identify_round_and_worker():
    labels = build_job_manifest(_spec())["spec"]["template"]["metadata"]["labels"]
    assert labels["app"] == "active-fl-worker"
    assert labels["fl-round"] == "3"
    assert labels["worker-id"] == "2"
```

- [ ] **Step 3: Run to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_orchestration_activities.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestration.activities'`

- [ ] **Step 4: Implement `src/orchestration/activities.py`**

```python
"""
Temporal activities: the only orchestration code that touches Kubernetes.

Workflow code must be deterministic because Temporal replays it, so every
side effect lives here. `build_job_manifest` and `job_name_for` are pure and
separately testable; the activities themselves are thin wrappers around the
Kubernetes API plus heartbeating.
"""

from __future__ import annotations

import asyncio
import logging

from temporalio import activity

from src.orchestration.types import WorkerResult, WorkerSpec

log = logging.getLogger(__name__)

# How long a worker pod may run before we give up on it.
POD_WATCH_TIMEOUT_S = 3600
# How often to poll pod state and emit a heartbeat.
POLL_INTERVAL_S = 5


def job_name_for(spec: WorkerSpec) -> str:
    """Deterministic Job name.

    Deterministic on purpose: a retried activity must re-attach to the existing
    Job rather than launch a second fleet. The previous implementation embedded
    uuid4() and raced 2N workers onto the same MinIO keys on retry.
    """
    return f"aflw-{spec.kfp_run_id[:8]}-r{spec.fl_round}-w{spec.worker_id}"


def build_job_manifest(spec: WorkerSpec) -> dict:
    """Render the batch/v1 Job for one worker. Pure — no I/O."""
    name = job_name_for(spec)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": spec.namespace,
            "labels": {
                "app": "active-fl-worker",
                "fl-round": str(spec.fl_round),
                "worker-id": str(spec.worker_id),
            },
        },
        "spec": {
            "backoffLimit": 2,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "active-fl-worker",
                        "fl-round": str(spec.fl_round),
                        "worker-id": str(spec.worker_id),
                    }
                },
                "spec": {
                    "restartPolicy": "OnFailure",
                    "containers": [
                        {
                            "name": "worker",
                            "image": spec.worker_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["uv", "run", "python", "-m", "src.agent.train_worker"],
                            "args": [
                                "--fl-round", str(spec.fl_round),
                                "--local-episodes", str(spec.local_episodes),
                                "--device", "cpu",
                            ],
                            "env": [
                                # train_worker.py reads RANK first, falling back to
                                # --worker-id, so no change is needed there.
                                {"name": "RANK", "value": str(spec.worker_id)},
                                {"name": "MINIO_ENDPOINT", "value": spec.minio_endpoint},
                                {"name": "MINIO_ACCESS_KEY", "value": spec.minio_access_key},
                                {"name": "MINIO_SECRET_KEY", "value": spec.minio_secret_key},
                                {"name": "MINIO_BUCKET", "value": spec.minio_bucket},
                                {"name": "MLFLOW_TRACKING_URI", "value": spec.mlflow_tracking_uri},
                                {
                                    "name": "MLFLOW_EXPERIMENT_NAME",
                                    "value": spec.mlflow_experiment_name,
                                },
                            ],
                        }
                    ],
                },
            },
        },
    }
```

Then append the two activities:

```python
def _k8s_batch_and_core():
    """Import and configure the Kubernetes client lazily.

    Kept out of module scope so the pure functions above remain importable
    (and unit-testable) without a kubeconfig present.
    """
    from kubernetes import client, config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.BatchV1Api(), client.CoreV1Api()


@activity.defn
async def launch_and_watch_pod(spec: WorkerSpec) -> WorkerResult:
    """Create the worker Job if absent, then watch it to completion.

    Heartbeats every poll so Temporal can distinguish a slow worker from a
    wedged one — this is what replaces the old fixed 20-minute ceiling.
    """
    from kubernetes.client.exceptions import ApiException

    batch, core = _k8s_batch_and_core()
    name = job_name_for(spec)
    manifest = build_job_manifest(spec)

    try:
        batch.create_namespaced_job(namespace=spec.namespace, body=manifest)
        activity.logger.info(f"created Job {name}")
    except ApiException as e:
        if e.status != 409:  # 409 = already exists; retry re-attaches
            raise
        activity.logger.info(f"Job {name} already exists, re-attaching")

    waited = 0
    while waited < POD_WATCH_TIMEOUT_S:
        job = batch.read_namespaced_job_status(name=name, namespace=spec.namespace)
        status = job.status
        if status.succeeded:
            await _log_tail(core, spec, name)
            return WorkerResult(
                worker_id=spec.worker_id, succeeded=True,
                attempts=int(status.failed or 0) + 1, failure_reason="", job_name=name,
            )
        if status.failed and status.failed > manifest["spec"]["backoffLimit"]:
            reason = await _failure_reason(core, spec, name)
            return WorkerResult(
                worker_id=spec.worker_id, succeeded=False,
                attempts=int(status.failed), failure_reason=reason, job_name=name,
            )

        activity.heartbeat(
            {"worker_id": spec.worker_id, "active": int(status.active or 0), "waited_s": waited}
        )
        await asyncio.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S

    return WorkerResult(
        worker_id=spec.worker_id, succeeded=False, attempts=0,
        failure_reason=f"timed out after {POD_WATCH_TIMEOUT_S}s", job_name=name,
    )


async def _failure_reason(core, spec: WorkerSpec, job_name: str) -> str:
    """Best-effort human-readable cause from the most recent pod."""
    try:
        pods = core.list_namespaced_pod(
            namespace=spec.namespace, label_selector=f"job-name={job_name}"
        )
        for pod in pods.items:
            for cs in pod.status.container_statuses or []:
                term = cs.state.terminated
                if term is not None and term.reason:
                    return f"{term.reason} (exit {term.exit_code})"
        return "unknown"
    except Exception as e:  # diagnostics must never mask the real failure
        return f"unavailable: {e}"


async def _log_tail(core, spec: WorkerSpec, job_name: str, lines: int = 20) -> None:
    try:
        pods = core.list_namespaced_pod(
            namespace=spec.namespace, label_selector=f"job-name={job_name}"
        )
        for pod in pods.items:
            text = core.read_namespaced_pod_log(
                name=pod.metadata.name, namespace=spec.namespace, tail_lines=lines
            )
            activity.logger.info(f"[{pod.metadata.name}] {text}")
    except Exception as e:
        activity.logger.warning(f"could not read logs for {job_name}: {e}")


@activity.defn
async def cleanup_worker_job(spec: WorkerSpec) -> None:
    """Delete the Job and its pods. Safe to call when already gone."""
    from kubernetes.client.exceptions import ApiException

    batch, _ = _k8s_batch_and_core()
    try:
        batch.delete_namespaced_job(
            name=job_name_for(spec), namespace=spec.namespace, propagation_policy="Background"
        )
    except ApiException as e:
        if e.status != 404:
            raise
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_orchestration_activities.py -q`
Expected: PASS — 8 tests

- [ ] **Step 6: Commit**

```bash
uv run ruff check src/orchestration tests/
git add src/orchestration/activities.py tests/test_orchestration_activities.py pyproject.toml uv.lock
git commit -m "feat(orchestration): kubernetes job activities with deterministic naming"
```

---

## Task 4: `active-fed` — workflows

**Files:**
- Create: `active-fed/src/orchestration/workflows.py`, `tests/test_orchestration_workflows.py`

**Interfaces:**
- Consumes: types (Task 2), `launch_and_watch_pod` / `cleanup_worker_job` (Task 3).
- Produces: `WorkerWorkflow` (run → `WorkerResult`, query `status` → `WorkerStatus`); `TrainRoundWorkflow` (run → `RoundReport`, query `status` → `dict[int, WorkerStatus]`). Task queue name constant `TASK_QUEUE = "active-fed"`.

**Testing note:** `temporalio.testing.WorkflowEnvironment.start_time_skipping()` runs the whole retry/timeout logic in milliseconds with no cluster and no real sleeping. Activities are mocked, so these tests cover orchestration decisions only — which is exactly the boundary we want.

- [ ] **Step 1: Write the failing test**

Create `tests/test_orchestration_workflows.py`:

```python
import uuid

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from src.orchestration.types import RoundSpec, WorkerResult, WorkerSpec
from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow, WorkerWorkflow


def _round_spec(num_workers=3, min_workers=2) -> RoundSpec:
    return RoundSpec(
        fl_round=0, num_workers=num_workers, min_workers=min_workers, local_episodes=5,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234",
    )


def _ok(spec: WorkerSpec) -> WorkerResult:
    return WorkerResult(worker_id=spec.worker_id, succeeded=True, attempts=1,
                        failure_reason="", job_name=f"j{spec.worker_id}")


async def _run(env: WorkflowEnvironment, acts, spec: RoundSpec):
    async with Worker(
        env.client, task_queue=TASK_QUEUE,
        workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=acts,
    ):
        return await env.client.execute_workflow(
            TrainRoundWorkflow.run, spec,
            id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
        )


@pytest.mark.asyncio
async def test_all_workers_succeed():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec())
    assert report.succeeded_ids == [0, 1, 2]
    assert report.failed_ids == []


@pytest.mark.asyncio
async def test_partial_failure_above_quorum_still_returns_survivors():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 1:
            raise RuntimeError("pod OOMKilled")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec(min_workers=2))
    assert report.succeeded_ids == [0, 2]
    assert report.failed_ids == [1]
    assert report.meets_quorum(2) is True


@pytest.mark.asyncio
async def test_below_quorum_fails_the_round():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 0:
            return _ok(spec)
        raise RuntimeError("boom")

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    from temporalio.client import WorkflowFailureError

    async with await WorkflowEnvironment.start_time_skipping() as env:
        with pytest.raises(WorkflowFailureError):
            await _run(env, [launch, cleanup], _round_spec(min_workers=2))


@pytest.mark.asyncio
async def test_worker_retries_then_succeeds():
    calls: dict[int, int] = {}

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        calls[spec.worker_id] = calls.get(spec.worker_id, 0) + 1
        if spec.worker_id == 1 and calls[spec.worker_id] == 1:
            raise RuntimeError("transient")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec(min_workers=3))
    assert report.succeeded_ids == [0, 1, 2]
    assert calls[1] == 2  # proves it actually retried rather than passing first time


@pytest.mark.asyncio
async def test_cleanup_runs_for_every_worker():
    cleaned: list[int] = []

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        cleaned.append(spec.worker_id)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        await _run(env, [launch, cleanup], _round_spec())
    assert sorted(cleaned) == [0, 1, 2]
```

Add `pytest-asyncio` to the dev extras and `asyncio_mode = "auto"` under `[tool.pytest.ini_options]` in `pyproject.toml`.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_orchestration_workflows.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestration.workflows'`

- [ ] **Step 3: Implement `src/orchestration/workflows.py`**

```python
"""
Temporal workflows owning the worker fleet within one federated round.

Scope boundary: KFP sequences rounds and owns the DAG and artifact lineage;
these workflows own the fleet inside a round. The two never overlap, so there
is no two-schedulers conflict.

Workflow code is replayed by Temporal and must stay deterministic — no I/O, no
clocks, no randomness. Everything with a side effect is an activity.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.orchestration.types import (
        RoundReport,
        RoundSpec,
        WorkerResult,
        WorkerSpec,
        WorkerStatus,
    )

TASK_QUEUE = "active-fed"


@workflow.defn
class WorkerWorkflow:
    """One durable entity per (round, worker).

    Gives per-pod retry, a per-pod failure reason in Temporal history, and a
    queryable live status the dashboard and the KFP component both read.
    """

    def __init__(self) -> None:
        self._status = WorkerStatus(worker_id=-1)

    @workflow.run
    async def run(self, spec: WorkerSpec) -> WorkerResult:
        self._status = WorkerStatus(worker_id=spec.worker_id, phase="Pending")
        try:
            result: WorkerResult = await workflow.execute_activity(
                "launch_and_watch_pod",
                spec,
                start_to_close_timeout=timedelta(seconds=3900),
                heartbeat_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=10),
                ),
            )
            self._status.phase = "Succeeded" if result.succeeded else "Failed"
            self._status.message = result.failure_reason
            return result
        finally:
            # Runs on success, failure and cancellation. Without it a failed
            # round leaves orphaned Jobs that collide with the next attempt's
            # deterministic names.
            await workflow.execute_activity(
                "cleanup_worker_job",
                spec,
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

    @workflow.query
    def status(self) -> WorkerStatus:
        return self._status


@workflow.defn
class TrainRoundWorkflow:
    """Fans out one WorkerWorkflow child per worker and gathers the outcomes."""

    def __init__(self) -> None:
        self._statuses: dict[int, WorkerStatus] = {}

    @workflow.run
    async def run(self, spec: RoundSpec) -> RoundReport:
        parent_id = workflow.info().workflow_id

        async def _one(worker_id: int) -> WorkerResult:
            return await workflow.execute_child_workflow(
                WorkerWorkflow.run,
                spec.worker_spec(worker_id),
                id=f"{parent_id}-w{worker_id}",
                task_queue=TASK_QUEUE,
            )

        # return_exceptions=True so one dead worker does not abort the fleet;
        # the quorum check below decides whether the round can still proceed.
        raw = await asyncio.gather(
            *[_one(i) for i in range(spec.num_workers)], return_exceptions=True
        )

        results: list[WorkerResult] = []
        for worker_id, item in enumerate(raw):
            if isinstance(item, BaseException):
                results.append(
                    WorkerResult(
                        worker_id=worker_id, succeeded=False, attempts=0,
                        failure_reason=str(item), job_name="",
                    )
                )
            else:
                results.append(item)

        report = RoundReport(fl_round=spec.fl_round, results=results)
        if not report.meets_quorum(spec.min_workers):
            raise workflow.ApplicationError(
                f"round {spec.fl_round}: only {len(report.succeeded_ids)} of "
                f"{spec.num_workers} workers succeeded, need {spec.min_workers}",
                non_retryable=True,
            )
        return report

    @workflow.query
    def status(self) -> dict[int, WorkerStatus]:
        return self._statuses
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_orchestration_workflows.py -q`
Expected: PASS — 5 tests, in seconds (time-skipping means no real waiting)

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/orchestration tests/
git add src/orchestration/workflows.py tests/test_orchestration_workflows.py pyproject.toml uv.lock
git commit -m "feat(orchestration): per-worker durable workflows with quorum handling"
```

---

## Task 5: `active-fed` — Temporal worker process and RBAC

**Files:**
- Create: `active-fed/src/orchestration/worker_main.py`, `active-fed/k8s/temporal-worker.yaml`
- Modify: `active-fed/k8s/rbac.yaml`, `active-fed/docker/Dockerfile.aggregator`, `active-fed/Makefile`

**Interfaces:**
- Consumes: `TASK_QUEUE`, both workflows (Task 4), both activities (Task 3).
- Produces: a runnable module `python -m src.orchestration.worker_main`, and an in-cluster Deployment running it.

- [ ] **Step 1: Implement `src/orchestration/worker_main.py`**

```python
"""
Temporal worker process: registers the workflows and activities and polls.

Runs as a Deployment in the consumer namespace. Its ServiceAccount needs
create/delete on jobs and get/list/watch on pods plus pods/log, because
`launch_and_watch_pod` does exactly those things.
"""

from __future__ import annotations

import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from src.orchestration.activities import cleanup_worker_job, launch_and_watch_pod
from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow, WorkerWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


async def main() -> None:
    address = os.environ.get("TEMPORAL_ADDRESS", "temporal-frontend:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    log.info(f"connecting to Temporal at {address} (namespace={namespace})")

    client = await Client.connect(address, namespace=namespace)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TrainRoundWorkflow, WorkerWorkflow],
        activities=[launch_and_watch_pod, cleanup_worker_job],
    )
    log.info(f"worker started on task queue '{TASK_QUEUE}'")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Extend `k8s/rbac.yaml`**

Add to the existing ClusterRole's `rules`:

```yaml
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["get", "list", "watch", "create", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
```

The existing file already grants `pytorchjobs` and a narrower `jobs` rule — **replace** the narrower rule rather than adding a duplicate, and keep the `pytorchjobs` rule (the `worker_launcher: pytorchjob` fallback still needs it during migration).

- [ ] **Step 3: Create `k8s/temporal-worker.yaml`**

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: temporal-worker
  namespace: active-fed
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: active-fed-temporal-worker
subjects:
  - kind: ServiceAccount
    name: temporal-worker
    namespace: active-fed
roleRef:
  kind: ClusterRole
  name: active-fed-pipeline-role
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: temporal-worker
  namespace: active-fed
  labels:
    app: temporal-worker
spec:
  replicas: 1
  selector:
    matchLabels:
      app: temporal-worker
  template:
    metadata:
      labels:
        app: temporal-worker
    spec:
      serviceAccountName: temporal-worker
      containers:
        - name: worker
          image: active-fed-aggregator:v1
          imagePullPolicy: IfNotPresent
          command: ["uv", "run", "python", "-m", "src.orchestration.worker_main"]
          env:
            - name: TEMPORAL_ADDRESS
              value: "temporal-frontend.active-fed.svc.cluster.local:7233"
            - name: TEMPORAL_NAMESPACE
              value: "default"
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
```

- [ ] **Step 4: Add `temporalio` and `kubernetes` to the aggregator image**

In `docker/Dockerfile.aggregator`, the `uv sync --no-dev` already installs from `pyproject.toml`, so adding the dependencies in Task 3 is sufficient. Verify by building:

```bash
docker build -t active-fed-aggregator:v1 -f docker/Dockerfile.aggregator .
docker run --rm active-fed-aggregator:v1 python -c "import temporalio, kubernetes; print('deps ok')"
```

Expected: `deps ok`

- [ ] **Step 5: Add Makefile targets**

```makefile
temporal-ui:
	@echo "Temporal UI → http://localhost:8233"

run-temporal-worker:
	uv run python -m src.orchestration.worker_main
```

- [ ] **Step 6: Commit**

```bash
git add src/orchestration/worker_main.py k8s/ docker/ Makefile
git commit -m "feat(orchestration): in-cluster temporal worker deployment and rbac"
```

---

## Task 6: `active-fed` — rewire the KFP component

**Files:**
- Modify: `active-fed/src/pipelines/active_fl_pipeline.py`, `active-fed/config/k8s.yaml`, `active-fed/infra.env`

**Interfaces:**
- Consumes: `TrainRoundWorkflow`, `TASK_QUEUE`, `RoundSpec`.
- Produces: `train_workers` KFP component that starts the workflow and blocks; new pipeline parameters `min_workers`, `worker_launcher`, `temporal_address`.

**Migration flag:** `worker_launcher` defaults to `temporal` but accepts `pytorchjob`, retaining the old path so a Temporal outage cannot block experiments. **Phase P2 deletes the flag and the dead path** — it is not permanent.

- [ ] **Step 1: Add config**

In `config/k8s.yaml` under `training:` add `min_workers: 2`; add a new block:

```yaml
# --- Orchestration ---
orchestration:
  worker_launcher: temporal      # temporal | pytorchjob (pytorchjob removed in P2)
  temporal_address: temporal-frontend.active-fed.svc.cluster.local:7233
```

In `infra.env`, add `temporal` to `FED_COMPONENTS` so it reads `kfp,training,temporal,minio,mlflow`, and add:

```sh
FED_TEMPORAL_VERSION=0.62.0
FED_NODEPORT_TEMPORAL_UI=30733
FED_HOSTPORT_TEMPORAL_UI=8233
```

- [ ] **Step 2: Rewrite the `train_workers` component**

Replace the existing component body. The signature gains `min_workers`, `worker_launcher`, `temporal_address`, and `kfp_run_id`:

```python
@component(base_image="active-fed-aggregator:v1", packages_to_install=[])
def train_workers(
    fl_round: int,
    num_workers: int,
    min_workers: int,
    local_episodes: int,
    namespace: str,
    worker_launcher: str,
    temporal_address: str,
    kfp_run_id: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    worker_image: str,
    worker_report: Output[Artifact],
) -> None:
    """Run one round's worker fleet.

    With worker_launcher='temporal' this starts a TrainRoundWorkflow and blocks
    on it, streaming per-worker status into this node's logs. The workflow ID is
    deterministic, so a retried component re-attaches to the running fleet
    instead of launching a second one.
    """
    import asyncio
    import json
    import sys

    sys.path.insert(0, "/app")

    if worker_launcher == "pytorchjob":
        # Migration fallback, removed in P2.
        raise NotImplementedError(
            "the pytorchjob launcher path is retained only for rollback; "
            "restore it from git history if you need it"
        )

    from temporalio.client import Client

    from src.orchestration.types import RoundSpec
    from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow

    spec = RoundSpec(
        fl_round=fl_round, num_workers=num_workers, min_workers=min_workers,
        local_episodes=local_episodes, namespace=namespace, worker_image=worker_image,
        minio_endpoint=minio_endpoint, minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key, minio_bucket=minio_bucket,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name, kfp_run_id=kfp_run_id,
    )

    async def _run() -> dict:
        client = await Client.connect(temporal_address)
        handle = await client.start_workflow(
            TrainRoundWorkflow.run,
            spec,
            id=f"train-{kfp_run_id[:8]}-r{fl_round}",
            task_queue=TASK_QUEUE,
        )
        print(f"started Temporal workflow {handle.id}")
        report = await handle.result()
        return {
            "fl_round": report.fl_round,
            "succeeded": report.succeeded_ids,
            "failed": report.failed_ids,
            "results": [vars(r) for r in report.results],
            "temporal_workflow_id": handle.id,
        }

    payload = asyncio.run(_run())
    print(json.dumps(payload, indent=2))
    with open(worker_report.path, "w") as f:
        json.dump(payload, f, indent=2)
```

- [ ] **Step 3: Thread the new parameters through the pipeline function**

Add `min_workers: int = 2`, `worker_launcher: str = "temporal"`, and `temporal_address: str = "temporal-frontend.active-fed.svc.cluster.local:7233"` to `active_fl_pipeline`'s signature, and pass them plus `kfp_run_id` into every `train_workers(...)` call. Use `dsl.PIPELINE_JOB_ID_PLACEHOLDER` for `kfp_run_id` so the deterministic workflow ID is stable within a run and distinct across runs.

- [ ] **Step 4: Add `.set_retry()` to all three tasks**

On each of `train_op`, `agg_op`, `eval_op`:

```python
    .set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
```

This is safe now: the deterministic workflow ID and Job names make a retried `train_workers` re-attach rather than duplicate.

- [ ] **Step 5: Verify compilation and commit**

```bash
make compile-pipeline
grep -c 'train_workers' /tmp/active_fl_pipeline.yaml   # one per round
git add src/pipelines/active_fl_pipeline.py config/k8s.yaml infra.env
git commit -m "feat(pipeline): drive the worker fleet through temporal"
```

---

## Task 7: Phase gate — real-cluster verification

**Files:**
- Modify: `active-fed/README.md`

**Interfaces:** none. This is the gate authorising P2.

Needs Docker running with ~9.7 GiB. Temporal plus PostgreSQL adds roughly 1.5 GiB on top of KFP.

- [ ] **Step 1: Bring up the stack**

```bash
cd active-fed
make local-teardown || true
git submodule update --init --recursive
make local-setup
```

Then confirm: `kubectl get pods -n active-fed` shows `temporal-frontend`, `temporal-history`, `temporal-matching`, `temporal-worker` (yours), `temporal-web`, and a PostgreSQL pod, all Running.

- [ ] **Step 2: Confirm the Temporal UI answers**

```bash
curl -sf -o /dev/null -w '%{http_code}\n' http://localhost:8233
```

Expected: `200`

- [ ] **Step 3: Run a two-round pipeline**

Set `fl_rounds: 2`, `num_workers: 2`, `local_episodes: 10` in `config/k8s.yaml`, then `make run-pipeline`.

- [ ] **Step 4: Verify the per-worker visibility this phase exists for**

In the Temporal UI: one `TrainRoundWorkflow` per round, each with N `WorkerWorkflow` children; open a child and confirm its pending activity shows **heartbeat details** with `worker_id` and `waited_s`. That live progress is the requirement this phase was built to satisfy.

Then confirm from the CLI:

```bash
kubectl get jobs -n active-fed -l app=active-fl-worker
kubectl get pods -n active-fed -l app=active-fl-worker
```

Expected: one Job per (round, worker) with deterministic names of the form `aflw-<8 hex>-r<N>-w<M>`.

- [ ] **Step 5: Verify per-pod failure attribution**

While a round is running, kill one worker pod:

```bash
kubectl delete pod -n active-fed -l worker-id=1 --force --grace-period=0
```

Expected: `restartPolicy: OnFailure` restarts it, or the activity retries; the round still completes because `min_workers=2` is met by the survivors, and the Temporal UI shows the failure and retry for that worker specifically — not an opaque whole-fleet error.

- [ ] **Step 6: Update the README and commit**

Document: the Temporal UI URL, the three-surface split (KFP = round DAG, Temporal = fleet, MLflow = metrics), and how to read per-worker progress.

```bash
git add README.md
git commit -m "docs: temporal orchestration and per-worker observability"
```

- [ ] **Step 7: Confirm the gate**

P1 is complete only when **all** hold:

- `fed-infra`: `make check` green.
- `active-fed`: full pytest suite green, including the new orchestration tests; `make compile-pipeline` succeeds.
- A 2-round pipeline completes with `TrainRoundWorkflow` + N `WorkerWorkflow` children visible in Temporal.
- Heartbeat details are visible on a pending activity.
- A killed worker pod produces a per-worker failure and retry, and the round survives at quorum.
- `src/experiment/local_runner.py` and `config/local.yaml` are unchanged.

**Do not start P2 until this gate passes.**

---

## Out of scope for P1

- Removing the `worker_launcher` flag and the PyTorchJob path → **P2**
- Round-0 global init, `start_round` resume, `restartPolicy` on the old path → **P2**
- Karmada multi-cluster and `PropagationPolicy` dispatch → **P3**
- Kubernetes Dashboard, Karmada Dashboard, MLflow cross-link tags → **P4**
- Any change to the local runner, `config/local.yaml`, or RL/ML code — permanently out of scope for this line of work.
