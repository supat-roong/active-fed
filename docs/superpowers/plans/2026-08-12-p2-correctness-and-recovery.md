# Correctness and Recovery Implementation Plan (Phase P2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix five defects found while reading `active-fed` — one of which silently corrupts every K8s experiment's round 0 — and add resume-from-checkpoint so a lost run costs one round rather than the whole experiment.

**Architecture:** No new components. This phase hardens what P0 and P1 built: a seeded round-0 global model written before any worker starts, `start_round` computed from MinIO so a resubmitted run covers only the remaining rounds, honest failure reporting in the post-processing path, and removal of the `worker_launcher` migration flag.

**Tech Stack:** Existing — Python 3.10, KFP, MinIO, MLflow, pytest. No new dependencies.

## Global Constraints

- **Prerequisite:** Phase P1 complete and its gate passed. Temporal drives the worker fleet; `worker_launcher` still exists and is removed here.
- **Do not change** `src/experiment/local_runner.py`, `config/local.yaml`, or any RL/ML algorithm code — with one deliberate exception in Task 1, called out explicitly there.
- Python: ruff line-length 100, `select = ["E","F","I","UP"]`, mypy `python_version = 3.10`.
- `fl_rounds` / `fl_round` are the project's own domain terms — never rename.
- Commit messages: Conventional Commits, no trailers.
- Work on `main`; **no `git push`, no `gh` commands** unless the human partner says otherwise.

## The defects this phase fixes

All five were identified by reading the code during the P0 design and confirmed against the current tree.

1. **Round-0 divergence (correctness, silent, highest priority).** In the K8s path round 0 has no `round_0/global.pt`, so `_fetch_global_weights` returns `None` (`src/agent/train_worker.py:76-79`) and every worker keeps **its own independently random** `ActorCritic` — no seed is set in the worker entrypoint. Round 0 therefore averages N unrelated random networks. `src/experiment/local_runner.py:148` does it correctly: it builds `global_weights` once and calls `set_weights` on every worker. **Local and K8s results are not comparable at round 0 today**, which invalidates any cross-mode conclusion drawn from round-0 behaviour.
2. **Non-idempotent job naming** — **already resolved in P1.** Job names are deterministic (`aflw-<run_uid>-r<N>-w<M>`, verified live) and the workflow ID is deterministic with `USE_EXISTING` conflict handling. Nothing left to do.
3. **`restartPolicy: Never` on the legacy PyTorchJob path** — **already resolved in P1.** That path is gone. Note the current `restartPolicy: Never` in `build_job_manifest` is a *different, deliberate* choice made in P1's gate fixes: with `backoffLimit: 0`, Kubernetes does not retry and Temporal owns retry exclusively. **Do not change it.**
4. **Hard 20-minute timeout with no retry** — **already resolved in P1** by heartbeat-based liveness. Nothing left to do.

5. **Silently swallowed post-processing failures.** `subprocess.run(..., check=False)` in `src/pipelines/run_pipeline.py:163-193` discards fetch and plot errors, so a sweep reports success while producing no plots.

**Verified remaining scope for P2** (checked against the tree after P1 landed): items 1 and 5 below, plus `start_round` resume and removal of the `worker_launcher` flag. Items 2-4 are done.

---

## File Structure

**Modified**

| Path | Change |
|---|---|
| `src/agent/train_worker.py` | Seed the fallback init so a missing global model is at least deterministic; log loudly when it happens |
| `src/pipelines/active_fl_pipeline.py` | New `init_global_model` component at the pipeline head; `start_round` parameter; delete the `worker_launcher` fallback |
| `src/pipelines/run_pipeline.py` | Compute `start_round` from MinIO; surface post-processing failures |
| `src/aggregator/collect.py` | Add `write_initial_global_weights` helper |
| `config/k8s.yaml` | Remove `orchestration.worker_launcher`; add `seed` |

**New tests**

| Path | Covers |
|---|---|
| `tests/test_global_init.py` | Seeded init is deterministic and identical across workers |
| `tests/test_resume.py` | `start_round` computation from MinIO object listings |
| `tests/test_run_pipeline_errors.py` | Post-processing failures propagate |

---

## Task 1: Seeded round-0 global model

**Files:**
- Modify: `active-fed/src/aggregator/collect.py`, `active-fed/src/agent/train_worker.py`
- Create: `active-fed/tests/test_global_init.py`

**Interfaces:**
- Produces: `write_initial_global_weights(minio_client, bucket, seed) -> str` in `src/aggregator/collect.py`, returning the written key `round_0/global.pt`. Idempotent: if the key already exists it returns without overwriting, so a retried pipeline head does not reset training.

**The one deliberate ML-adjacent change:** `train_worker.py` gains a seeded fallback. This is not an algorithm change — it makes an existing implicit behaviour explicit and reproducible. The real fix is that `round_0/global.pt` now always exists; the seed is a second line of defence that also makes the failure mode diagnosable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_global_init.py`:

```python
import io

import torch

from src.agent.model import ActorCritic
from src.aggregator.collect import write_initial_global_weights


class FakeMinio:
    """Minimal stand-in recording puts and answering existence checks."""

    def __init__(self, existing=None):
        self.objects: dict[str, bytes] = dict(existing or {})
        self.puts: list[str] = []

    def bucket_exists(self, bucket):
        return True

    def stat_object(self, bucket, key):
        if key not in self.objects:
            from minio.error import S3Error

            raise S3Error("NoSuchKey", "missing", key, "rid", "hid", None)
        return object()

    def put_object(self, bucket, key, data, length):
        self.puts.append(key)
        self.objects[key] = data.read()

    def get_object(self, bucket, key):
        class R:
            def __init__(self, b):
                self._b = b

            def read(self):
                return self._b

        return R(self.objects[key])


def _load(client, key):
    return torch.load(io.BytesIO(client.objects[key]), map_location="cpu", weights_only=True)


def test_writes_round_zero_global_weights():
    c = FakeMinio()
    key = write_initial_global_weights(c, "bkt", seed=42)
    assert key == "round_0/global.pt"
    assert c.puts == ["round_0/global.pt"]


def test_written_weights_load_into_the_real_model():
    c = FakeMinio()
    write_initial_global_weights(c, "bkt", seed=42)
    model = ActorCritic()
    model.load_state_dict(_load(c, "round_0/global.pt"))  # raises if shapes mismatch


def test_same_seed_produces_identical_weights():
    a, b = FakeMinio(), FakeMinio()
    write_initial_global_weights(a, "bkt", seed=42)
    write_initial_global_weights(b, "bkt", seed=42)
    wa, wb = _load(a, "round_0/global.pt"), _load(b, "round_0/global.pt")
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k


def test_different_seeds_produce_different_weights():
    a, b = FakeMinio(), FakeMinio()
    write_initial_global_weights(a, "bkt", seed=1)
    write_initial_global_weights(b, "bkt", seed=2)
    wa, wb = _load(a, "round_0/global.pt"), _load(b, "round_0/global.pt")
    assert any(not torch.equal(wa[k], wb[k]) for k in wa)


def test_is_idempotent_and_does_not_overwrite_existing_weights():
    # A retried pipeline head must not reset training back to round 0.
    c = FakeMinio(existing={"round_0/global.pt": b"sentinel"})
    key = write_initial_global_weights(c, "bkt", seed=42)
    assert key == "round_0/global.pt"
    assert c.puts == []
    assert c.objects["round_0/global.pt"] == b"sentinel"
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_global_init.py -q`
Expected: FAIL — `ImportError: cannot import name 'write_initial_global_weights'`

- [ ] **Step 3: Implement the helper**

Append to `src/aggregator/collect.py`:

```python
def write_initial_global_weights(
    minio_client: Minio,
    bucket: str,
    seed: int = 42,
) -> str:
    """Write a single seeded round-0 global model, if one is not already there.

    Without this, round 0 has no global model, every worker falls back to its
    own independently-random ActorCritic, and the aggregator averages N
    unrelated networks. The local runner never had this bug because it builds
    one model and pushes it to every worker; only the K8s path diverged.

    Idempotent: an existing round_0/global.pt is left untouched, so a retried
    pipeline head cannot reset training.
    """
    key = "round_0/global.pt"
    try:
        minio_client.stat_object(bucket, key)
        log.info(f"initial global weights already present at {key}, leaving them alone")
        return key
    except S3Error as e:
        if e.code not in ("NoSuchKey", "NoSuchObject"):
            raise

    from src.agent.model import ActorCritic

    torch.manual_seed(seed)
    weights = {k: v.clone() for k, v in ActorCritic().state_dict().items()}

    buf = io.BytesIO()
    torch.save(weights, buf)
    buf.seek(0)
    minio_client.put_object(bucket, key, buf, length=buf.getbuffer().nbytes)
    log.info(f"wrote seeded initial global weights (seed={seed}) → {key}")
    return key
```

Add `from minio.error import S3Error` to the imports at the top of the file.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_global_init.py -q`
Expected: PASS — 5 tests

- [ ] **Step 5: Make the worker's fallback loud and deterministic**

In `src/agent/train_worker.py`, in `train()`, immediately before `agent = PPOAgent(device=args.device)`:

```python
        # Deterministic fallback. round_0/global.pt is written by the pipeline's
        # init_global_model step, so reaching the None branch below means that
        # step did not run — seed anyway so the failure is reproducible rather
        # than a different random net per worker.
        torch.manual_seed(args.seed + worker_id)
```

and after the `if global_weights is not None:` block add:

```python
            else:
                log.warning(
                    f"no global weights for round {args.fl_round}; worker {worker_id} is "
                    "starting from a locally-seeded model. If this appears at round 0 the "
                    "init_global_model step did not run and aggregation will average "
                    "unrelated networks."
                )
```

Add `--seed` to `parse_args()` with `type=int, default=42`.

- [ ] **Step 6: Confirm nothing else regressed, then commit**

```bash
PYTHONPATH=. uv run pytest tests/ -q
git add src/aggregator/collect.py src/agent/train_worker.py tests/test_global_init.py
git commit -m "fix: seed and persist the round-0 global model

Round 0 previously had no global model in the K8s path, so every worker kept
its own random init and the aggregator averaged N unrelated networks. The
local runner was always correct, which made local and K8s results
incomparable at round 0."
```

---

## Task 2: Pipeline head and `start_round`

**Files:**
- Modify: `active-fed/src/pipelines/active_fl_pipeline.py`

**Interfaces:**
- Produces: `init_global_model` KFP component; `start_round: int = 0` pipeline parameter; rounds unrolled over `range(start_round, fl_rounds)`.

- [ ] **Step 1: Add the component**

```python
@component(base_image="active-fed-aggregator:v1", packages_to_install=[])
def init_global_model(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    seed: int,
) -> None:
    """Ensure a single seeded round-0 global model exists before any worker runs."""
    import sys

    sys.path.insert(0, "/app")

    import logging

    from minio import Minio

    from src.aggregator.collect import write_initial_global_weights

    logging.basicConfig(level=logging.INFO)
    client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )
    if not client.bucket_exists(minio_bucket):
        client.make_bucket(minio_bucket)
    write_initial_global_weights(client, minio_bucket, seed=seed)
```

- [ ] **Step 2: Wire it in as the head and add `start_round`**

In `active_fl_pipeline`, add parameters `start_round: int = 0` and `seed: int = 42`, then:

```python
    init_op = init_global_model(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
        seed=seed,
    )

    prev_op = init_op
    for round_idx in range(start_round, compile_time_rounds):
        ...
```

Every round's `train_op` chains `.after(prev_op)` as before, so the first round now waits on `init_op`.

- [ ] **Step 3: Delete the `worker_launcher` fallback**

Remove the `worker_launcher` parameter from `train_workers`, from `active_fl_pipeline`, and from `config/k8s.yaml`. Delete the `NotImplementedError` branch. The Temporal path is now the only path.

- [ ] **Step 4: Verify and commit**

```bash
make compile-pipeline
python3 -c "
import yaml
d = yaml.safe_load(open('/tmp/active_fl_pipeline.yaml'))
params = d['root']['inputDefinitions']['parameters']
assert 'start_round' in params and 'seed' in params, sorted(params)
assert 'worker_launcher' not in params, 'migration flag still present'
print('pipeline parameters OK')
"
git add src/pipelines/active_fl_pipeline.py config/k8s.yaml
git commit -m "feat(pipeline): seeded init head, start_round resume, drop migration flag"
```

---

## Task 3: Resume computation and honest post-processing

**Files:**
- Modify: `active-fed/src/pipelines/run_pipeline.py`
- Create: `active-fed/tests/test_resume.py`, `active-fed/tests/test_run_pipeline_errors.py`

**Interfaces:**
- Produces: `compute_start_round(minio_client, bucket) -> int` and `run_step(argv, description) -> None` in `src/pipelines/run_pipeline.py`.

**Why not KFP caching:** KFP's caching is left at its default and is too implicit to build resume on — a cache hit depends on input hashing we do not control. Reading the highest `round_N/global.pt` from MinIO is explicit and inspectable.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resume.py`:

```python
from src.pipelines.run_pipeline import compute_start_round


class FakeObj:
    def __init__(self, name):
        self.object_name = name


class FakeMinio:
    def __init__(self, names):
        self._names = names

    def list_objects(self, bucket, prefix="", recursive=False):
        return [FakeObj(n) for n in self._names if n.startswith(prefix)]


def test_empty_bucket_starts_at_zero():
    assert compute_start_round(FakeMinio([]), "b") == 0


def test_only_initial_weights_starts_at_zero():
    assert compute_start_round(FakeMinio(["round_0/global.pt"]), "b") == 0


def test_resumes_at_highest_completed_round():
    names = ["round_0/global.pt", "round_1/global.pt", "round_2/global.pt"]
    assert compute_start_round(FakeMinio(names), "b") == 2


def test_ignores_worker_artifacts():
    names = [
        "round_0/global.pt",
        "round_1/global.pt",
        "round_1/workers/worker_0_weights.pt",
        "round_1/workers/worker_1_delta.pt",
    ]
    assert compute_start_round(FakeMinio(names), "b") == 1


def test_ignores_unparseable_keys():
    names = ["round_0/global.pt", "roundX/global.pt", "scratch/global.pt", "round_3/global.pt"]
    assert compute_start_round(FakeMinio(names), "b") == 3


def test_non_contiguous_rounds_resume_at_the_maximum():
    # Gaps mean an earlier attempt died mid-experiment; the newest checkpoint wins.
    assert compute_start_round(FakeMinio(["round_0/global.pt", "round_5/global.pt"]), "b") == 5
```

Create `tests/test_run_pipeline_errors.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_resume.py tests/test_run_pipeline_errors.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_start_round'`

- [ ] **Step 3: Implement both helpers**

Add to `src/pipelines/run_pipeline.py`:

```python
def compute_start_round(minio_client, bucket: str) -> int:
    """Highest round N for which round_N/global.pt exists, else 0.

    Deliberately does not rely on KFP caching, which is left at its default and
    depends on input hashing we do not control. Reading the checkpoint that
    actually exists is explicit and inspectable.
    """
    highest = 0
    for obj in minio_client.list_objects(bucket, prefix="round_", recursive=True):
        name = obj.object_name
        if not name.endswith("/global.pt"):
            continue
        head = name.split("/", 1)[0]
        if not head.startswith("round_"):
            continue
        try:
            highest = max(highest, int(head[len("round_") :]))
        except ValueError:
            continue
    return highest


def run_step(argv: list[str], description: str) -> None:
    """Run a post-processing step, failing loudly.

    These calls previously used check=False, so a failed fetch or plot was
    discarded and the sweep reported success while producing nothing.
    """
    log.info(f"{description}: {' '.join(argv)}")
    subprocess.run(argv, check=True)
```

- [ ] **Step 4: Use them**

Replace the three `subprocess.run(..., check=False)` calls in the `--auto-download` block with `run_step(...)`, each with a descriptive label. Wrap the block so one failed combination does not abort the rest, but **does** surface:

```python
        failures: list[str] = []
        for _, run_name in submitted_runs:
            try:
                run_step(
                    ["uv", "run", "python", "analysis/fetch_k8s_runs.py",
                     "--experiment-prefix", run_name],
                    f"fetch results for {run_name}",
                )
            except Exception as e:
                failures.append(f"{run_name}: {e}")
        ...
        if failures:
            log.error("post-processing failed for %d run(s):", len(failures))
            for f in failures:
                log.error("  %s", f)
            raise SystemExit(1)
```

Then, before submitting, compute and pass `start_round`:

```python
    start_round = compute_start_round(minio_client, bucket_name)
    if start_round:
        log.info(f"resuming from round {start_round} (checkpoint found in MinIO)")
    arguments["start_round"] = start_round
```

Note the bucket is currently generated per combination (`fed-<uuid>`), so resume only applies when a bucket is reused — say so in a comment, and add a `--bucket` override so a resumed run can target the original bucket.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=. uv run pytest tests/ -q
git add src/pipelines/run_pipeline.py tests/test_resume.py tests/test_run_pipeline_errors.py
git commit -m "feat(pipeline): resume from the last MinIO checkpoint and surface post-processing failures"
```

---

## Task 4: Phase gate

- [ ] **Step 1: Full suite green**

```bash
cd active-fed && PYTHONPATH=. uv run pytest tests/ -q && make compile-pipeline
```

- [ ] **Step 2: Prove round 0 is now identical across workers**

Run a 1-round, 2-worker pipeline, then compare each worker's *pre-training* view. The simplest proof: confirm `round_0/global.pt` exists before any worker Job starts, and that both workers' logs lack the "no global weights" warning added in Task 1.

```bash
kubectl logs -n active-fed -l app=active-fl-worker --tail=-1 | grep -c "no global weights"
```

Expected: `0`

- [ ] **Step 3: Prove resume works**

Start a 4-round run, kill it after round 2 (`kubectl delete workflow -n kubeflow --all`), then resubmit with the same bucket. Confirm the log reports `resuming from round 2` and the compiled DAG contains 2 rounds, not 4.

- [ ] **Step 4: Prove post-processing failures surface**

Temporarily point `--experiment-prefix` at a non-existent experiment and confirm the run exits non-zero with the failure named, rather than reporting success.

- [ ] **Step 5: Confirm the gate**

P2 is complete when: the suite is green; round 0 produces no fallback warnings; a resumed run covers only remaining rounds; a failing post-processing step exits non-zero; `worker_launcher` appears nowhere in the tree; and `src/experiment/local_runner.py` and `config/local.yaml` are unchanged.

---

## Out of scope for P2

- Karmada multi-cluster → **P3**
- Dashboards and MLflow cross-link tags → **P4**
- Any change to the local runner, `config/local.yaml`, or RL/ML algorithm code.
