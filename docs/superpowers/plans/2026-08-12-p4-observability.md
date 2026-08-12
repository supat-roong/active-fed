# Observability and Cross-Linking Implementation Plan (Phase P4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the Kubernetes Dashboard as a `fed-infra` component, promote `fed-twin`'s Karmada Dashboard into the library, and tag every MLflow run so the four observability surfaces link to one another instead of being four disconnected islands.

**Architecture:** No new custom UI. Four existing surfaces, each answering a distinct question, connected by identifiers: KFP shows the round DAG and artifact lineage; Temporal shows the worker fleet with per-pod failure reasons and live heartbeat progress; MLflow shows the ML result; the Kubernetes Dashboard shows pod phase, restarts and events. In multi-cluster mode the Karmada Dashboard adds propagation state and member health.

**Tech Stack:** Existing. Kubernetes Dashboard v2.7.0 and the Karmada Dashboard, both installed from upstream manifests. No custom dashboard code.

## Global Constraints

- **Prerequisite:** Phases P0–P3 complete and gated.
- **`fed-infra` MUST NOT contain the strings `active-fed` or `fed-twin`.** Enforced by `tests/agnostic.bats`.
- `lib/*.sh` sourced, no shell options; `bin/*` and `tests/stubs/*` set them. Dry-run guards use `if/then/return 0`. Both enforced by tests.
- **Any new `FED_*` variable used in a template MUST be added to `FED_TEMPLATE_VARS`** in `lib/render.sh`, or it silently renders empty.
- Test-harness facts: `STUB_*_FAIL_GLOB` supports spaces, **not** `|` alternation.
- **Do not change** the local runner, `config/local.yaml`, or RL/ML algorithm code.
- Commit messages: Conventional Commits, no trailers. Work on `main`; no push.

## Explicitly rejected during design — do not build

- **Grafana + kube-state-metrics.** Deferred deliberately: it adds historical resource metrics, which is not the current need, and costs memory on an already-tight VM. Revisit only if resource history becomes a daily question.
- **A custom single-pane dashboard.** Rejected as code we would own and maintain in exchange for convenience only. The built-in surfaces cover the requirement.
- **Building pod-status monitoring on Temporal.** Temporal is a durable execution engine, not a monitoring system. It backs the *workflow*-level view — which worker-round is retrying and why. Pod health, restarts and resource usage belong to the Kubernetes Dashboard.

If during implementation you find yourself wanting one of these, stop and raise it rather than building it.

---

## File Structure

**`fed-infra` — new**

| Path | Responsibility |
|---|---|
| `lib/dashboard.sh` | `fed_k8s_dashboard_install`, `fed_karmada_dashboard_install`, `fed_dashboard_token` |
| `manifests/dashboard-admin.yaml.tpl` | ServiceAccount + ClusterRoleBinding for dashboard access |
| `tests/dashboard.bats` | Unit tests with stubs |

**`fed-infra` — modified:** `lib/config.sh` (dashboard NodePorts/versions), `lib/components.sh` (dispatch), `lib/render.sh` (whitelist), `kind/*.tpl` (host port mappings), both `bin/*`, `README.md`.

**`active-fed` — modified**

| Path | Change |
|---|---|
| `src/tracking/mlflow_logger.py` | `log_run_context()` — writes the cross-link tags |
| `src/pipelines/active_fl_pipeline.py` | `evaluate_global` sets tags; `train_workers` prints the Temporal workflow URL |
| `src/orchestration/workflows.py` | Carry `kfp_run_id` into workflow memo for reverse lookup |
| `infra.env`, `infra.env.multi` | `FED_COMPONENTS` gains `k8s-dashboard` (and `karmada-dashboard` for multi) |
| `README.md` | The four-surface observability section |

**`fed-twin` — modified:** `setup/install_multi_cluster_local.sh` drops its inlined Karmada Dashboard block in favour of the library component; `infra.env.multi` gains the component.

---

## Task 1: `fed-infra` — dashboard components

**Files:**
- Create: `fed-infra/lib/dashboard.sh`, `manifests/dashboard-admin.yaml.tpl`, `tests/dashboard.bats`
- Modify: `lib/config.sh`, `lib/render.sh`

**Interfaces:**
- Produces: `fed_k8s_dashboard_install(version)`, `fed_karmada_dashboard_install(karmada_config)`, `fed_dashboard_token(namespace, service_account) -> token on stdout`.
- New config: `FED_K8S_DASHBOARD_VERSION` (default `v2.7.0`), `FED_NODEPORT_K8S_DASHBOARD` (default `30443`), `FED_HOSTPORT_K8S_DASHBOARD` (default `8443`), `FED_NODEPORT_KARMADA_DASHBOARD` (default `32000`), `FED_HOSTPORT_KARMADA_DASHBOARD` (default `32000`).

**Source material:** `fed-twin/setup/install_multi_cluster_local.sh:236-296` already installs the Karmada Dashboard, exposes it on NodePort 32000, creates `karmada-admin-sa` in the federation context, and prints a 24-hour token. Port that faithfully; it works today.

- [ ] **Step 1: Write the failing tests**

`tests/dashboard.bats` should cover:
- `fed_k8s_dashboard_install` applies the upstream manifest pinned to the requested version, and applies the admin ServiceAccount template.
- It is idempotent — a second call with the dashboard namespace present does not re-apply.
- `fed_karmada_dashboard_install` applies the dashboard manifest, creates the kubeconfig secret in both `karmada-system` and `kubeflow`, and patches the Service to the configured NodePort.
- `fed_dashboard_token` calls `kubectl create token` with the given SA and namespace.
- All three are complete no-ops under `FED_DRY_RUN=1`.
- Each fails fast when its underlying command fails.

Note the token function returns a credential on stdout: assert it is **not** written to `$STUB_LOG` or any file, so it cannot leak into logs.

- [ ] **Step 2: Implement `lib/dashboard.sh`**

Follow library conventions exactly: dry-run guard first, existence probe for idempotency, `|| return 1` on every external command, no `set -euo pipefail`.

For the token function, log a redacted line (`fed_log "created 24h dashboard token (not logged)"`) and print the token itself only on stdout, so a caller can capture it deliberately.

- [ ] **Step 3: Write `manifests/dashboard-admin.yaml.tpl`**

A ServiceAccount plus a `cluster-admin` ClusterRoleBinding, namespaced by `${FED_NAMESPACE}`. Add a comment noting this is a **local development convenience** and would be inappropriate in a shared cluster — `cluster-admin` for a dashboard SA is deliberate here and nowhere else.

- [ ] **Step 4: Add config defaults, whitelist entries, kind port mappings**

Add all five variables to `fed_config_defaults` and its export list, add the two `NODEPORT`/`HOSTPORT` pairs to `FED_TEMPLATE_VARS`, and add the mappings to `kind/single-cluster.yaml.tpl` and `kind/multi-host.yaml.tpl`.

- [ ] **Step 5: Wire into dispatch**

In `lib/components.sh`, after the existing component blocks:

```bash
  if fed_has_component k8s-dashboard; then
    fed_k8s_dashboard_install "$FED_K8S_DASHBOARD_VERSION"
  fi

  if fed_has_component karmada-dashboard; then
    fed_karmada_dashboard_install "$FED_KARMADA_CONFIG"
  fi
```

Add `dashboard` to the module list in both `bin/*`, and add dashboard URLs to `fed_up_summary` using the `if fed_has_component …; then … fi` form.

- [ ] **Step 6: Confirm goldens are unchanged, run tests, commit**

The dashboards render no manifests into the render dir, so `git diff tests/golden/` must be empty. If anything moved, investigate before committing.

```bash
make check
git add lib/dashboard.sh manifests/ lib/config.sh lib/render.sh lib/components.sh bin/ kind/ tests/
git commit -m "feat: kubernetes and karmada dashboard components"
```

---

## Task 2: `active-fed` — MLflow cross-link tags

**Files:**
- Modify: `active-fed/src/tracking/mlflow_logger.py`, `src/pipelines/active_fl_pipeline.py`, `src/orchestration/workflows.py`
- Create: `active-fed/tests/test_mlflow_cross_links.py`

**Interfaces:**
- Produces:

```python
def log_run_context(
    kfp_run_id: str,
    temporal_workflow_id: str,
    topology: str,
    kfp_base_url: str = "http://localhost:8080",
    temporal_base_url: str = "http://localhost:8233",
) -> None: ...
```

Sets MLflow tags `kfp_run_id`, `temporal_workflow_id`, `topology`, `kfp_run_url`, `temporal_workflow_url`.

**Why this matters:** without it, answering "this reward curve looks wrong — which worker failed?" means manually correlating three UIs by timestamp. With it, an MLflow run links straight to its DAG and its fleet.

- [ ] **Step 1: Write the failing tests**

Cover: all five tags are set; URLs are well-formed and contain the corresponding IDs; empty IDs are skipped rather than written as empty tags; and an MLflow failure is caught and logged rather than aborting the pipeline — tracking must never fail a training run. Use a fake/monkeypatched `mlflow.set_tags`.

- [ ] **Step 2: Implement `log_run_context`**

Wrap the whole body in `try`/`except` with a warning, matching the defensive style already used in `fed-twin/src/core/tracking.py:39-46`. Skip empty values instead of writing empty tags.

- [ ] **Step 3: Call it from `evaluate_global`**

`evaluate_global` already opens an MLflow run per round (`active_fl_pipeline.py:430`). Add the call inside that run, threading `kfp_run_id` (from `dsl.PIPELINE_JOB_ID_PLACEHOLDER`) and the `temporal_workflow_id` that `train_workers` writes into its `worker_report` artifact — read it from the report the aggregator already receives.

- [ ] **Step 4: Print the Temporal URL from `train_workers`**

After starting the workflow, print a clickable line into the KFP node logs:

```
Temporal workflow: http://localhost:8233/namespaces/default/workflows/<id>
```

That single line closes the KFP → Temporal direction, which is the one a user follows most often when a round looks wrong.

- [ ] **Step 5: Carry `kfp_run_id` in the workflow memo**

In `TrainRoundWorkflow`, attach `kfp_run_id` as a workflow memo so the reverse direction — Temporal → KFP — is also available from the Temporal UI.

- [ ] **Step 6: Run tests and commit**

---

## Task 3: Consumer wiring and documentation

**Files:**
- Modify: `active-fed/infra.env`, `infra.env.multi`, `README.md`; `fed-twin/infra.env.multi`, `setup/install_multi_cluster_local.sh`, `README.md`; `fed-infra/README.md`

- [ ] **Step 1: Enable the components**

`active-fed/infra.env`: `FED_COMPONENTS` gains `k8s-dashboard`. `infra.env.multi` gains both `k8s-dashboard` and `karmada-dashboard`.

- [ ] **Step 2: Replace `fed-twin`'s inlined dashboard block**

Delete the Karmada Dashboard install, secret creation, NodePort patch and token generation from `setup/install_multi_cluster_local.sh` (currently lines ~236-296), and add `karmada-dashboard` to its `infra.env.multi`. **Verify the token is still printed** — that is how the user logs in, and losing it would make the dashboard useless without an obvious error. If the library's `fed_dashboard_token` output is not surfaced by `fed-infra-up`, call it explicitly from the consumer script.

- [ ] **Step 3: Document the four surfaces**

In `active-fed/README.md`, a table:

| Surface | URL | Answers |
|---|---|---|
| Kubeflow Pipelines | http://localhost:8080 | Round DAG, node logs, artifact lineage |
| Temporal | http://localhost:8233 | Which worker failed and why; retry counts; live progress |
| MLflow | http://localhost:5050 | Reward curves, client scores, acceptance rate, active-data usage |
| Kubernetes Dashboard | http://localhost:8443 | Pod phase, restarts, events, exec |
| Karmada Dashboard *(multi only)* | http://localhost:32000 | Propagation state, member cluster health |

Plus a short "start from a bad reward curve" walkthrough: open the MLflow run → follow `temporal_workflow_url` to the fleet → find the failed `WorkerWorkflow` → read its failure reason → follow `kfp_run_url` for the round's artifacts.

In `fed-infra/README.md`, document the two dashboard components and the security note that the admin binding is a local-development convenience.

- [ ] **Step 4: Commit**

---

## Task 4: Phase gate

- [ ] **Step 1: Single-cluster bring-up with dashboards**

```bash
cd active-fed && make local-teardown || true && make local-setup
curl -sk -o /dev/null -w '%{http_code}\n' https://localhost:8443
```

Expected: `200` or `401` (the dashboard requires a token — either proves it is serving).

- [ ] **Step 2: Token works**

Generate a token via the library function and confirm it authenticates against the dashboard.

- [ ] **Step 3: Cross-links resolve**

Run a 2-round pipeline. In MLflow, confirm each round's run carries all five tags. Follow `temporal_workflow_url` and confirm it opens the correct workflow. Follow `kfp_run_url` and confirm it opens the correct run. **Follow them by actually opening the URLs**, not by inspecting the strings — a plausible-looking URL that 404s is the failure mode here.

- [ ] **Step 4: Multi-cluster dashboards**

`make multi-setup`, then confirm the Karmada Dashboard serves on 32000, the token authenticates, and member clusters appear healthy.

- [ ] **Step 5: `fed-twin` multi-cluster still works after the dashboard move**

`make multi-cluster-setup`, confirm the dashboard is reachable and the token is printed, then `./run_pipeline.sh fed_twin_multi_cluster` reaches `Succeeded`.

- [ ] **Step 6: Confirm the gate**

P4 is complete when: both dashboards deploy as `fed-infra` components in both profiles; tokens authenticate; every MLflow run carries all five cross-link tags and both URLs resolve to the correct pages when opened; `fed-twin`'s multi-cluster path works with the dashboard supplied by the library and still prints its token; all prior gates still pass; and `fed-infra`'s `make check` is green with goldens unchanged.

---

## After P4

All five planned phases are complete. The remaining backlog is `docs/superpowers/reviews/2026-08-12-P0-deferred-findings.md` — 15 minor findings deliberately deferred during P0, several of which concern weak test assertions that later phases may have already displaced. Re-triage that list before deciding whether any still warrant work.
