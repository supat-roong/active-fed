# Phase P4, Task 3 — Consumer wiring and documentation

Status: DONE

## What was built

**`fed-infra`** (docs only — Task 1 had already landed the library code at
`f36f2b0`):
- `README.md`: added the two dashboard components (`the Kubernetes
  Dashboard`, `the Karmada Dashboard`) to the top-level overview sentence.
  The per-component docs, the `## Dashboard access` section and its
  local-development-convenience security note were already written in Task
  1's commit `f36f2b0` — verified present by reading the README and by
  `git show f36f2b0 -- README.md` before touching anything, so no further
  content was needed there.
- Confirmed `tests/agnostic.bats` still passes and a repo-wide
  `grep -rn "active-fed\|fed-twin"` (excluding `.git`) matches only the guard
  test's own pattern string.

**`active-fed`**:
- `infra.env`: `FED_COMPONENTS` gained `k8s-dashboard`.
- `infra.env.multi`: `FED_COMPONENTS` gained `k8s-dashboard,karmada-dashboard`
  (kept every pre-existing component).
- `vendor/fed-infra` bumped `e19c969` → `f36f2b0`.
- `README.md`: "Observability: Three Surfaces" → "Observability: Four
  Surfaces", table gained the Kubernetes Dashboard row and the
  multi-only Karmada Dashboard row (URLs/answers matching the plan's Step 3
  table verbatim), a note on the five MLflow cross-link tags, a new "Start
  from a bad reward curve" walkthrough (MLflow run → `temporal_workflow_url`
  → failed `WorkerWorkflow` → failure reason → `kfp_run_url`), and the
  Use-Case-2 "Open UIs" comment block gained the Kubernetes Dashboard line.

**`fed-twin`**:
- `setup/install_multi_cluster_local.sh`: deleted the entire inlined Karmada
  Dashboard block (manifest apply, both kubeconfig Secrets, the NodePort
  Service patch, the admin ServiceAccount/ClusterRoleBinding, and the
  original token-mint) — all of that is now `fed-infra`'s
  `fed_karmada_dashboard_install`, invoked automatically by the existing
  `fed-infra-up` call once `karmada-dashboard` is in `FED_COMPONENTS`.
  Replaced it with an explicit `fed_dashboard_token` call (sourcing
  `lib/dashboard.sh`) that mints and prints the same 24h token as before.
  129 → 82 lines.
- `infra.env.multi`: `FED_COMPONENTS` gained `karmada-dashboard`; updated the
  header comment describing what each field does.
- `vendor/fed-infra` bumped `e19c969` → `f36f2b0`.
- `README.md`: updated the Karmada Dashboard bullet — it's now reachable
  directly at `http://localhost:32000` (fed-infra's `kind/multi-host.yaml.tpl`
  maps the host port at cluster-creation time as of Task 1), so the
  `kubectl port-forward` instruction was removed.

## How I verified the dashboard token still reaches the user

This was the task's flagged risk, so I checked it by direct code comparison,
not by inference:

1. Read the pre-change script with `git show HEAD:setup/install_multi_cluster_local.sh`
   (before any edits) and confirmed exactly what reached the user: a
   `"Dashboard Access Token (expires in 24h):"` header, the raw token via
   `echo "$DASHBOARD_TOKEN"` between two dashed lines, where
   `DASHBOARD_TOKEN=$(kubectl --kubeconfig="${FED_KARMADA_CONFIG}" create token
   karmada-admin-sa -n karmada-system --duration=24h)`.
2. Read `fed-infra/lib/dashboard.sh`'s `fed_dashboard_token`: it runs
   `token=$(kubectl create token "$sa" -n "$ns" --duration=24h)` and
   `printf '%s' "$token"` to stdout only (never through `fed_log`, so it can't
   leak into a redirected log) — same SA (`karmada-admin-sa`), same namespace
   (`karmada-system`), same `--duration=24h`.
3. Rewrote the deleted block's call site to
   `DASHBOARD_TOKEN=$(KUBECONFIG="${FED_KARMADA_CONFIG}" fed_dashboard_token karmada-system karmada-admin-sa)`
   — `KUBECONFIG=` as an env var is the exact substitute for the old
   `--kubeconfig=` flag on a single kubectl invocation, and matches the
   pattern `fed-infra/README.md`'s own "Dashboard access" section documents
   for the karmada-dashboard token. The surrounding `echo` lines (header,
   token, dashes) were left byte-for-byte identical to the original, so the
   information reaching the terminal is unchanged.
4. Confirmed the SA/ClusterRoleBinding the token authenticates against still
   gets created — now by `fed_karmada_dashboard_install` itself (against
   `$karmada_config` via `--kubeconfig`, same `karmada-admin-sa` name, same
   `cluster-admin` role) instead of by the deleted inline `kubectl create
   serviceaccount`/`clusterrolebinding` calls.
5. `bash -n setup/install_multi_cluster_local.sh` (syntax check) and
   `shellcheck setup/install_multi_cluster_local.sh` (exit 0) — the script
   compiles and lints clean. I did not execute the script end-to-end since
   that would require a real cluster (excluded by this task's constraints);
   the `--dry-run` contract check below exercises the library side of the
   same call chain (`fed_karmada_dashboard_install` is invoked; dry-run makes
   it log-only and it correctly no-ops, since it applies no local template).

Net effect: the user-visible token block is unchanged, and the final
summary line was upgraded from a `kubectl port-forward` workaround to a
direct URL, which is a strict improvement (Task 1 added the missing kind
hostPort mapping that made the workaround necessary in the first place).

## Test counts

- `fed-infra`: 198 bats tests, `make check` run unpiped — exit code 0.
- `active-fed`: `uv run pytest tests/ -q` → 246 passed, 2 warnings (pre-existing
  Pydantic deprecation warnings from `mlflow`, unrelated to this change).
- `fed-twin`: `uv run pytest tests/ -q` → 26 passed, including
  `tests/test_multi_cluster_contract.py` (4 passed) specifically re-run to
  confirm the `FED_COMPONENTS` addition didn't disturb the
  member-count/prefix/kubeconfig-path/cluster-name contract checks (none of
  which touch `FED_COMPONENTS`).

## Dry-run verification

All three ran via
`bash vendor/fed-infra/bin/fed-infra-up --env <contract> --dry-run --render-dir <dir>`,
exit code 0 in every case, no real kind/docker/kubectl calls made:

- `active-fed/infra.env` (single): log shows
  `dry-run: would install the Kubernetes Dashboard v2.7.0`, a rendered
  `dashboard-admin.yaml`, an `expose kubernetes-dashboard ... as NodePort`
  line, and the summary line
  `Kubernetes Dashboard : https://localhost:8443`.
- `active-fed/infra.env.multi`: same k8s-dashboard lines, plus
  `dry-run: would install the Karmada Dashboard and expose it on NodePort 32000`
  and summary line `Karmada Dashboard    : http://localhost:32000`.
- `fed-twin/infra.env.multi`: `dry-run: would install the Karmada Dashboard
  and expose it on NodePort 32000` and summary line
  `Karmada Dashboard    : http://localhost:32000`; no k8s-dashboard lines,
  correctly, since fed-twin's contract doesn't enable that component.

`fed_config_validate`'s `multi` → `karmada` requirement still passes for both
multi contracts (neither dropped `karmada` while adding the dashboard
components).

## Constraints respected

- `fed-infra` contains no `active-fed`/`fed-twin` strings — re-checked after
  every edit; `tests/agnostic.bats` (bats test #1) passed.
- No Python/pipeline/RL/ML source was touched in any repo.
- No real cluster was created or touched; all three verifications above are
  `--dry-run` or static (shellcheck/pytest/bats).
- Commits are Conventional Commits, no trailers, one logical change each, on
  `main` in each repo; no push.

## Concerns

None outstanding. The one open item from Task 2's report — `kfp_run_url`
built from `run_uid` (a locally generated ID) rather than the KFP-backend
`run.run_id` — is unchanged by this task and remains Task 4's to hit when it
follows the URLs against a real KFP UI.
