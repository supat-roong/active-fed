# fed-infra Foundation Implementation Plan (Phase P0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the duplicated kind/KFP/MLflow/MinIO bootstrap out of `active-fed` and `fed-twin` into a new repo-agnostic `fed-infra` repository, consumed by both as a git submodule, with both single-cluster setups still working.

**Architecture:** `fed-infra` is a pure Bash library plus `envsubst` manifest templates, driven entirely by a consumer-supplied `infra.env`. A single `bin/fed-infra-up` entrypoint reads that file, installs only the components listed in `FED_COMPONENTS`, and renders every manifest from the consumer's values. Neither consumer's Python, pipelines, or ML code is touched in this phase.

**Tech Stack:** Bash 3.2+ (macOS default), `envsubst` (gettext), `bats-core` for unit tests, `shellcheck` for linting, `kind`, `kubectl`, `docker`. No Helm in P0 — Helm arrives with Temporal in P1.

## Global Constraints

- `fed-infra` MUST NOT contain the literal strings `active-fed` or `fed-twin`. Enforced by a CI grep. Consumer identity lives only in each consumer's `infra.env`.
- Every `fed-infra` function MUST be idempotent — safe to run repeatedly against an existing cluster.
- Every script starts with `set -euo pipefail`.
- All shell MUST pass `shellcheck -x` with no warnings.
- KFP version is `2.4.0` for both consumers.
- Target Bash is macOS's system Bash 3.2 — no associative arrays, no `${var,,}`, no `mapfile`.
- Consumers pin `fed-infra` by commit SHA via git submodule. Bumping the SHA requires both consumers' setup smoke tests to pass (Task 12).
- No changes to Python, pipelines, Dockerfiles for workers/aggregator, or ML code in this phase.
- Namespace ownership moves to `fed-infra`: it creates `${FED_NAMESPACE}`, which `active-fed/k8s/minio.yaml` currently does.
- **Naming:** the repo, directory, and scripts are `fed-infra` / `fed-infra-up` / `fed-infra-down`. Shell functions use the `fed_` prefix; environment variables use the `FED_` prefix. `fl_rounds` and `fl_round` are **active-fed's own** domain terms and must stay as they are — do not rename them.
- **Repo locations (absolute):** `fed-infra` → `/Users/supat/workspace/project/fed-infra`; `active-fed` → `/Users/supat/workspace/project/active-fed`; `fed-twin` → `/Users/supat/workspace/project/fed-twin`.
- **Branching:** all three repos commit directly to `main`. The human partner chose this explicitly. Do not create feature branches.
- **NO OUTWARD-FACING ACTIONS.** Never run `git push`, `gh repo create`, `gh pr create`, or anything else that leaves the machine. Local commits only. `https://github.com/supat-roong/fed-infra` already exists and is empty; it is only ever added as a remote, never pushed to.
- **Commit messages:** Conventional Commits, no trailers. Do not add `Co-Authored-By` — neither consumer repo has ever used it.
- Environment is macOS on Apple Silicon with colima (4 CPU, ~9.7 GiB, Docker 28.4.0). `kind`, `kubectl`, `uv`, `gh`, `envsubst` are installed; `bats` and `shellcheck` are installed by Task 1; `helm` is absent and not needed in this phase.

---

## File Structure

**New repo — `fed-infra/`**

| Path | Responsibility |
|---|---|
| `bin/fed-infra-up` | Entrypoint: load config, dispatch components in order |
| `bin/fed-infra-down` | Entrypoint: delete cluster(s) |
| `lib/common.sh` | Logging, fatal errors, command checks, retry |
| `lib/config.sh` | Load `infra.env`, apply defaults, validate, component predicate |
| `lib/render.sh` | `envsubst` template rendering; dry-run redirection |
| `lib/kind.sh` | Cluster create/delete; digest-checked image loading |
| `lib/kfp.sh` | KFP install, ARM image patches, KFP-MinIO patches, readiness |
| `lib/training.sh` | Kubeflow Training Operator install (registers the `pytorchjobs` CRD) |
| `lib/minio.sh` | Standalone MinIO StatefulSet; bucket creation via `mc` pod |
| `lib/mlflow.sh` | Build `fed-mlflow` image; deploy server/PVC/service |
| `lib/nodeport.sh` | Patch a Service to NodePort with a caller-supplied ports array |
| `lib/components.sh` | Ordered component dispatch used by `bin/fed-infra-up` |
| `manifests/namespace.yaml.tpl` | `${FED_NAMESPACE}` |
| `manifests/minio.yaml.tpl` | MinIO StatefulSet + Service |
| `manifests/mlflow-server.yaml.tpl` | MLflow Deployment + PVC + Service |
| `kind/single-cluster.yaml.tpl` | kind cluster with NodePort→hostPort mappings |
| `tests/helper.bash` | bats helper: stub `PATH`, call log |
| `tests/stubs/{kubectl,kind,docker,envsubst_probe}` | Command stubs recording argv |
| `tests/*.bats` | Unit tests per lib module |
| `tests/golden/{consumer-a,consumer-b}/` | Golden rendered manifests for dry-run diffing |

**Modified — `active-fed/`**

| Path | Change |
|---|---|
| `infra.env` | New — consumer contract |
| `vendor/fed-infra` | New — submodule |
| `setup/install_local.sh` | Reduced to `fed-infra-up` wrapper |
| `setup/teardown_local.sh` | Reduced to `fed-infra-down` wrapper |
| `setup/kind-cluster.yaml` | Deleted |
| `k8s/minio.yaml`, `k8s/mlflow-server.yaml` | Deleted |
| `Makefile` | `local-setup`/`local-teardown` pass `infra.env` |
| `.gitmodules` | New |

**Modified — `fed-twin/`**

| Path | Change |
|---|---|
| `infra.env` | New — `FED_COMPONENTS=kfp,mlflow` (uses KFP's built-in MinIO) |
| `vendor/fed-infra` | New — submodule |
| `setup/install_single_cluster_local.sh` | Reduced to `fed-infra-up` wrapper |
| `setup/teardown_single_cluster_local.sh` | Reduced to `fed-infra-down` wrapper |
| `setup/kind-single-cluster.yaml` | Deleted |
| `k8s/mlflow-server.yaml` | Deleted |
| `docker/Dockerfile.mlflow` | Deleted — `fed-infra` builds the mlflow+boto3 image |
| `Makefile` | `single-cluster-setup`/`-teardown` pass `infra.env` |
| `.gitmodules` | New |

**Key consolidation decision.** The two consumers differ in MinIO topology: `active-fed` runs a dedicated StatefulSet in its own namespace, while `fed-twin` reuses KFP's built-in MinIO in `kubeflow`. This is why `minio` is a *component* — `active-fed` enables it, `fed-twin` does not. Patches to KFP's own MinIO belong to `lib/kfp.sh`, not `lib/minio.sh`.

The MLflow image also differs today: `active-fed` runs `pip install boto3` inline at pod start (slow, needs network), `fed-twin` prebuilds `local-mlflow-boto3:v2.12.2`. `fed-infra` standardises on the prebuilt approach for both and builds the image itself, which is why `fed-twin/docker/Dockerfile.mlflow` is deleted.

---

## Task 1: Repo skeleton, test harness, and `lib/common.sh`

**Files:**
- Create: `fed-infra/lib/common.sh`
- Create: `fed-infra/tests/helper.bash`
- Create: `fed-infra/tests/stubs/kubectl`, `fed-infra/tests/stubs/kind`, `fed-infra/tests/stubs/docker`
- Create: `fed-infra/tests/common.bats`
- Create: `fed-infra/Makefile`, `fed-infra/.gitignore`, `fed-infra/README.md`

**Interfaces:**
- Produces: `fed_log(msg)`, `fed_warn(msg)`, `fed_die(msg)` (exit 1), `fed_require_cmd(cmd...)`, `fed_retry(attempts, delay_s, cmd...)`. All logging goes to **stderr** so stdout stays clean for rendered manifests.

- [ ] **Step 1: Initialise the repo and install test tooling**

```bash
mkdir -p fed-infra/{bin,lib,manifests,kind,tests/stubs,tests/golden}
cd fed-infra && git init
brew install bats-core shellcheck gettext && brew link --force gettext
```

- [ ] **Step 2: Write the stub commands**

Each stub records its argv to `$STUB_LOG` and succeeds by default. A `*_FAIL_GLOB`
variable makes **only matching subcommands** fail — this matters because the libraries use
failing probes (`kubectl get deploy`, `docker image inspect`) to decide whether something is
already installed. A blanket `RC=1` would also fail the real work that follows and give
misleading test results.

Create `tests/stubs/kubectl`:

```bash
#!/usr/bin/env bash
echo "kubectl $*" >> "$STUB_LOG"
case "$*" in
  ${STUB_KUBECTL_FAIL_GLOB:-__never_matches__}) exit 1 ;;
esac
[ -n "${STUB_KUBECTL_OUT:-}" ] && printf '%s' "$STUB_KUBECTL_OUT"
exit 0
```

Create `tests/stubs/kind`:

```bash
#!/usr/bin/env bash
echo "kind $*" >> "$STUB_LOG"
case "$*" in
  ${STUB_KIND_FAIL_GLOB:-__never_matches__}) exit 1 ;;
esac
[ -n "${STUB_KIND_OUT:-}" ] && printf '%s' "$STUB_KIND_OUT"
exit 0
```

Create `tests/stubs/docker`:

```bash
#!/usr/bin/env bash
echo "docker $*" >> "$STUB_LOG"
case "$*" in
  ${STUB_DOCKER_FAIL_GLOB:-__never_matches__}) exit 1 ;;
esac
[ -n "${STUB_DOCKER_OUT:-}" ] && printf '%s' "$STUB_DOCKER_OUT"
exit 0
```

Then: `chmod +x tests/stubs/*`

- [ ] **Step 3: Write the bats helper**

Create `tests/helper.bash`:

```bash
FED_INFRA_ROOT="$(cd "$(dirname "$BATS_TEST_FILENAME")/.." && pwd)"
export FED_INFRA_ROOT

setup_stubs() {
  export STUB_LOG="$BATS_TEST_TMPDIR/stub.log"
  : > "$STUB_LOG"
  export PATH="$FED_INFRA_ROOT/tests/stubs:$PATH"
}

calls() { cat "$STUB_LOG"; }

assert_called() {
  if ! grep -qF -- "$1" "$STUB_LOG"; then
    echo "expected call not found: $1" >&2
    echo "--- actual calls ---" >&2
    cat "$STUB_LOG" >&2
    return 1
  fi
}

refute_called() {
  if grep -qF -- "$1" "$STUB_LOG"; then
    echo "unexpected call found: $1" >&2
    return 1
  fi
}
```

- [ ] **Step 4: Write the failing test for `lib/common.sh`**

Create `tests/common.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  source "$FED_INFRA_ROOT/lib/common.sh"
}

@test "fed_log writes to stderr, not stdout" {
  run --separate-stderr bash -c "source '$FED_INFRA_ROOT/lib/common.sh'; fed_log hello"
  [ "$output" = "" ]
  [[ "$stderr" == *"hello"* ]]
}

@test "fed_die exits 1 with the message on stderr" {
  run bash -c "source '$FED_INFRA_ROOT/lib/common.sh'; fed_die boom"
  [ "$status" -eq 1 ]
  [[ "$output" == *"boom"* ]]
}

@test "fed_require_cmd succeeds when all commands exist" {
  run fed_require_cmd kubectl kind
  [ "$status" -eq 0 ]
}

@test "fed_require_cmd dies naming the missing command" {
  run bash -c "source '$FED_INFRA_ROOT/lib/common.sh'; fed_require_cmd definitely_not_a_real_cmd"
  [ "$status" -eq 1 ]
  [[ "$output" == *"definitely_not_a_real_cmd"* ]]
}

@test "fed_retry returns 0 as soon as the command succeeds" {
  run fed_retry 3 0 true
  [ "$status" -eq 0 ]
}

@test "fed_retry gives up after the requested attempts" {
  run fed_retry 3 0 false
  [ "$status" -eq 1 ]
}
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `bats tests/common.bats`
Expected: FAIL — `lib/common.sh: No such file or directory`

- [ ] **Step 6: Implement `lib/common.sh`**

```bash
#!/usr/bin/env bash
# common.sh — logging, fatal errors, command checks, retry.
# All output goes to stderr so stdout stays clean for rendered manifests.

fed_log()  { printf '\033[0;34m[fed-infra]\033[0m %s\n' "$*" >&2; }
fed_warn() { printf '\033[0;33m[fed-infra]\033[0m %s\n' "$*" >&2; }
fed_die()  { printf '\033[0;31m[fed-infra]\033[0m %s\n' "$*" >&2; exit 1; }

fed_require_cmd() {
  local c
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || fed_die "required command not found: $c"
  done
}

fed_retry() {
  local attempts=$1 delay=$2
  shift 2
  local i=1
  while [ "$i" -le "$attempts" ]; do
    if "$@"; then return 0; fi
    if [ "$i" -lt "$attempts" ]; then
      fed_warn "attempt $i/$attempts failed: $* (retrying in ${delay}s)"
      sleep "$delay"
    fi
    i=$((i + 1))
  done
  return 1
}
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `bats tests/common.bats`
Expected: PASS — 6 tests

- [ ] **Step 8: Add the Makefile**

```makefile
.PHONY: test lint check
test:
	bats tests/
lint:
	shellcheck -x bin/* lib/*.sh tests/stubs/*
check: lint test
```

- [ ] **Step 9: Verify lint is clean and commit**

```bash
make check
git add -A
git commit -m "feat: repo skeleton, bats stub harness, and common.sh"
```

---

## Task 2: `lib/config.sh` — load, default, and validate `infra.env`

**Files:**
- Create: `fed-infra/lib/config.sh`
- Create: `fed-infra/tests/config.bats`

**Interfaces:**
- Consumes: `fed_die` from Task 1.
- Produces: `fed_config_load(env_file)`, `fed_config_defaults()`, `fed_config_validate()`, `fed_has_component(name)` (returns 0 if present in `FED_COMPONENTS`).

- [ ] **Step 1: Write the failing test**

Create `tests/config.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  source "$FED_INFRA_ROOT/lib/common.sh"
  source "$FED_INFRA_ROOT/lib/config.sh"
  ENVFILE="$BATS_TEST_TMPDIR/infra.env"
  cat > "$ENVFILE" <<'EOF'
FED_CLUSTER_NAME=demo
FED_NAMESPACE=demo-ns
FED_PROFILE=single
FED_COMPONENTS=kfp,minio,mlflow
EOF
}

@test "fed_config_load exports variables from the env file" {
  fed_config_load "$ENVFILE"
  [ "$FED_CLUSTER_NAME" = "demo" ]
  [ "$FED_NAMESPACE" = "demo-ns" ]
}

@test "fed_config_load applies documented defaults" {
  fed_config_load "$ENVFILE"
  [ "$FED_KFP_VERSION" = "2.4.0" ]
  [ "$FED_NODEPORT_KFP" = "30080" ]
  [ "$FED_HOSTPORT_MLFLOW" = "5050" ]
  [ "$FED_KIND_WORKERS" = "0" ]
  [ "$FED_DRY_RUN" = "0" ]
}

@test "fed_config_load does not override values already set in the env file" {
  echo "FED_KFP_VERSION=9.9.9" >> "$ENVFILE"
  fed_config_load "$ENVFILE"
  [ "$FED_KFP_VERSION" = "9.9.9" ]
}

@test "fed_config_load dies when the env file is missing" {
  run bash -c "source '$FED_INFRA_ROOT/lib/common.sh'; source '$FED_INFRA_ROOT/lib/config.sh'; fed_config_load /nope/infra.env"
  [ "$status" -eq 1 ]
  [[ "$output" == *"not found"* ]]
}

@test "fed_config_validate dies naming a missing required variable" {
  echo "FED_NAMESPACE=" > "$ENVFILE"
  echo "FED_CLUSTER_NAME=demo" >> "$ENVFILE"
  echo "FED_PROFILE=single" >> "$ENVFILE"
  echo "FED_COMPONENTS=kfp" >> "$ENVFILE"
  run bash -c "source '$FED_INFRA_ROOT/lib/common.sh'; source '$FED_INFRA_ROOT/lib/config.sh'; fed_config_load '$ENVFILE'"
  [ "$status" -eq 1 ]
  [[ "$output" == *"FED_NAMESPACE"* ]]
}

@test "fed_config_validate rejects an invalid profile" {
  sed -i.bak 's/FED_PROFILE=single/FED_PROFILE=sideways/' "$ENVFILE"
  run bash -c "source '$FED_INFRA_ROOT/lib/common.sh'; source '$FED_INFRA_ROOT/lib/config.sh'; fed_config_load '$ENVFILE'"
  [ "$status" -eq 1 ]
  [[ "$output" == *"FED_PROFILE"* ]]
}

@test "fed_has_component matches only whole component names" {
  fed_config_load "$ENVFILE"
  run fed_has_component minio
  [ "$status" -eq 0 ]
  run fed_has_component temporal
  [ "$status" -eq 1 ]
  run fed_has_component mini
  [ "$status" -eq 1 ]
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bats tests/config.bats`
Expected: FAIL — `lib/config.sh: No such file or directory`

- [ ] **Step 3: Implement `lib/config.sh`**

```bash
#!/usr/bin/env bash
# config.sh — load, default, and validate a consumer's infra.env.

FED_REQUIRED_VARS="FED_CLUSTER_NAME FED_NAMESPACE FED_PROFILE FED_COMPONENTS"

fed_config_load() {
  local env_file=$1
  [ -f "$env_file" ] || fed_die "infra.env not found: $env_file"
  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
  fed_config_defaults
  fed_config_validate
}

fed_config_defaults() {
  : "${FED_KFP_VERSION:=2.4.0}"
  : "${FED_TRAINING_OPERATOR_VERSION:=v1.7.0}"
  : "${FED_KIND_WORKERS:=0}"
  : "${FED_MLFLOW_VERSION:=2.12.2}"
  : "${FED_MLFLOW_IMAGE:=fed-mlflow:${FED_MLFLOW_VERSION}}"
  : "${FED_IMAGES:=}"
  : "${FED_S3_BUCKET:=mlflow-artifacts}"
  : "${FED_NODEPORT_KFP:=30080}"
  : "${FED_NODEPORT_MLFLOW:=30500}"
  : "${FED_NODEPORT_MINIO_API:=30900}"
  : "${FED_NODEPORT_MINIO_CONSOLE:=30901}"
  : "${FED_HOSTPORT_KFP:=8080}"
  : "${FED_HOSTPORT_MLFLOW:=5050}"
  : "${FED_HOSTPORT_MINIO_API:=9000}"
  : "${FED_HOSTPORT_MINIO_CONSOLE:=9001}"
  : "${FED_DRY_RUN:=0}"
  : "${FED_RENDER_DIR:=}"
  export FED_KFP_VERSION FED_TRAINING_OPERATOR_VERSION FED_KIND_WORKERS FED_MLFLOW_VERSION FED_MLFLOW_IMAGE \
         FED_IMAGES FED_S3_BUCKET FED_NODEPORT_KFP FED_NODEPORT_MLFLOW \
         FED_NODEPORT_MINIO_API FED_NODEPORT_MINIO_CONSOLE FED_HOSTPORT_KFP \
         FED_HOSTPORT_MLFLOW FED_HOSTPORT_MINIO_API FED_HOSTPORT_MINIO_CONSOLE \
         FED_DRY_RUN FED_RENDER_DIR
}

fed_config_validate() {
  local v
  for v in $FED_REQUIRED_VARS; do
    eval "local value=\${$v:-}"
    # shellcheck disable=SC2154
    [ -n "$value" ] || fed_die "missing required variable: $v"
  done
  case "$FED_PROFILE" in
    single|multi) ;;
    *) fed_die "FED_PROFILE must be 'single' or 'multi', got: $FED_PROFILE" ;;
  esac
}

fed_has_component() {
  case ",${FED_COMPONENTS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bats tests/config.bats`
Expected: PASS — 7 tests

- [ ] **Step 5: Commit**

```bash
make check
git add lib/config.sh tests/config.bats
git commit -m "feat: infra.env loading, defaults, and validation"
```

---

## Task 3: `lib/render.sh` — template rendering and dry-run

**Files:**
- Create: `fed-infra/lib/render.sh`
- Create: `fed-infra/manifests/namespace.yaml.tpl`
- Create: `fed-infra/tests/render.bats`

**Interfaces:**
- Consumes: `fed_die` (Task 1); `FED_DRY_RUN`, `FED_RENDER_DIR` (Task 2).
- Produces: `fed_render(template_path)` → rendered YAML on **stdout**; `fed_apply(template_path, label)` → applies via `kubectl`, or writes `${FED_RENDER_DIR}/${label}.yaml` when `FED_DRY_RUN=1`.

**Why an explicit variable list:** `envsubst` with no arguments substitutes *every* `$NAME` in the file, which would mangle shell snippets inside manifests. Passing an explicit list confines substitution to `FED_*` variables.

- [ ] **Step 1: Write the failing test**

Create `tests/render.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  source "$FED_INFRA_ROOT/lib/common.sh"
  source "$FED_INFRA_ROOT/lib/config.sh"
  source "$FED_INFRA_ROOT/lib/render.sh"
  fed_config_defaults
  export FED_NAMESPACE=demo-ns
  TPL="$BATS_TEST_TMPDIR/t.yaml.tpl"
}

@test "fed_render substitutes FED_ variables" {
  echo 'namespace: ${FED_NAMESPACE}' > "$TPL"
  run fed_render "$TPL"
  [ "$output" = "namespace: demo-ns" ]
}

@test "fed_render leaves non-FL variables untouched" {
  echo 'cmd: echo $HOSTNAME ${PATH}' > "$TPL"
  run fed_render "$TPL"
  [ "$output" = 'cmd: echo $HOSTNAME ${PATH}' ]
}

@test "fed_render dies on a missing template" {
  run bash -c "source '$FED_INFRA_ROOT/lib/common.sh'; source '$FED_INFRA_ROOT/lib/render.sh'; fed_render /nope.tpl"
  [ "$status" -eq 1 ]
  [[ "$output" == *"template not found"* ]]
}

@test "fed_apply pipes rendered output into kubectl apply" {
  echo 'namespace: ${FED_NAMESPACE}' > "$TPL"
  fed_apply "$TPL" thing
  assert_called "kubectl apply -f -"
}

@test "fed_apply writes a file instead of calling kubectl when dry-running" {
  echo 'namespace: ${FED_NAMESPACE}' > "$TPL"
  export FED_DRY_RUN=1 FED_RENDER_DIR="$BATS_TEST_TMPDIR/out"
  fed_apply "$TPL" thing
  refute_called "kubectl apply"
  [ "$(cat "$FED_RENDER_DIR/thing.yaml")" = "namespace: demo-ns" ]
}

@test "fed_apply dies when dry-running without a render dir" {
  echo 'x: 1' > "$TPL"
  export FED_DRY_RUN=1 FED_RENDER_DIR=""
  run fed_apply "$TPL" thing
  [ "$status" -eq 1 ]
  [[ "$output" == *"FED_RENDER_DIR"* ]]
}

@test "namespace template renders to a valid Namespace object" {
  run fed_render "$FED_INFRA_ROOT/manifests/namespace.yaml.tpl"
  [[ "$output" == *"kind: Namespace"* ]]
  [[ "$output" == *"name: demo-ns"* ]]
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bats tests/render.bats`
Expected: FAIL — `lib/render.sh: No such file or directory`

- [ ] **Step 3: Implement `lib/render.sh`**

```bash
#!/usr/bin/env bash
# render.sh — envsubst-based manifest rendering with a dry-run mode.

# Explicit substitution list. envsubst with no args would replace every $NAME
# in the template, including shell variables inside container commands.
FED_TEMPLATE_VARS='${FED_CLUSTER_NAME} ${FED_NAMESPACE} ${FED_KIND_WORKERS}
${FED_S3_ENDPOINT} ${FED_S3_ACCESS_KEY} ${FED_S3_SECRET_KEY} ${FED_S3_BUCKET}
${FED_MLFLOW_IMAGE} ${FED_MLFLOW_VERSION}
${FED_NODEPORT_KFP} ${FED_NODEPORT_MLFLOW} ${FED_NODEPORT_MINIO_API} ${FED_NODEPORT_MINIO_CONSOLE}
${FED_HOSTPORT_KFP} ${FED_HOSTPORT_MLFLOW} ${FED_HOSTPORT_MINIO_API} ${FED_HOSTPORT_MINIO_CONSOLE}'

fed_render() {
  local tpl=$1
  [ -f "$tpl" ] || fed_die "template not found: $tpl"
  envsubst "$FED_TEMPLATE_VARS" < "$tpl"
}

fed_apply() {
  local tpl=$1 label=$2
  if [ "${FED_DRY_RUN:-0}" = "1" ]; then
    [ -n "${FED_RENDER_DIR:-}" ] || fed_die "FED_DRY_RUN=1 requires FED_RENDER_DIR to be set"
    mkdir -p "$FED_RENDER_DIR"
    fed_render "$tpl" > "$FED_RENDER_DIR/${label}.yaml"
    fed_log "dry-run: rendered ${label} -> ${FED_RENDER_DIR}/${label}.yaml"
  else
    fed_render "$tpl" | kubectl apply -f -
  fi
}
```

- [ ] **Step 4: Create `manifests/namespace.yaml.tpl`**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ${FED_NAMESPACE}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `bats tests/render.bats`
Expected: PASS — 7 tests

- [ ] **Step 6: Commit**

```bash
make check
git add lib/render.sh manifests/namespace.yaml.tpl tests/render.bats
git commit -m "feat: envsubst manifest rendering with dry-run support"
```

---

## Task 4: `lib/kind.sh` — cluster lifecycle and digest-checked image loading

**Files:**
- Create: `fed-infra/lib/kind.sh`
- Create: `fed-infra/kind/single-cluster.yaml.tpl`
- Create: `fed-infra/tests/kind.bats`

**Interfaces:**
- Consumes: `fed_log`, `fed_die` (Task 1); `fed_render` (Task 3).
- Produces: `fed_kind_ensure_cluster(name, template)`, `fed_kind_load_image(image, cluster)`, `fed_kind_delete_cluster(name)`.

**Why the digest check:** `kind load docker-image` re-uploads a multi-hundred-MB image every run. `fed-twin` already avoids this by comparing image IDs (`setup/install_single_cluster_local.sh:57-67`); this generalises that logic and gives it test coverage.

- [ ] **Step 1: Write the failing test**

Create `tests/kind.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  source "$FED_INFRA_ROOT/lib/common.sh"
  source "$FED_INFRA_ROOT/lib/config.sh"
  source "$FED_INFRA_ROOT/lib/render.sh"
  source "$FED_INFRA_ROOT/lib/kind.sh"
  fed_config_defaults
  export FED_CLUSTER_NAME=demo
}

@test "fed_kind_ensure_cluster creates the cluster when absent" {
  export STUB_KIND_OUT="other-cluster"
  fed_kind_ensure_cluster demo "$FED_INFRA_ROOT/kind/single-cluster.yaml.tpl"
  assert_called "kind create cluster --name demo --config -"
}

@test "fed_kind_ensure_cluster is a no-op when the cluster exists" {
  export STUB_KIND_OUT="demo"
  fed_kind_ensure_cluster demo "$FED_INFRA_ROOT/kind/single-cluster.yaml.tpl"
  refute_called "kind create cluster"
}

@test "fed_kind_ensure_cluster does not match a cluster by prefix" {
  export STUB_KIND_OUT="demo-other"
  fed_kind_ensure_cluster demo "$FED_INFRA_ROOT/kind/single-cluster.yaml.tpl"
  assert_called "kind create cluster --name demo"
}

@test "fed_kind_load_image skips loading when the digest already matches" {
  export STUB_DOCKER_OUT="sha256:abcdef123456789"
  # 'docker image inspect' returns the id above, trimmed to 12 chars by the
  # implementation. FED_KIND_CRICTL_OUT injects the in-cluster id so the test
  # does not need a real 'docker exec ... crictl images'.
  export FED_KIND_CRICTL_OUT="abcdef123456"
  fed_kind_load_image myimg:v1 demo
  refute_called "kind load docker-image"
}

@test "fed_kind_load_image loads when the cluster copy is stale" {
  export STUB_DOCKER_OUT="sha256:abcdef123456789"
  export FED_KIND_CRICTL_OUT="999999999999"
  fed_kind_load_image myimg:v1 demo
  assert_called "kind load docker-image myimg:v1 --name demo"
}

@test "fed_kind_load_image loads when the image is absent from the cluster" {
  export STUB_DOCKER_OUT="sha256:abcdef123456789"
  export FED_KIND_CRICTL_OUT=""
  fed_kind_load_image myimg:v1 demo
  assert_called "kind load docker-image myimg:v1 --name demo"
}

@test "fed_kind_delete_cluster deletes by name" {
  fed_kind_delete_cluster demo
  assert_called "kind delete cluster --name demo"
}

@test "kind template renders the configured host port mappings" {
  export FED_HOSTPORT_KFP=8080 FED_NODEPORT_KFP=30080
  run fed_render "$FED_INFRA_ROOT/kind/single-cluster.yaml.tpl"
  [[ "$output" == *"containerPort: 30080"* ]]
  [[ "$output" == *"hostPort: 8080"* ]]
  [[ "$output" == *"name: demo"* ]]
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bats tests/kind.bats`
Expected: FAIL — `lib/kind.sh: No such file or directory`

- [ ] **Step 3: Implement `lib/kind.sh`**

`FED_KIND_CRICTL_OUT` exists solely so tests can inject the in-cluster image id; in production it is unset and the real `docker exec ... crictl images` runs.

```bash
#!/usr/bin/env bash
# kind.sh — kind cluster lifecycle and digest-checked image loading.

fed_kind_ensure_cluster() {
  local name=$1 tpl=$2
  if kind get clusters 2>/dev/null | grep -qx "$name"; then
    fed_log "kind cluster '$name' already exists"
    return 0
  fi
  fed_log "creating kind cluster '$name'"

  # Worker nodes are appended rather than templated, because the count varies
  # per consumer and YAML has no repeat construct.
  local rendered i=0
  rendered=$(fed_render "$tpl")
  while [ "$i" -lt "${FED_KIND_WORKERS:-0}" ]; do
    rendered="${rendered}
  - role: worker"
    i=$((i + 1))
  done
  printf '%s\n' "$rendered" | kind create cluster --name "$name" --config -
}

# Returns the 12-char image id of $1 inside cluster $2, or empty if absent.
fed_kind_cluster_image_id() {
  local image=$1 cluster=$2
  if [ -n "${FED_KIND_CRICTL_OUT+x}" ]; then
    printf '%s' "$FED_KIND_CRICTL_OUT"
    return 0
  fi
  docker exec "${cluster}-control-plane" crictl images 2>/dev/null \
    | awk -v r="${image%:*}" -v t="${image#*:}" '$1==r && $2==t {print $3}' \
    | head -n 1
}

fed_kind_load_image() {
  local image=$1 cluster=$2
  local local_id cluster_id
  local_id=$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null \
    | cut -d: -f2 | cut -c1-12)
  [ -n "$local_id" ] || fed_die "image not found locally: $image (build it first)"

  cluster_id=$(fed_kind_cluster_image_id "$image" "$cluster")
  if [ -n "$cluster_id" ] && [ "$(printf '%s' "$cluster_id" | cut -c1-12)" = "$local_id" ]; then
    fed_log "image $image already current in cluster $cluster"
    return 0
  fi

  fed_log "loading image $image into cluster $cluster"
  kind load docker-image "$image" --name "$cluster"
}

fed_kind_delete_cluster() {
  local name=$1
  fed_log "deleting kind cluster '$name'"
  kind delete cluster --name "$name"
}
```

- [ ] **Step 4: Create `kind/single-cluster.yaml.tpl`**

`active-fed` currently adds a worker node and an `ingress-ready` label; `fed-twin` uses control-plane only. `FED_KIND_WORKERS` covers the difference. The `ingress-ready` label is dropped because no ingress controller is installed by either consumer.

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ${FED_CLUSTER_NAME}
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: ${FED_NODEPORT_KFP}
        hostPort: ${FED_HOSTPORT_KFP}
        protocol: TCP
      - containerPort: ${FED_NODEPORT_MLFLOW}
        hostPort: ${FED_HOSTPORT_MLFLOW}
        protocol: TCP
      - containerPort: ${FED_NODEPORT_MINIO_API}
        hostPort: ${FED_HOSTPORT_MINIO_API}
        protocol: TCP
      - containerPort: ${FED_NODEPORT_MINIO_CONSOLE}
        hostPort: ${FED_HOSTPORT_MINIO_CONSOLE}
        protocol: TCP
```

Worker nodes are appended at runtime by `fed_kind_ensure_cluster` (Step 3) when
`FED_KIND_WORKERS > 0`, so the template itself declares only the control plane.

- [ ] **Step 5: Add a test for worker-node appending**

Append to `tests/kind.bats`:

```bash
@test "fed_kind_ensure_cluster appends the requested worker nodes" {
  export STUB_KIND_OUT="" FED_KIND_WORKERS=2
  run fed_kind_ensure_cluster demo "$FED_INFRA_ROOT/kind/single-cluster.yaml.tpl"
  [ "$status" -eq 0 ]
  assert_called "kind create cluster --name demo"
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `bats tests/kind.bats`
Expected: PASS — 9 tests

- [ ] **Step 7: Commit**

```bash
make check
git add lib/kind.sh kind/single-cluster.yaml.tpl tests/kind.bats
git commit -m "feat: kind cluster lifecycle with digest-checked image loading"
```

---

## Task 5: `lib/minio.sh` — standalone MinIO and bucket creation

**Files:**
- Create: `fed-infra/lib/minio.sh`
- Create: `fed-infra/manifests/minio.yaml.tpl`
- Create: `fed-infra/tests/minio.bats`

**Interfaces:**
- Consumes: `fed_log`, `fed_warn` (Task 1); `fed_apply` (Task 3).
- Produces: `fed_minio_install()`, `fed_minio_ensure_bucket(namespace, endpoint, access_key, secret_key, bucket)`.

This module covers **only** the standalone StatefulSet that `active-fed` uses. Patches to KFP's own bundled MinIO live in `lib/kfp.sh` (Task 7).

- [ ] **Step 1: Write the failing test**

Create `tests/minio.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  source "$FED_INFRA_ROOT/lib/common.sh"
  source "$FED_INFRA_ROOT/lib/config.sh"
  source "$FED_INFRA_ROOT/lib/render.sh"
  source "$FED_INFRA_ROOT/lib/minio.sh"
  fed_config_defaults
  export FED_NAMESPACE=demo-ns
  export FED_S3_ACCESS_KEY=ak FED_S3_SECRET_KEY=sk
}

@test "fed_minio_install applies namespace and minio, then waits for the statefulset" {
  fed_minio_install
  assert_called "kubectl apply -f -"
  assert_called "kubectl rollout status statefulset/minio -n demo-ns"
}

@test "fed_minio_install skips the rollout wait when dry-running" {
  export FED_DRY_RUN=1 FED_RENDER_DIR="$BATS_TEST_TMPDIR/out"
  fed_minio_install
  refute_called "kubectl rollout status"
  [ -f "$FED_RENDER_DIR/minio.yaml" ]
  [ -f "$FED_RENDER_DIR/namespace.yaml" ]
}

@test "fed_minio_ensure_bucket runs mc with the given endpoint and credentials" {
  fed_minio_ensure_bucket demo-ns minio-service:9000 ak sk mybucket
  assert_called "kubectl run"
  assert_called "mc alias set t http://minio-service:9000 ak sk"
  assert_called "mc mb t/mybucket --ignore-existing"
}

@test "fed_minio_ensure_bucket deletes any leftover pod before and after" {
  fed_minio_ensure_bucket demo-ns minio-service:9000 ak sk mybucket
  run bash -c "grep -c 'kubectl delete pod' '$STUB_LOG'"
  [ "$output" -ge 2 ]
}

@test "fed_minio_ensure_bucket is a no-op when dry-running" {
  export FED_DRY_RUN=1 FED_RENDER_DIR="$BATS_TEST_TMPDIR/out"
  fed_minio_ensure_bucket demo-ns minio-service:9000 ak sk mybucket
  refute_called "kubectl run"
}

@test "minio template renders credentials and namespace" {
  run fed_render "$FED_INFRA_ROOT/manifests/minio.yaml.tpl"
  [[ "$output" == *"namespace: demo-ns"* ]]
  [[ "$output" == *'value: "ak"'* ]]
  [[ "$output" == *"kind: StatefulSet"* ]]
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bats tests/minio.bats`
Expected: FAIL — `lib/minio.sh: No such file or directory`

- [ ] **Step 3: Implement `lib/minio.sh`**

```bash
#!/usr/bin/env bash
# minio.sh — standalone MinIO StatefulSet and bucket provisioning.
# KFP's own bundled MinIO is handled in kfp.sh, not here.

# Note: use `if ...; then return 0; fi`, never `[ ... ] && return 0`. Under
# `set -e` the latter aborts the whole script whenever the test is false,
# because the && compound itself evaluates to a non-zero status.
fed_minio_install() {
  fed_apply "${FED_INFRA_ROOT}/manifests/namespace.yaml.tpl" namespace
  fed_apply "${FED_INFRA_ROOT}/manifests/minio.yaml.tpl" minio
  if [ "${FED_DRY_RUN:-0}" = "1" ]; then return 0; fi
  fed_log "waiting for MinIO statefulset"
  kubectl rollout status statefulset/minio -n "$FED_NAMESPACE" --timeout=180s
}

fed_minio_ensure_bucket() {
  local ns=$1 endpoint=$2 access=$3 secret=$4 bucket=$5
  if [ "${FED_DRY_RUN:-0}" = "1" ]; then
    fed_log "dry-run: would ensure bucket '${bucket}' at ${endpoint}"
    return 0
  fi
  local pod
  pod="fed-mc-$(printf '%s' "$bucket" | tr -cd 'a-z0-9')"
  fed_log "ensuring bucket '${bucket}' at ${endpoint}"
  kubectl delete pod "$pod" -n "$ns" --ignore-not-found >/dev/null 2>&1 || true
  kubectl run "$pod" --image=minio/mc:latest -n "$ns" --restart=Never --command -- \
    sh -c "mc alias set t http://${endpoint} ${access} ${secret} && mc mb t/${bucket} --ignore-existing"
  kubectl wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$pod" -n "$ns" --timeout=120s \
    || fed_warn "bucket pod for '${bucket}' did not report Succeeded; continuing"
  kubectl delete pod "$pod" -n "$ns" --ignore-not-found >/dev/null 2>&1 || true
}
```

- [ ] **Step 4: Create `manifests/minio.yaml.tpl`**

Ported from `active-fed/k8s/minio.yaml` with namespace and credentials parameterised, and the `Namespace` object removed (now `namespace.yaml.tpl`).

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: minio
  namespace: ${FED_NAMESPACE}
spec:
  serviceName: minio-service
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
        - name: minio
          image: quay.io/minio/minio:latest
          args: ["server", "/data", "--console-address", ":9001"]
          env:
            - name: MINIO_ROOT_USER
              value: "${FED_S3_ACCESS_KEY}"
            - name: MINIO_ROOT_PASSWORD
              value: "${FED_S3_SECRET_KEY}"
          ports:
            - containerPort: 9000
            - containerPort: 9001
          volumeMounts:
            - name: minio-data
              mountPath: /data
          readinessProbe:
            httpGet:
              path: /minio/health/live
              port: 9000
            initialDelaySeconds: 10
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata:
        name: minio-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: minio-service
  namespace: ${FED_NAMESPACE}
spec:
  selector:
    app: minio
  ports:
    - name: api
      port: 9000
      targetPort: 9000
    - name: console
      port: 9001
      targetPort: 9001
  type: ClusterIP
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `bats tests/minio.bats`
Expected: PASS — 6 tests

- [ ] **Step 6: Commit**

```bash
make check
git add lib/minio.sh manifests/minio.yaml.tpl tests/minio.bats
git commit -m "feat: standalone MinIO statefulset and bucket provisioning"
```

---

## Task 6: `lib/mlflow.sh` — image build and server deployment

**Files:**
- Create: `fed-infra/lib/mlflow.sh`
- Create: `fed-infra/manifests/mlflow-server.yaml.tpl`
- Create: `fed-infra/tests/mlflow.bats`

**Interfaces:**
- Consumes: `fed_log` (Task 1); `fed_apply` (Task 3).
- Produces: `fed_mlflow_build_image(image, version)`, `fed_mlflow_install()`.

**Consolidation:** `active-fed` runs `pip install boto3` inline in the container command at pod start, which is slow and needs network on every restart. `fed-twin` prebuilds an image. `fed-infra` prebuilds for both, so `fed-twin/docker/Dockerfile.mlflow` is deleted in Task 11.

- [ ] **Step 1: Write the failing test**

Create `tests/mlflow.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  source "$FED_INFRA_ROOT/lib/common.sh"
  source "$FED_INFRA_ROOT/lib/config.sh"
  source "$FED_INFRA_ROOT/lib/render.sh"
  source "$FED_INFRA_ROOT/lib/mlflow.sh"
  fed_config_defaults
  export FED_NAMESPACE=demo-ns
  export FED_S3_ENDPOINT=minio-service:9000
  export FED_S3_ACCESS_KEY=ak FED_S3_SECRET_KEY=sk FED_S3_BUCKET=arts
}

@test "fed_mlflow_build_image builds when the image is absent" {
  # Only the existence probe fails; the build itself must still succeed.
  export STUB_DOCKER_FAIL_GLOB="image inspect*"
  fed_mlflow_build_image fed-mlflow:2.12.2 2.12.2
  assert_called "docker build -t fed-mlflow:2.12.2"
}

@test "fed_mlflow_build_image skips the build when the image exists" {
  fed_mlflow_build_image fed-mlflow:2.12.2 2.12.2   # probe succeeds by default
  refute_called "docker build"
}

@test "fed_mlflow_install applies the manifests and waits for rollout" {
  fed_mlflow_install
  assert_called "kubectl apply -f -"
  assert_called "kubectl rollout status deployment/mlflow-server -n demo-ns"
}

@test "fed_mlflow_install skips the rollout wait when dry-running" {
  export FED_DRY_RUN=1 FED_RENDER_DIR="$BATS_TEST_TMPDIR/out"
  fed_mlflow_install
  refute_called "kubectl rollout status"
  [ -f "$FED_RENDER_DIR/mlflow-server.yaml" ]
}

@test "mlflow template renders image, S3 endpoint, bucket, and nodePort" {
  export FED_MLFLOW_IMAGE=fed-mlflow:2.12.2 FED_NODEPORT_MLFLOW=30500
  run fed_render "$FED_INFRA_ROOT/manifests/mlflow-server.yaml.tpl"
  [[ "$output" == *"image: fed-mlflow:2.12.2"* ]]
  [[ "$output" == *"http://minio-service:9000"* ]]
  [[ "$output" == *"s3://arts"* ]]
  [[ "$output" == *"nodePort: 30500"* ]]
  [[ "$output" == *"namespace: demo-ns"* ]]
}

@test "mlflow template does not embed a pip install at pod start" {
  run fed_render "$FED_INFRA_ROOT/manifests/mlflow-server.yaml.tpl"
  [[ "$output" != *"pip install"* ]]
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bats tests/mlflow.bats`
Expected: FAIL — `lib/mlflow.sh: No such file or directory`

- [ ] **Step 3: Implement `lib/mlflow.sh`**

```bash
#!/usr/bin/env bash
# mlflow.sh — build the mlflow+boto3 image and deploy the tracking server.

fed_mlflow_build_image() {
  local image=$1 version=$2
  if docker image inspect "$image" >/dev/null 2>&1; then
    fed_log "mlflow image $image already built"
    return 0
  fi
  fed_log "building mlflow image $image"
  local ctx
  ctx=$(mktemp -d)
  cat > "$ctx/Dockerfile" <<EOF
FROM ghcr.io/mlflow/mlflow:v${version}
RUN pip install --no-cache-dir boto3
EOF
  docker build -t "$image" "$ctx"
  rm -rf "$ctx"
}

fed_mlflow_install() {
  fed_apply "${FED_INFRA_ROOT}/manifests/namespace.yaml.tpl" namespace
  fed_apply "${FED_INFRA_ROOT}/manifests/mlflow-server.yaml.tpl" mlflow-server
  if [ "${FED_DRY_RUN:-0}" = "1" ]; then return 0; fi
  fed_log "waiting for MLflow server"
  kubectl rollout status deployment/mlflow-server -n "$FED_NAMESPACE" --timeout=300s
}
```

- [ ] **Step 4: Create `manifests/mlflow-server.yaml.tpl`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: ${FED_NAMESPACE}
  labels:
    app: mlflow-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
        - name: mlflow
          image: ${FED_MLFLOW_IMAGE}
          imagePullPolicy: IfNotPresent
          command:
            - mlflow
            - server
            - --host
            - "0.0.0.0"
            - --port
            - "5000"
            - --backend-store-uri
            - /mlflow/mlruns
            - --default-artifact-root
            - s3://${FED_S3_BUCKET}
          env:
            - name: MLFLOW_S3_ENDPOINT_URL
              value: "http://${FED_S3_ENDPOINT}"
            - name: AWS_ACCESS_KEY_ID
              value: "${FED_S3_ACCESS_KEY}"
            - name: AWS_SECRET_ACCESS_KEY
              value: "${FED_S3_SECRET_KEY}"
            - name: MLFLOW_S3_IGNORE_TLS
              value: "true"
          ports:
            - containerPort: 5000
          volumeMounts:
            - name: mlflow-storage
              mountPath: /mlflow
      volumes:
        - name: mlflow-storage
          persistentVolumeClaim:
            claimName: mlflow-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-pvc
  namespace: ${FED_NAMESPACE}
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: ${FED_NAMESPACE}
spec:
  selector:
    app: mlflow-server
  ports:
    - port: 5000
      targetPort: 5000
      nodePort: ${FED_NODEPORT_MLFLOW}
  type: NodePort
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `bats tests/mlflow.bats`
Expected: PASS — 6 tests

- [ ] **Step 6: Commit**

```bash
make check
git add lib/mlflow.sh manifests/mlflow-server.yaml.tpl tests/mlflow.bats
git commit -m "feat: prebuilt mlflow image and tracking server deployment"
```

---

## Task 7: `lib/kfp.sh` — install, ARM patches, and readiness

**Files:**
- Create: `fed-infra/lib/kfp.sh`
- Create: `fed-infra/tests/kfp.bats`

**Interfaces:**
- Consumes: `fed_log` (Task 1).
- Produces: `fed_kfp_install(version)`, `fed_kfp_patch_arm(version)`, `fed_kfp_patch_minio()`, `fed_kfp_wait()`.

**Idempotency fix carried in this task:** `fed-twin/setup/install_single_cluster_local.sh:99` uses a JSON-patch `add` to `/spec/template/spec/containers/0/ports/-`, which appends a duplicate container port on every re-run. `fed_kfp_patch_minio` uses `replace` on the whole ports array instead, making it genuinely idempotent.

- [ ] **Step 1: Write the failing test**

Create `tests/kfp.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  source "$FED_INFRA_ROOT/lib/common.sh"
  source "$FED_INFRA_ROOT/lib/config.sh"
  source "$FED_INFRA_ROOT/lib/kfp.sh"
  fed_config_defaults
}

@test "fed_kfp_install applies cluster-scoped resources then core when absent" {
  # Only the existence probe fails; the applies that follow must still succeed,
  # otherwise the second apply would never be reached.
  export STUB_KUBECTL_FAIL_GLOB="get deploy*"
  fed_kfp_install 2.4.0
  assert_called "cluster-scoped-resources?ref=2.4.0"
  assert_called "platform-agnostic?ref=2.4.0"
}

@test "fed_kfp_install is a no-op when KFP is already present" {
  fed_kfp_install 2.4.0   # probe succeeds by default
  refute_called "cluster-scoped-resources"
}

@test "fed_kfp_patch_arm repoints all four images at ghcr" {
  fed_kfp_patch_arm 2.4.0
  assert_called "ml-pipeline-ui=ghcr.io/kubeflow/kfp-frontend:2.4.0"
  assert_called "ml-pipeline-api-server=ghcr.io/kubeflow/kfp-api-server:2.4.0"
  assert_called "ml-pipeline-visualizationserver=ghcr.io/kubeflow/kfp-visualization-server:2.4.0"
  assert_called "V2_LAUNCHER_IMAGE=ghcr.io/kubeflow/kfp-launcher:2.4.0"
}

@test "fed_kfp_patch_arm pins the argo executor image" {
  fed_kfp_patch_arm 2.4.0
  assert_called "quay.io/argoproj/argoexec:v3.4.17"
}

@test "fed_kfp_patch_minio replaces rather than appends the ports array" {
  fed_kfp_patch_minio
  assert_called '"op":"replace","path":"/spec/template/spec/containers/0/ports"'
  refute_called '"path":"/spec/template/spec/containers/0/ports/-"'
}

@test "fed_kfp_patch_minio enables the console address" {
  fed_kfp_patch_minio
  assert_called '"--console-address"'
}

@test "fed_kfp_wait waits on all four core deployments" {
  fed_kfp_wait
  assert_called "rollout status deployment/workflow-controller -n kubeflow"
  assert_called "rollout status deployment/minio -n kubeflow"
  assert_called "rollout status deployment/ml-pipeline -n kubeflow"
  assert_called "rollout status deployment/ml-pipeline-ui -n kubeflow"
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bats tests/kfp.bats`
Expected: FAIL — `lib/kfp.sh: No such file or directory`

- [ ] **Step 3: Implement `lib/kfp.sh`**

```bash
#!/usr/bin/env bash
# kfp.sh — Kubeflow Pipelines install, ARM/kind stability patches, readiness.
# The patches exist because the upstream images published to gcr.io are either
# amd64-only or no longer served; ghcr.io hosts working multi-arch equivalents.

FED_KFP_NAMESPACE=kubeflow
FED_ARGOEXEC_IMAGE="quay.io/argoproj/argoexec:v3.4.17"

fed_kfp_install() {
  local ver=$1
  if kubectl get deploy -n "$FED_KFP_NAMESPACE" ml-pipeline >/dev/null 2>&1; then
    fed_log "KFP already installed"
    return 0
  fi
  fed_log "installing KFP ${ver} cluster-scoped resources"
  kubectl apply -k "https://github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${ver}"
  kubectl wait --for condition=established --timeout=300s crd/applications.app.k8s.io
  fed_log "installing KFP ${ver} core"
  kubectl apply -k "https://github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${ver}"
}

fed_kfp_patch_arm() {
  local ver=$1 ns=$FED_KFP_NAMESPACE
  fed_log "patching KFP images for ARM/kind stability"
  kubectl set image deployment/ml-pipeline-ui \
    "ml-pipeline-ui=ghcr.io/kubeflow/kfp-frontend:${ver}" -n "$ns"
  kubectl set image deployment/ml-pipeline \
    "ml-pipeline-api-server=ghcr.io/kubeflow/kfp-api-server:${ver}" -n "$ns"
  kubectl set image deployment/ml-pipeline-visualizationserver \
    "ml-pipeline-visualizationserver=ghcr.io/kubeflow/kfp-visualization-server:${ver}" -n "$ns"
  kubectl set env deployment/ml-pipeline \
    "V2_LAUNCHER_IMAGE=ghcr.io/kubeflow/kfp-launcher:${ver}" -n "$ns"
  kubectl patch deployment workflow-controller -n "$ns" --type=json \
    -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args/3\",\"value\":\"${FED_ARGOEXEC_IMAGE}\"}]"
}

# Patches KFP's own bundled MinIO (namespace kubeflow), which is separate from
# the standalone MinIO in minio.sh. Uses 'replace' on the whole ports array so
# repeated runs cannot append duplicate container ports.
fed_kfp_patch_minio() {
  local ns=$FED_KFP_NAMESPACE
  fed_log "patching KFP MinIO image and console port"
  kubectl set image deployment/minio minio=minio/minio:latest -n "$ns"
  kubectl patch deployment minio -n "$ns" --type=json \
    -p='[{"op":"replace","path":"/spec/template/spec/containers/0/ports","value":[{"containerPort":9000},{"containerPort":9001}]}]'
  kubectl patch deployment minio -n "$ns" --type=json \
    -p='[{"op":"replace","path":"/spec/template/spec/containers/0/args","value":["server","/data","--console-address",":9001"]}]'
}

fed_kfp_wait() {
  local ns=$FED_KFP_NAMESPACE d
  fed_log "waiting for KFP core deployments"
  for d in workflow-controller minio ml-pipeline ml-pipeline-ui; do
    kubectl rollout status "deployment/$d" -n "$ns" --timeout=15m
  done
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bats tests/kfp.bats`
Expected: PASS — 7 tests

- [ ] **Step 5: Commit**

```bash
make check
git add lib/kfp.sh tests/kfp.bats
git commit -m "feat: KFP install with idempotent ARM stability patches"
```

---

## Task 8: NodePort helper, component dispatch, and both entrypoints

**Files:**
- Create: `fed-infra/lib/nodeport.sh`
- Create: `fed-infra/lib/components.sh`
- Create: `fed-infra/bin/fed-infra-up`
- Create: `fed-infra/bin/fed-infra-down`
- Create: `fed-infra/tests/components.bats`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: `fed_expose_nodeport(service, namespace, ports_json)`; `fed_up()`; `fed_down()`. Entrypoint usage: `fed-infra-up --env <path/to/infra.env> [--dry-run --render-dir DIR]`.

- [ ] **Step 1: Write the failing test**

Create `tests/components.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() {
  setup_stubs
  ENVFILE="$BATS_TEST_TMPDIR/infra.env"
  RENDER="$BATS_TEST_TMPDIR/render"
}

write_env() {
  cat > "$ENVFILE" <<EOF
FED_CLUSTER_NAME=demo
FED_NAMESPACE=demo-ns
FED_PROFILE=single
FED_COMPONENTS=$1
FED_S3_ENDPOINT=minio-service.demo-ns.svc.cluster.local:9000
FED_S3_ACCESS_KEY=ak
FED_S3_SECRET_KEY=sk
EOF
}

@test "fed-infra-up with all components installs kfp, minio, and mlflow" {
  write_env "kfp,minio,mlflow"
  run "$FED_INFRA_ROOT/bin/fed-infra-up" --env "$ENVFILE" --dry-run --render-dir "$RENDER"
  [ "$status" -eq 0 ]
  [ -f "$RENDER/minio.yaml" ]
  [ -f "$RENDER/mlflow-server.yaml" ]
}

@test "fed-infra-up omits minio when the component is not listed" {
  write_env "kfp,mlflow"
  run "$FED_INFRA_ROOT/bin/fed-infra-up" --env "$ENVFILE" --dry-run --render-dir "$RENDER"
  [ "$status" -eq 0 ]
  [ ! -f "$RENDER/minio.yaml" ]
  [ -f "$RENDER/mlflow-server.yaml" ]
}

@test "fed-infra-up fails with a clear message when --env is missing" {
  run "$FED_INFRA_ROOT/bin/fed-infra-up"
  [ "$status" -eq 1 ]
  [[ "$output" == *"--env"* ]]
}

@test "fed-infra-down deletes the configured cluster" {
  write_env "kfp"
  run "$FED_INFRA_ROOT/bin/fed-infra-down" --env "$ENVFILE"
  [ "$status" -eq 0 ]
  assert_called "kind delete cluster --name demo"
}

@test "fed_expose_nodeport patches a service to NodePort with the given ports" {
  source "$FED_INFRA_ROOT/lib/common.sh"
  source "$FED_INFRA_ROOT/lib/nodeport.sh"
  fed_expose_nodeport mysvc myns '[{"port":80,"targetPort":3000,"nodePort":30080}]'
  assert_called "kubectl patch service mysvc -n myns"
  assert_called '"type":"NodePort"'
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bats tests/components.bats`
Expected: FAIL — `bin/fed-infra-up: No such file or directory`

- [ ] **Step 3: Implement `lib/nodeport.sh`**

```bash
#!/usr/bin/env bash
# nodeport.sh — expose a Service as NodePort with a caller-supplied ports array.

fed_expose_nodeport() {
  local svc=$1 ns=$2 ports_json=$3
  if [ "${FED_DRY_RUN:-0}" = "1" ]; then
    fed_log "dry-run: would expose ${svc} in ${ns} as NodePort"
    return 0
  fi
  fed_log "exposing ${svc} in ${ns} as NodePort"
  kubectl patch service "$svc" -n "$ns" \
    -p "{\"spec\":{\"type\":\"NodePort\",\"ports\":${ports_json}}}"
}
```

- [ ] **Step 4: Implement `lib/components.sh`**

Ordering matters: KFP installs first because it creates the `kubeflow` namespace and its own MinIO, which later bucket creation depends on.

```bash
#!/usr/bin/env bash
# components.sh — ordered component dispatch for fed-infra-up / fed-infra-down.

fed_up() {
  fed_require_cmd kind kubectl docker envsubst

  fed_kind_ensure_cluster "$FED_CLUSTER_NAME" "${FED_INFRA_ROOT}/kind/single-cluster.yaml.tpl"
  if [ "${FED_DRY_RUN:-0}" != "1" ]; then
    kubectl config use-context "kind-${FED_CLUSTER_NAME}"
  fi

  if fed_has_component kfp; then
    fed_kfp_install "$FED_KFP_VERSION"
    fed_kfp_patch_arm "$FED_KFP_VERSION"
    fed_kfp_patch_minio
  fi

  if fed_has_component training; then
    fed_training_install "$FED_TRAINING_OPERATOR_VERSION"
  fi

  if fed_has_component minio; then
    fed_minio_install
  fi

  if fed_has_component mlflow; then
    fed_mlflow_build_image "$FED_MLFLOW_IMAGE" "$FED_MLFLOW_VERSION"
    fed_kind_load_image "$FED_MLFLOW_IMAGE" "$FED_CLUSTER_NAME"
    fed_mlflow_install
  fi

  local img
  for img in $FED_IMAGES; do
    fed_kind_load_image "$img" "$FED_CLUSTER_NAME"
  done

  if fed_has_component kfp; then
    fed_kfp_wait
    fed_minio_ensure_bucket "$FED_KFP_NAMESPACE" \
      "minio-service.${FED_KFP_NAMESPACE}.svc.cluster.local:9000" \
      minio minio123 mlpipeline
    fed_expose_nodeport ml-pipeline-ui "$FED_KFP_NAMESPACE" \
      "[{\"port\":80,\"targetPort\":3000,\"nodePort\":${FED_NODEPORT_KFP}}]"
  fi

  if fed_has_component mlflow; then
    fed_minio_ensure_bucket "$FED_NAMESPACE" "$FED_S3_ENDPOINT" \
      "$FED_S3_ACCESS_KEY" "$FED_S3_SECRET_KEY" "$FED_S3_BUCKET"
  fi

  if fed_has_component minio; then
    fed_expose_nodeport minio-service "$FED_NAMESPACE" \
      "[{\"name\":\"api\",\"port\":9000,\"targetPort\":9000,\"nodePort\":${FED_NODEPORT_MINIO_API}},{\"name\":\"console\",\"port\":9001,\"targetPort\":9001,\"nodePort\":${FED_NODEPORT_MINIO_CONSOLE}}]"
  fi

  fed_up_summary
}

fed_up_summary() {
  fed_log ""
  fed_log "Setup complete. Services:"
  if fed_has_component kfp; then
    fed_log "  Kubeflow Pipelines : http://localhost:${FED_HOSTPORT_KFP}"
  fi
  if fed_has_component mlflow; then
    fed_log "  MLflow             : http://localhost:${FED_HOSTPORT_MLFLOW}"
  fi
  if fed_has_component minio; then
    fed_log "  MinIO Console      : http://localhost:${FED_HOSTPORT_MINIO_CONSOLE}"
  fi
}

fed_down() {
  fed_require_cmd kind
  pkill -f "kubectl port-forward" 2>/dev/null || true
  fed_kind_delete_cluster "$FED_CLUSTER_NAME"
}
```

- [ ] **Step 5: Implement `bin/fed-infra-up`**

```bash
#!/usr/bin/env bash
set -euo pipefail

FED_INFRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FED_INFRA_ROOT

# shellcheck source=../lib/common.sh
. "${FED_INFRA_ROOT}/lib/common.sh"
for m in config render kind kfp minio mlflow nodeport components; do
  # shellcheck disable=SC1090
  . "${FED_INFRA_ROOT}/lib/${m}.sh"
done

ENV_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --env)        ENV_FILE=$2; shift 2 ;;
    --dry-run)    export FED_DRY_RUN=1; shift ;;
    --render-dir) export FED_RENDER_DIR=$2; shift 2 ;;
    -h|--help)
      echo "usage: fed-infra-up --env <infra.env> [--dry-run --render-dir DIR]" >&2
      exit 0 ;;
    *) fed_die "unknown argument: $1" ;;
  esac
done

[ -n "$ENV_FILE" ] || fed_die "--env <path/to/infra.env> is required"
fed_config_load "$ENV_FILE"
fed_up
```

- [ ] **Step 6: Implement `bin/fed-infra-down`**

```bash
#!/usr/bin/env bash
set -euo pipefail

FED_INFRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FED_INFRA_ROOT

# shellcheck source=../lib/common.sh
. "${FED_INFRA_ROOT}/lib/common.sh"
for m in config render kind kfp minio mlflow nodeport components; do
  # shellcheck disable=SC1090
  . "${FED_INFRA_ROOT}/lib/${m}.sh"
done

ENV_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_FILE=$2; shift 2 ;;
    -h|--help) echo "usage: fed-infra-down --env <infra.env>" >&2; exit 0 ;;
    *) fed_die "unknown argument: $1" ;;
  esac
done

[ -n "$ENV_FILE" ] || fed_die "--env <path/to/infra.env> is required"
fed_config_load "$ENV_FILE"
fed_down
```

Then: `chmod +x bin/fed-infra-up bin/fed-infra-down`

- [ ] **Step 7: Run the test to verify it passes**

Run: `bats tests/components.bats`
Expected: PASS — 5 tests

- [ ] **Step 8: Run the full suite and commit**

```bash
make check
git add lib/nodeport.sh lib/components.sh bin/ tests/components.bats
git commit -m "feat: component dispatch and fed-infra-up/down entrypoints"
```

---

## Task 9: Golden-file dry-run tests, repo-agnosticism guard, and CI

**Files:**
- Create: `fed-infra/tests/fixtures/consumer-a.env`, `fed-infra/tests/fixtures/consumer-b.env`
- Create: `fed-infra/tests/golden/consumer-a/*.yaml`, `fed-infra/tests/golden/consumer-b/*.yaml`
- Create: `fed-infra/tests/golden.bats`
- Create: `fed-infra/tests/agnostic.bats`
- Create: `fed-infra/.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `bin/fed-infra-up` (Task 8).
- Produces: no runtime interface — a regression harness. Fixture names are deliberately `consumer-a`/`consumer-b`, not the real repo names, because of the repo-agnosticism constraint.

- [ ] **Step 1: Write the repo-agnosticism guard test**

Create `tests/agnostic.bats`:

```bash
#!/usr/bin/env bats
load helper

@test "fed-infra contains no consumer-specific identifiers" {
  run grep -rIl --exclude-dir=.git -e 'active-fed' -e 'fed-twin' "$FED_INFRA_ROOT"
  [ "$status" -ne 0 ] || {
    echo "consumer-specific strings found in: $output" >&2
    return 1
  }
}

@test "no library uses the set -e hostile '[ test ] && return' idiom" {
  # Under `set -e`, `[ cond ] && return 0` aborts the whole script whenever
  # cond is false, because the && compound evaluates to non-zero.
  run grep -nE '^\s*\[.*\]\s*&&\s*return' "$FED_INFRA_ROOT"/lib/*.sh
  [ "$status" -ne 0 ] || {
    echo "use 'if ...; then return 0; fi' instead:" >&2
    echo "$output" >&2
    return 1
  }
}
```

- [ ] **Step 2: Write the fixtures**

`tests/fixtures/consumer-a.env` — mirrors a dedicated-MinIO consumer:

```sh
FED_CLUSTER_NAME=consumer-a
FED_NAMESPACE=consumer-a
FED_PROFILE=single
FED_COMPONENTS=kfp,training,minio,mlflow
FED_KIND_WORKERS=1
FED_S3_ENDPOINT=minio-service.consumer-a.svc.cluster.local:9000
FED_S3_ACCESS_KEY=minioadmin
FED_S3_SECRET_KEY=minioadmin
FED_S3_BUCKET=mlflow-artifacts
```

`tests/fixtures/consumer-b.env` — mirrors a consumer reusing KFP's MinIO:

```sh
FED_CLUSTER_NAME=consumer-b
FED_NAMESPACE=kubeflow
FED_PROFILE=single
FED_COMPONENTS=kfp,mlflow
FED_S3_ENDPOINT=minio-service.kubeflow.svc.cluster.local:9000
FED_S3_ACCESS_KEY=minio
FED_S3_SECRET_KEY=minio123
FED_S3_BUCKET=mlflow-artifacts
```

- [ ] **Step 3: Write the golden test**

Create `tests/golden.bats`:

```bash
#!/usr/bin/env bats
load helper

setup() { setup_stubs; }

render_consumer() {
  local name=$1 out=$2
  "$FED_INFRA_ROOT/bin/fed-infra-up" \
    --env "$FED_INFRA_ROOT/tests/fixtures/${name}.env" \
    --dry-run --render-dir "$out"
}

@test "consumer-a renders byte-identically to its golden files" {
  local out="$BATS_TEST_TMPDIR/a"
  render_consumer consumer-a "$out"
  run diff -r "$FED_INFRA_ROOT/tests/golden/consumer-a" "$out"
  [ "$status" -eq 0 ]
}

@test "consumer-b renders byte-identically to its golden files" {
  local out="$BATS_TEST_TMPDIR/b"
  render_consumer consumer-b "$out"
  run diff -r "$FED_INFRA_ROOT/tests/golden/consumer-b" "$out"
  [ "$status" -eq 0 ]
}

@test "the two consumers render materially different manifests" {
  local a="$BATS_TEST_TMPDIR/a" b="$BATS_TEST_TMPDIR/b"
  render_consumer consumer-a "$a"
  render_consumer consumer-b "$b"
  run diff "$a/mlflow-server.yaml" "$b/mlflow-server.yaml"
  [ "$status" -ne 0 ]
}
```

- [ ] **Step 4: Run to verify it fails, then generate the golden files**

Run: `bats tests/golden.bats`
Expected: FAIL — golden directories do not exist

Generate them, then **read every rendered file** and confirm the namespaces, credentials, and ports are correct before committing:

```bash
STUB_LOG=/dev/null bin/fed-infra-up --env tests/fixtures/consumer-a.env \
  --dry-run --render-dir tests/golden/consumer-a
STUB_LOG=/dev/null bin/fed-infra-up --env tests/fixtures/consumer-b.env \
  --dry-run --render-dir tests/golden/consumer-b
cat tests/golden/consumer-a/*.yaml tests/golden/consumer-b/*.yaml
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `bats tests/`
Expected: PASS — all suites green

- [ ] **Step 6: Add CI**

Create `.github/workflows/ci.yml`:

The nightly job is what catches upstream drift — KFP manifests, image tags, and the argoexec
pin all live outside this repo and can break without any commit here.

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }
  schedule:
    - cron: "0 3 * * *"

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install bats and shellcheck
        run: sudo apt-get update && sudo apt-get install -y bats shellcheck gettext-base
      - name: Lint
        run: shellcheck -x bin/* lib/*.sh tests/stubs/*
      - name: Test
        run: bats tests/

  smoke:
    # Nightly only — brings up a real kind cluster, which is far too slow for PRs.
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    timeout-minutes: 40
    steps:
      - uses: actions/checkout@v4
      - uses: helm/kind-action@v1
        with:
          install_only: true
      - name: Install gettext
        run: sudo apt-get update && sudo apt-get install -y gettext-base
      - name: Bring up the single profile
        run: bin/fed-infra-up --env tests/fixtures/consumer-a.env
      - name: Verify core workloads are running
        run: |
          kubectl get pods -n consumer-a
          kubectl wait --for=condition=available deployment/mlflow-server \
            -n consumer-a --timeout=300s
      - name: Verify the run is idempotent
        run: bin/fed-infra-up --env tests/fixtures/consumer-a.env
      - name: Tear down
        if: always()
        run: bin/fed-infra-down --env tests/fixtures/consumer-a.env
```

- [ ] **Step 7: Commit**

```bash
make check
git add tests/ .github/
git commit -m "test: golden dry-run fixtures, agnosticism guard, and CI"
```

- [ ] **Step 8: Add the remote and record the SHA — do NOT push**

`https://github.com/supat-roong/fed-infra` already exists and is empty. Add it as a remote
so the submodule URL in Tasks 10 and 11 resolves once you choose to push, but leave the
push for later — nothing in this plan leaves the machine.

```bash
git remote add origin https://github.com/supat-roong/fed-infra.git 2>/dev/null || true
git branch -M main
git rev-parse HEAD   # record this SHA — Tasks 10 and 11 pin it
```

Because the remote has no commits yet, Tasks 10 and 11 add the submodule from the **local**
path and then retarget its URL, which is covered in their own steps.

---

## Task 10: Convert `active-fed` to `fed-infra`

**Files:**
- Create: `active-fed/infra.env`
- Create: `active-fed/vendor/fed-infra` (submodule)
- Modify: `active-fed/setup/install_local.sh` (full rewrite)
- Modify: `active-fed/setup/teardown_local.sh` (full rewrite)
- Modify: `active-fed/Makefile:29-33`
- Delete: `active-fed/setup/kind-cluster.yaml`, `active-fed/k8s/minio.yaml`, `active-fed/k8s/mlflow-server.yaml`

**Interfaces:**
- Consumes: `bin/fed-infra-up --env`, `bin/fed-infra-down --env` (Task 8).
- Produces: `make local-setup` and `make local-teardown` with unchanged behaviour from the caller's point of view.

Note `active-fed/k8s/rbac.yaml` is **kept** — it is consumer-specific and stays in the consumer repo.

- [ ] **Step 1: Add the submodule pinned to the Task 9 SHA**

Work happens directly on `main` — the human partner explicitly chose this and authorised no
pushes, so nothing leaves the machine. The submodule is added from the **local** path
because the GitHub remote has no commits yet; the URL is then retargeted so `.gitmodules`
records the eventual public URL.

Note the absolute path and `-c protocol.file.allow=always`. A *relative* submodule URL like
`../fed-infra` is resolved by git against the **origin remote**, not the filesystem — it would
resolve to `github.com/supat-roong/fed-infra` and silently clone the empty remote instead of
your local work. Recent git also refuses `file://` submodule transport unless that config is set.

```bash
cd /Users/supat/workspace/project/active-fed
git -c protocol.file.allow=always submodule add /Users/supat/workspace/project/fed-infra vendor/fed-infra
cd vendor/fed-infra && git checkout <SHA-from-task-9> && cd ../..
git config -f .gitmodules submodule.vendor/fed-infra.url https://github.com/supat-roong/fed-infra.git
git add .gitmodules vendor/fed-infra
```

- [ ] **Step 2: Write `infra.env`**

Values are taken from the current `setup/install_local.sh` and `k8s/*.yaml` so behaviour is preserved exactly.

```sh
# infra.env — consumer contract for vendor/fed-infra
FED_CLUSTER_NAME=active-fed
FED_NAMESPACE=active-fed
FED_PROFILE=single
FED_COMPONENTS=kfp,training,minio,mlflow
FED_KFP_VERSION=2.4.0
FED_TRAINING_OPERATOR_VERSION=v1.7.0
FED_KIND_WORKERS=1

FED_S3_ENDPOINT=minio-service.active-fed.svc.cluster.local:9000
FED_S3_ACCESS_KEY=minioadmin
FED_S3_SECRET_KEY=minioadmin
FED_S3_BUCKET=mlflow-artifacts

FED_IMAGES="active-fed-worker:v1 active-fed-aggregator:v1"

FED_NODEPORT_KFP=30080
FED_NODEPORT_MLFLOW=30500
FED_NODEPORT_MINIO_API=30900
FED_NODEPORT_MINIO_CONSOLE=30901
FED_HOSTPORT_KFP=8080
FED_HOSTPORT_MLFLOW=5050
FED_HOSTPORT_MINIO_API=9000
FED_HOSTPORT_MINIO_CONSOLE=9001
```

- [ ] **Step 3: Rewrite `setup/install_local.sh`**

The consumer keeps ownership of what is genuinely its own: building its images, and applying its own RBAC.

```bash
#!/usr/bin/env bash
# install_local.sh — bootstrap the Active-FL stack via vendor/fed-infra.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building Active-FL images..."
docker build -t active-fed-worker:v1 -f "${ROOT_DIR}/docker/Dockerfile.worker" "${ROOT_DIR}"
docker build -t active-fed-aggregator:v1 -f "${ROOT_DIR}/docker/Dockerfile.aggregator" "${ROOT_DIR}"

"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-up" --env "${ROOT_DIR}/infra.env"

echo "Applying Active-FL RBAC..."
kubectl apply -f "${ROOT_DIR}/k8s/rbac.yaml"

echo "Run 'make local-teardown' to destroy the cluster."
```

- [ ] **Step 4: Rewrite `setup/teardown_local.sh`**

```bash
#!/usr/bin/env bash
# teardown_local.sh — destroy the local cluster via vendor/fed-infra.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-down" --env "${ROOT_DIR}/infra.env"
```

- [ ] **Step 5: Delete the superseded files**

```bash
git rm setup/kind-cluster.yaml k8s/minio.yaml k8s/mlflow-server.yaml
```

- [ ] **Step 6: Verify the dry-run renders correctly**

```bash
vendor/fed-infra/bin/fed-infra-up --env infra.env --dry-run --render-dir /tmp/af-render
cat /tmp/af-render/*.yaml
```

Expected: `namespace.yaml` names `active-fed`; `minio.yaml` carries `minioadmin` credentials; `mlflow-server.yaml` points at `http://minio-service.active-fed.svc.cluster.local:9000` with `nodePort: 30500`.

- [ ] **Step 7: Confirm no Python or pipeline files changed, then commit**

```bash
git status --porcelain | grep -E '\.py$' && echo "UNEXPECTED PYTHON CHANGES" && exit 1
make test        # existing suite must still pass — it is untouched
git add -A
git commit -m "refactor: bootstrap local cluster via fed-infra submodule

Replaces the inlined kind/KFP/MinIO/MLflow setup with vendor/fed-infra,
driven by infra.env. No Python, pipeline, or ML changes."
```

---

## Task 11: Convert `fed-twin` to `fed-infra`

**Files:**
- Create: `fed-twin/infra.env`
- Create: `fed-twin/vendor/fed-infra` (submodule)
- Modify: `fed-twin/setup/install_single_cluster_local.sh` (full rewrite)
- Modify: `fed-twin/setup/teardown_single_cluster_local.sh` (full rewrite)
- Modify: `fed-twin/Makefile:33-37`
- Delete: `fed-twin/setup/kind-single-cluster.yaml`, `fed-twin/k8s/mlflow-server.yaml`, `fed-twin/docker/Dockerfile.mlflow`

**Interfaces:**
- Consumes: `bin/fed-infra-up --env`, `bin/fed-infra-down --env` (Task 8).
- Produces: `make single-cluster-setup` / `make single-cluster-teardown` with unchanged behaviour.

`fed-twin` sets `FED_COMPONENTS=kfp,mlflow` — no standalone MinIO, because it reuses KFP's bundled instance in the `kubeflow` namespace. `FED_NAMESPACE=kubeflow` accordingly. Only the multi-cluster script is left untouched in this phase; it is converted in Phase P3.

- [ ] **Step 1: Add the submodule pinned to the same SHA**

Work happens directly on `main`, with no pushes — same arrangement as Task 10.

```bash
cd /Users/supat/workspace/project/fed-twin
git -c protocol.file.allow=always submodule add /Users/supat/workspace/project/fed-infra vendor/fed-infra
cd vendor/fed-infra && git checkout <SHA-from-task-9> && cd ../..
git config -f .gitmodules submodule.vendor/fed-infra.url https://github.com/supat-roong/fed-infra.git
git add .gitmodules vendor/fed-infra
```

- [ ] **Step 2: Write `infra.env`**

```sh
# infra.env — consumer contract for vendor/fed-infra
FED_CLUSTER_NAME=single-cluster
FED_NAMESPACE=kubeflow
FED_PROFILE=single
FED_COMPONENTS=kfp,training,mlflow
FED_KFP_VERSION=2.4.0
FED_TRAINING_OPERATOR_VERSION=v1.7.0
FED_KIND_WORKERS=0

FED_S3_ENDPOINT=minio-service.kubeflow.svc.cluster.local:9000
FED_S3_ACCESS_KEY=minio
FED_S3_SECRET_KEY=minio123
FED_S3_BUCKET=mlflow-artifacts

FED_IMAGES="fed-twin-app:v1"

FED_NODEPORT_KFP=30080
FED_NODEPORT_MLFLOW=30500
FED_NODEPORT_MINIO_API=30900
FED_NODEPORT_MINIO_CONSOLE=30901
FED_HOSTPORT_KFP=8080
FED_HOSTPORT_MLFLOW=5050
FED_HOSTPORT_MINIO_API=9000
FED_HOSTPORT_MINIO_CONSOLE=9001
```

- [ ] **Step 3: Rewrite `setup/install_single_cluster_local.sh`**

The `pipeline-runner-extend` ClusterRoleBinding and the KFP-MinIO NodePort are consumer concerns and stay here. The MinIO NodePort patch is needed because `fed-twin` exposes KFP's MinIO rather than a standalone one, so `fed-infra`'s `minio` component does not run.

```bash
#!/usr/bin/env bash
# install_single_cluster_local.sh — bootstrap the Fed-Twin stack via vendor/fed-infra.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing uv environment..."
command -v uv >/dev/null 2>&1 && uv sync || echo "uv not found, skipping sync"

echo "Building Fed-Twin image..."
docker build -t fed-twin-app:v1 -f "${ROOT_DIR}/docker/Dockerfile.app" "${ROOT_DIR}"

"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-up" --env "${ROOT_DIR}/infra.env"

echo "Granting the KFP default service account cluster-admin..."
kubectl create clusterrolebinding pipeline-runner-extend \
  --clusterrole=cluster-admin --serviceaccount=kubeflow:default \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Exposing KFP MinIO..."
kubectl patch service minio-service -n kubeflow --type=json -p='[
  {"op":"replace","path":"/spec/type","value":"NodePort"},
  {"op":"replace","path":"/spec/ports","value":[
    {"name":"api","port":9000,"protocol":"TCP","targetPort":9000,"nodePort":30900},
    {"name":"console","port":9001,"protocol":"TCP","targetPort":9001,"nodePort":30901}]}]'

echo "Setup complete."
```

- [ ] **Step 4: Rewrite `setup/teardown_single_cluster_local.sh`**

```bash
#!/usr/bin/env bash
# teardown_single_cluster_local.sh — destroy the local cluster via vendor/fed-infra.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-down" --env "${ROOT_DIR}/infra.env"
```

- [ ] **Step 5: Delete the superseded files**

```bash
git rm setup/kind-single-cluster.yaml k8s/mlflow-server.yaml docker/Dockerfile.mlflow
```

- [ ] **Step 6: Verify the dry-run renders correctly**

```bash
vendor/fed-infra/bin/fed-infra-up --env infra.env --dry-run --render-dir /tmp/ft-render
ls /tmp/ft-render
cat /tmp/ft-render/mlflow-server.yaml
```

Expected: **no** `minio.yaml` is produced; `mlflow-server.yaml` uses namespace `kubeflow`, credentials `minio`/`minio123`, and image `fed-mlflow:2.12.2`.

- [ ] **Step 7: Confirm no Python changes, then commit**

```bash
git status --porcelain | grep -E '\.py$' && echo "UNEXPECTED PYTHON CHANGES" && exit 1
PYTHONPATH=src uv run pytest tests/ -q
git add -A
git commit -m "refactor: bootstrap single-cluster via fed-infra submodule

Replaces the inlined kind/KFP/MLflow setup with vendor/fed-infra. The
mlflow+boto3 image is now built by fed-infra, so Dockerfile.mlflow is
removed. Multi-cluster setup is unchanged and converts in a later phase."
```

---

## Task 12: Phase gate — real-cluster smoke test for both consumers

**Files:**
- Create: `fed-infra/README.md` (usage and the consumer contract table)
- Modify: `active-fed/README.md` (setup section references the submodule)
- Modify: `fed-twin/README.md` (setup section references the submodule)

**Interfaces:**
- Consumes: everything above. Produces no code — this is the gate that authorises Phase P1.

This task is a real-cluster verification, not a unit test. It needs Docker running with at least 8 GB of memory available.

- [ ] **Step 1: Verify `active-fed` from a clean slate**

```bash
cd active-fed
make local-teardown || true
git submodule update --init --recursive
make local-setup
```

Expected: completes without error. Then confirm:

```bash
kubectl get pods -n active-fed          # minio + mlflow-server Running
kubectl get pods -n kubeflow            # ml-pipeline, ml-pipeline-ui, workflow-controller Running
curl -sf -o /dev/null http://localhost:8080 && echo "KFP UI OK"
curl -sf -o /dev/null http://localhost:5050 && echo "MLflow UI OK"
curl -sf -o /dev/null http://localhost:9001 && echo "MinIO console OK"
```

- [ ] **Step 2: Verify idempotency**

```bash
make local-setup   # second run against the existing cluster
```

Expected: completes without error; log shows "already exists" / "already installed" / "already current" rather than re-creating anything. Confirm no duplicate container ports were appended:

```bash
kubectl get deployment minio -n kubeflow \
  -o jsonpath='{.spec.template.spec.containers[0].ports}'; echo
```

Expected: exactly two ports, `9000` and `9001`.

- [ ] **Step 3: Tear down and verify `fed-twin` from a clean slate**

```bash
cd active-fed && make local-teardown
cd ../fed-twin
make single-cluster-teardown || true
git submodule update --init --recursive
make single-cluster-setup
```

Expected: completes without error. Then confirm:

```bash
kubectl get pods -n kubeflow            # mlflow-server + KFP core Running
curl -sf -o /dev/null http://localhost:8080 && echo "KFP UI OK"
curl -sf -o /dev/null http://localhost:5050 && echo "MLflow UI OK"
```

- [ ] **Step 4: Run one real fed-twin pipeline end to end**

This is the strongest available signal that the extraction did not break anything.

```bash
./run_pipeline.sh single_twin_single_cluster
```

Expected: the run reaches `Succeeded` and a metrics CSV appears under `metrics/`.

- [ ] **Step 5: Write `fed-infra/README.md`**

Must document: the `infra.env` contract (every `FED_*` variable, its default, and its meaning); the `FED_COMPONENTS` list; the repo-agnosticism rule and why it exists; how to run the dry-run; and how consumers pin and bump the SHA.

- [ ] **Step 6: Update both consumer READMEs**

In `active-fed/README.md`, the setup section gains `git submodule update --init --recursive` before `make local-setup`. Same for `fed-twin/README.md` before `make single-cluster-setup`.

- [ ] **Step 7: Commit locally — do NOT push, do NOT open PRs**

The human partner authorised local commits only. No `git push`, no `gh pr create`, no other
outward-facing command anywhere in this plan. Pushing is a separate decision they make later.

```bash
cd /Users/supat/workspace/project/fed-infra
git add README.md && git commit -m "docs: infra.env contract and usage"

cd /Users/supat/workspace/project/active-fed
git add README.md && git commit -m "docs: note submodule init in setup"

cd /Users/supat/workspace/project/fed-twin
git add README.md && git commit -m "docs: note submodule init in setup"
```

Verify nothing was pushed:

```bash
for r in fed-infra active-fed fed-twin; do
  echo "== $r =="
  git -C "/Users/supat/workspace/project/$r" status -sb | head -1
done
```

Expected: each reports `## main` with no `[ahead N]` tracking a pushed remote — `fed-infra`
has an `origin` configured but no upstream branch, and the two consumers show unpushed
commits ahead of their origin/main.

- [ ] **Step 8: Confirm the phase gate**

P0 is complete only when **all** of these hold:

- `fed-infra`: `make check` green (shellcheck + all bats suites, including the agnosticism guard).
- `active-fed`: `make local-setup` green twice in a row; `make test` green.
- `fed-twin`: `make single-cluster-setup` green twice in a row; `pytest` green; one pipeline run reaches `Succeeded`.
- Neither consumer has any `.py` diff on its branch.

Phase P1 (Temporal) MUST NOT start until this gate passes.

---

## Deferred to later phases

These are in the spec but deliberately out of scope for P0, listed here so no one implements them early:

- `lib/temporal.sh`, the Temporal worker Deployment, and `src/orchestration/` → **P1**
- `restartPolicy: OnFailure`, KFP `.set_retry()`, round-0 global init, deterministic job naming, `start_round` resume → **P2**
- `lib/karmada.sh`, multi-cluster kind templates, `PropagationPolicy` dispatch, and converting `fed-twin/setup/install_multi_cluster_local.sh` → **P3**
- `lib/dashboard.sh`, Kubernetes Dashboard, Karmada Dashboard, MLflow cross-link tags → **P4**
