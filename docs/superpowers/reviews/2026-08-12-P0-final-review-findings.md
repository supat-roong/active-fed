# Final whole-branch review — fed-infra foundation (P0)

Scope: `fed-infra` (21 commits, entire history), `active-fed` (5 commits), `fed-twin` (4 commits),
reviewed as one body of work against `docs/superpowers/plans/2026-08-12-fed-infra-foundation.md`.

Working trees confirmed clean and identical to the review packages (`fed-infra` and `fed-twin`
have empty `git status`; `active-fed` has only the pre-existing uncommitted `.gitignore` +
untracked `docs/`, deferred minor #14). All 69 bats tests re-run green during this review.

Items listed as "known and accepted" in the review brief (nothing pushed, `fed-twin`'s untracked
`pipeline_specs/`+`metrics/`, the header-only metrics CSV, later-phase Karmada/Temporal/dashboards)
are not repeated as findings.

---

## Verdict summary

| Question | Answer |
|---|---|
| Delivers stated purpose | **Yes**, with one residual duplication (Important 3) and one collateral break (Important 2) |
| Cross-cutting consistency | Good. Naming, idempotency and dry-run guarding are consistent to a fault; the only real gaps are the *down* path and two unguarded command substitutions |
| Agnosticism invariant holds | **Yes** — verified independently and more broadly than the test does |
| Counts | 0 Critical, 3 Important, 10 Minor |
| Deferred-minor triage | 2 fix before merge, 13 safe to defer |
| Merge recommendation | **Fix first** — Important 1 and 2 are each a small, contained edit |

---

## 1. Cross-cutting correctness of the ten library modules

**What is genuinely consistent, and worth saying plainly:**

- **Naming.** Every function is `fed_*`, every variable `FED_*`, every module named for its
  component. There are no stragglers: the `a230104 fix: rename stale fl- literals to fed-`
  commit caught the last of them, and a fresh grep finds no `fl_`/`fl-` identifiers left in
  `lib/`, `bin/`, `manifests/` or `kind/`.
- **Idempotency.** Every install function has an existence probe that returns 0 (`kfp.sh:15`,
  `training.sh:10`, `mlflow.sh:10`, `kind.sh:10`, `kind.sh:52`). `fed_kfp_patch_minio` uses
  `replace` rather than `add` on both the ports and args arrays, which is what makes repeated
  runs converge instead of accumulating ports. `fed_minio_ensure_bucket` deletes any leftover
  pod *before* it runs, so it is recoverable even if a prior run was interrupted mid-flight.
- **The sourced-library-must-not-set-shell-options rule.** Held everywhere. `lib/common.sh:5-10`
  documents it, `tests/common.bats:10` is a real regression test for it (it asserts `$-` contains
  neither `e` nor `u` after sourcing), and no `lib/*.sh` file contains `set -`. Both `bin/*`
  entrypoints and all four `tests/stubs/*` set their own `set -euo pipefail`. This is one of the
  better-executed invariants in the repo.
- **The `[ cond ] && return` prohibition** is not just documented in `lib/minio.sh:5-7`, it is
  enforced by a second test in `tests/agnostic.bats:15`. Good: a comment that would have rotted
  is instead a guard.
- **Dry-run guarding on the up path** is complete. Every mutating function checks
  `"${FED_DRY_RUN:-0}" = "1"` and returns early, including `kubectl config use-context` in
  `components.sh:8-10`, which is easy to forget.

**Contradictions and drift found:** see Important 1 and Minors 5, 6, 8, 9 below. There is no
module that contradicts another in behaviour. The only duplicated logic that has drifted is the
`FED_KFP_NAMESPACE` constant (Minor 6), and it is currently harmless because `kfp.sh` is always
sourced before `training.sh`.

---

## 2. Does it deliver the stated purpose?

Yes. The measure is whether the *next* fix to the bootstrap lands in one place, and for almost
everything it now does. Concretely, comparing the two old install scripts against the library:

| Previously duplicated (and divergent) | Now |
|---|---|
| kind cluster + NodePort→hostPort mappings | `kind/single-cluster.yaml.tpl`, parameterised |
| KFP install + `crd/applications.app.k8s.io` wait | `fed_kfp_install` |
| Four ARM/ghcr image patches + argoexec pin | `fed_kfp_patch_arm` |
| KFP-MinIO image/ports/args patches | `fed_kfp_patch_minio`, with the `add`→`replace` fix that only `active-fed` never had and `fed-twin` had wrong |
| `mlpipeline` bucket provisioning via `mc` pod | `fed_minio_ensure_bucket` |
| MLflow image (`active-fed` did `pip install` at pod start, `fed-twin` prebuilt) | `fed_mlflow_build_image` — standardised on the prebuilt approach for both |
| Digest-checked `kind load` (only `fed-twin` had it) | `fed_kind_load_image` — `active-fed` gains it |
| Rollout waits | `fed_kfp_wait`, `fed_minio_install`, `fed_mlflow_install` |
| `pkill` of stale port-forwards on teardown (only `active-fed` had it) | `fed_down` — `fed-twin` gains it |

Three genuine divergences were resolved *in favour of the better version*, which is the strongest
evidence the extraction was real rather than cosmetic. Both teardown scripts are now byte-for-byte
equivalent modulo the filename.

The `FED_COMPONENTS` design does carry the two consumers' different needs without a single
conditional on consumer identity, exactly as intended.

**What is still duplicated between the two consumers that arguably should have moved:**

1. **The KFP-MinIO NodePort exposure** — Important 3. This is the one real miss.
2. **The KFP service-account RBAC grant.** `active-fed` applies `k8s/rbac.yaml` (a scoped
   ClusterRole for `pytorchjobs`/pods/jobs); `fed-twin` inline-creates
   `clusterrolebinding pipeline-runner-extend --clusterrole=cluster-admin`. Same underlying need
   ("let KFP pipeline pods create PyTorchJobs"), two implementations. I do **not** recommend
   moving this into the library: the two are deliberately different security postures, and a
   library that granted `cluster-admin` for everyone would be worse. Correctly left as a consumer
   concern — but worth a line in each consumer's README saying so, since it is now the only piece
   of cluster bootstrap left outside `fed-infra`.
3. **Image tags declared three times per consumer** — Minor 12.

**What is in the library that is really only one consumer's concern:** nothing. I checked each
candidate. The `minio` component is genuinely optional and only `active-fed` enables it. The
`--console-address :9001` arg in `fed_kfp_patch_minio` is only *used* by `fed-twin` but is
harmless and generic. `FED_NODEPORT_MINIO_*` defaults exist for a component either consumer could
enable. The kind template's unconditional MinIO host-port claims are the one thing that leaks a
minio assumption into a minio-less consumer (Minor 13), and even that is a wart rather than a
consumer-specific special case.

---

## 3. The agnosticism invariant

**It holds, and it holds more strongly than the test proves.** I ran the invariant grep directly
and then went well beyond it:

```
grep -rIl --exclude-dir=.git --exclude=agnostic.bats -e 'active-fed' -e 'fed-twin' .   → no matches
grep -rIn --exclude-dir=.git -i -E 'active[-_]?fed|fed[-_]?twin|digital[-_]twin|karmada|twin|supat|aktsk' .
    → exactly one hit: tests/agnostic.bats:8, the test's own pattern
```

No consumer name, no product term (`digital-twin`, `karmada`, `twin`), no author or org identifier
appears anywhere in the tree. `tests/fixtures/*.env` and `tests/golden/*` use `consumer-a` /
`consumer-b`, and the README's examples do too, deliberately (README:148-150). The
`kind/single-cluster.yaml.tpl` filename coinciding with `fed-twin`'s `FED_CLUSTER_NAME=single-cluster`
is coincidence on a generic term, not a leak.

**Does `--exclude=agnostic.bats` leave a meaningful hole?** Two answers:

- *The literal hole is trivial.* `--exclude` matches by basename anywhere in the tree, so a second
  file named `agnostic.bats` in any subdirectory would also be skipped. Nobody is going to do
  that. If you want it airtight for free, `--exclude-dir` is not needed — building the patterns
  from concatenated fragments (`-e "active""-fed"`) would let the file check itself. Not worth a
  commit on its own.
- *The hole that actually exists is elsewhere, and that exclusion is not what causes it.*
  `--exclude-dir=.git` means the guard can never see **commit messages**, and commit `b31aefe`'s
  body reads *"Both existing consumers (fed-twin, active-fed) waited…"* and *"active-fed's
  original install_local.sh applied the Training Operator…"*. So the shipped repo does carry
  consumer identity — in its history, not its tree. No code branches on it and nothing behaves
  differently, so this is a provenance blemish rather than a correctness problem, and it is
  unfixable without rewriting history. Worth knowing before the repo is ever made public.

A third, subtler point: the guard enforces **two string literals**, while README:146 states the
rule as "no consumer repo name, product name, or business-specific identifier". The test is
narrower than the stated invariant and does not generalise to a third consumer. That is fine for
today (I verified the broader property by hand above) but the test will need editing, not just
extending, when a third consumer appears.

---

## 4. Findings

### Critical

None.

### Important

**I1 — `fed_kind_load_image` aborts silently under the strict mode every entrypoint sets; its
own error message is unreachable in production.**
`fed-infra/lib/kind.sh:47-48` and `:51`

```bash
local_id=$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null | cut -d: -f2 | cut -c1-12)
[ -n "$local_id" ] || fed_die "image not found locally: $image (build it first)"
```

Both `bin/fed-infra-up` and `bin/fed-infra-down` set `set -euo pipefail`. When the image is not
present locally, `docker image inspect` exits 1; `pipefail` propagates that to the whole pipeline;
the assignment therefore returns non-zero and `set -e` kills the script **on the assignment line**,
before the `fed_die` on the next line can run. The carefully written
`"image not found locally: … (build it first)"` message is dead code in production.

Verified empirically:

```
$ bash -c 'set -euo pipefail; . lib/common.sh; . lib/config.sh; . lib/kind.sh; fed_config_defaults
           echo before; fed_kind_load_image definitely-not-a-real-image:v9 somecluster; echo after'
before
EXIT=1          # no message at all
```

Line 51 has the same shape with a worse failure mode: if `docker exec <cluster>-control-plane
crictl images` fails — the control-plane container stopped or removed, e.g. after a colima
restart, with the cluster still listed by `kind get clusters` — `cluster_id=$(…)` aborts the whole
setup instead of falling through to `kind load docker-image`, which is the correct recovery.
Verified the same way (exit 1, no output).

Why no test caught it: bats calls these functions directly, without `set -e`, so
`tests/kind.bats` exercises exactly the path production never takes. This is precisely the class
of thing a per-task review cannot see — the module is correct in isolation and only misbehaves
under the entrypoint's shell options.

The user-visible symptom is a bare `exit 1` with zero diagnostics from the tool whose entire
purpose is to make the bootstrap less mysterious. Both consumers' install scripts build their
images first so the happy path is unaffected, but README:189 documents running
`vendor/fed-infra/bin/fed-infra-up --env infra.env` directly, which walks straight into it.

Fix (two lines):
```bash
local_id=$(docker image inspect … | cut -d: -f2 | cut -c1-12) || local_id=""
cluster_id=$(fed_kind_cluster_image_id "$image" "$cluster") || cluster_id=""
```
Add a bats case that runs the function through `bash -c 'set -euo pipefail; …'` so the regression
is covered under the real shell options.

---

**I2 — this work deleted two files that `fed-twin`'s untouched multi-cluster script still
consumes, so `make multi-cluster-setup` now fails immediately.**
`fed-twin/setup/install_multi_cluster_local.sh:160` and `:217`

```
line 160: docker build -t local-mlflow-boto3:v2.12.2 -f docker/Dockerfile.mlflow .   → file deleted
line 217: kubectl apply -f k8s/mlflow-server.yaml                                    → file deleted
```

Both files were removed by commit `d5197ba` (as the plan directed, plan:2093). The multi-cluster
script has `set -euo pipefail` at line 5, so it now dies at line 160 — after creating the Karmada
host cluster and member clusters, leaving a half-built multi-cluster environment behind.

The plan says the multi-cluster path is "left untouched in this phase" (plan:2002) and "unchanged
and converts in a later phase" (plan:2116). *Unchanged* was achieved; *still working* was not, and
the plan itself did not notice that its own deletion list intersected the script it was promising
not to disturb. `fed-twin/README.md` was edited in this branch and still offers
`make multi-cluster-setup` as a documented alternative directly beneath the single-cluster
instructions, so a user is actively pointed at the broken command.

No per-task review could have caught this: the task that deleted the files was scoped to the
single-cluster path, and the task that edited the README was scoped to submodule-init prose.

Fix — pick one, all small:
- Restore the two files (they are 2 and 70 lines) purely for the multi-cluster path, with a
  comment that P3 deletes them; **or**
- Point line 160/217 at the library (`vendor/fed-infra` can build the image and render the
  manifest against a multi-cluster `infra.env`); **or**
- At minimum, make the script fail fast with a clear message and add a one-line note to
  `fed-twin/README.md` that multi-cluster is temporarily out of service pending P3.

The last option is acceptable only if the human partner is content to have the documented command
be broken between P0 and P3.

---

**I3 — `fed-twin` still hand-rolls the KFP-MinIO NodePort exposure, duplicating
`fed_expose_nodeport` and silently ignoring its own `infra.env`.**
`fed-twin/setup/install_single_cluster_local.sh:20-25`

```bash
kubectl patch service minio-service -n kubeflow --type=json -p='[
  {"op":"replace","path":"/spec/type","value":"NodePort"},
  {"op":"replace","path":"/spec/ports","value":[
    {"name":"api", …,"nodePort":30900},
    {"name":"console", …,"nodePort":30901}]}]'
```

`fed-twin/infra.env:19-20` declares `FED_NODEPORT_MINIO_API=30900` and
`FED_NODEPORT_MINIO_CONSOLE=30901`, and those values *are* consumed — but only by
`kind/single-cluster.yaml.tpl`, which builds the host→node port mapping. The service patch that
puts the service on those node ports hardcodes them. The two agree today by coincidence of both
being 30900/30901.

Change `FED_NODEPORT_MINIO_CONSOLE` in `fed-twin/infra.env` and the kind cluster maps host 9001 to
node port 30902 while the service still publishes 30901: the console silently stops resolving,
with nothing in either file to indicate why. That is the same failure shape — one bootstrap fact
stated in two places, drifting apart — that this whole plan exists to eliminate, reintroduced in
the consumer that was supposed to be the proof the design works.

The root cause is a gap in the component model: the `minio` component conflates *deploy a
standalone MinIO* with *expose a MinIO service on the configured node ports*. `fed-twin` needs the
second without the first, and `FED_COMPONENTS` has no way to say that, so it reimplements it.

Note this is a latent desync, not a live bug — the cluster works today. Ranked Important because
it is a direct hit on the plan's stated purpose, and because the fix is genuinely small.

Fix options, in order of preference:
1. Have the library expose the KFP-bundled MinIO whenever `kfp` is enabled and
   `FED_NODEPORT_MINIO_*` are set — one more `fed_expose_nodeport` call in the `kfp` branch of
   `components.sh`, guarded so `active-fed` (whose own minio-service lives in a different
   namespace) is unaffected.
2. Split a `minio-expose` sub-behaviour out of the `minio` component.
3. Cheapest stopgap: have `fed-twin`'s script read the values from `infra.env` rather than
   hardcoding them, so at least there is one source of truth.

### Minor

**M1 — both consumers are pinned one commit behind `fed-infra` HEAD, and the missing commit is the
entire contract documentation.**
`active-fed/vendor/fed-infra`, `fed-twin/vendor/fed-infra` → both `b31aefe`; `fed-infra` HEAD is
`d0e8dc4 docs: infra.env contract and usage` (README.md, +213/-9, docs only).
Functionally irrelevant — `b31aefe` is the newest functional commit — but it means the
`vendor/fed-infra/README.md` that a developer opens inside either consumer is the stub version,
without the `infra.env` variable tables, the `FED_COMPONENTS` reference, the dry-run section, or
the submodule pin/bump instructions. Since documenting the contract was a deliverable of this
plan, the deliverable is not in what the consumers actually vendor. Fix: bump both pins to
`d0e8dc4` (two one-line commits).

**M2 — the teardown path is the only place dry-run guarding is missing.**
`lib/components.sh:73-77` (`fed_down`), `lib/kind.sh:61-65` (`fed_kind_delete_cluster`).
Every function on the up path guards on `FED_DRY_RUN`; neither of these does. `bin/fed-infra-down`
has no `--dry-run` flag, but `FED_DRY_RUN` is a documented `infra.env` variable
(`config.sh:33`, README:74) and an exported environment variable, so
`FED_DRY_RUN=1 fed-infra-down --env infra.env` really deletes the cluster. Either guard both
functions or document that dry-run is an up-path-only concept.

**M3 — `FED_KFP_NAMESPACE` has two sources of truth.**
`lib/kfp.sh:6` sets it unconditionally (`FED_KFP_NAMESPACE=kubeflow`); `lib/training.sh:24` re-defaults
it (`-n "${FED_KFP_NAMESPACE:-kubeflow}"`). Harmless today because `bin/*` always sources `kfp.sh`
before `training.sh`, but it is the same constant written twice, which is the drift pattern this
project exists to remove. Also note `kfp.sh:6` uses a bare assignment rather than `:=`, so it
overrides any consumer value — while `training.sh` respects one. `tests/training.bats:26` asserts
the override works, so the two modules disagree about whether the variable is overridable.

**M4 — a failed `uv sync` is reported as "uv not found".**
`fed-twin/setup/install_single_cluster_local.sh:8`
```bash
command -v uv >/dev/null 2>&1 && uv sync || echo "uv not found, skipping sync"
```
The `A && B || C` shape means C also runs when B fails. A genuine dependency-resolution failure
prints "uv not found, skipping sync" and setup continues to build a Docker image against a stale
environment. Use an `if` block.

**M5 — mixed error-propagation convention across modules.**
`kfp.sh` and `training.sh` append `|| return 1` to every command; `kind.sh`, `minio.sh`,
`mlflow.sh`, `nodeport.sh` rely on the caller's `set -e`. Both work in production. The difference
is that only the first style makes a function fail correctly when called *without* `set -e` —
which is exactly how bats calls them. `tests/kfp.bats:58` can test "stops on the first failed
rollout"; no equivalent test is possible for `fed_minio_install` or `fed_mlflow_install` as
written. Worth standardising on the explicit style, both for testability and because Important 1
is a symptom of the same ambiguity about who owns error handling.

**M6 — no test asserts that every `${FED_*}` in a template is in the substitution whitelist.**
`lib/render.sh:7-11`. I verified the invariant holds today — all 15 variables referenced across
`kind/*.tpl` and `manifests/*.tpl` are whitelisted, with `FED_KIND_WORKERS` and
`FED_MLFLOW_VERSION` whitelisted but unused (harmless). This is the structural fix for deferred
minor #2 and is about three lines of bats:

```bash
@test "every template variable is in the substitution whitelist" {
  for v in $(grep -rhoE '\$\{FED_[A-Z0-9_]+\}' "$FED_INFRA_ROOT"/kind/*.tpl "$FED_INFRA_ROOT"/manifests/*.tpl | sort -u); do
    [[ "$FED_TEMPLATE_VARS" == *"$v"* ]] || { echo "not whitelisted: $v" >&2; return 1; }
  done
}
```

Also note `FED_KFP_NAMESPACE` is not whitelisted, and is at least as likely as
`FED_KFP_VERSION` (deferred #2) to be wanted by a future template — a template referencing it
would render `namespace: ` and apply into `default`.

**M7 — a partially applied KFP install can never be repaired by re-running.**
`lib/kfp.sh:15`. The existence probe is a single deployment (`get deploy ml-pipeline`). If the
cluster-scoped apply succeeded and the core apply died halfway — plausible on a network blip
mid-`kubectl apply -k` against GitHub — a subsequent run reports "KFP already installed", skips
the reinstall, and proceeds to patch a broken installation. Recovery requires deleting the
cluster. Inherited verbatim from both old scripts (`active-fed/setup/install_local.sh` and
`fed-twin/setup/install_single_cluster_local.sh` had the same probe), so not a regression, and
`fed_kfp_wait` at least fails loudly rather than silently. Listed because the library's stated
contract is idempotency, and this is the one place where "run it again" does not converge.

**M8 — Makefile asymmetry between the two consumers.**
`active-fed/Makefile:29-35` runs `git submodule update --init --recursive` in both `local-setup`
and `local-teardown`; `fed-twin/Makefile:33-37` does not, relying on README prose only. Same
integration, two behaviours. A fresh `fed-twin` clone runs `uv sync` and a full `docker build`
before failing at line 13 with "no such file or directory". (The submodule init failing for other
reasons is the known-accepted not-pushed item; the asymmetry is separate.)

**M9 — consumer image tags are declared in three places.**
E.g. `active-fed`: `setup/install_local.sh:8-9` (`docker build -t`), `Makefile:38-44`
(`build-images`/`load-images`), and `infra.env:15` (`FED_IMAGES`). `fed-twin` likewise across
`setup/install_single_cluster_local.sh:11`, `Makefile:47-51`, `infra.env:15`. The `Makefile`
`load-images` targets additionally hardcode the cluster name that `infra.env` also declares. Drift
surfaces as a `fed_kind_load_image` failure — which, per Important 1, is currently silent.

**M10 — two cosmetic gaps in the up-path summary and the kind template.**
`lib/components.sh:68-70` prints the MinIO console URL only when the `minio` component is enabled,
so `fed-twin` — which does expose a working MinIO console on 9001 — never sees it mentioned.
`kind/single-cluster.yaml.tpl:13-17` unconditionally claims host ports 9000 and 9001 even for a
consumer with no `minio` component; port 9000 is commonly occupied on a developer laptop, and
`kind create cluster` fails outright if it is.

---

## 5. Things a per-task review could not see — explicit checks performed

| Check | Result |
|---|---|
| Ordering dependencies between components in `fed_up` | **Sound.** minio→mlflow (endpoint exists before the server that uses it); `fed_kfp_wait` before the `mlpipeline` bucket (the bundled MinIO is rolled out before `mc` targets it); `fed-twin`'s mlflow bucket is provisioned after `fed_kfp_wait`, so its dependency on the *kubeflow*-namespace MinIO is satisfied. `namespace.yaml.tpl` is applied by both `fed_minio_install` and `fed_mlflow_install` — duplicate but idempotent and correct, since either component can be enabled alone. |
| `infra.env` variable declared but never consumed | `FED_PROFILE` — validated as `single\|multi` (`config.sh:50-53`) but never branched on, so `FED_PROFILE=multi` silently builds a single cluster. Documented as such at README:52; acceptable, but consider `fed_die` on `multi` until P3 implements it. `FED_KIND_WORKERS` and `FED_MLFLOW_VERSION` are whitelisted for substitution but used only in shell, which is harmless. In `fed-twin/infra.env`, `FED_NODEPORT_MINIO_*` are consumed by the kind template but not by the service patch — see Important 3. |
| `infra.env` variable consumed but never declared | `FED_S3_ENDPOINT` / `FED_S3_ACCESS_KEY` / `FED_S3_SECRET_KEY` — see deferred minor #3, promoted to fix-before-merge below. Both real consumers declare all three. |
| Template variable missing from the whitelist | None today; all 15 referenced variables are whitelisted. No test enforces it — Minor 6. |
| Error paths leaving a half-configured cluster | The general design is sound: `fed_up` is a linear sequence under `set -e`, every step is idempotent, and recovery is "run it again". Three exceptions: Important 1 (aborts with no diagnostic), Minor 7 (partial KFP install never converges), and deferred #3 (missing `FED_S3_ENDPOINT` deploys a live MLflow with `MLFLOW_S3_ENDPOINT_URL: "http://"` and *then* dies on an unbound variable). Deferred #8 (`kubectl run` failure skips the trailing pod cleanup) is self-healing, since the function deletes leftovers on entry. |
| `set -e` / `pipefail` interaction across the sourced/executed boundary | This is where Important 1 lives. Worth a targeted sweep: the only two command substitutions over pipelines in `lib/` are `kind.sh:47` and `kind.sh:51`, and both are affected. `minio.sh:23` (`tr -cd`) and `mlflow.sh:16` (`mktemp -d`) are safe. |
| Consumer-side behaviour silently dropped in the conversion | Checked both old scripts line by line against the library. Everything is accounted for except the `ingress-ready=true` node label (deferred #12, verified unused) and `active-fed`'s `kubectl cluster-info` echo (replaced by `kubectl config use-context`). `fed-twin` additionally *gains* the argoexec pin's `|| return 1` fail-fast and the `pkill` teardown; `active-fed` gains digest-checked image loading. |
| Working tree vs. review package | Match, modulo the pre-existing `active-fed` `.gitignore`/`docs/` (deferred #14). |

---

## 6. Deferred-minor triage

| # | Task | Item (abbreviated) | Verdict | Reasoning |
|---|---|---|---|---|
| 1 | 1 | `fed_require_cmd succeeds` passes with or without the stub PATH | **Safe to defer** | The assertion is weak but the companion die-path test (`common.bats:33`) is strong and is the half that can actually regress. |
| 2 | 3 | `FED_KFP_VERSION` defaulted but absent from `FED_TEMPLATE_VARS` | **Safe to defer** | No template references it today. Fix the *class* rather than the instance via the whitelist-coverage test in Minor 6, and add `FED_KFP_NAMESPACE` while you are there. |
| 3 | 3 | `FED_S3_ENDPOINT/ACCESS_KEY/SECRET_KEY` whitelisted, no defaults, not in `FED_REQUIRED_VARS` | **Fix before merge** | The only deferred item that leaves a cluster half-configured. Omit `FED_S3_ENDPOINT` with `mlflow` enabled and `envsubst` (which does not honour `set -u`) renders `MLFLOW_S3_ENDPOINT_URL: "http://"` into a **live, successfully rolled-out** Deployment; the run then dies at `components.sh:47` with bash's raw `FED_S3_ENDPOINT: unbound variable` rather than a `fed_die`. The user is left with a running-but-broken MLflow and a cryptic message. ~4 lines: validate the three variables in `fed_config_validate` when `fed_has_component mlflow`. |
| 4 | 4 | kind worker-append test asserts only exit 0; the stub drops piped stdin | **Fix before merge** | See the blind-spot note below. `FED_KIND_WORKERS` is the single structural difference between the two consumers' clusters and has zero automated coverage. Fix is small: have `tests/stubs/kind` append stdin to the log when `--config -` is passed, then assert `role: worker` appears exactly `FED_KIND_WORKERS` times. That one change also unlocks a golden file for the rendered kind config. |
| 5 | 4 | `${image%:*}`/`${image#*:}` misparses ported registries | **Safe to defer** | Fails safe (redundant reload, never a stale-skip), and no current image has a port. Note it interacts with Important 1: after that fix, a misparse costs a redundant `kind load`, which is the benign outcome. |
| 6 | 5 | bucket pod name could degenerate via `tr -cd 'a-z0-9'` | **Safe to defer** | Requires a bucket name with no alphanumerics; both real buckets are fine and the failure is an immediate, loud `kubectl run` rejection. |
| 7 | 5 | "deletes pod before and after" asserts only `grep -c >= 2` | **Safe to defer** | Ordering is correct by inspection and the before-delete is what makes the function recoverable; a regression would show up as a live `AlreadyExists` failure on the next run. |
| 8 | 5 | `kubectl run` unguarded, could skip trailing cleanup under `set -e` | **Safe to defer** | Self-healing: the function's first action deletes any leftover pod, so the next run recovers. Consistent with the module convention (see Minor 5 for the broader point). |
| 9 | 6 | dry-run test checks only `mlflow-server.yaml`, not `namespace.yaml` | **Safe to defer** | `minio.bats:26` asserts `namespace.yaml` on the parallel path, and `render.bats:54` covers the template itself. |
| 10 | 7 | `kfp.bats` test 1 asserts neither apply ordering nor the CRD wait between them | **Safe to defer** | Ordering is load-bearing but a regression is immediately fatal on any real run (the core manifests reference CRDs from the cluster-scoped apply), and the nightly smoke job exercises it. |
| 11 | 8 | 2 of 6 `refute_called` assertions in the dry-run test do not discriminate | **Safe to defer** | Real gap, zero risk. When touching `components.bats` for #4, replace the six refutes with `[ ! -s "$STUB_LOG" ]` — strictly stronger, one line, and it makes all six discriminate at once. |
| 12 | 10 | kind `ingress-ready=true` node label dropped | **Safe to defer** | Verified unused; no ingress controller is installed by either consumer. |
| 13 | 10 | MLflow rollout wait 180s → 300s | **Safe to defer** | Strictly more lenient; the only cost is a slower failure on a genuinely broken deploy. |
| 14 | 11 | `active-fed`'s pre-existing uncommitted `.gitignore` + `docs/` left alone | **Safe to defer** | Correct call by the implementer — out of scope, and I re-confirmed it is the *only* working-tree deviation across all three repos. |
| 15 | 12 | metrics CSV header-only | **Safe to defer** | Pre-existing race in `fed-twin`'s own pipeline code (last touched by `7e347c6`, predating every conversion commit); explicitly out of scope, and the metrics were confirmed via the MLflow API instead. |

**Total: 2 fix before merge (#3, #4), 13 safe to defer.**

### Does the aggregate of the weak-assertion items leave a dangerous blind spot?

Six of the fifteen concern weak test assertions (#1, #4, #7, #9, #10, #11). Five of those six are
individually harmless *and* harmless in aggregate, because in every case the behaviour is covered
somewhere else — a sibling test, the nightly smoke job, or the live-cluster verification already
accepted as established. Weak unit assertions over behaviour that a real run exercises loudly are
a cosmetic debt, not a risk.

**#4 is the exception, and it is a real blind spot when combined with what is missing around it.**
Three facts stack:

1. `tests/stubs/kind` logs only `"$*"`, never piped stdin, so the entire config document handed to
   `kind create cluster --config -` is invisible to every test.
2. There is consequently no golden file for the rendered kind config — `tests/golden/` contains
   only `namespace.yaml`, `minio.yaml`, `mlflow-server.yaml`, and `fed_kind_ensure_cluster` returns
   early under dry-run (`kind.sh:6-9`) before rendering anything, so a golden could not be produced
   even if wanted.
3. `tests/kind.bats:61` spot-checks the *template render* for exactly one of the four port
   mappings (KFP: 30080→8080) and nothing else.

So the artifact that determines whether **any** UI is reachable — four NodePort→hostPort mappings —
is one-quarter covered, and the worker-node count, the only structural difference between
`active-fed`'s and `fed-twin`'s clusters, is not covered at all. Delete the append loop at
`kind.sh:20-24` and the full suite still passes green; `active-fed` would then silently come up
control-plane-only and its worker pods would schedule onto the control-plane node (or not
schedule), with the first symptom appearing deep inside a pipeline run.

That is the one place where "69 tests pass" overstates the safety net, and it is why #4 is
promoted. The fix is genuinely small — tee stdin in the stub — and it converts the least-tested
file in the repo into the best-tested one, since a golden kind config falls out for free.

---

## 7. Recommended pre-merge worklist

Ordered by value-to-effort. Total is well under an hour.

1. **I1** — two `|| var=""` guards in `lib/kind.sh:47,51`, plus one bats case that runs the
   function under `bash -c 'set -euo pipefail; …'`.
2. **I2** — decide between restoring the two files, converting the multi-cluster script's two
   lines, or a fail-fast plus a README note in `fed-twin`.
3. **Deferred #3** — validate `FED_S3_*` in `fed_config_validate` when `mlflow` is enabled.
4. **Deferred #4** — tee stdin in `tests/stubs/kind`; assert the worker count; add a golden kind
   config while you are in there.
5. **I3** — at minimum, stop hardcoding 30900/30901 in `fed-twin`'s install script; ideally move
   the exposure into `components.sh`.
6. **M1** — bump both consumers' submodule pins to `d0e8dc4` so the vendored README is the real one.

Minors 2-10 and the remaining twelve deferred items are all safe to carry into P1.
