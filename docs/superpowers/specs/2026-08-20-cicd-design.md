# CI/CD design — fed-infra, active-fed, fed-twin

**Status:** approved in brainstorming, 2026-08-20. Next step: implementation plan.

**Goal.** Give all three repos CI that gates every change, integration testing at the
depth each environment can actually support, and CD that publishes the artifacts a
bring-up needs — then validate the whole thing by pushing and watching it run.

---

## Current state (verified, not assumed)

All three repos already have `.github/workflows/ci.yml`. **None has ever executed**,
because nothing has been pushed. Running each workflow's own commands locally:

| Repo | Result | Detail |
|---|---|---|
| fed-infra | green | `shellcheck -x -P SCRIPTDIR` exits 0; 199 bats tests; smoke fixtures present |
| active-fed | green | `make lint` 0, 248 tests, `make compile-pipeline` 0 |
| fed-twin | **red** | 72 ruff errors, 4 files unformatted |

fed-twin's failures sit in files untouched by recent work (`automate_run.py`,
`single_twin_*`, `core/engine.py`), so this is pre-existing. Confirmed with
`ruff check . --isolated` that a clean runner sees the identical 72 — it is not an
artifact of local configuration.

Two further gaps:

- Neither consumer's `actions/checkout` sets `submodules: recursive`. Harmless today
  (no test reads `vendor/`), fatal as soon as contract testing exists.
- fed-infra's `smoke` job is nightly-only, single-profile, and unproven.

### The real defect in fed-twin's lint

Not "72 errors". fed-twin has **no ruff configuration** and runs `uvx ruff` **unpinned**,
so its enabled rule set is whatever the newest ruff ships that day: CI can turn red with
no code change. active-fed pins the tool and selects `["E","F","I","UP"]`, which is both
narrower and stable.

---

## Decisions taken

| Question | Decision |
|---|---|
| Where do integration tests run? | GitHub-hosted, scoped to what fits |
| What does CD do? | Publish images to GHCR; tagged releases for fed-infra |
| How is the fed-infra ↔ consumer coupling verified? | Both directions |
| Can we push to validate? | Yes, once the workflows are written |

---

## Architecture: four tiers

Each tier runs where it can actually run; nothing claims coverage it does not have.

| Tier | Where | When | Contents |
|---|---|---|---|
| 1 — Unit | GitHub-hosted | every PR | lint, unit tests, pipeline compile, image build |
| 2 — Contract | GitHub-hosted | every PR | `fed-infra-up --dry-run` over the contracts that repo owns; golden renders; config-agreement tests |
| 3 — Smoke | GitHub-hosted | nightly | one real kind cluster: single-profile bring-up, then a second run proving idempotency |
| 4 — Gate | developer machine | manual | full KFP/Temporal/MLflow/MinIO stack; multi-cluster Karmada topology |

**Why Tier 2 carries its weight.** It needs no cluster and runs in seconds, yet most
defects found during P3/P4 were reachable from it, because they were contract
mismatches rather than runtime failures: the `karmada`-component validation, the
`FED_KARMADA_DASHBOARD_PORT` → NODEPORT/HOSTPORT rename, `member_prefix` drifting from
`FED_MEMBER_PREFIX`, and every "the config layers disagree" bug.

**Why Tier 4 stays out of CI.** A single-cluster stack consumed ~33 GB of Docker disk
and 10 GB RAM locally; the multi-cluster gate needed three clusters and a VM grown to
118 GB. A GitHub-hosted runner offers roughly an order of magnitude less disk than that
(commonly cited as 14 GB free, **unverified in this session** — the Tier 3 task must
measure it on a real run and adjust, rather than trust this figure). A nightly that
thrashes is worse than an honest manual gate, because people learn to ignore it. Tier 4 is already
documented in `docs/superpowers/gates/`.

---

## Cross-repo verification (both directions)

**fed-infra → consumers**, nightly, in fed-infra: dry-run `active-fed`'s `infra.env` and
`infra.env.multi` plus `fed-twin`'s `infra.env.multi` against fed-infra's current `main`.
Sparse-checkout the contracts rather than vendoring copies — a checked-in copy drifts,
which is the exact failure this is meant to catch. A library change that breaks a
consumer then fails in the repo that caused it, before any submodule bump.

Concretely, "the contracts that repo owns" means: fed-infra dry-runs its three test
fixtures (`consumer-a/b/c.env`); active-fed dry-runs `infra.env` and `infra.env.multi`;
fed-twin dry-runs `infra.env` and `infra.env.multi`.

**consumers → fed-infra**, every PR, in each consumer: dry-run that consumer's own
contracts against the **pinned** submodule SHA, so a bad bump fails in the PR performing it.

---

## CD

| Trigger | Action |
|---|---|
| push to `main` (consumers) | build + push images to GHCR, tagged `sha-<short>` and `main` |
| tag `v*` (consumers) | same images tagged `vX.Y.Z`; GitHub Release |
| tag `v*` (fed-infra) | no images; a Release consumers pin their submodule to |

Justified by evidence rather than convention: every bring-up this week rebuilt torch from
scratch, and that build broke outright when a package index shifted under a Dockerfile
that had worked six days earlier (fixed by `--extra-index-url`). Published images make a
bring-up reproducible and turn a ~10-minute rebuild into a pull.

**Known limitation.** Runner-built images are `linux/amd64`; the local kind clusters run
on Apple Silicon. Published images therefore serve CI and x86 machines, while local
bring-up keeps building natively. Multi-arch via `docker/build-push-action` roughly
doubles build time and is deferred until someone actually needs to pull these locally.

---

## Prerequisite fixes

1. **fed-twin lint.** Adopt active-fed's ruff configuration (`select = ["E","F","I","UP"]`,
   `line-length = 100`) and pin ruff in dev dependencies. 79 of the 81 resulting issues
   are mechanical (import sorting, redundant open modes, line length). Do **not** blanket-fix
   the 19 `BLE001` blind-except warnings the default rule set raises: those are deliberate,
   and `src/core/tracking.py` is the codebase's stated model for "tracking must never fail a
   training run".
2. **`submodules: recursive`** in both consumers' checkout steps.
3. **fed-infra smoke job**: add disk reclamation, and prove it runs at all.
4. **Workflow hygiene**: `concurrency` to cancel superseded runs, `timeout-minutes` on every
   job, least-privilege `permissions` (`packages: write` only where GHCR is pushed).

### Unifying rule: CI invokes Makefile targets, never inline commands

This is why active-fed's CI is trustworthy and fed-twin's drifted — `make lint` is something
a developer runs locally; `uvx ruff check .` buried in YAML is not. Every tier gets a target,
so what CI runs is reproducible on a laptop, and the two cannot diverge unnoticed.

---

## Validation

Nothing here is real until it runs. The final task is: push all three repos to their
remotes, watch every workflow execute, and fix what the runner exposes that local
execution cannot — YAML validity, runner environment, GHCR authentication, submodule
checkout, and Tier 3's disk headroom. **The plan is not complete until each workflow has
been observed green on a real run**, including one deliberate red (a broken commit on a
branch) to confirm the gates actually block rather than merely report.

---

## Out of scope

- Deploying to any environment; there is no target.
- Multi-arch images (see limitation above).
- Self-hosted runners.
- Tier 4 automation.
- Coverage thresholds or required-reviewer policy — worth revisiting once the pipelines
  have run for a while and their real failure rate is known.

## Success criteria

- Every PR to any repo runs lint + unit tests + contract dry-runs, and a failure blocks.
- A fed-infra change that breaks either consumer's contract fails fed-infra's own nightly.
- A submodule bump that breaks a consumer fails that consumer's PR.
- Images appear in GHCR for `main` and for tags; fed-infra cuts Releases.
- fed-twin's CI is green, with an explicit and pinned lint configuration.
- Each of the above observed on a real run, not inferred.
