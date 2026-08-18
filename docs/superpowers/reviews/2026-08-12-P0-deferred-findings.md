# P0 deferred findings — carry into later phases

Recorded during the Phase P0 subagent-driven run. None block P0; several
are worth folding into P1-P4 when touching the same code.

- Task 1: minor (deferred): tests/common.bats "fed_require_cmd succeeds" passes with or without the stub PATH prepend (real kubectl/kind exist on host) - weak assertion, inherited from plan's own test body.
- Task 3: minor (deferred): FED_KFP_VERSION is defaulted/exported in config.sh but absent from FED_TEMPLATE_VARS. Harmless in this plan (Task 7 uses it as a shell var in `kubectl apply -k` URLs, never in a template), but a future KFP *template* referencing it would silently render empty.
- Task 3: minor (deferred): FED_S3_ENDPOINT/ACCESS_KEY/SECRET_KEY are whitelisted but have no defaults and are not in FED_REQUIRED_VARS - consumers must supply them; nothing validates that today.
- Task 4: minor (deferred): tests/kind.bats worker-append test asserts only exit 0 + that `kind create cluster` ran. The kind stub logs argv but NOT piped stdin, so the test would pass even if the append loop were deleted. Behaviour verified manually by controller instead.
- Task 4: minor (deferred): awk repo/tag split via ${image%:*}/${image#*:} misparses untagged refs and registries with a port (localhost:5000/x:tag). Fails SAFE (redundant reload, never a stale-skip). Irrelevant for this plan's images (all plain repo:tag); matters only if a consumer adds a ported registry to FED_IMAGES.
- Task 5: minor (deferred): fed_minio_ensure_bucket pod name via `tr -cd 'a-z0-9'` could degenerate to an invalid k8s name for a bucket with no alnum chars. Irrelevant for the two real buckets (mlpipeline, mlflow-artifacts).
- Task 5: minor (deferred): the "deletes pod before and after" test asserts only `grep -c >= 2`, not ordering - a double-delete on one side would pass. Implementation is correct by inspection.
- Task 5: minor (deferred): `kubectl run` unguarded against a set -e caller could skip trailing cleanup on failure; consistent with existing lib convention.
- Task 6: minor (deferred): dry-run test checks only mlflow-server.yaml, not namespace.yaml (parallel minio.bats test checks both). Namespace rendering covered elsewhere.
- Task 7: minor (deferred): tests/kfp.bats test 1 does not assert ordering of the two applies nor that the CRD-establishment wait happens between them; a reordering regression would pass.
- Task 8: minor (deferred): in the new dry-run test, 2 of the 6 refute_called assertions (`docker build`, `kubectl apply`) do not discriminate - fed_kfp_install's `kubectl get deploy` probe and fed_mlflow_build_image's `docker image inspect` probe both succeed by default in the stub, so those lines are never reached even with guards removed. Stronger assertion would be "stub log is empty". Controller's own empty-log check does cover these paths.
- Task 10: minor (deferred): kind `ingress-ready=true` node label dropped (verified unused anywhere - no ingress controller installed by either consumer).
- Task 10: minor (deferred): MLflow rollout wait 180s -> 300s (more lenient, inherited from fed-infra's mlflow module).
- Task 11: minor (deferred): active-fed's pre-existing uncommitted .gitignore + docs/ correctly left alone by the implementer.
- Task 12: minor (deferred): metrics CSV came out header-only (32 bytes). Pre-existing race in fed-twin's log-tailing/scraping code in src/pipelines/single_twin_single_cluster_pipeline.py - last touched by commit 7e347c6, which PREDATES all conversion commits. Real values confirmed via MLflow API + pod logs instead.

> **Correction (P0 backlog closeout):** the claim above that an unlisted `FED_TEMPLATE_VARS` entry "silently renders empty" is wrong, and was repeated from here into the P1 and P4 plans and several task briefs. `envsubst` leaves the placeholder *literally* (`${FED_X}` stays as that text), so the failure is visible: `kubectl apply` rejects it on a typed field. The whitelist is still mandatory; the hazard was overstated. Verified directly against `envsubst`.
