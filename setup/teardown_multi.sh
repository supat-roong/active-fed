#!/usr/bin/env bash
# teardown_multi.sh — destroy the multi-topology host + member clusters via
# vendor/fed-infra. Mirrors teardown_local.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-down" --env "${ROOT_DIR}/infra.env.multi"
