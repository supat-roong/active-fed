#!/usr/bin/env bash
# install_multi.sh — bootstrap the Active-FL stack (multi topology: host +
# Karmada member clusters) via vendor/fed-infra. Mirrors install_local.sh;
# see there for the single-topology comments this doesn't repeat.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/infra.env.multi"

echo "Building Active-FL images..."
docker build -t active-fed-worker:v1 -f "${ROOT_DIR}/docker/Dockerfile.worker" "${ROOT_DIR}"
docker build -t active-fed-aggregator:v1 -f "${ROOT_DIR}/docker/Dockerfile.aggregator" "${ROOT_DIR}"

"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-up" --env "${ENV_FILE}"

# fed-infra-up (fed_up_multi, vendor/fed-infra/lib/components.sh) leaves the
# current kubectl context on the host cluster once it returns, same as the
# single-profile path -- every kubectl call below is unqualified, same as
# install_local.sh.

# This script runs in its own process, so it does not inherit fed-infra-up's
# exported FED_* vars -- load the same consumer contract here too, just for
# FED_NAMESPACE/FED_KARMADA_CONFIG, needed below. Falls back to fed-infra's
# own FED_KARMADA_CONFIG default (lib/config.sh) if infra.env.multi ever
# stops setting it explicitly.
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
: "${FED_KARMADA_CONFIG:=${HOME}/.karmada/karmada-apiserver.config}"

echo "Registering Temporal namespace 'default'..."
if kubectl exec -n "${FED_NAMESPACE}" deploy/temporal-admintools -- \
    temporal operator namespace describe --namespace default >/dev/null 2>&1; then
  echo "Temporal namespace 'default' already registered."
else
  kubectl exec -n "${FED_NAMESPACE}" deploy/temporal-admintools -- \
    temporal operator namespace create --namespace default
fi

echo "Applying Active-FL RBAC..."
kubectl apply -f "${ROOT_DIR}/k8s/rbac.yaml"

# Idempotent: `create secret --dry-run=client -o yaml | apply` re-running
# this against an already-provisioned cluster updates the Secret in place
# instead of failing (a plain `kubectl create secret` 409s on a second run).
echo "Creating/updating the Karmada kubeconfig Secret for the Temporal worker..."
kubectl create secret generic karmada-kubeconfig -n "${FED_NAMESPACE}" \
  --from-file=karmada-apiserver.config="${FED_KARMADA_CONFIG}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Applying Temporal worker..."
kubectl apply -f "${ROOT_DIR}/k8s/temporal-worker.yaml"

echo "Run 'make multi-teardown' to destroy the clusters."
