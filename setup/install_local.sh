#!/usr/bin/env bash
# install_local.sh — bootstrap the Active-FL stack via vendor/fed-infra.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building Active-FL images..."
docker build -t active-fed-worker:v1 -f "${ROOT_DIR}/docker/Dockerfile.worker" "${ROOT_DIR}"
docker build -t active-fed-aggregator:v1 -f "${ROOT_DIR}/docker/Dockerfile.aggregator" "${ROOT_DIR}"

"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-up" --env "${ROOT_DIR}/infra.env"

# F3 (gate fix): fed_temporal_install installs the chart and waits for the
# frontend to roll out, but never registers a Temporal namespace. Both
# worker_main.py and active_fl_pipeline.py hard-code "default" (neither
# passes a namespace= argument, so the Temporal SDK's own default applies),
# and this chart configuration does not create it automatically -- every
# client.start_workflow() call would otherwise fail outright with
# NamespaceNotFound. Registered here via the chart's own bundled
# temporal-admintools deployment. Idempotent: `namespace create` exits
# non-zero ("already exists") on a second run, so check first via `describe`
# and only create when it's actually absent -- re-running setup against an
# already-provisioned cluster must not fail.
echo "Registering Temporal namespace 'default'..."
if kubectl exec -n active-fed deploy/temporal-admintools -- \
    temporal operator namespace describe --namespace default >/dev/null 2>&1; then
  echo "Temporal namespace 'default' already registered."
else
  kubectl exec -n active-fed deploy/temporal-admintools -- \
    temporal operator namespace create --namespace default
fi

echo "Applying Active-FL RBAC..."
kubectl apply -f "${ROOT_DIR}/k8s/rbac.yaml"

echo "Applying Temporal worker..."
kubectl apply -f "${ROOT_DIR}/k8s/temporal-worker.yaml"

echo "Run 'make local-teardown' to destroy the cluster."
