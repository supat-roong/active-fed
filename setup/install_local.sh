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

echo "Applying Temporal worker..."
kubectl apply -f "${ROOT_DIR}/k8s/temporal-worker.yaml"

echo "Run 'make local-teardown' to destroy the cluster."
