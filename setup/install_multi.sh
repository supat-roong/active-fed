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

# The kubeconfig fed-infra writes points at https://127.0.0.1:<nodeport>,
# which is right for this machine (kind maps that port to the host) but
# meaningless inside a pod, where 127.0.0.1 is the pod's own loopback. The
# Temporal worker runs as a pod on the host cluster, so handing it that file
# verbatim would make every topology='multi' dispatch fail to connect. This
# is the same trap vendor/fed-infra/lib/karmada.sh's fed_karmada_join already
# patches around for member clusters -- read its comment; it is the clearest
# statement of the problem.
#
# Rewrite the server to the Karmada apiserver's in-cluster Service DNS, whose
# port is read from the live Service rather than hardcoded so this cannot
# drift from whatever karmadactl actually created. The cluster entry's name
# is likewise read from the file instead of assumed.
echo "Rewriting the Karmada kubeconfig for in-cluster use..."
KARMADA_SVC_PORT=$(kubectl -n karmada-system get svc karmada-apiserver \
  -o jsonpath='{.spec.ports[0].port}') || KARMADA_SVC_PORT=""
if [ -z "$KARMADA_SVC_PORT" ]; then
  echo "ERROR: could not read the karmada-apiserver Service port in karmada-system." >&2
  echo "       Is the Karmada control plane installed on this cluster?" >&2
  exit 1
fi

KARMADA_INCLUSTER_CONFIG=$(mktemp)
trap 'rm -f "$KARMADA_INCLUSTER_CONFIG"' EXIT
cp "${FED_KARMADA_CONFIG}" "$KARMADA_INCLUSTER_CONFIG"
KARMADA_CLUSTER_NAME=$(kubectl --kubeconfig="$KARMADA_INCLUSTER_CONFIG" \
  config view -o jsonpath='{.clusters[0].name}')
kubectl --kubeconfig="$KARMADA_INCLUSTER_CONFIG" config set-cluster \
  "$KARMADA_CLUSTER_NAME" \
  --server="https://karmada-apiserver.karmada-system.svc.cluster.local:${KARMADA_SVC_PORT}"

# Idempotent: `create secret --dry-run=client -o yaml | apply` re-running
# this against an already-provisioned cluster updates the Secret in place
# instead of failing (a plain `kubectl create secret` 409s on a second run).
echo "Creating/updating the Karmada kubeconfig Secret for the Temporal worker..."
kubectl create secret generic karmada-kubeconfig -n "${FED_NAMESPACE}" \
  --from-file=karmada-apiserver.config="$KARMADA_INCLUSTER_CONFIG" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Applying Temporal worker..."
kubectl apply -f "${ROOT_DIR}/k8s/temporal-worker.yaml"

echo "Run 'make multi-teardown' to destroy the clusters."
