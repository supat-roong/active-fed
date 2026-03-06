#!/usr/bin/env bash
# install_local.sh — Bootstraps the full Active-FL stack on a local kind cluster
# Prerequisites: kind, kubectl, helm, docker

set -euo pipefail

CLUSTER_NAME="active-fed"
NAMESPACE="active-fed"
KFP_VERSION="2.4.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "  Active-FL Local Setup"
echo "=================================================="

# ---- 1. Create kind cluster ----
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "[1/7] kind cluster '${CLUSTER_NAME}' already exists, skipping creation"
else
  echo "[1/7] Creating kind cluster '${CLUSTER_NAME}'..."
  kind create cluster --config "${SCRIPT_DIR}/kind-cluster.yaml"
fi

kubectl cluster-info --context "kind-${CLUSTER_NAME}"

# ---- 2. Install Kubeflow Pipelines (standalone) ----
echo "[2/7] Installing Kubeflow Pipelines v${KFP_VERSION}..."
KFP_MANIFEST="https://raw.githubusercontent.com/kubeflow/pipelines/refs/tags/${KFP_VERSION}/manifests/kustomize/cluster-scoped-resources/kustomization.yaml"
kubectl apply -k "https://github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KFP_VERSION}" || true
kubectl wait --for condition=established --timeout=300s crd/applications.app.k8s.io || true
kubectl apply -k "https://github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${KFP_VERSION}" || true

echo "🛠 Patching KFP Deployments for M1/ARM stability..."
kubectl set image deployment/ml-pipeline-ui ml-pipeline-ui=ghcr.io/kubeflow/kfp-frontend:${KFP_VERSION} -n kubeflow
kubectl set image deployment/ml-pipeline ml-pipeline-api-server=ghcr.io/kubeflow/kfp-api-server:${KFP_VERSION} -n kubeflow
kubectl set image deployment/ml-pipeline-visualizationserver ml-pipeline-visualizationserver=ghcr.io/kubeflow/kfp-visualization-server:${KFP_VERSION} -n kubeflow

# Fix Launcher Image (M1/Kind Stability)
kubectl set env deployment/ml-pipeline V2_LAUNCHER_IMAGE=ghcr.io/kubeflow/kfp-launcher:${KFP_VERSION} -n kubeflow

# Fix Minio Image for KFP (Kind/M1 issue)
kubectl set image deployment/minio minio=minio/minio:latest -n kubeflow

# Fix Workflow Controller Executor Image (Init Container Issue on ARM)
kubectl patch deployment workflow-controller -n kubeflow --type=json -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/args/3", "value": "quay.io/argoproj/argoexec:v3.4.17"}]'

# ---- 3. Install PyTorch Training Operator ----
echo "[3/7] Installing Kubeflow Training Operator..."
kubectl apply -k "https://github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0" || true

# ---- 4. Create namespace + apply infra YAMLs ----
echo "[4/7] Deploying MinIO + MLflow + RBAC to namespace '${NAMESPACE}'..."
kubectl apply -f "${ROOT_DIR}/k8s/minio.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/mlflow-server.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/rbac.yaml"

# KFP/Argo requires a 'mlpipeline' bucket in the kubeflow-namespace MinIO instance.
# On a fresh KFP install the bucket may not be pre-created.
echo "   Ensuring 'mlpipeline' bucket exists in KFP MinIO..."
kubectl run minio-setup-kfp --image=minio/mc:latest --namespace=kubeflow --restart=Never \
  --command -- sh -c "mc alias set kfp http://minio-service.kubeflow.svc.cluster.local:9000 minio minio123 && mc mb kfp/mlpipeline --ignore-existing" 2>/dev/null || true
kubectl wait --for=condition=completed pod/minio-setup-kfp -n kubeflow --timeout=60s 2>/dev/null || true
kubectl delete pod minio-setup-kfp -n kubeflow 2>/dev/null || true

# ---- 5. Build + load Docker images into kind ----
echo "[5/7] Building Docker images and loading into kind..."
docker build -t active-fed-worker:v1 -f "${ROOT_DIR}/docker/Dockerfile.worker" "${ROOT_DIR}"
docker build -t active-fed-aggregator:v1 -f "${ROOT_DIR}/docker/Dockerfile.aggregator" "${ROOT_DIR}"
kind load docker-image active-fed-worker:v1 --name "${CLUSTER_NAME}"
kind load docker-image active-fed-aggregator:v1 --name "${CLUSTER_NAME}"

# ---- 6. Wait for services to be ready ----
echo "[6/7] Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment/mlflow-server -n "${NAMESPACE}" --timeout=180s
kubectl rollout status statefulset/minio -n "${NAMESPACE}" --timeout=180s

# ---- 7. Expose Services via NodePort ----
echo "[7/7] Exposing Services via NodePort..."
kubectl patch service ml-pipeline-ui -n kubeflow -p '{"spec": {"type": "NodePort", "ports": [{"port": 80, "targetPort": 3000, "nodePort": 30080}]}}' || true
kubectl patch service mlflow-service -n active-fed -p '{"spec": {"type": "NodePort", "ports": [{"port": 5000, "targetPort": 5000, "nodePort": 30500}]}}' || true
kubectl patch service minio-service -n active-fed -p '{"spec": {"type": "NodePort", "ports": [{"name": "api", "port": 9000, "targetPort": 9000, "nodePort": 30900}, {"name": "console", "port": 9001, "targetPort": 9001, "nodePort": 30901}]}}' || true

echo ""
echo "✅ Setup complete! Services are exposed locally via Docker/Kind port-mapping:"
echo "  → Kubeflow Pipelines UI : http://localhost:8080"
echo "  → MLflow UI             : http://localhost:5050"
echo "  → MinIO Console         : http://localhost:9001  (user: minioadmin / minioadmin)"
echo ""
echo "   Run 'bash setup/teardown_local.sh' to destroy the cluster."
