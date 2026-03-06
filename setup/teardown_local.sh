#!/usr/bin/env bash
# teardown_local.sh — Destroys the local kind cluster and cleans up port-forwards

set -euo pipefail

echo "Tearing down kind cluster 'active-fed'..."
pkill -f "kubectl port-forward" 2>/dev/null || true
kind delete cluster --name active-fed
echo "✅ Cluster deleted."
