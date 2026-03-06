#!/usr/bin/env bash
# run_pipeline.sh — Compile and submit the Active-FL Kubeflow pipeline
#
# This script is a wrapper around the Python orchestrator.
# It uses settings from config/k8s.yaml
#
# Usage:
#   bash run_pipeline.sh [options]
#
# Options:
#   --config PATH      Path to config yaml (default: config/k8s.yaml)
#   --kfp-host URL     Kubeflow Pipelines host (default: http://localhost:8080)
#   --wait             Wait for pipeline runs to complete
#   --auto-download    Auto-download results from MLflow after waiting (forces --wait)

set -euo pipefail

uv run python src/pipelines/run_pipeline.py "$@"
