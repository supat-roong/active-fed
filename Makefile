.PHONY: install install-dev test lint fmt type-check clean build-images \
        local-setup local-teardown compile-pipeline mlflow-ui

# ---- Deps ----
install:
	uv sync --no-dev

install-dev:
	uv sync --extra dev

# ---- Tests ----
test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -k "not test_short_training and not test_returns_result_for_each"

# ---- Code Quality ----
lint:
	uv run ruff check src/ tests/

fmt:
	uv run ruff format src/ tests/

type-check:
	uv run mypy src/

# ---- Local k8s ----
local-setup:
	bash setup/install_local.sh

local-teardown:
	bash setup/teardown_local.sh

# ---- Docker ----
build-images:
	docker build -t active-fed-worker:v1 -f docker/Dockerfile.worker .
	docker build -t active-fed-aggregator:v1 -f docker/Dockerfile.aggregator .

load-images:
	kind load docker-image active-fed-worker:v1 --name active-fed
	kind load docker-image active-fed-aggregator:v1 --name active-fed

# ---- Pipeline ----
compile-pipeline:
	uv run python src/pipelines/active_fl_pipeline.py --output /tmp/active_fl_pipeline.yaml
	@echo "Compiled → /tmp/active_fl_pipeline.yaml"

run-pipeline:
	bash run_pipeline.sh $(ARGS)

# ---- Local dry-run (no K8s) ----
dry-run-worker:
	uv run python -m src.agent.train_worker \
		--fl-round 0 \
		--local-episodes 20 \
		--dry-run

# ---- Experiments (local, no K8s) ----
run-experiments:
	uv run python experiments/run_experiments.py $(ARGS)

run-single:
	uv run python experiments/run_experiments.py --mode single \
		--weight-mode $(WEIGHT_MODE) \
		--active-data-mode $(ACTIVE_DATA_MODE)
# Usage: make run-single WEIGHT_MODE=data_only ACTIVE_DATA_MODE=bc

mlflow-ui:
	uv run mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000 to view experiment results

compare:
	uv run python analysis/compare_runs.py
# Output: results/plots/*.png

compare-k8s:
	uv run python analysis/fetch_k8s_runs.py --tracking-uri http://localhost:5050 --output-dir results/k8s_results/
	uv run python analysis/compare_runs.py --input-dir results/k8s_results/ --output-dir results/k8s_plots/
# Output: results/k8s_plots/*.png
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete

clean-results:
	rm -rf results/
