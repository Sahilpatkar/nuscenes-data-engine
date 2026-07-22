# nuScenes data engine — developer entrypoints.
#
# Topology: GPU-server targets run compute and need NO infra (no docker/MinIO/MLflow
# server). Infra-machine targets run the local ops stack. See README "Two-machine topology".

.DEFAULT_GOAL := help
.PHONY: help setup infra-up infra-down ingest validate train evaluate serve monitor \
        manifest embed search gpu-run gpu-train gpu-embed gpu-manifest sync-down \
        test lint format typecheck check clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install base + dev deps.
	uv sync --extra dev

# --- Local INFRA MACHINE only (needs docker; do NOT run on the GPU server) ---
infra-up:  ## [infra machine] Start MinIO + MLflow.
	docker compose up -d minio minio-setup mlflow

infra-down:  ## [infra machine] Stop MinIO + MLflow.
	docker compose down

# --- Remote GPU-server execution (no manual ssh/pull; see docs/GPU_SERVER.md) ---
GPU_SSH  ?= TRINITY
GPU_NODE ?= trinity-2-18
GPU_REPO ?= /home/mgaur/sahil/nuscenes_project

gpu-run:  ## Run a CLI command on the GPU node at the current branch: make gpu-run CMD="embed --limit-scenes 5"
	scripts/gpu-run.sh $(CMD)

gpu-train:  ## Launch training on the GPU node (background; logs to gpu-run-*.log there).
	scripts/gpu-run.sh --bg train

gpu-embed:  ## Launch the full frame-embedding job on the GPU node (background).
	scripts/gpu-run.sh --bg embed

gpu-manifest:  ## Build the data-availability manifest on the server (CPU; head node is fine).
	GPU_NODE=$(GPU_SSH) scripts/gpu-run.sh manifest

sync-down:  ## Pull compute outputs off the GPU server (parquet, mlruns, lancedb).
	rsync -a --partial --timeout=120 --exclude yolo/ $(GPU_SSH):$(GPU_REPO)/data/processed/ ./data/processed/
	rsync -a --partial --timeout=120 $(GPU_SSH):$(GPU_REPO)/mlruns/ ./mlruns/
	rsync -a --partial --timeout=120 $(GPU_SSH):$(GPU_REPO)/data/lancedb/ ./data/lancedb/

# --- GPU-SERVER compute stages — infra-free (Phase 1+) ---
ingest:  ## Phase 1: parse nuScenes -> Parquet + 2D projections.
	uv run nuscenes-data-engine ingest

validate:  ## Phase 1: run Great Expectations suites.
	uv run nuscenes-data-engine validate

train:  ## Phase 2: run the training pipeline.
	uv run nuscenes-data-engine train

evaluate:  ## Phase 3: compute mAP + condition-sliced metrics.
	uv run nuscenes-data-engine evaluate

serve:  ## Phase 4: launch the FastAPI serving app.
	uv run nuscenes-data-engine serve

monitor:  ## Phase 5: drift report from the serving capture log.
	uv run nuscenes-data-engine monitor report

manifest:  ## Phase 6: data-availability manifest (run where the dataset lives).
	uv run nuscenes-data-engine manifest

embed:  ## Phase 6a: embed camera keyframes into the LanceDB store (GPU server).
	uv run nuscenes-data-engine embed

search:  ## Phase 6a: semantic frame search, e.g. make search Q="night construction".
	uv run nuscenes-data-engine search "$(Q)"

# --- Quality gates ---
test:  ## Run the test suite.
	uv run pytest

lint:  ## Lint with ruff.
	uv run ruff check .

format:  ## Auto-format with ruff.
	uv run ruff format .

typecheck:  ## Static type-check with mypy.
	uv run mypy

check: lint typecheck test  ## Run all quality gates.

clean:  ## Remove caches and build artifacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage \
	       build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
