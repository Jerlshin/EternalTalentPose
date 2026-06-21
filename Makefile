# REDSTACK — task aliases over `uv` and the `redstack` CLI.
# No business logic lives here; every target shells to uv or a CLI verb.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Determinism: pin BLAS/OMP threads for every invoked target so local runs
# match the network-isolated sandbox byte-for-byte.
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

UV ?= uv

.PHONY: help install format lint typecheck test test-unit test-integration \
        build rank validate imports lock clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Sync the full env (core+offline+dev) from the committed lock
	$(UV) sync --frozen --group core --group offline --group dev
	$(UV) pip install -e .

lock: ## Re-resolve and rewrite uv.lock after a dependency bump
	$(UV) lock

format: ## Apply ruff formatting + import sorting
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

lint: ## Lint without mutating (ruff)
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

typecheck: ## mypy --strict over the source package and test suites
	$(UV) run mypy --strict src tests

imports: ## Enforce the eight architecture boundary contracts
	$(UV) run lint-imports

test: ## Full test suite with coverage
	$(UV) run pytest --cov=src --cov-report=term-missing tests/

test-unit: ## Fast unit + property suites only
	$(UV) run pytest -m "unit or property" tests/

test-integration: ## Integration + determinism suites
	$(UV) run pytest -m "integration or determinism" tests/

build: ## Offline: O0..O18 -> artifacts/ + MANIFEST.json (locked heuristic weights; no gold-label search)
	@mkdir -p artifacts/embeddings artifacts/models artifacts/gates artifacts/weights artifacts/calibration artifacts/lexicon artifacts/archetypes
	$(UV) run redstack build --config configs/runtime/offline.yaml --no-golden-labels

rank: ## Online: R0..R9 -> submission.csv + run_report.json
	@mkdir -p artifacts
	$(UV) run redstack rank \
		--input data/raw/candidates.jsonl \
		--output artifacts/submission.csv

validate: ## Validate a finished submission.csv against validate_submission.py rules
	$(UV) run redstack validate --submission artifacts/submission.csv

clean: ## Remove caches and build noise (never touches data/ or configurations)
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov typecover \
		dist build
	find . -type d -name "__pycache__" -exec rm -rf {} +