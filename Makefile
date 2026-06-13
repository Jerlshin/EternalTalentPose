# REDSTACK — task aliases over `uv` and the `redstack` CLI.
# No business logic lives here; every target shells to uv or a CLI verb.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Determinism: pin BLAS/OMP threads for every invoked target so local runs
# match the network-isolated sandbox byte-for-byte.
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1

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

typecheck: ## mypy --strict over the package
	$(UV) run mypy

imports: ## Enforce the eight architecture boundary contracts
	$(UV) run lint-imports

test: ## Full test suite with coverage
	$(UV) run pytest --cov --cov-report=term-missing

test-unit: ## Fast unit + property suites only
	$(UV) run pytest -m "unit or property"

test-integration: ## Integration + determinism suites
	$(UV) run pytest -m "integration or determinism"

build: ## Offline: O0..O18 -> artifacts/ + MANIFEST.json
	$(UV) run redstack build --config configs/runtime/offline.yaml

rank: ## Online: R0..R9 -> submission.csv + run_report.json (spec 10.3)
	$(UV) run redstack rank \
		--candidates data/raw/candidates.jsonl \
		--out submission.csv

validate: ## Validate a finished submission.csv
	$(UV) run redstack validate --submission submission.csv

clean: ## Remove caches and build noise (never touches data/ or artifacts/)
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov \
		dist build **/__pycache__
