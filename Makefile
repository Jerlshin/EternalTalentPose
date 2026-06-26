# REDSTACK — task aliases routed through the cross-platform Python task runner.
# Every target delegates to scripts/run.py so behaviour is identical on
# macOS, Linux, and native Windows — no shell-specific binary assumptions.

.DEFAULT_GOAL := help

# Determinism: pin BLAS/OMP threads for the make session itself (subprocesses
# inherit these; scripts/run.py also sets them programmatically for coverage).
export OMP_NUM_THREADS          := 1
export MKL_NUM_THREADS          := 1
export OPENBLAS_NUM_THREADS     := 1
export VECLIB_MAXIMUM_THREADS   := 1

UV ?= uv

.PHONY: help install lock format lint typecheck test test-unit test-integration \
        imports build rank validate clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
install: ## Sync the full env (all groups) from the committed lockfile
	$(UV) sync --frozen --all-groups

lock: ## Re-resolve and rewrite uv.lock after a dependency bump
	$(UV) lock

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Pipeline lifecycle (routed through the cross-platform task runner)
# ---------------------------------------------------------------------------
build: ## Offline: O0..O18 -> artifacts/ + MANIFEST.json
	python scripts/run.py build

rank: ## Online: R0..R9 -> artifacts/submission.csv + run_report.json
	python scripts/run.py rank

validate: ## Validate a finished submission.csv against submission spec rules
	python scripts/run.py validate

clean: ## Remove caches and build noise (never touches data/ or artifacts/)
	python scripts/run.py clean
