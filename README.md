# REDSTACK

Redrob Evidence-Driven Symbolic + Semantic Ranker. CPU-only, network-isolated,
deterministic, reproducible. Architecture is frozen; see `ARCHITECTURE.md`.

## Setup

```bash
uv sync --frozen --group core --group dev   # hot path + tooling
uv pip install -e .                          # src/ layout: install before tests
```

## Reproduce (Stage-3, spec 10.3)

```bash
make rank
# == uv run redstack rank --candidates data/raw/candidates.jsonl --out submission.csv
# == scripts/reproduce.sh
```

See `docs/runbook.md` for populating `data/` and `make build` for artifacts.
