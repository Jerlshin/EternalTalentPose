# `data/` — Raw Inputs

Gitignored and **read-only at runtime** — no pipeline stage, engine, or adapter ever writes into this directory. This is the system's "raw fact," as distinct from `configs/` ("intent") and `artifacts/` ("derived fact"). See [`/ARCHITECTURE.md` §3](../ARCHITECTURE.md#3-repository-layout) and [`docs/runbook.md`](../docs/runbook.md) for how to populate it.

## Layout

| Path | README |
|---|---|
| [`raw/`](raw/README.md) | The candidate pool to rank or build artifacts from. |
| [`golden/`](golden/README.md) | Hand-labeled relevance data used only for the offline gold-label weight calibration search. |

## Why this directory is gitignored

The candidate pool and any gold labels are operator-provided facts, not code — committing them would mean committing potentially large, frequently-changing, and possibly sensitive data into version control. `.gitkeep` files preserve the empty directory structure so a fresh checkout has the right shape to populate.
