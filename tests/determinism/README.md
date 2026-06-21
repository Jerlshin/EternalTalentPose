# `tests/determinism/` — Reproducibility Checks

Proves that identical inputs always produce identical output — the property the whole online pipeline is designed around. Determinism tests compare only the `reproducible` block of a run report (see [`src/redstack/observability/README.md`](../../src/redstack/observability/README.md)); the `audit` block (timestamps, run id, host label) is explicitly excluded from every comparison here.

## What belongs here

Per [`docs/specs/REDSTACK_TESTING_STRATEGY.md` §13](../../docs/specs/REDSTACK_TESTING_STRATEGY.md):

| Test | Method | Passing criterion |
|---|---|---|
| Repeated runs | Run R0–R9 twice on identical inputs, same process | Byte-identical `submission.csv`; identical `reproducible` block |
| Hash equality | Compare `output_sha256`, `manifest_hash`, per-artifact hashes, `config_hash` across runs | All equal |
| Single- vs. multi-thread | Run once at 1 thread, once at N threads | Identical ranking and `reproducible` block |
| Restart reproducibility | Run in two fresh processes | Identical `reproducible` block — proves no in-process state leaks between runs |
| Offline/online feature parity | Recompute features online for the sample pool and diff against the offline feature snapshot | Exact equality for deterministic features; cosine-similarity epsilon for semantic columns |

## Why this category exists

Without a live, continuously-scored environment, a non-reproducible ranking run is not just a bug — it's an undetectable one until it's too late to fix. This category exists specifically to make non-determinism a loud, pre-merge CI failure instead of a silent risk. This directory currently holds only its package marker (`__init__.py`); [`tests/integration/test_online_pipeline.py`](../integration/test_online_pipeline.py) is the nearest existing exercise of the full pipeline these tests would run twice and diff.
