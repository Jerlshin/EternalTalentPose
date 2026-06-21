# `observability/` — Logging, Timing, and the Run Report

Cross-cutting concerns, importable only from `domain/` plus the standard library. This package builds an object that structurally conforms to the `RunReport` protocol owned by `ports/_types.py` — `observability/` never imports `ports/`, and `ports/` never imports `observability/` (an explicit independence contract). See [`/ARCHITECTURE.md` §10](../../../ARCHITECTURE.md#10-determinism-and-performance) and [`/ARCHITECTURE.md` §11](../../../ARCHITECTURE.md#11-quality-gates).

## File inventory

| File | Contents |
|---|---|
| [`logging.py`](logging.py) | `get_logger()` and `stage_metric()` — structured, deterministic logging. A custom formatter and stream handler ensure no reproducibility-relevant log line carries a wall-clock timestamp; audit-only lines are clearly separated from anything that could affect a reproducibility comparison. |
| [`timing.py`](timing.py) | `StageTimer`, `BudgetGuard`, `BudgetExceededError` — per-stage wall-time measurement and peak-RSS sampling (`_sample_rss_mb`), and the hard budget guard that computes `within_budget` for the run report. |
| [`run_report.py`](run_report.py) | `RunReportBuilder` and its component snapshots: `ReproducibleSnapshot` (code/config/manifest/artifact/input/output hashes, candidate count, honeypot rate, eligibility summary, score-distribution digest — the only block compared by determinism tests), `AuditSnapshot` (run id, wall-clock start/end, host label — explicitly excluded from any reproducibility comparison), `BudgetSnapshot` (limit, used, `within_budget`, peak RSS), assembled into a `RunReportSnapshot`. `RunReportBuilderError` is raised if a snapshot is assembled with missing required fields. |

## The reproducible/audit split

Every run report is built with two deliberately separated regions: a **reproducible** block, which must be byte-identical across two runs with identical inputs (and is exactly what the determinism test suite diffs), and an **audit** block (timestamps, run id, host label), which is expected to differ run-to-run and is explicitly excluded from any reproducibility check. This split exists so that "the run is reproducible" can be asserted as `reproducible_block_a == reproducible_block_b` without having to special-case timestamp fields in the comparison.

## Who calls this package

Only `pipelines/` and `cli/` import `observability/` directly — `engines/` and `features/` never log or time themselves; the pipeline wraps each stage call in a `StageTimer` from the outside. This keeps every engine's unit tests free of logging side effects.
