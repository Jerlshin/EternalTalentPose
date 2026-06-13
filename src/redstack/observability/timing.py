"""Stage timing + hard budget guard: per-stage wall-ms, within_budget, peak_rss_mb.

Owner layer: observability.
Allowed imports: stdlib, resource.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "StageTimer",
    "BudgetGuard",
)
