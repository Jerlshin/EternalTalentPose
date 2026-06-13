"""ScoringEngine — weighted CQV x behavioral x logistics, integrity/eligibility-gated; base==sum(weighted); floored => FLOOR.

Owner layer: engines.
Allowed imports: domain, ports (injected), features, config.schema.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "ScoringEngine",
)
