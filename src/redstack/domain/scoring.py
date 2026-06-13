"""Per-component breakdown + scored candidate; invariants weighted==raw*weight, base==sum(weighted), gating => FLOOR.

Owner layer: domain.
Allowed imports: ids, enums, provenance.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "ScoringWeights",
    "ScoreComponentValue",
    "GateOutcome",
    "ScoreBreakdown",
    "ScoredCandidate",
)
