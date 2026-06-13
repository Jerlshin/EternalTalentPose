"""Evidence-grounded reasoning models; a clause cannot construct without >=1 EvidenceRef.

Owner layer: domain.
Allowed imports: ids, enums, provenance.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "ReasoningClause",
    "CandidateReasoning",
)
