"""Raw dict -> RawCandidate (tolerant of drift, never silently coerces); mints EvidenceRef paths.

Owner layer: features.
Allowed imports: domain.source, domain.errors.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "parse_candidate",
)
