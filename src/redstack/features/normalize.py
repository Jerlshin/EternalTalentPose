"""Canonical text/date/skill-token/company normalization + composed embedding document (field order pinned by embedding_manifest).

Owner layer: features.
Allowed imports: domain, config.schema.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "NormalizedCandidate",
    "normalize_candidate",
)
