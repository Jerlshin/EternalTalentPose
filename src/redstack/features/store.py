"""Snapshot/lineage/cache metadata + explainability chain.

Owner layer: features.
Allowed imports: registry, domain.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "FeatureSnapshot",
    "FeatureLineage",
    "FeatureCache",
    "FeatureProvenance",
    "FeatureAuditRecord",
    "FeatureImportance",
    "FeatureContracts",
    "FeatureValidation",
)
