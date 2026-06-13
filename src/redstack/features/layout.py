"""Ordered, versioned CQV index map binding FeatureId -> FeatureIndex; pins layout_version. References domain.candidate.quality.FeatureLayout.

Owner layer: features.
Allowed imports: domain.candidate.quality, domain.ids.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "FEATURE_LAYOUT",
    "layout_version",
)
