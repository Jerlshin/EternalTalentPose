"""Populated FeatureRegistry: every FeatureDefinition. Single source of truth; rejects unknown ids.

Owner layer: features.
Allowed imports: layout, domain.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "FeatureRegistry",
    "FeatureDefinition",
    "FeatureSchema",
    "FeatureMetadata",
    "FeatureVersion",
    "FeatureManifest",
)
