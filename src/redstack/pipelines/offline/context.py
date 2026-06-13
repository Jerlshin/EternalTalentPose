"""OfflinePipelineContext: resolved build environment (config, bound ports, seed, as_of, output roots, registry/layout). Immutable.

Owner layer: pipelines.
Allowed imports: config, ports, features, domain.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "OfflinePipelineContext",
)
