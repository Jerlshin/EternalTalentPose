"""SemanticEngine (Retrieval) — anchor cosine + nearest-centroid; lookup-first, encode-fallback. The only port-dependent engine.

Owner layer: engines.
Allowed imports: domain, ports (injected), features, config.schema.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "SemanticEngine",
)
