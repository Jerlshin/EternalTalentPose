"""FilesystemArtifactStoreAdapter implements ArtifactStorePort (offline O17 verify; online R0). Streamed sha256, no degraded mode.

Owner layer: adapters.
Allowed imports: ports, domain, config.schema, pathlib, hashlib, numpy.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "FilesystemArtifactStoreAdapter",
)
