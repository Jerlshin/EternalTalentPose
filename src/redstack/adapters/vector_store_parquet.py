"""ParquetSemanticVectorStoreAdapter implements SemanticVectorStorePort (online R3). Mmap read-only + in-memory id index.

Owner layer: adapters.
Allowed imports: ports, domain, pyarrow, numpy.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "ParquetSemanticVectorStoreAdapter",
)
