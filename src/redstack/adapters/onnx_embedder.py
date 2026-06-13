"""OnnxEmbeddingModelAdapter implements EmbeddingModelPort (online R3 fallback). CPU EP, pinned threads, no network.

Owner layer: adapters.
Allowed imports: ports, domain, onnxruntime, numpy.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "OnnxEmbeddingModelAdapter",
)
