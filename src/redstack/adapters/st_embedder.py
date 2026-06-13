"""SentenceTransformerEmbeddingAdapter implements EmbeddingModelPort (offline O13 only). Import-guarded: raises under online markers.

Owner layer: adapters.
Allowed imports: ports, domain, sentence_transformers, torch.

Defence-in-depth: raises on import when an online environment marker (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) signals the hot path.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "SentenceTransformerEmbeddingAdapter",
)
