"""R0..R9 orchestrator — the Stage-3 reproduce spine. Binds online ports, runs stages sequentially, copy-on-write, fail-fast.

Owner layer: pipelines.
Allowed imports: engines, adapters, config, observability, features.

Online containment: this subgraph must not import sentence_transformers, sklearn, adapters.st_embedder, requests/httpx/urllib3, or socket.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "OnlinePipeline",
    "OnlineRunContext",
)
