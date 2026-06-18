"""Online pipeline — R0..R9 hot path.

Owner layer: pipelines.
Allowed imports: engines, config, observability, and any adapter except
``adapters.st_embedder`` (sentence-transformers is offline-only; Online
Pipeline Containment, enforced by import-linter contract 7). ``pipeline.py``
and ``stages.py`` themselves depend only on the port Protocols; the
composition root that binds concrete adapters to those ports is
``pipelines.online.compose``.
"""
from __future__ import annotations

__all__: tuple[str, ...] = ()
