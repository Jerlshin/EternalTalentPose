"""OfflinePipeline: declares the O0..O18 stage set + dependency DAG; pure orchestration over injected stage callables.

Owner layer: pipelines.
Allowed imports: engines, adapters, config, observability.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "OfflinePipeline",
)
