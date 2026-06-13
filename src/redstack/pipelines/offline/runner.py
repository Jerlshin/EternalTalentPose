"""OfflinePipelineRunner: executes the plan with resume, checkpointing, per-stage timing/metrics, failure quarantine.

Owner layer: pipelines.
Allowed imports: context, registry, graph.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "OfflinePipelineRunner",
    "StageReceipt",
)
