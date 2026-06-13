"""One pure callable per R-stage: r0_load..r9_report. R2/R4/R5/R6/R7 make no port calls.

Owner layer: pipelines.
Allowed imports: engines, domain, features.

Online containment applies transitively here too.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "r0_load",
    "r1_ingest",
    "r2_features",
    "r3_semantic",
    "r4_gates",
    "r5_score",
    "r6_rank",
    "r7_reason",
    "r8_submit",
    "r9_report",
)
