"""RunContext base: resolved config + bound ports + seeds + loaded manifest. Immutable carrier handed to stages.

Owner layer: pipelines.
Allowed imports: config, ports, domain.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "RunContext",
)
