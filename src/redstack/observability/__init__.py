"""Observability layer — cross-cutting; domain only.

Owner layer: observability.
Allowed imports: domain, stdlib.

Package marker only — ``ports`` and ``observability`` are mutually
independent (Repository Layout §8c), so this package never re-exports
anything that would require importing ``ports``. Consumers import the
concrete submodules directly: :mod:`redstack.observability.logging`,
:mod:`redstack.observability.timing`, :mod:`redstack.observability.run_report`.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
