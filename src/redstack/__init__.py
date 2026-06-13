"""REDSTACK — Redrob Evidence-Driven Symbolic + Semantic Ranker.

Top-level package marker. Carries the version constant only; no side
effects, no imports of subpackages at import time (keeps the online
hot path free of accidental eager loading).
"""
from __future__ import annotations

__version__: str = "1.1.0"

__all__: tuple[str, ...] = ("__version__",)
