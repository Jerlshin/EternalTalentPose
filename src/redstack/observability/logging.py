
from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import Final, TextIO

__all__: tuple[str, ...] = (
    "get_logger",
    "stage_metric",
)

_REPRO_SAFE_FORMAT: Final[str] = "%(levelname)s %(name)s %(message)s"


class _DeterministicFormatter(logging.Formatter):
    """A ``Formatter`` that never renders wall-clock time into a record."""

    def formatTime(  # noqa: N802 — overrides logging.Formatter's mixed-case API.
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        """Return an empty string; repro-relevant lines carry no timestamp."""
        return ""


class _DeterministicStreamHandler(logging.StreamHandler[TextIO]):
    """Marker subclass so :func:`get_logger` can detect prior setup idempotently."""


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a stdlib logger configured for deterministic, timestamp-free output.

    Idempotent: calling this repeatedly for the same ``name`` reuses the
    already-attached handler instead of stacking duplicate handlers, and
    disables propagation to the root logger so output is not duplicated by a
    caller's own root configuration.

    Args:
        name: The logger name, conventionally the calling module's stage id
            (e.g. ``"R3"`` or ``"redstack.engines.scoring"``).
        level: The minimum level this logger emits at.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, _DeterministicStreamHandler) for h in logger.handlers):
        handler = _DeterministicStreamHandler(sys.stderr)
        handler.setFormatter(_DeterministicFormatter(_REPRO_SAFE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def _render_fields(fields: Mapping[str, object]) -> str:
    """Render structured fields as ``key=value`` pairs in ascending key order.

    Sorting by key — rather than trusting call-site or dict insertion order —
    is what makes the rendered line stable across runs and Python versions.
    """
    return " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))


def stage_metric(
    logger: logging.Logger, stage: str, event: str, **fields: object
) -> None:
    """Emit one deterministic structured record: ``[stage] event key=value ...``.

    This is the per-stage metric capture surface: stages report facts
    (``stage_metric(log, "R6", "ranked", honeypot_count=0, size=100)``) rather
    than hand-formatting strings, so every emitted line has a stable shape.

    Args:
        logger: A logger obtained from :func:`get_logger`.
        stage: The stage identifier the metric belongs to (e.g. ``"R6"``).
        event: A short, stable event name (e.g. ``"ranked"``, ``"started"``).
        **fields: Arbitrary structured payload; rendered sorted by key.
    """
    rendered = _render_fields(fields)
    message = f"[{stage}] {event}" + (f" {rendered}" if rendered else "")
    logger.info(message)
