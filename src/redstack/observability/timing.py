"""Stage timing + hard budget guard: per-stage wall-ms, within_budget, peak_rss_mb.

Owner layer: observability.
Allowed imports: stdlib, resource.

``StageTimer`` measures one stage's wall-clock duration with the monotonic
``time.perf_counter`` clock (not ``datetime.now`` — duration measurement is
not a wall-clock dependency in the sense Repository Layout §1 forbids).
``BudgetGuard`` samples the process's peak resident set size via the POSIX
``resource`` module and raises as soon as either the wall-time or RSS ceiling
is crossed, so a runaway online stage fails fast against the <=16GB ceiling
instead of degrading the ranking silently.
"""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import MutableMapping
from types import TracebackType
from typing import final

__all__: tuple[str, ...] = (
    "BudgetExceededError",
    "BudgetGuard",
    "StageTimer",
)


def _sample_rss_mb() -> float:
    """Return the process's peak RSS so far, normalized to MB.

    ``ru_maxrss`` is kibibytes on Linux and bytes on macOS/BSD (Darwin); both
    are normalized here so the figure is platform-stable.
    """
    raw_units = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw_units / divisor


@final
class StageTimer:
    """Context manager measuring one stage's wall-clock duration in milliseconds.

    On ``__exit__`` the elapsed time is written into ``ledger[stage]``, so a
    composition root can accumulate every stage's timing into one dict that
    feeds ``RunReport.timings`` (Ports §13) without this class holding any
    reference back to the report itself.
    """

    __slots__ = ("_ledger", "_stage", "_start")

    def __init__(self, stage: str, ledger: MutableMapping[str, float]) -> None:
        self._stage = stage
        self._ledger = ledger
        self._start = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._ledger[self._stage] = elapsed_ms


class BudgetExceededError(RuntimeError):
    """The wall-clock or RSS budget for the run was exceeded."""


@final
class BudgetGuard:
    """Hard wall-time + RSS budget guard for one run (CLAUDE.md §1: <=16GB RAM).

    Constructed once at the start of a run with the two ceilings; stages call
    :meth:`check` at safe boundaries (between rows, between stages) so a
    breach is caught promptly rather than only at the very end.
    """

    __slots__ = ("_limit_seconds", "_max_rss_mb", "_peak_rss_mb", "_started_at")

    def __init__(self, limit_seconds: float, max_rss_mb: float) -> None:
        self._limit_seconds = limit_seconds
        self._max_rss_mb = max_rss_mb
        self._started_at = time.perf_counter()
        self._peak_rss_mb = 0.0

    def check(self) -> None:
        """Sample elapsed time + RSS; raise :class:`BudgetExceededError` if over.

        Raises:
            BudgetExceededError: the wall-clock or RSS ceiling has been crossed.
        """
        self._peak_rss_mb = max(self._peak_rss_mb, _sample_rss_mb())
        if self.used_seconds > self._limit_seconds:
            raise BudgetExceededError(
                f"wall-clock budget exceeded: used {self.used_seconds:.3f}s "
                f"> limit {self._limit_seconds:.3f}s"
            )
        if self._peak_rss_mb > self._max_rss_mb:
            raise BudgetExceededError(
                f"RSS budget exceeded: peak {self._peak_rss_mb:.1f}MB "
                f"> ceiling {self._max_rss_mb:.1f}MB"
            )

    @property
    def limit_seconds(self) -> float:
        """The configured wall-clock ceiling, in seconds."""
        return self._limit_seconds

    @property
    def used_seconds(self) -> float:
        """Elapsed wall-clock time since construction, in seconds."""
        return time.perf_counter() - self._started_at

    @property
    def peak_rss_mb(self) -> float:
        """The highest RSS observed across all :meth:`check` calls, in MB."""
        self._peak_rss_mb = max(self._peak_rss_mb, _sample_rss_mb())
        return self._peak_rss_mb

    @property
    def within_budget(self) -> bool:
        """Whether both the time and RSS ceilings currently hold."""
        return (
            self.used_seconds <= self._limit_seconds
            and self.peak_rss_mb <= self._max_rss_mb
        )
