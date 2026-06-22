
from __future__ import annotations

import sys
import time
from collections.abc import MutableMapping
from types import TracebackType
from typing import final

__all__: tuple[str, ...] = (
    "BudgetExceededError",
    "BudgetGuard",
    "StageTimer",
    "format_duration",
    "sample_peak_rss_mb",
)


def _posix_peak_rss_mb() -> float:
    """POSIX peak RSS via ``resource.getrusage`` (Linux/Darwin/BSD only).

    ``ru_maxrss`` is kibibytes on Linux and bytes on macOS/BSD (Darwin); both
    are normalized here so the figure is platform-stable.
    """
    import resource

    raw_units = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw_units / divisor


def _windows_peak_rss_mb() -> float:
    """Windows peak working-set size via the ``psapi`` ``GetProcessMemoryInfo`` call.

    stdlib-only (``ctypes``); no ``pywin32``/``psutil`` dependency. Returns 0.0
    on any failure -- a peak-RSS reading is observability, not a behavior-
    critical value, so a failed sample must never abort a run.
    """
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )

    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        psapi = ctypes.WinDLL("psapi")  # type: ignore[attr-defined]
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return 0.0
        return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
    except (OSError, AttributeError, ValueError):
        return 0.0


def sample_peak_rss_mb() -> float:
    """Return the process's peak RSS so far, normalized to MB, on any platform.

    Dispatches to the POSIX (``resource``) or Windows (``ctypes``/``psapi``)
    reading so neither ``redstack build`` nor ``redstack rank`` crashes at
    import time on Windows (CLAUDE.md §1: cross-platform execution stability) —
    the POSIX-only ``resource`` module does not exist there at all.
    """
    if sys.platform == "win32":
        return _windows_peak_rss_mb()
    return _posix_peak_rss_mb()


def format_duration(seconds: float) -> str:
    """Format a non-negative duration in seconds as ``MM:SS.mmm``.

    Minutes are not capped at 59 (a multi-hour offline build still renders as
    e.g. ``137:04.250`` rather than wrapping), matching the MM:SS.ms telemetry
    format used for offline/online pass durations.
    """
    total_ms = round(seconds * 1000.0)
    minutes, remainder_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(remainder_ms, 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


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
        self._peak_rss_mb = max(self._peak_rss_mb, sample_peak_rss_mb())
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
        self._peak_rss_mb = max(self._peak_rss_mb, sample_peak_rss_mb())
        return self._peak_rss_mb

    @property
    def within_budget(self) -> bool:
        """Whether both the time and RSS ceilings currently hold."""
        return (
            self.used_seconds <= self._limit_seconds
            and self.peak_rss_mb <= self._max_rss_mb
        )
