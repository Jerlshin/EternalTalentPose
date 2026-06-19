

from __future__ import annotations

from typing import Protocol, runtime_checkable

from redstack.domain.errors import DomainError
from redstack.ports._types import ReportReceipt, RunReport


class ReportWriteError(DomainError):
    """An IO failure while writing the run report."""


@runtime_checkable
class RunReportSinkPort(Protocol):
    """Write a ``RunReport`` to durable, deterministic storage."""

    def write(self, report: RunReport) -> ReportReceipt:
        """Serialize ``report`` to JSON atomically and return a receipt.

        The output uses stable key order and fixed float formatting so the
        ``reproducible`` region is byte-stable across runs with identical
        inputs.

        Args:
            report: The structural run report to persist.

        Returns:
            A ``ReportReceipt`` carrying the report's sha256.

        Raises:
            ReportWriteError: the report could not be written.
        """
        ...


__all__: tuple[str, ...] = ("ReportWriteError", "RunReportSinkPort")
