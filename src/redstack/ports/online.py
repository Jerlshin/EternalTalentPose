
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from redstack.ports._types import SubmissionReceipt
from redstack.ports.run_report_sink import RunReportSinkPort
from redstack.ports.semantic_index import SemanticVectorStorePort
from redstack.ports.submission_sink import SubmissionSinkPort

__all__: tuple[str, ...] = (
    "OnlineEntropyPort",
    "RunReportSinkPort",
    "SemanticVectorStorePort",
    "SubmissionReceipt",
    "SubmissionSinkPort",
)


@runtime_checkable
class OnlineEntropyPort(Protocol):
    """``seed`` + ``as_of`` only — the online run's sole permitted entropy seam."""

    @property
    def seed(self) -> int: ...

    def as_of(self) -> date: ...
