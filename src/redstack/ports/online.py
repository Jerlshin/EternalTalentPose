"""Curated re-export surface for the ports the online package touches.

Owner layer: ports.
Allowed imports: domain, stdlib typing, sibling ports modules.

The online subgraph (``pipelines.online``) imports its ports through this one
module rather than reaching into five separate port files, so the
online-containment review surface is a single, small import line. Everything
here is a re-export of an already-frozen port except :class:`OnlineEntropyPort`,
a *narrower* Protocol than :class:`~redstack.ports.rng.DeterministicEntropyPort`:
it declares only ``seed``/``as_of`` and omits ``derive``/``numpy_generator``
entirely, so the online code cannot even *name* the RNG methods (compile-time
containment, on top of the runtime ``EntropyDisabledError`` the real adapter
raises). Both ``OfflineEntropy`` and ``OnlineEntropy`` (and any
``DeterministicEntropyPort`` implementation) satisfy this Protocol structurally.
"""

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
