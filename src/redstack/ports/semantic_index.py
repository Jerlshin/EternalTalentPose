
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from redstack.domain.errors import DomainError
from redstack.domain.ids import CandidateId
from redstack.ports._types import BulkVectorResult, FloatMatrix, FloatVector


class VectorStoreError(DomainError):
    """The store is corrupt or unreadable, or its dimensionality is wrong.

    A *missing candidate* is never this error — it is reported as ``None`` /
    ``missing`` so the pipeline can fall back to the encoder.
    """


@runtime_checkable
class SemanticVectorStorePort(Protocol):
    """Read-only, memory-mapped access to precomputed candidate vectors."""

    @property
    def dim(self) -> int:
        """The vector dimensionality (asserted against the embedding contract)."""
        ...

    def contains(self, cid: CandidateId) -> bool:
        """Return whether ``cid`` has a precomputed vector in the store."""
        ...

    def get(self, cid: CandidateId) -> FloatVector | None:
        """Return ``cid``'s read-only normalized ``(dim,)`` vector, or ``None``.

        ``None`` means the candidate is absent (a normal event that triggers
        the encode fallback), never a failure.
        """
        ...

    def get_many(self, cids: Sequence[CandidateId]) -> BulkVectorResult:
        """Gather many vectors at once, partitioning found from missing.

        The result's ``found`` ids align row-for-row with its ``vectors``
        matrix and preserve the order of the requested ``cids``; ``missing``
        lists the ids with no precomputed vector.
        """
        ...

    def view_all(self) -> tuple[FloatMatrix, Sequence[CandidateId]]:
        """Return the full read-only matrix and its aligned id order.

        ``id_order[i]`` labels row ``i`` of the matrix. Intended for the
        vectorized columnar scoring path; the matrix is the mmap'd store with
        no copy.

        Raises:
            VectorStoreError: the store is corrupt or unreadable.
        """
        ...


__all__: tuple[str, ...] = ("SemanticVectorStorePort", "VectorStoreError")
