

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from redstack.domain.errors import DomainError
from redstack.ports._types import SubmissionReceipt

if TYPE_CHECKING:
    from redstack.domain.ranking import Ranking


class SubmissionWriteError(DomainError):
    """An IO failure while writing the submission (a partial file is never left)."""


class SubmissionContractError(DomainError):
    """Defence-in-depth: a serialized invariant would be broken.

    Should be impossible given a valid ``Ranking``. After formatting, the sink
    re-checks non-increasing-by-rank and the id-ascending tie-break **at the
    emitted precision** and raises this rather than emit a rejectable file.
    """


@runtime_checkable
class SubmissionSinkPort(Protocol):
    """Faithfully serialize a ``Ranking`` to the submission CSV."""

    def write(self, ranking: Ranking) -> SubmissionReceipt:
        """Write ``ranking`` to a validator-compliant UTF-8 (no BOM) CSV.

        Emits ``candidate_id,rank,score,reasoning`` plus exactly 100 data rows
        with ``\\n`` endings and deterministic ``score`` precision; the
        id-ascending order within equal scores is taken verbatim from
        ``ranking`` (never re-ordered). Writes to a temp file then atomically
        renames so a partial submission is never observable.

        Args:
            ranking: An invariant-checked ``Ranking`` (exactly 100, unique
                ranks/ids, non-increasing scores, id-ascending tie-break).

        Returns:
            A ``SubmissionReceipt`` with the row count, bytes written, and
            output sha256.

        Raises:
            SubmissionWriteError: the file could not be written.
            SubmissionContractError: a serialized invariant would be violated
                at the emitted precision.
        """
        ...


__all__: tuple[str, ...] = (
    "SubmissionContractError",
    "SubmissionSinkPort",
    "SubmissionWriteError",
)
