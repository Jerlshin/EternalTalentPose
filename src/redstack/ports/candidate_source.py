
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from redstack.domain.errors import DomainError
from redstack.ports._types import SourceRecord


class CandidateSourceError(DomainError):
    """Cannot open or decompress the source, or a low-level IO error occurred.

    Per-line decode failures are *not* this error — they surface as
    ``SourceMalformed`` records so the pipeline can apply skip-vs-abort policy.
    """


@runtime_checkable
class CandidateSourcePort(Protocol):
    """Lazy, single-pass, order-preserving stream of raw candidate records."""

    def stream(self) -> Iterator[SourceRecord]:
        """Yield records lazily in file order, one at a time.

        Each ``SourceOk`` carries the undecoded record plus ``source_index``
        (enumeration order) and ``line_no``; each ``SourceMalformed`` carries
        ``line_no`` and an error string. Calling ``stream`` again starts a
        fresh, independent pass (a new iterator).

        Raises:
            CandidateSourceError: the source cannot be opened or decompressed,
                or an IO error occurs mid-stream.
        """
        ...

    def count(self) -> int | None:
        """Return the record count, or ``None`` if unknown without a full pass."""
        ...


__all__: tuple[str, ...] = ("CandidateSourceError", "CandidateSourcePort")
