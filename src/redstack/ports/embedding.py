"""``EmbeddingModelPort`` — the single text->vector seam (§1).

Owner layer: ports.
Allowed imports: stdlib typing, ``domain.errors``, ``_types``, numpy typing.

Turns already-composed text documents into normalized dense vectors. Used
offline to *build* candidate/anchor vectors and online only as a *fallback*
for candidates absent from the vector store. The port does not store vectors,
compute similarity, compose the input text, choose pooling, or touch the
network — those live elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from redstack.domain.errors import DomainError
from redstack.ports._types import FloatMatrix


class EmbeddingError(DomainError):
    """An ``encode`` runtime failure.

    The pipeline decides whether a fallback miss is fatal. The adapter must
    raise rather than return zero vectors to mask a failure.
    """


@runtime_checkable
class EmbeddingModelPort(Protocol):
    """Deterministic, L2-normalizing encoder with a stable model identity."""

    @property
    def dim(self) -> int:
        """The fixed output dimensionality (must agree with the manifest)."""
        ...

    @property
    def model_id(self) -> str:
        """A stable identifier for the underlying model, for provenance."""
        ...

    def encode(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> FloatMatrix:
        """Encode pre-composed documents into a ``(len(texts), dim)`` matrix.

        The output is ``float32`` with each row L2-normalized to unit length
        within epsilon, and **row order equals input order** irrespective of
        any internal batching. An empty string is valid input and encodes to
        the model's representation (never raises, never returns zeros to mask a
        failure). ``batch_size`` is an adapter throughput hint that does not
        affect results.

        Args:
            texts: The pre-composed input documents; the caller assembles them.
            batch_size: Optional adapter batching hint; never changes output.

        Returns:
            A read-only ``float32`` ``FloatMatrix`` of shape ``(len(texts), dim)``.

        Raises:
            EmbeddingError: the encode operation failed at runtime.
        """
        ...


__all__: tuple[str, ...] = ("EmbeddingError", "EmbeddingModelPort")
