
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np

from redstack.domain.errors import DomainError


class EntropyDisabledError(DomainError):
    """Raised by the online variant when ``derive``/``numpy_generator`` is called.

    Enforces the online RNG-free guarantee: only ``as_of`` is permitted on the
    hot path, since output ordering must never depend on randomness.
    """


@runtime_checkable
class DeterministicEntropyPort(Protocol):
    """Seeds, seeded numpy generators, labeled substreams, and the ``as_of`` date."""

    @property
    def seed(self) -> int:
        """The run seed (an int sourced from config)."""
        ...

    def as_of(self) -> date:
        """The fixed reference date from config (the only clock engines see)."""
        ...

    def derive(self, label: str) -> int:
        """Return a stable sub-seed deterministically derived from ``(seed, label)``.

        Args:
            label: Names an independent substream.

        Returns:
            A reproducible integer sub-seed.

        Raises:
            EntropyDisabledError: called on the online (RNG-disabled) variant.
        """
        ...

    def numpy_generator(self, label: str) -> np.random.Generator:
        """Return a seeded numpy ``Generator`` for the named substream.

        Generators are not thread-safe; a caller needing per-thread randomness
        requests a distinct ``label`` rather than sharing one generator.

        Args:
            label: Names an independent, reproducible substream.

        Returns:
            A seeded :class:`numpy.random.Generator`.

        Raises:
            EntropyDisabledError: called on the online (RNG-disabled) variant.
        """
        ...


__all__: tuple[str, ...] = ("DeterministicEntropyPort", "EntropyDisabledError")
