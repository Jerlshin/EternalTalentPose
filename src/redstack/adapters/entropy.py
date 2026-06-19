

from __future__ import annotations

import hashlib
from datetime import date
from typing import TYPE_CHECKING, Final

import numpy as np

from redstack.ports.rng import EntropyDisabledError

#: Width (bytes) of the sha256 prefix folded into a numpy seed (64-bit).
_SUBSEED_BYTES: Final[int] = 8


def _derive_subseed(seed: int, label: str) -> int:
    """Deterministically fold ``(seed, label)`` into a stable 64-bit sub-seed."""
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:_SUBSEED_BYTES], "big")


class OfflineEntropy:
    """Seeded, labeled, reproducible entropy for the offline pipeline."""

    __slots__ = ("_seed", "_as_of")

    def __init__(self, seed: int, as_of: date) -> None:
        """Bind the run seed and the fixed reference date.

        Args:
            seed: The run seed from config.
            as_of: The fixed reference date from config (the only clock).
        """
        self._seed: Final[int] = seed
        self._as_of: Final[date] = as_of

    @property
    def seed(self) -> int:
        """The run seed."""
        return self._seed

    def as_of(self) -> date:
        """The fixed reference date from config."""
        return self._as_of

    def derive(self, label: str) -> int:
        """Return a stable sub-seed deterministically derived from ``(seed, label)``."""
        return _derive_subseed(self._seed, label)

    def numpy_generator(self, label: str) -> np.random.Generator:
        """Return a seeded PCG64 ``Generator`` for the named substream."""
        return np.random.default_rng(self.derive(label))


class OnlineEntropy:
    """RNG-disabled entropy for the online pipeline: ``as_of`` only.

    Any attempt to draw randomness raises :class:`EntropyDisabledError`,
    enforcing the online RNG-free guarantee (ties break by ``candidate_id``).
    """

    __slots__ = ("_seed", "_as_of")

    def __init__(self, seed: int, as_of: date) -> None:
        """Bind the recorded seed (provenance only) and the fixed reference date."""
        self._seed: Final[int] = seed
        self._as_of: Final[date] = as_of

    @property
    def seed(self) -> int:
        """The recorded run seed (audit/provenance; never used to draw randomness)."""
        return self._seed

    def as_of(self) -> date:
        """The fixed reference date from config."""
        return self._as_of

    def derive(self, label: str) -> int:
        """Always raise: online randomness is forbidden.

        Raises:
            EntropyDisabledError: online RNG is disabled.
        """
        raise EntropyDisabledError(
            f"online RNG is disabled; derive({label!r}) is not permitted"
        )

    def numpy_generator(self, label: str) -> np.random.Generator:
        """Always raise: online randomness is forbidden.

        Raises:
            EntropyDisabledError: online RNG is disabled.
        """
        raise EntropyDisabledError(
            f"online RNG is disabled; numpy_generator({label!r}) is not permitted"
        )


if TYPE_CHECKING:
    from redstack.ports.rng import DeterministicEntropyPort

    # Compile-time structural conformance to the frozen port surface.
    _OFFLINE_CONFORMANCE: type[DeterministicEntropyPort] = OfflineEntropy
    _ONLINE_CONFORMANCE: type[DeterministicEntropyPort] = OnlineEntropy


__all__: tuple[str, ...] = ("OfflineEntropy", "OnlineEntropy")