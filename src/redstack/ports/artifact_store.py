"""``ArtifactStorePort`` — load and integrity-verify build artifacts (§3).

Owner layer: ports.
Allowed imports: stdlib typing, ``domain.errors``, ``_types``, numpy typing.

The contract backbone between the offline build and the online run. Every
loader verifies the artifact's sha256 against the manifest before returning;
integrity or contract violations raise (fail-fast, no degraded mode). The
adapter — not this port — touches the filesystem and mmaps locators.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import DomainError
from redstack.ports._types import ArtifactKey, ArtifactLocator, Manifest


class ManifestError(DomainError):
    """The manifest is missing, unparseable, or fails its self-hash check.

    Distinct from :class:`ArtifactContractError`, which signals a per-artifact
    integrity or cross-artifact coherence failure. Both are fatal at R0.
    """


@runtime_checkable
class ArtifactStorePort(Protocol):
    """Verified, typed access to the immutable artifact set by manifest key.

    Implementations parse and self-verify ``MANIFEST.json``, then verify each
    artifact's sha256 while streaming it during load. Verification is
    idempotent and may be result-cached per key. No method interprets artifact
    semantics — callers receive bytes, text, JSON, ndarrays, or a verified
    locator and decode meaning themselves.

    Verdicts-vs-failures: there is no "missing as data" here — an absent or
    corrupt artifact is always a failure that aborts the run.
    """

    def manifest(self) -> Manifest:
        """Return the parsed, self-verified manifest.

        Raises:
            ManifestError: the manifest is missing, unparseable, or its
                ``manifest_sha256`` self-hash does not match.
        """
        ...

    def verify_all(self) -> None:
        """Eagerly verify every required artifact's sha256 and coherence (§8).

        An optional fail-fast pass at R0 that forces integrity failures before
        any downstream work begins.

        Raises:
            ManifestError: manifest self-integrity failure.
            ArtifactContractError: a missing key, sha256 mismatch, incompatible
                schema/layout version, or cross-artifact incoherence.
        """
        ...

    def load_bytes(self, key: ArtifactKey) -> bytes:
        """Return the verified raw bytes for ``key``.

        Raises:
            ArtifactContractError: missing key or sha256 mismatch.
        """
        ...

    def load_text(self, key: ArtifactKey) -> str:
        """Return the verified UTF-8 text for ``key``.

        Raises:
            ArtifactContractError: missing key or sha256 mismatch.
        """
        ...

    def load_json(self, key: ArtifactKey) -> Mapping[str, object]:
        """Return the verified, parsed JSON object for ``key``.

        Raises:
            ArtifactContractError: missing key or sha256 mismatch.
        """
        ...

    def load_npy(self, key: ArtifactKey) -> npt.NDArray[np.float32]:
        """Return the verified ndarray for a ``.npy`` ``key``.

        The offline build pipeline guarantees all arrays are serialized as float32,
        ensuring online semantic operations are strictly typed end-to-end.

        Raises:
            ArtifactContractError: missing key or sha256 mismatch.
        """
        ...

    def locate(self, key: ArtifactKey) -> ArtifactLocator:
        """Return a hash-verified, opaque locator for a consumer to mmap.

        Used for large artifacts (vectors, ONNX model) the store does not
        materialize itself.

        Raises:
            ArtifactContractError: missing key or sha256 mismatch.
        """
        ...


__all__: tuple[str, ...] = (
    "ArtifactStorePort",
    "ManifestError",
)