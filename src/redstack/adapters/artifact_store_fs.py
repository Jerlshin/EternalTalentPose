"""``FilesystemArtifactStoreAdapter`` — implements ``ArtifactStorePort`` (Adapters §2).

Owner layer: adapters (infrastructure — impure IO).
Allowed imports: stdlib ``pathlib``/``hashlib``/``json``/``io``/``threading``;
numpy; ``domain.errors``; ``ports``.
Forbidden: ``engines``, ``pipelines``, any ML/network runtime, ``pickle``.

The fail-fast integrity gate between the offline build and the online run. Reads
and self-verifies ``MANIFEST.json`` at construction; every loader resolves a key
to a path strictly contained within the artifact root, verifies the artifact's
sha256 against the manifest entry (streaming for ``locate``/``verify_all``,
read-once for the small typed loaders), and only then materializes. Integrity or
contract violations raise — there is no degraded mode.

Manifest self-hash contract (mirrors the offline packager, Ports §8): the
``manifest_sha256`` is recomputed over the canonical JSON serialization of the
on-disk manifest object with the ``manifest_sha256`` and ``created_at`` keys
removed (``created_at`` is audit-only), using sorted keys and compact
separators, UTF-8 encoded.

Safe deserialization only: ``numpy.load(allow_pickle=False)``, stdlib ``json``,
UTF-8 strict text. No ``pickle``; no arbitrary code execution from any artifact.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import ArtifactContractError
from redstack.ports._types import (
    ArtifactEntry,
    ArtifactKey,
    ArtifactKind,
    ArtifactLocator,
    EmbeddingManifest,
    Manifest,
)
from redstack.ports.artifact_store import ManifestError

#: Read buffer for streaming sha256 of large artifacts.
_HASH_CHUNK: Final[int] = 1 << 20  # 1 MiB
#: Manifest keys excluded from the self-hash canonical serialization.
_SELF_HASH_EXCLUDED: Final[frozenset[str]] = frozenset(
    {"manifest_sha256", "created_at"}
)
#: The closed set of permitted artifact encodings (mirrors ``ArtifactKind``).
_ALLOWED_KINDS: Final[tuple[ArtifactKind, ...]] = (
    "npy",
    "parquet",
    "json",
    "onnx",
    "yaml",
)


def _canonical_json(obj: object) -> bytes:
    """Serialize ``obj`` deterministically: sorted keys, compact, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _as_mapping(value: object, ctx: str) -> dict[str, object]:
    """Narrow a decoded JSON value to a ``dict[str, object]`` or fail."""
    if not isinstance(value, dict):
        raise ManifestError(f"{ctx}: expected a JSON object")
    return {str(key): item for key, item in value.items()}


def _req_str(mapping: Mapping[str, object], key: str, ctx: str) -> str:
    if key not in mapping:
        raise ManifestError(f"{ctx}: missing required field {key!r}")
    value = mapping[key]
    if not isinstance(value, str):
        raise ManifestError(f"{ctx}: field {key!r} must be a string")
    return value


def _req_int(mapping: Mapping[str, object], key: str, ctx: str) -> int:
    if key not in mapping:
        raise ManifestError(f"{ctx}: missing required field {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{ctx}: field {key!r} must be an integer")
    return value


def _req_kind(mapping: Mapping[str, object], key: str, ctx: str) -> ArtifactKind:
    raw = _req_str(mapping, key, ctx)
    # Explicit per-literal returns keep the value typed as ``ArtifactKind``
    # with no cast and no ``Any``.
    if raw == "npy":
        return "npy"
    if raw == "parquet":
        return "parquet"
    if raw == "json":
        return "json"
    if raw == "onnx":
        return "onnx"
    if raw == "yaml":
        return "yaml"
    raise ManifestError(
        f"{ctx}: field {key!r} has unknown kind {raw!r}; "
        f"expected one of {list(_ALLOWED_KINDS)}"
    )


def _major(version: str) -> int:
    """Parse the leading integer major component of a ``major.minor`` version."""
    head = version.split(".", 1)[0]
    try:
        return int(head)
    except ValueError as exc:
        raise ArtifactContractError(
            f"unparseable version major in {version!r}"
        ) from exc


class FilesystemArtifactStoreAdapter:
    """Filesystem-backed, hash-verifying artifact store keyed by manifest key.

    Constructed by the pipeline composition root with the artifact root; reads
    and self-verifies the manifest at init. Read-only thereafter; concurrent
    loads are safe and per-key verification is idempotent and result-cached.
    """

    __slots__ = (
        "_root",
        "_root_resolved",
        "_manifest_path",
        "_supported_manifest_major",
        "_required_keys",
        "_manifest",
        "_verified",
        "_lock",
    )

    def __init__(
        self,
        root: Path,
        *,
        manifest_name: str = "MANIFEST.json",
        supported_manifest_major: int = 1,
        required_keys: frozenset[ArtifactKey] = frozenset(),
    ) -> None:
        """Bind to the artifact root and self-verify the manifest.

        Args:
            root: Artifact root directory containing ``MANIFEST.json`` and the
                artifact tree it references.
            manifest_name: Manifest filename within ``root``.
            supported_manifest_major: The manifest schema major version this
                loader supports; checked in :meth:`verify_all`.
            required_keys: Keys whose presence :meth:`verify_all` enforces.

        Raises:
            ManifestError: the manifest is missing, unparseable, or its self-hash
                does not match.
        """
        self._root: Final[Path] = root
        self._root_resolved: Final[Path] = root.resolve()
        self._manifest_path: Final[Path] = root / manifest_name
        self._supported_manifest_major: Final[int] = supported_manifest_major
        self._required_keys: Final[frozenset[ArtifactKey]] = required_keys
        self._verified: set[ArtifactKey] = set()
        self._lock: Final[threading.Lock] = threading.Lock()
        self._manifest: Final[Manifest] = self._parse_and_verify_manifest()

    # ------------------------------------------------------------------ #
    # Manifest parsing + self-verification.
    # ------------------------------------------------------------------ #
    def _parse_and_verify_manifest(self) -> Manifest:
        try:
            raw_text = self._manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(
                f"cannot read manifest {self._manifest_path!s}: {exc}"
            ) from exc

        try:
            decoded: object = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                f"manifest {self._manifest_path!s} is not valid JSON: {exc.msg}"
            ) from exc

        root_obj = _as_mapping(decoded, "manifest")

        declared = _req_str(root_obj, "manifest_sha256", "manifest")
        canonical_obj = {
            key: value
            for key, value in root_obj.items()
            if key not in _SELF_HASH_EXCLUDED
        }
        computed = hashlib.sha256(_canonical_json(canonical_obj)).hexdigest()
        if computed.lower() != declared.lower():
            raise ManifestError(
                "manifest self-hash mismatch: "
                f"expected {declared!r}, computed {computed!r}"
            )

        embedding_obj = _as_mapping(
            root_obj.get("embedding", None), "manifest.embedding"
        )
        embedding = EmbeddingManifest(
            model_id=_req_str(embedding_obj, "model_id", "manifest.embedding"),
            dim=_req_int(embedding_obj, "dim", "manifest.embedding"),
        )

        artifacts_obj = _as_mapping(
            root_obj.get("artifacts", None), "manifest.artifacts"
        )
        artifacts: dict[ArtifactKey, ArtifactEntry] = {}
        for raw_key, raw_entry in artifacts_obj.items():
            key = ArtifactKey(raw_key)
            ctx = f"manifest.artifacts[{raw_key!r}]"
            entry_obj = _as_mapping(raw_entry, ctx)
            artifacts[key] = ArtifactEntry(
                key=key,
                path=_req_str(entry_obj, "path", ctx),
                sha256=_req_str(entry_obj, "sha256", ctx),
                size_bytes=_req_int(entry_obj, "bytes", ctx),
                schema_version=_req_str(entry_obj, "schema_version", ctx),
                kind=_req_kind(entry_obj, "kind", ctx),
            )

        return Manifest(
            manifest_schema_version=_req_str(
                root_obj, "manifest_schema_version", "manifest"
            ),
            builder_version=_req_str(root_obj, "builder_version", "manifest"),
            created_at=_req_str(root_obj, "created_at", "manifest"),
            embedding=embedding,
            layout_version=_req_str(root_obj, "layout_version", "manifest"),
            artifacts=artifacts,
            manifest_sha256=declared,
        )

    # ------------------------------------------------------------------ #
    # Path containment + hashing.
    # ------------------------------------------------------------------ #
    def _entry(self, key: ArtifactKey) -> ArtifactEntry:
        entry = self._manifest.artifacts.get(key)
        if entry is None:
            raise ArtifactContractError(f"missing artifact key: {key!r}")
        return entry

    def _resolve(self, entry: ArtifactEntry) -> Path:
        """Resolve ``entry.path`` strictly within the artifact root.

        Rejects absolute paths, ``..`` traversal, and symlinks whose real target
        escapes the root.

        Raises:
            ArtifactContractError: the path would escape the artifact root.
        """
        rel = PurePosixPath(entry.path)
        if rel.is_absolute() or any(part == ".." for part in rel.parts):
            raise ArtifactContractError(
                f"artifact {entry.key!r} path escapes root: {entry.path!r}"
            )
        resolved = (self._root / Path(*rel.parts)).resolve()
        if not resolved.is_relative_to(self._root_resolved):
            raise ArtifactContractError(
                f"artifact {entry.key!r} resolves outside root: {entry.path!r}"
            )
        return resolved

    def _read_and_verify(self, key: ArtifactKey) -> bytes:
        """Read the whole artifact, verifying sha256 (read-once load path)."""
        entry = self._entry(key)
        path = self._resolve(entry)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ArtifactContractError(
                f"cannot read artifact {key!r} at {path!s}: {exc}"
            ) from exc
        if key not in self._verified:
            digest = hashlib.sha256(data).hexdigest()
            self._compare_digest(key, entry, digest)
            self._mark_verified(key)
        return data

    def _stream_verify(self, key: ArtifactKey) -> Path:
        """Stream the artifact computing sha256 without materializing it."""
        entry = self._entry(key)
        path = self._resolve(entry)
        if key in self._verified:
            return path
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                while True:
                    chunk = handle.read(_HASH_CHUNK)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except OSError as exc:
            raise ArtifactContractError(
                f"cannot read artifact {key!r} at {path!s}: {exc}"
            ) from exc
        self._compare_digest(key, entry, hasher.hexdigest())
        self._mark_verified(key)
        return path

    @staticmethod
    def _compare_digest(
        key: ArtifactKey, entry: ArtifactEntry, digest: str
    ) -> None:
        if digest.lower() != entry.sha256.lower():
            raise ArtifactContractError(
                f"sha256 mismatch for {key!r}: "
                f"expected {entry.sha256!r}, computed {digest!r}"
            )

    def _mark_verified(self, key: ArtifactKey) -> None:
        with self._lock:
            self._verified.add(key)

    # ------------------------------------------------------------------ #
    # Port surface.
    # ------------------------------------------------------------------ #
    def manifest(self) -> Manifest:
        """Return the parsed, self-verified manifest."""
        return self._manifest

    def verify_all(self) -> None:
        """Eagerly verify schema compatibility, required keys, and every sha256.

        Raises:
            ManifestError: manifest self-integrity failure.
            ArtifactContractError: incompatible manifest schema version, a
                missing required key, or any per-artifact sha256 mismatch.
        """
        major = _major(self._manifest.manifest_schema_version)
        if major != self._supported_manifest_major:
            raise ArtifactContractError(
                "incompatible manifest_schema_version "
                f"{self._manifest.manifest_schema_version!r}: "
                f"loader supports major {self._supported_manifest_major}"
            )

        missing = sorted(
            str(key)
            for key in self._required_keys
            if key not in self._manifest.artifacts
        )
        if missing:
            raise ArtifactContractError(f"missing required artifact keys: {missing}")

        for key in self._manifest.artifacts:
            self._stream_verify(key)

    def load_bytes(self, key: ArtifactKey) -> bytes:
        """Return the verified raw bytes for ``key``."""
        return self._read_and_verify(key)

    def load_text(self, key: ArtifactKey) -> str:
        """Return the verified UTF-8 text for ``key``."""
        data = self._read_and_verify(key)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactContractError(
                f"artifact {key!r} is not valid UTF-8 text: {exc}"
            ) from exc

    def load_json(self, key: ArtifactKey) -> Mapping[str, object]:
        """Return the verified, parsed JSON object for ``key``."""
        text = self.load_text(key)
        try:
            decoded: object = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(
                f"artifact {key!r} is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(decoded, dict):
            raise ArtifactContractError(
                f"artifact {key!r} top-level JSON is not an object"
            )
        return {str(field): value for field, value in decoded.items()}

    def load_npy(self, key: ArtifactKey) -> npt.NDArray[np.generic]:
        """Return the verified ndarray for a ``.npy`` ``key`` (no pickle)."""
        data = self._read_and_verify(key)
        try:
            loaded = np.load(io.BytesIO(data), allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise ArtifactContractError(
                f"artifact {key!r} is not a loadable .npy array: {exc}"
            ) from exc
        if not isinstance(loaded, np.ndarray):
            raise ArtifactContractError(
                f"artifact {key!r} did not decode to a single ndarray"
            )
        array: npt.NDArray[np.generic] = loaded
        return array

    def locate(self, key: ArtifactKey) -> ArtifactLocator:
        """Return a hash-verified, opaque locator (resolved path) for ``key``."""
        path = self._stream_verify(key)
        return ArtifactLocator(key=key, verified=True, opaque_handle=path)


if TYPE_CHECKING:
    from redstack.ports.artifact_store import ArtifactStorePort

    # Compile-time structural conformance to the frozen port surface.
    _PORT_CONFORMANCE: type[ArtifactStorePort] = FilesystemArtifactStoreAdapter


__all__: tuple[str, ...] = ("FilesystemArtifactStoreAdapter",)