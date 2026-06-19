

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.registry import OfflineArtifactRegistry
from redstack.ports._types import (
    ArtifactEntry,
    ArtifactKey,
    ArtifactLocator,
    EmbeddingManifest,
    Manifest,
)

if TYPE_CHECKING:
    from redstack.ports.artifact_store import ArtifactStorePort

__all__: tuple[str, ...] = ("BuildArtifactStore",)

#: Placeholder manifest fields a mid-build synthesis cannot know (audit-only).
_BUILD_TIME_EMBEDDING: Final[EmbeddingManifest] = EmbeddingManifest(model_id="", dim=0)


class BuildArtifactStore:
    """Reads artifacts directly off ``artifacts_root`` by registry key, no manifest.

    Bound once to ``artifacts_root`` + the frozen registry by the offline
    composition root and carried on :class:`OfflinePipelineContext` for the
    whole run. Stateless and re-entrant; every call re-resolves the current
    on-disk content (a later stage's read sees an earlier stage's freshly
    written bytes).
    """

    __slots__ = ("_registry", "_root")

    def __init__(
        self, artifacts_root: Path, registry: OfflineArtifactRegistry
    ) -> None:
        self._root: Final[Path] = artifacts_root
        self._registry: Final[OfflineArtifactRegistry] = registry

    def _resolve(self, key: str) -> Path:
        """Resolve ``key`` to its on-disk path via the registry spec.

        Raises:
            ArtifactContractError: ``key`` is not a registered artifact, or its
                file does not yet exist on disk (the owning stage has not run).
        """
        try:
            spec = self._registry.spec(key)
        except KeyError as exc:
            raise ArtifactContractError(str(exc)) from exc
        path = self._root / spec.relative_path
        if not path.is_file():
            raise ArtifactContractError(
                f"artifact {key!r} not yet produced (expected at {path})"
            )
        return path

    # ------------------------------------------------------------------ #
    # Port surface.
    # ------------------------------------------------------------------ #
    def manifest(self) -> Manifest:
        """Best-effort synthesis over whatever artifacts currently exist.

        Mid-build there is no signed manifest yet (O17 writes the real one);
        this assembles one from the registry specs whose files are already
        present, purely for introspection. No offline stage calls this today.
        """
        entries: dict[ArtifactKey, ArtifactEntry] = {}
        for spec in self._registry.specs:
            path = self._root / spec.relative_path
            if not path.is_file():
                continue
            data = path.read_bytes()
            key = ArtifactKey(spec.key)
            entries[key] = ArtifactEntry(
                key=key,
                path=spec.relative_path,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                schema_version=spec.schema_version,
                kind=spec.kind.value,
            )
        body: dict[str, object] = {
            "manifest_schema_version": "0.0-build",
            "builder_version": "build-in-progress",
            "layout_version": "",
            "embedding": {
                "model_id": _BUILD_TIME_EMBEDDING.model_id,
                "dim": _BUILD_TIME_EMBEDDING.dim,
            },
            "artifacts": {
                str(k): {
                    "path": e.path,
                    "sha256": e.sha256,
                    "bytes": e.size_bytes,
                    "schema_version": e.schema_version,
                    "kind": e.kind,
                }
                for k, e in entries.items()
            },
        }
        canonical = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        self_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return Manifest(
            manifest_schema_version="0.0-build",
            builder_version="build-in-progress",
            created_at="",
            embedding=_BUILD_TIME_EMBEDDING,
            layout_version="",
            artifacts=entries,
            manifest_sha256=self_hash,
        )

    def verify_all(self) -> None:
        """Confirm every currently-present registered artifact is readable.

        No offline stage calls this; provided for structural completeness.
        """
        for spec in self._registry.specs:
            path = self._root / spec.relative_path
            if path.is_file():
                path.read_bytes()

    def load_bytes(self, key: ArtifactKey) -> bytes:
        """Return the raw bytes currently on disk for ``key``."""
        return self._resolve(key).read_bytes()

    def load_text(self, key: ArtifactKey) -> str:
        """Return the UTF-8 text currently on disk for ``key``."""
        path = self._resolve(key)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactContractError(
                f"artifact {key!r} is not valid UTF-8 text: {exc}"
            ) from exc

    def load_json(self, key: ArtifactKey) -> Mapping[str, object]:
        """Return the parsed JSON object currently on disk for ``key``."""
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

    def load_npy(self, key: ArtifactKey) -> npt.NDArray[np.float32]:
        """Return the ndarray currently on disk for a ``.npy`` ``key``."""
        path = self._resolve(key)
        try:
            loaded = np.load(path, allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise ArtifactContractError(
                f"artifact {key!r} is not a loadable .npy array: {exc}"
            ) from exc
        array: npt.NDArray[np.float32] = np.asarray(loaded, dtype=np.float32)
        return array

    def locate(self, key: ArtifactKey) -> ArtifactLocator:
        """Return an opaque locator (the resolved path) for ``key``."""
        path = self._resolve(key)
        return ArtifactLocator(key=key, verified=True, opaque_handle=path)


if TYPE_CHECKING:
    # Compile-time structural conformance to the frozen port surface.
    _PORT_CONFORMANCE: type[ArtifactStorePort] = BuildArtifactStore
