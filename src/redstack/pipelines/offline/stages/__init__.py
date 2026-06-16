"""Offline compute stages (O0–O18) + the common typed stage base.

Owner layer: pipelines (offline). Each module here is one stage; this package
``__init__`` owns the shared :class:`OfflineStage` base every stage inherits, so
the "no loose dicts" rule (Sprint-2 bound) is structural: a stage cannot enter
the :class:`OfflinePipelineRunner` plan without conforming to the typed
``StageCallable`` contract this base implements.

``OfflineStage`` provides:

* the ``stage_id`` / ``stage_version`` properties the runner's staleness key
  reads (Offline Pipeline Part 11);
* the ``__call__(ctx, upstream) -> StageResult`` entrypoint the runner invokes,
  delegating to the subclass's typed :meth:`OfflineStage._run`;
* deterministic artifact emission helpers (:meth:`OfflineStage.emit_json`,
  :meth:`OfflineStage.emit_array`) that write under ``ctx.artifacts_root`` via
  an atomic temp+rename, compute the streaming sha256 that lands in
  ``MANIFEST.json``, and return the registry-validatable :class:`ArtifactPayload`
  the runner checks before manifesting.

Stages never construct loose dicts as their public surface: inputs arrive on the
immutable :class:`OfflinePipelineContext`, outputs leave as registered
:class:`ArtifactPayload`s keyed against the :data:`OFFLINE_ARTIFACT_REGISTRY`.
The base centralizes determinism (sorted-key canonical JSON, fixed float
formatting, ``\\n`` newlines, no trailing whitespace) so artifact bytes — and
therefore their sha256 — are byte-stable across rebuilds.

All artifact paths are resolved through the registry spec, never hand-built, and
are containment-checked against ``artifacts_root`` so a malformed relative path
cannot escape the build tree (defence-in-depth; the adapters enforce this too).
"""

from __future__ import annotations

import hashlib
import io
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import yaml

from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.registry import (
    OFFLINE_ARTIFACT_REGISTRY,
    ArtifactKind,
    OfflineArtifactRegistry,
)
from redstack.pipelines.offline.runner import (
    ArtifactPayload,
    StageReceipt,
    StageResult,
)

if TYPE_CHECKING:
    from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = ("OfflineStage",)

#: Minimum accepted st<->onnx parity cosine for the exported encoder (Ports §9).
_ONNX_PARITY_MIN: float = 0.999


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize a mapping to deterministic, byte-stable UTF-8 JSON.

    Sorted keys, compact separators, ``ensure_ascii=False`` (UTF-8), trailing
    ``\\n``. Non-JSON-native values (e.g. enums) coerce via ``str`` so the writer
    never raises on an audit value; stages keep payloads JSON-native by design.
    """
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return (text + "\n").encode("utf-8")


def _to_plain(value: object) -> object:
    """Recursively coerce mappings/sequences to plain dict/list for YAML dumping.

    Keeps the serializer free of pydantic/enum surprises: enums become their
    ``.value`` (via ``str`` on ``StrEnum``), mappings become sorted-key dicts,
    tuples/lists become lists. Scalars pass through. Pure and deterministic.
    """
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


class OfflineStage(ABC):
    """Common typed base for every offline stage (structural ``StageCallable``).

    Subclasses set :attr:`stage_id` / :attr:`stage_version` as class attributes
    and implement :meth:`_run`. The base wires the runner contract and the
    deterministic artifact-emission helpers; it holds no per-run mutable state
    (a stage instance is reusable and reentrant-safe across the data-parallel
    feature passes — Engine Layer §threading).

    Attributes:
        registry: the artifact catalog used to resolve a key's on-disk path,
            kind, and validator view shape. Defaults to the frozen catalog.
    """

    #: The canonical stage id (``"O0"`` … ``"O18"``); overridden per subclass.
    stage_id: str = ""
    #: The stage's own semver; part of its staleness key (bump to force recompute).
    stage_version: str = "1.0"

    def __init__(
        self, registry: OfflineArtifactRegistry = OFFLINE_ARTIFACT_REGISTRY
    ) -> None:
        if not self.stage_id:
            msg = f"{type(self).__name__} must define a non-empty stage_id"
            raise ValueError(msg)
        self.registry = registry

    # ------------------------------------------------------------------ #
    # Runner entrypoint                                                  #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        """The runner-invoked entrypoint; delegates to the typed :meth:`_run`."""
        return self._run(ctx, upstream)

    @abstractmethod
    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        """Execute the stage's logic and return its produced artifacts + metrics.

        Implementations read inputs from ``ctx`` (ports, roots, registry, layout)
        and the ``upstream`` receipts of their dependencies; they must stream the
        candidate pool (O(1) memory) and emit artifacts via :meth:`emit_json` /
        :meth:`emit_array`. Returning a :class:`StageResult` whose artifacts fail
        the registry validator causes the runner to raise ``ArtifactContractError``.
        """

    # ------------------------------------------------------------------ #
    # Path resolution + containment                                      #
    # ------------------------------------------------------------------ #
    def _resolve_path(self, ctx: OfflinePipelineContext, key: str) -> Path:
        """Resolve a registry key to its absolute, containment-checked path.

        Raises:
            ArtifactContractError: if the resolved path escapes ``artifacts_root``
                (a malformed registry path — defence-in-depth).
        """
        spec = self.registry.spec(key)
        root = ctx.artifacts_root.resolve()
        target = (root / spec.relative_path).resolve()
        if root not in target.parents and target != root:
            msg = (
                f"artifact path for {key!r} escapes artifacts_root: "
                f"{spec.relative_path!r}"
            )
            raise ArtifactContractError(msg)
        return target

    # ------------------------------------------------------------------ #
    # Deterministic artifact emission                                    #
    # ------------------------------------------------------------------ #
    def emit_json(
        self,
        ctx: OfflinePipelineContext,
        key: str,
        payload: Mapping[str, object],
    ) -> ArtifactPayload:
        """Write a JSON artifact deterministically and return its payload record.

        The bytes are canonical (sorted keys, compact, UTF-8, trailing newline)
        so the sha256 is byte-stable across rebuilds. The ``validation_view`` is
        the payload itself (the registry validator inspects the same mapping the
        runner will hash). Write is atomic (temp + ``replace``).

        Raises:
            ArtifactContractError: if the key's registered kind is not JSON.
        """
        spec = self.registry.spec(key)
        if spec.kind is not ArtifactKind.JSON:
            msg = f"emit_json called for {key!r} of kind {spec.kind.value!r}"
            raise ArtifactContractError(msg)
        data = _canonical_json_bytes(payload)
        path = self._resolve_path(ctx, key)
        self._atomic_write_bytes(path, data)
        return ArtifactPayload(
            key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes_written=len(data),
            validation_view=payload,
        )

    def emit_yaml(
        self,
        ctx: OfflinePipelineContext,
        key: str,
        payload: Mapping[str, object],
    ) -> ArtifactPayload:
        """Write a YAML artifact deterministically and return its payload record.

        Used for the authored ``gates/eligibility_rules.yaml`` (O6). The YAML is
        block-style with sorted keys and no flow collections, so the bytes — and
        the sha256 — are stable across rebuilds and pyyaml versions. The
        ``validation_view`` is the same mapping the registry validator inspects
        (the runner validates the structure, not the YAML text). Write is atomic.

        Raises:
            ArtifactContractError: if the key's registered kind is not YAML.
        """
        spec = self.registry.spec(key)
        if spec.kind is not ArtifactKind.YAML:
            msg = f"emit_yaml called for {key!r} of kind {spec.kind.value!r}"
            raise ArtifactContractError(msg)
        text = yaml.safe_dump(
            _to_plain(payload),
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
            width=4096,
        )
        data = text.encode("utf-8")
        path = self._resolve_path(ctx, key)
        self._atomic_write_bytes(path, data)
        return ArtifactPayload(
            key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes_written=len(data),
            validation_view=payload,
        )

    def emit_array(
        self,
        ctx: OfflinePipelineContext,
        key: str,
        array: npt.NDArray[np.float32],
    ) -> ArtifactPayload:
        """Write a ``.npy`` artifact deterministically and return its payload record.

        The array is persisted with ``allow_pickle`` implicitly off (numpy never
        pickles plain float32 arrays) in C-order; the ``validation_view`` carries
        the shape + dtype metadata the registry's npy validators inspect (never
        the raw array — keeps the validator IO-free and numpy-light).

        Raises:
            ArtifactContractError: if the key's kind is not NPY, or the array is
                not float32 / contains non-finite values.
        """
        spec = self.registry.spec(key)
        if spec.kind is not ArtifactKind.NPY:
            msg = f"emit_array called for {key!r} of kind {spec.kind.value!r}"
            raise ArtifactContractError(msg)
        if array.dtype != np.float32:
            msg = f"artifact {key!r} array must be float32, got {array.dtype}"
            raise ArtifactContractError(msg)
        if not np.all(np.isfinite(array)):
            msg = f"artifact {key!r} array contains NaN/inf"
            raise ArtifactContractError(msg)
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
        data = buffer.getvalue()
        path = self._resolve_path(ctx, key)
        self._atomic_write_bytes(path, data)
        view: dict[str, object] = {
            "shape": list(array.shape),
            "dtype": "float32",
        }
        return ArtifactPayload(
            key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes_written=len(data),
            validation_view=view,
        )

    def emit_vector_parquet(
        self,
        ctx: OfflinePipelineContext,
        key: str,
        *,
        ids: Sequence[str],
        vectors: npt.NDArray[np.float32],
        unit_norm_eps: float = 1e-3,
    ) -> ArtifactPayload:
        """Write an embedding-vector ``.parquet`` (id + vector cols), mmap-ready.

        Columns: ``id`` (string) + ``v0..v{dim-1}`` (float32), one row per id in
        the supplied order — the stable id→row order the parquet vector-store
        adapter mmaps and indexes (Adapters §5). Asserts float32, finite, **unit
        L2 norm per row** within ``unit_norm_eps`` (Ports §9 normalization
        guarantee), and **unique ids** (Part 10 ``candidate_vectors`` validation).
        Write is deterministic (fixed column/row order, uncompressed pages) and
        atomic. The ``validation_view`` carries ``columns``/``row_count``/
        ``unique_ids``/``dim`` for ``_v_candidate_vectors``.

        Raises:
            ArtifactContractError: kind/dtype/shape mismatch, non-finite or
                non-unit-norm rows, or duplicate ids.
        """
        spec = self.registry.spec(key)
        if spec.kind is not ArtifactKind.PARQUET:
            msg = f"emit_vector_parquet called for {key!r} of kind {spec.kind.value!r}"
            raise ArtifactContractError(msg)
        n = len(ids)
        if vectors.dtype != np.float32:
            msg = f"artifact {key!r} vectors must be float32, got {vectors.dtype}"
            raise ArtifactContractError(msg)
        if vectors.ndim != 2 or vectors.shape[0] != n:
            msg = f"artifact {key!r} vectors must be (N, dim) aligned to ids"
            raise ArtifactContractError(msg)
        if n == 0:
            msg = f"artifact {key!r} must contain at least one vector"
            raise ArtifactContractError(msg)
        if len(set(ids)) != n:
            msg = f"artifact {key!r} ids must be unique"
            raise ArtifactContractError(msg)
        if not np.all(np.isfinite(vectors)):
            msg = f"artifact {key!r} vectors contain NaN/inf"
            raise ArtifactContractError(msg)
        norms = np.linalg.norm(vectors, axis=1)
        if not np.all(np.abs(norms - 1.0) <= unit_norm_eps):
            worst = float(np.max(np.abs(norms - 1.0)))
            msg = f"artifact {key!r} rows not unit-normalized (max |‖v‖-1|={worst:.2e})"
            raise ArtifactContractError(msg)

        import pyarrow as pa
        import pyarrow.parquet as pq

        dim = int(vectors.shape[1])
        columns: dict[str, pa.Array] = {"id": pa.array(list(ids), type=pa.string())}
        for j in range(dim):
            columns[f"v{j}"] = pa.array(vectors[:, j], type=pa.float32())
        table = pa.table(columns)

        path = self._resolve_path(ctx, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(table, tmp, compression="none")  # type: ignore[no-untyped-call]
            tmp.replace(path)
        except OSError as exc:
            msg = f"failed to write vector parquet at {path}: {exc}"
            raise ArtifactContractError(msg) from exc

        data = path.read_bytes()
        view: dict[str, object] = {
            "columns": list(columns.keys()),
            "row_count": n,
            "unique_ids": True,
            "dim": dim,
        }
        return ArtifactPayload(
            key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes_written=len(data),
            validation_view=view,
        )

    def emit_encoder(
        self,
        ctx: OfflinePipelineContext,
        key: str,
        *,
        onnx_path: Path,
        model_id: str,
        dim: int,
        parity_cosine: float,
    ) -> ArtifactPayload:
        """Register an already-exported ``model/encoder.onnx`` as an artifact.

        The offline embedding adapter exports + parity-verifies the ONNX twin
        (Adapters §4); this helper moves the exported file into the artifact tree
        atomically and records its sha256. The ``validation_view`` carries
        ``model_id`` + ``dim`` (``_v_encoder_onnx``) plus the recorded
        ``parity_cosine`` for the build report.

        Raises:
            ArtifactContractError: kind mismatch, missing source file, or a
                parity cosine below the accepted ε bound (Ports §9, ≥ 0.999).
        """
        spec = self.registry.spec(key)
        if spec.kind is not ArtifactKind.ONNX:
            msg = f"emit_encoder called for {key!r} of kind {spec.kind.value!r}"
            raise ArtifactContractError(msg)
        if not onnx_path.is_file():
            msg = f"exported onnx file not found at {onnx_path}"
            raise ArtifactContractError(msg)
        if parity_cosine < _ONNX_PARITY_MIN:
            msg = (
                f"onnx parity cosine {parity_cosine:.6f} below accepted bound "
                f"{_ONNX_PARITY_MIN} (Ports §9)"
            )
            raise ArtifactContractError(msg)
        data = onnx_path.read_bytes()
        dest = self._resolve_path(ctx, key)
        self._atomic_write_bytes(dest, data)
        view: dict[str, object] = {
            "model_id": model_id,
            "dim": dim,
            "parity_cosine": round(parity_cosine, 6),
        }
        return ArtifactPayload(
            key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes_written=len(data),
            validation_view=view,
        )

    def emit_parquet(
        self,
        ctx: OfflinePipelineContext,
        key: str,
        *,
        ids: Sequence[str],
        values: npt.NDArray[np.float32],
        confidence: npt.NDArray[np.float32],
        group_names: Sequence[str],
    ) -> ArtifactPayload:
        """Write a feature-snapshot ``.parquet`` and return its payload record.

        Columns: ``id`` (string), one ``f<idx>`` column per feature value
        (float32), and one ``c<group>`` column per group confidence (float32).
        The write is deterministic — fixed column order, fixed row order (the
        supplied id order), zstd-free uncompressed pages so bytes are stable
        across pyarrow builds — and atomic (temp + replace). The
        ``validation_view`` carries the schema metadata the registry's
        ``_v_feature_snapshot`` inspects (columns, row_count, has_nan,
        feature_dim) without re-reading the file.

        Raises:
            ArtifactContractError: kind mismatch, dtype/shape mismatch, or any
                non-finite cell (no NaN/inf allowed in the snapshot).
        """
        spec = self.registry.spec(key)
        if spec.kind is not ArtifactKind.PARQUET:
            msg = f"emit_parquet called for {key!r} of kind {spec.kind.value!r}"
            raise ArtifactContractError(msg)
        n = len(ids)
        if values.dtype != np.float32 or confidence.dtype != np.float32:
            msg = f"artifact {key!r} matrices must be float32"
            raise ArtifactContractError(msg)
        if values.ndim != 2 or values.shape[0] != n:
            msg = f"artifact {key!r} values must be (N, D) aligned to ids"
            raise ArtifactContractError(msg)
        if confidence.ndim != 2 or confidence.shape[0] != n:
            msg = f"artifact {key!r} confidence must be (N, G) aligned to ids"
            raise ArtifactContractError(msg)
        if confidence.shape[1] != len(group_names):
            msg = f"artifact {key!r} confidence width must equal group count"
            raise ArtifactContractError(msg)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(confidence)):
            msg = f"artifact {key!r} snapshot contains NaN/inf"
            raise ArtifactContractError(msg)

        import pyarrow as pa
        import pyarrow.parquet as pq

        feature_dim = int(values.shape[1])
        columns: dict[str, pa.Array] = {"id": pa.array(list(ids), type=pa.string())}
        for j in range(feature_dim):
            columns[f"f{j}"] = pa.array(values[:, j], type=pa.float32())
        for g, name in enumerate(group_names):
            columns[f"c_{name}"] = pa.array(confidence[:, g], type=pa.float32())
        table = pa.table(columns)

        path = self._resolve_path(ctx, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(table, tmp, compression="none")  # type: ignore[no-untyped-call]
            tmp.replace(path)
        except OSError as exc:
            msg = f"failed to write parquet artifact at {path}: {exc}"
            raise ArtifactContractError(msg) from exc

        data = path.read_bytes()
        view: dict[str, object] = {
            "columns": list(columns.keys()),
            "row_count": n,
            "has_nan": False,
            "feature_dim": feature_dim,
        }
        return ArtifactPayload(
            key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes_written=len(data),
            validation_view=view,
        )

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        """Write ``data`` to ``path`` via temp + atomic rename; never leave partials.

        Raises:
            ArtifactContractError: on any IO failure (surfaced as a contract
                breach so the runner quarantines the partial output).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as exc:
            msg = f"failed to write artifact at {path}: {exc}"
            raise ArtifactContractError(msg) from exc