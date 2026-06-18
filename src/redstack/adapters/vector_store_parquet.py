"""``ParquetSemanticVectorStoreAdapter`` — implements ``SemanticVectorStorePort`` (Adapters §5).

Owner layer: adapters (infrastructure — impure IO).
Allowed imports: stdlib typing; ``pyarrow``/numpy; ``domain.ids``; ``ports``.
Forbidden: ``engines``, ``pipelines``, business logic.

O(1) retrieval of precomputed candidate vectors by ``CandidateId`` plus a
full-matrix view for the columnar scoring path — the reason R3 is "lookup,"
not "encode." The artifact (``embeddings/candidate_vectors.parquet``, schema
``id, v0..v{dim-1}``; unit-norm, id-unique) is reached through the injected
``ArtifactStorePort.locate(...)`` (hash-verified) — never by importing the
artifact-store adapter — preserving the port seam between adapters.

On-disk contract read here: a parquet file with a string id column and one
``float32`` column per vector dimension (``v0, v1, ..., v{dim-1}`` — the
layout :meth:`~redstack.pipelines.offline.stages.OfflineStage.emit_vector_parquet`
writes), not a single ``fixed_size_list`` column. The file is opened
memory-mapped; the per-dimension columns are stacked into a ``(N, dim)``
numpy array. The id column loads fully into an in-memory ``CandidateId ->
row`` index. Vectors are already L2-normalized offline; the adapter never
re-normalizes. A missing candidate is reported as ``None`` / ``missing``
(triggers the encode fallback); structural corruption or a dimensionality
mismatch raises ``VectorStoreError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from redstack.domain.ids import CandidateId
from redstack.ports._types import (
    ArtifactKey,
    BulkVectorResult,
    FloatMatrix,
    FloatVector,
)
from redstack.ports.artifact_store import ArtifactStorePort
from redstack.ports.semantic_index import VectorStoreError


class ParquetSemanticVectorStoreAdapter:
    """Read-only, memory-mapped parquet vector store with an in-memory id index.

    Constructed at R0 from a hash-verified locator obtained via the injected
    ``ArtifactStorePort``. Read-only after open; concurrent reads are safe.
    """

    __slots__ = (
        "_dim",
        "_id_order",
        "_index",
        "_matrix",
    )

    def __init__(
        self,
        store: ArtifactStorePort,
        *,
        vectors_key: ArtifactKey,
        expected_dim: int,
        id_column: str = "id",
        vector_column_prefix: str = "v",
    ) -> None:
        """Open and validate the vector store via a hash-verified locator.

        Args:
            store: Injected artifact store; ``locate(vectors_key)`` yields a
                verified path the adapter memory-maps.
            vectors_key: Manifest key of the candidate-vectors parquet.
            expected_dim: The embedding dimensionality from the manifest; also
                the number of per-dimension vector columns expected.
            id_column: Name of the string id column.
            vector_column_prefix: Prefix of the per-dimension float32 columns
                (``f"{prefix}{j}"`` for ``j`` in ``range(expected_dim)``).

        Raises:
            VectorStoreError: the store is unreadable/corrupt, a column is
                missing or mis-typed, ids are duplicated, or the column count
                disagrees with ``expected_dim``.
        """
        locator = store.locate(vectors_key)
        path = self._locator_path(locator.opaque_handle, vectors_key)
        vector_columns = [f"{vector_column_prefix}{j}" for j in range(expected_dim)]

        try:
            # pyarrow.parquet.read_table is untyped in pyarrow's stubs; the
            # result is bound to a typed local so no Any propagates downstream.
            # The repo pyproject additionally scopes a `pyarrow.*` mypy override.
            table: pa.Table = pq.read_table(  # type: ignore[no-untyped-call]
                str(path), columns=[id_column, *vector_columns], memory_map=True
            )
        except (OSError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise VectorStoreError(f"cannot read vector store {path!s}: {exc}") from exc

        missing = [
            c for c in (id_column, *vector_columns) if c not in table.column_names
        ]
        if missing:
            raise VectorStoreError(
                f"vector store {path!s} missing column(s) {missing}; "
                f"has {table.column_names}"
            )

        ids = self._read_ids(table.column(id_column))
        matrix = self._read_matrix(table, vector_columns, expected_dim, len(ids))

        index: dict[CandidateId, int] = {}
        for row, cid in enumerate(ids):
            if cid in index:
                raise VectorStoreError(f"duplicate candidate id in store: {cid!r}")
            index[cid] = row

        self._dim: Final[int] = expected_dim
        self._index: Final[dict[CandidateId, int]] = index
        self._id_order: Final[tuple[CandidateId, ...]] = ids
        self._matrix: Final[FloatMatrix] = matrix

    # ------------------------------------------------------------------ #
    # Construction helpers.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _locator_path(handle: object, key: ArtifactKey) -> Path:
        if isinstance(handle, Path):
            return handle
        if isinstance(handle, str):
            return Path(handle)
        raise VectorStoreError(
            f"locator for {key!r} carries a non-path handle: {type(handle).__name__}"
        )

    @staticmethod
    def _read_ids(column: pa.ChunkedArray) -> tuple[CandidateId, ...]:
        if not (
            pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
        ):
            raise VectorStoreError(f"id column must be string-typed, got {column.type}")
        ids: list[CandidateId] = []
        for value in column.to_pylist():
            if not isinstance(value, str):
                raise VectorStoreError("id column contains a non-string / null value")
            ids.append(CandidateId(value))
        return tuple(ids)

    @staticmethod
    def _read_matrix(
        table: pa.Table,
        vector_columns: Sequence[str],
        expected_dim: int,
        n_rows: int,
    ) -> FloatMatrix:
        if n_rows == 0:
            empty = np.empty((0, expected_dim), dtype=np.float32)
            empty.flags.writeable = False
            return empty

        rows: list[np.ndarray] = []
        for name in vector_columns:
            column = table.column(name)
            chunk: pa.Array = (
                column.chunk(0)
                if column.num_chunks == 1
                else pa.concat_arrays(column.chunks)
            )
            if chunk.type != pa.float32():
                raise VectorStoreError(
                    f"vector column {name!r} must be float32, got {chunk.type}"
                )
            if chunk.null_count != 0:
                raise VectorStoreError(f"vector column {name!r} contains nulls")
            if len(chunk) != n_rows:
                raise VectorStoreError(
                    f"vector column {name!r} has {len(chunk)} rows; expected {n_rows}"
                )
            values = chunk.to_numpy(zero_copy_only=False)
            if values.dtype != np.float32:
                raise VectorStoreError(
                    f"vector column {name!r} dtype {values.dtype} != float32"
                )
            rows.append(values)

        matrix: FloatMatrix = np.ascontiguousarray(
            np.column_stack(rows).astype(np.float32, copy=False)
        )
        matrix.flags.writeable = False
        return matrix

    # ------------------------------------------------------------------ #
    # Port surface.
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        """The vector dimensionality."""
        return self._dim

    def contains(self, cid: CandidateId) -> bool:
        """Return whether ``cid`` has a precomputed vector in the store."""
        return cid in self._index

    def get(self, cid: CandidateId) -> FloatVector | None:
        """Return ``cid``'s read-only ``(dim,)`` vector, or ``None`` if absent."""
        row = self._index.get(cid)
        if row is None:
            return None
        vector: FloatVector = self._matrix[row]
        return vector

    def get_many(self, cids: Sequence[CandidateId]) -> BulkVectorResult:
        """Gather vectors in request order, partitioning found from missing."""
        rows: list[int] = []
        found: list[CandidateId] = []
        missing: list[CandidateId] = []
        for cid in cids:
            row = self._index.get(cid)
            if row is None:
                missing.append(cid)
            else:
                rows.append(row)
                found.append(cid)

        if rows:
            gathered: FloatMatrix = np.ascontiguousarray(
                self._matrix[rows], dtype=np.float32
            )
        else:
            gathered = np.empty((0, self._dim), dtype=np.float32)

        return BulkVectorResult(
            vectors=gathered, found=tuple(found), missing=tuple(missing)
        )

    def view_all(self) -> tuple[FloatMatrix, Sequence[CandidateId]]:
        """Return the full read-only matrix and its aligned id order (no copy)."""
        return self._matrix, self._id_order


if TYPE_CHECKING:
    from redstack.ports.semantic_index import SemanticVectorStorePort

    # Compile-time structural conformance to the frozen port surface.
    _PORT_CONFORMANCE: type[SemanticVectorStorePort] = ParquetSemanticVectorStoreAdapter


__all__: tuple[str, ...] = ("ParquetSemanticVectorStoreAdapter",)
