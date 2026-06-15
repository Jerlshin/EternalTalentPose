"""``ParquetSemanticVectorStoreAdapter`` — implements ``SemanticVectorStorePort`` (Adapters §5).

Owner layer: adapters (infrastructure — impure IO).
Allowed imports: stdlib typing; ``pyarrow``/numpy; ``domain.ids``; ``ports``.
Forbidden: ``engines``, ``pipelines``, business logic.

O(1) retrieval of precomputed candidate vectors by ``CandidateId`` plus a
zero-copy full-matrix view for the columnar scoring path — the reason R3 is
"lookup," not "encode." The artifact (``embeddings/candidate_vectors.parquet``,
schema ``id, vector(dim)``; unit-norm, id-unique) is reached through the injected
``ArtifactStorePort.locate(...)`` (hash-verified) — never by importing the
artifact-store adapter — preserving the port seam between adapters.

On-disk contract read here: a parquet file with a string id column and a
``fixed_size_list<float32>[dim]`` vector column. The file is opened
memory-mapped; the vector column's contiguous child buffer is exposed as a
read-only ``(N, dim)`` numpy view with no copy. The id column loads fully into an
in-memory ``CandidateId -> row`` index. Vectors are already L2-normalized
offline; the adapter never re-normalizes. A missing candidate is reported as
``None`` / ``missing`` (triggers the encode fallback); structural corruption or a
dimensionality mismatch raises ``VectorStoreError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from redstack.domain.ids import CandidateId
from redstack.ports._types import ArtifactKey, BulkVectorResult, FloatMatrix, FloatVector
from redstack.ports.artifact_store import ArtifactStorePort
from redstack.ports.semantic_index import VectorStoreError


class ParquetSemanticVectorStoreAdapter:
    """Read-only, memory-mapped parquet vector store with an in-memory id index.

    Constructed at R0 from a hash-verified locator obtained via the injected
    ``ArtifactStorePort``. Read-only after open; concurrent reads are safe.
    """

    __slots__ = (
        "_dim",
        "_index",
        "_id_order",
        "_matrix",
        "_table",
        "_values",
    )

    def __init__(
        self,
        store: ArtifactStorePort,
        *,
        vectors_key: ArtifactKey,
        expected_dim: int,
        id_column: str = "candidate_id",
        vector_column: str = "vector",
    ) -> None:
        """Open and validate the vector store via a hash-verified locator.

        Args:
            store: Injected artifact store; ``locate(vectors_key)`` yields a
                verified path the adapter memory-maps.
            vectors_key: Manifest key of the candidate-vectors parquet.
            expected_dim: The embedding dimensionality from the manifest;
                asserted against the parquet's fixed-size-list width.
            id_column: Name of the string id column.
            vector_column: Name of the ``fixed_size_list<float32>`` column.

        Raises:
            VectorStoreError: the store is unreadable/corrupt, the columns are
                missing or mis-typed, ids are duplicated, or the dimensionality
                disagrees with ``expected_dim``.
        """
        locator = store.locate(vectors_key)
        path = self._locator_path(locator.opaque_handle, vectors_key)

        try:
            # pyarrow.parquet.read_table is untyped in pyarrow's stubs; the
            # result is bound to a typed local so no Any propagates downstream.
            # The repo pyproject additionally scopes a `pyarrow.*` mypy override.
            table: pa.Table = pq.read_table(  # type: ignore[no-untyped-call]
                str(path), columns=[id_column, vector_column], memory_map=True
            )
        except (OSError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise VectorStoreError(
                f"cannot read vector store {path!s}: {exc}"
            ) from exc

        if id_column not in table.column_names or vector_column not in table.column_names:
            raise VectorStoreError(
                f"vector store {path!s} missing columns "
                f"{id_column!r}/{vector_column!r}; has {table.column_names}"
            )

        ids = self._read_ids(table.column(id_column))
        matrix, values, dim = self._read_matrix(
            table.column(vector_column), expected_dim, len(ids)
        )

        index: dict[CandidateId, int] = {}
        for row, cid in enumerate(ids):
            if cid in index:
                raise VectorStoreError(f"duplicate candidate id in store: {cid!r}")
            index[cid] = row

        self._dim: Final[int] = dim
        self._index: Final[dict[CandidateId, int]] = index
        self._id_order: Final[tuple[CandidateId, ...]] = ids
        self._matrix: Final[FloatMatrix] = matrix
        # Hold the arrow table + values buffer alive so the zero-copy numpy
        # view backing ``_matrix`` remains valid for the store's lifetime.
        self._table: Final[pa.Table] = table
        self._values: Final[pa.Array] = values

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
        if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
            raise VectorStoreError(
                f"id column must be string-typed, got {column.type}"
            )
        ids: list[CandidateId] = []
        for value in column.to_pylist():
            if not isinstance(value, str):
                raise VectorStoreError("id column contains a non-string / null value")
            ids.append(CandidateId(value))
        return tuple(ids)

    @staticmethod
    def _read_matrix(
        column: pa.ChunkedArray, expected_dim: int, n_rows: int
    ) -> tuple[FloatMatrix, pa.Array, int]:
        if column.num_chunks == 0:
            empty = np.empty((0, expected_dim), dtype=np.float32)
            empty.flags.writeable = False
            return empty, pa.array([], type=pa.float32()), expected_dim

        fsl: pa.Array = (
            column.chunk(0)
            if column.num_chunks == 1
            else pa.concat_arrays(column.chunks)
        )
        if not pa.types.is_fixed_size_list(fsl.type):
            raise VectorStoreError(
                f"vector column must be fixed_size_list<float32>, got {fsl.type}"
            )
        dim = int(fsl.type.list_size)
        if dim != expected_dim:
            raise VectorStoreError(
                f"vector dim {dim} != expected embedding dim {expected_dim}"
            )

        values = fsl.values
        if values.type != pa.float32():
            raise VectorStoreError(
                f"vector values must be float32, got {values.type}"
            )
        if values.null_count != 0:
            raise VectorStoreError("vector values contain nulls")
        if len(values) != n_rows * dim:
            raise VectorStoreError(
                f"vector buffer length {len(values)} != rows*dim {n_rows * dim}"
            )

        try:
            flat = values.to_numpy(zero_copy_only=True)
        except pa.ArrowInvalid as exc:
            raise VectorStoreError(
                f"vector buffer is not zero-copy contiguous: {exc}"
            ) from exc
        if flat.dtype != np.float32:
            raise VectorStoreError(f"vector dtype {flat.dtype} != float32")

        matrix = flat.reshape(n_rows, dim)
        matrix.flags.writeable = False
        return matrix, values, dim

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