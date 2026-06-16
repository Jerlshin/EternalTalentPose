"""``SemanticEngine`` — R3 retrieval / archetype assignment (``engines/semantic.py``).

Owner layer: engines. The **only** port-dependent engine: pure *given* its injected
``SemanticVectorStorePort`` (lookup) and ``EmbeddingModelPort`` (encode fallback).
Allowed imports: ``domain``, ``ports``, ``features``, ``config.schema``, numpy.
Forbidden: ``adapters``, ``pipelines``, ``observability``, sibling engines, network,
clock, online RNG.

Logical→physical (Repo §9): the ``CandidateRetrievalEngine``. Lookup-first
(``view_all``/``get_many`` over the mmap'd store), encode-fallback only on store
misses (``EmbeddingModelPort.encode`` — the onnx adapter is offline-capable; no
network online). All similarity/archetype math is **vectorized numpy** (matmul +
argmin), float32, fixed reduction order, ties resolved by ``AnchorId`` /
``ArchetypeId`` ascending — so output is thread-count-invariant.

Outputs ``SemanticProfile`` (anchor similarities, positive/negative fit,
``net_semantic_fit``, ``vector_ref``) and ``ArchetypeAssignment`` per candidate.
The bulk path (``compute``) returns columnar arrays the pipeline folds into the
``(N, D)`` CQV; per-candidate domain objects materialize for survivors/top-K only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, final

import numpy as np
import numpy.typing as npt

from redstack.domain.candidate.archetype import ArchetypeAssignment
from redstack.domain.candidate.semantic import SemanticProfile, VectorRef
from redstack.domain.ids import AnchorId, ArchetypeId, CandidateId, Similarity, UnitScore
from redstack.features.view import clamp_unit
from redstack.ports._types import FloatMatrix, FloatVector
from redstack.ports.embedding import EmbeddingModelPort
from redstack.ports.semantic_index import SemanticVectorStorePort

_SIM_LO: Final[float] = -1.0
_SIM_HI: Final[float] = 1.0


def _clamp_similarity(value: float) -> float:
    return _SIM_LO if value < _SIM_LO else _SIM_HI if value > _SIM_HI else value


@final
@dataclass(frozen=True)
class AnchorSet:
    """JD anchor vectors split by polarity (rows L2-normalized, ids id-ascending).

    Authored in ``configs/anchors/jd_anchors.yaml`` → ``anchor_vectors.npy`` (O6),
    loaded and injected by the pipeline at R0. Ids are sorted ascending so that a
    plain ``argmax`` resolves ties by ``AnchorId`` ascending (determinism).
    """

    positive_ids: tuple[AnchorId, ...]
    positive_matrix: FloatMatrix  # (P, dim)
    negative_ids: tuple[AnchorId, ...]
    negative_matrix: FloatMatrix  # (Nn, dim)

    def __post_init__(self) -> None:
        if list(self.positive_ids) != sorted(self.positive_ids):
            raise ValueError("AnchorSet.positive_ids must be ascending")
        if list(self.negative_ids) != sorted(self.negative_ids):
            raise ValueError("AnchorSet.negative_ids must be ascending")


@final
@dataclass(frozen=True)
class ArchetypeSpace:
    """O7 KMeans centroids + target-archetype set (ids id-ascending)."""

    centroids: FloatMatrix  # (K, dim)
    archetype_ids: tuple[ArchetypeId, ...]
    target_archetypes: frozenset[ArchetypeId]
    labels: Mapping[ArchetypeId, str]

    def __post_init__(self) -> None:
        if list(self.archetype_ids) != sorted(self.archetype_ids):
            raise ValueError("ArchetypeSpace.archetype_ids must be ascending")


@final
@dataclass(frozen=True)
class SemanticArrays:
    """Columnar bulk result aligned to the input matrix row order (for CQV fold)."""

    positive_fit: npt.NDArray[np.float32]
    negative_fit: npt.NDArray[np.float32]
    net_fit: npt.NDArray[np.float32]
    best_positive_index: npt.NDArray[np.int64]
    nearest_centroid_index: npt.NDArray[np.int64]
    centroid_distance: npt.NDArray[np.float32]
    membership_confidence: npt.NDArray[np.float32]


@final
class SemanticEngine:
    """Stateless retrieval engine; all state arrives via ports + injected anchors."""

    def __init__(
        self,
        *,
        store: SemanticVectorStorePort,
        embedder: EmbeddingModelPort,
        anchors: AnchorSet,
        archetypes: ArchetypeSpace,
        vector_store_key: str = "embeddings/candidate_vectors",
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._anchors = anchors
        self._archetypes = archetypes
        self._store_key = vector_store_key

    # ------------------------------------------------------------ vector source
    def resolve_vectors(
        self, ids: Sequence[CandidateId], documents: Mapping[CandidateId, str]
    ) -> tuple[FloatMatrix, tuple[CandidateId, ...]]:
        """Lookup-first gather; encode store misses via the fallback embedder.

        Returns a matrix aligned to ``ids`` order and the tuple of ids that fell
        back to the encoder (for metrics). A miss with no composed document is a
        pipeline error (the candidate cannot be scored) and is left to the caller.
        """
        bulk = self._store.get_many(ids)
        found_set = set(bulk.found)
        rows: list[FloatVector] = []
        found_iter = iter(bulk.vectors)
        misses: list[CandidateId] = []
        miss_docs: list[str] = []
        miss_positions: list[int] = []
        for position, cid in enumerate(ids):
            if cid in found_set:
                rows.append(next(found_iter))
            else:
                misses.append(cid)
                miss_docs.append(documents[cid])
                miss_positions.append(position)
                rows.append(np.zeros(self._store.dim, dtype=np.float32))
        if miss_docs:
            encoded = self._embedder.encode(miss_docs)
            for slot, position in enumerate(miss_positions):
                rows[position] = encoded[slot]
        matrix = np.vstack(rows).astype(np.float32, copy=False)
        return matrix, tuple(misses)

    # ----------------------------------------------------------- vectorized core
    def compute(self, matrix: FloatMatrix) -> SemanticArrays:
        """Vectorized similarities + nearest-centroid over an ``(N, dim)`` matrix."""
        pos = matrix @ self._anchors.positive_matrix.T  # (N, P)
        neg = matrix @ self._anchors.negative_matrix.T  # (N, Nn)

        positive_fit = np.clip(pos.max(axis=1), _SIM_LO, _SIM_HI).astype(np.float32)
        best_positive_index = pos.argmax(axis=1).astype(np.int64)  # first max == min id
        negative_fit = np.clip(neg.max(axis=1), _SIM_LO, _SIM_HI).astype(np.float32)

        net_fit = np.clip(
            (positive_fit - negative_fit + 1.0) * 0.5, 0.0, 1.0
        ).astype(np.float32)

        nearest_idx, distance, confidence = self._archetype_columns(matrix)

        return SemanticArrays(
            positive_fit=positive_fit,
            negative_fit=negative_fit,
            net_fit=net_fit,
            best_positive_index=best_positive_index,
            nearest_centroid_index=nearest_idx,
            centroid_distance=distance,
            membership_confidence=confidence,
        )

    def _archetype_columns(
        self, matrix: FloatMatrix
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        centroids = self._archetypes.centroids  # (K, dim)
        # Squared Euclidean distances via the (a-b)^2 expansion (vectorized).
        sq = (
            (matrix * matrix).sum(axis=1, keepdims=True)
            - 2.0 * (matrix @ centroids.T)
            + (centroids * centroids).sum(axis=1)[np.newaxis, :]
        )
        sq = np.maximum(sq, 0.0)
        dists = np.sqrt(sq)  # (N, K)
        nearest = dists.argmin(axis=1).astype(np.int64)  # first min == min ArchetypeId
        k = dists.shape[1]
        d1 = dists[np.arange(dists.shape[0]), nearest].astype(np.float32)
        if k == 1:
            confidence = np.ones(dists.shape[0], dtype=np.float32)
        else:
            partitioned = np.partition(dists, 1, axis=1)
            d2 = partitioned[:, 1].astype(np.float32)
            denom = d1 + d2
            confidence = np.where(denom > 0.0, d2 / denom, 1.0).astype(np.float32)
            confidence = np.clip(confidence, 0.0, 1.0)
        return nearest, d1, confidence

    # ---------------------------------------------------- per-candidate objects
    def profile_for(
        self, vector: FloatVector, *, row_index: int
    ) -> tuple[SemanticProfile, ArchetypeAssignment]:
        """Materialize the two domain slices for one candidate (survivors/top-K)."""
        row = vector.reshape(1, -1).astype(np.float32, copy=False)
        arrays = self.compute(row)

        anchor_sims = self._anchor_similarity_map(vector)
        best_idx = int(arrays.best_positive_index[0])
        best_anchor: AnchorId | None = (
            self._anchors.positive_ids[best_idx] if self._anchors.positive_ids else None
        )
        semantic = SemanticProfile(
            anchor_similarities=anchor_sims,
            positive_fit=Similarity(float(arrays.positive_fit[0])),
            negative_fit=Similarity(float(arrays.negative_fit[0])),
            net_semantic_fit=UnitScore(clamp_unit(float(arrays.net_fit[0]))),
            best_positive_anchor=best_anchor,
            vector_ref=VectorRef(
                store_key=self._store_key,
                row_index=row_index,
                dim=int(vector.shape[0]),
            ),
        )

        a_idx = int(arrays.nearest_centroid_index[0])
        archetype_id = self._archetypes.archetype_ids[a_idx]
        secondary = self._secondary_archetype(vector, primary_index=a_idx)
        archetype = ArchetypeAssignment(
            archetype_id=archetype_id,
            distance=float(arrays.centroid_distance[0]),
            membership_confidence=UnitScore(
                clamp_unit(float(arrays.membership_confidence[0]))
            ),
            secondary_archetype=secondary,
            label=self._archetypes.labels.get(archetype_id),
            is_target_archetype=archetype_id in self._archetypes.target_archetypes,
        )
        return semantic, archetype

    def _anchor_similarity_map(
        self, vector: FloatVector
    ) -> Mapping[AnchorId, Similarity]:
        sims: dict[AnchorId, Similarity] = {}
        pos = vector @ self._anchors.positive_matrix.T
        for anchor_id, value in zip(self._anchors.positive_ids, pos, strict=True):
            sims[anchor_id] = Similarity(_clamp_similarity(float(value)))
        neg = vector @ self._anchors.negative_matrix.T
        for anchor_id, value in zip(self._anchors.negative_ids, neg, strict=True):
            sims[anchor_id] = Similarity(_clamp_similarity(float(value)))
        return sims

    def _secondary_archetype(
        self, vector: FloatVector, *, primary_index: int
    ) -> ArchetypeId | None:
        centroids = self._archetypes.centroids
        if centroids.shape[0] < 2:
            return None
        diff = centroids - vector[np.newaxis, :]
        dists = np.sqrt(np.maximum((diff * diff).sum(axis=1), 0.0))
        order = np.argsort(dists, kind="stable")
        for idx in order:
            if int(idx) != primary_index:
                return self._archetypes.archetype_ids[int(idx)]
        return None


__all__: tuple[str, ...] = (
    "AnchorSet",
    "ArchetypeSpace",
    "SemanticArrays",
    "SemanticEngine",
)