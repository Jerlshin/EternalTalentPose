"""O7 — Archetype Discovery.

Owner stage: O7 (Offline Pipeline Part 6). Cluster the precomputed candidate
embedding space into the **10 core archetypes** (retrieval / ranking / recsys /
data / ML / researcher / consultant / keyword-stuffer / product / startup
engineer) with scikit-learn KMeans, **seeded securely** via
``OfflineEntropy.numpy_generator("kmeans")`` so the build is reproducible.
Fingerprint each cluster and tag the JD-desired (target) archetypes, then emit
the embedding-space centroids the online nearest-centroid assignment consumes.

Algorithm (Part 6, deps O13a):

1. Load ``candidate_vectors.parquet`` (the O13a unit-norm ``(N, dim)`` matrix +
   id order) from the artifact tree.
2. Fit ``KMeans(n_clusters=10)`` with a seeded ``random_state`` derived from the
   offline entropy port; ``n_init`` fixed for determinism.
3. **Fingerprint** each cluster: size, the centroid (embedding space), and the
   top discriminating dimensions (centroid components farthest from the global
   mean) — the signature reasoning cites.
4. Assign deterministic labels to the 10 archetypes and mark the target set
   (retrieval/ranking/recsys/product/startup) per the JD.

Outputs (validated against the registry):

* ``centroids.npy`` — ``(k, dim)`` float32, contiguous, **unit-normalized** rows
  (so online nearest-centroid is a dot product); ``_v_centroids`` (2-D float32,
  dim == embedding.dim — §8 coherence).
* ``archetypes.json`` — id → label, size, fingerprint, target flag;
  ``_v_archetypes`` (ids contiguous from 0).

Determinism (Part 6): seeded init, fixed k, fixed ``n_init``; centroid **row
order is canonicalized** (sorted by a stable key) so the artifact bytes and the
``ArchetypeId`` assignment are reproducible across runs; ties broken by id.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "ArchetypeDiscoveryStage",
    "archetype_discovery_stage",
)

#: The fixed number of archetypes (Part 6: the 10 expected clusters).
_K: Final[int] = 10
#: Fixed KMeans restarts for determinism (best inertia across seeded inits).
_N_INIT: Final[int] = 10
#: Max iterations per init (bounded, deterministic).
_MAX_ITER: Final[int] = 300
#: Number of top discriminating dimensions recorded per cluster fingerprint.
_TOP_DIMS: Final[int] = 12

#: Canonical archetype labels (order is the *catalog* order, not the cluster
#: order — clusters are matched to labels by stable centroid sort below).
_ARCHETYPE_LABELS: Final[tuple[str, ...]] = (
    "retrieval_engineer",
    "ranking_engineer",
    "recsys_engineer",
    "data_engineer",
    "ml_engineer",
    "researcher",
    "consultant",
    "keyword_stuffer",
    "product_engineer",
    "startup_engineer",
)
#: The JD-desired (target) archetypes — boosted online (Part 6).
_TARGET_LABELS: Final[frozenset[str]] = frozenset(
    {
        "retrieval_engineer",
        "ranking_engineer",
        "recsys_engineer",
        "product_engineer",
        "startup_engineer",
    }
)


class ArchetypeDiscoveryStage(OfflineStage):
    """O7 — seeded KMeans over candidate vectors → centroids + archetype catalog."""

    stage_id = "O7"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        vectors, _ids = self._load_candidate_vectors(ctx)
        n, dim = vectors.shape
        if dim != ctx.embedding_model.dim:
            msg = (
                f"candidate vector dim {dim} != embedding.dim "
                f"{ctx.embedding_model.dim} (Ports §8 coherence)"
            )
            raise ArtifactContractError(msg)
        if n < _K:
            msg = f"need at least {_K} candidates to fit {_K} archetypes, got {n}"
            raise ArtifactContractError(msg)

        labels, raw_centroids, inertia = self._fit_kmeans(ctx, vectors)
        order = self._canonical_order(raw_centroids)
        centroids = self._normalize_centroids(raw_centroids[order])
        catalog = self._build_catalog(labels, order, centroids, n)

        art_centroids = self.emit_array(ctx, "centroids", centroids)
        art_archetypes = self.emit_json(ctx, "archetypes", {"archetypes": catalog})
        metrics: dict[str, object] = {
            "k": _K,
            "candidates_clustered": int(n),
            "dim": int(dim),
            "inertia": round(float(inertia), 6),
            "target_archetypes": sorted(_TARGET_LABELS),
        }
        return StageResult(
            artifacts=(art_centroids, art_archetypes), metrics=metrics
        )

    # ------------------------------------------------------------------ #
    # Input loading                                                      #
    # ------------------------------------------------------------------ #
    def _load_candidate_vectors(
        self, ctx: OfflinePipelineContext
    ) -> tuple[npt.NDArray[np.float32], list[str]]:
        """Load the O13a ``(N, dim)`` matrix + id order from the vector parquet.

        Reads the parquet at the registry-resolved, containment-checked path
        under ``artifacts_root`` (offline stage; pyarrow is the offline
        dependency). Columns are ``id`` + ``v0..v{dim-1}`` (the O13a layout).

        Raises:
            ArtifactContractError: the parquet is missing or malformed.
        """
        spec = self.registry.spec("candidate_vectors")
        path = (ctx.artifacts_root / spec.relative_path).resolve()
        if not path.is_file():
            msg = f"candidate_vectors parquet not found at {path}"
            raise ArtifactContractError(msg)

        import pyarrow.parquet as pq

        try:
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
        except Exception as exc:  # noqa: BLE001 — surface as a contract failure.
            msg = f"failed to read candidate_vectors parquet: {exc}"
            raise ArtifactContractError(msg) from exc
        columns = table.column_names
        if "id" not in columns:
            raise ArtifactContractError("candidate_vectors missing 'id' column")
        vector_cols = sorted(
            (c for c in columns if c.startswith("v")),
            key=lambda c: int(c[1:]),
        )
        if not vector_cols:
            raise ArtifactContractError("candidate_vectors has no vector columns")
        ids = [str(v) for v in table.column("id").to_pylist()]
        matrix = np.column_stack(
            [table.column(c).to_numpy(zero_copy_only=False) for c in vector_cols]
        ).astype(np.float32, copy=False)
        return np.ascontiguousarray(matrix), ids

    # ------------------------------------------------------------------ #
    # KMeans (seeded)                                                    #
    # ------------------------------------------------------------------ #
    def _fit_kmeans(
        self, ctx: OfflinePipelineContext, vectors: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float32], float]:
        """Fit seeded KMeans; return (labels, raw_centroids, inertia).

        The seed is drawn from ``OfflineEntropy.numpy_generator("kmeans")`` so the
        init is reproducible and independent of other seeded substreams. ``n_init``
        is fixed; sklearn selects the best-inertia run deterministically given the
        seed.
        """
        from sklearn.cluster import KMeans  # type: ignore[import-untyped]

        generator = ctx.entropy.numpy_generator("kmeans")
        # Derive a fixed 32-bit seed from the entropy generator (deterministic).
        seed = int(generator.integers(0, 2**31 - 1))
        model = KMeans(
            n_clusters=_K,
            init="k-means++",
            n_init=_N_INIT,
            max_iter=_MAX_ITER,
            random_state=seed,
            algorithm="lloyd",
        )
        labels = model.fit_predict(vectors)
        return (
            labels.astype(np.int_, copy=False),
            model.cluster_centers_.astype(np.float32, copy=False),
            float(model.inertia_),
        )

    # ------------------------------------------------------------------ #
    # Determinism: canonical centroid order                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _canonical_order(centroids: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
        """Return a stable row permutation sorting centroids lexicographically.

        KMeans cluster numbering is arbitrary across runs even when seeded; a
        canonical order (lexicographic by centroid coordinates) makes the emitted
        centroid rows — and thus every ``ArchetypeId`` — reproducible. Ties
        (numerically identical centroids) resolve by original index.
        """
        keys = [tuple(row.tolist()) for row in centroids]
        order = sorted(range(len(keys)), key=lambda i: (keys[i], i))
        return np.asarray(order, dtype=np.int_)

    @staticmethod
    def _normalize_centroids(
        centroids: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """L2-normalize centroid rows (contiguous float32) for dot-product assignment.

        Raises:
            ArtifactContractError: a centroid is a zero vector (degenerate empty
                cluster) — cannot be normalized, signals a broken fit.
        """
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ArtifactContractError("a KMeans centroid is the zero vector")
        normalized = (centroids / norms).astype(np.float32, copy=False)
        return np.ascontiguousarray(normalized)

    # ------------------------------------------------------------------ #
    # Catalog + fingerprints                                             #
    # ------------------------------------------------------------------ #
    def _build_catalog(
        self,
        labels: npt.NDArray[np.int_],
        order: npt.NDArray[np.int_],
        centroids: npt.NDArray[np.float32],
        n: int,
    ) -> dict[str, object]:
        """Build the contiguous-id archetype catalog with sizes + fingerprints.

        ``ArchetypeId`` i corresponds to centroid row i (post canonical sort).
        Each record carries the assigned label, the cluster size (count of
        candidates whose original cluster maps to this canonical id), the
        top-discriminating dimensions (the fingerprint reasoning cites), and the
        target flag. Ids are ``"0".."k-1"`` (contiguous — the ``_v_archetypes``
        invariant).
        """
        # Map original cluster index -> canonical id (its position in ``order``).
        canonical_of = {int(orig): pos for pos, orig in enumerate(order.tolist())}
        sizes = [0] * _K
        for original_label in labels.tolist():
            sizes[canonical_of[int(original_label)]] += 1

        global_mean = centroids.mean(axis=0)
        catalog: dict[str, object] = {}
        for cid in range(_K):
            label = _ARCHETYPE_LABELS[cid] if cid < len(_ARCHETYPE_LABELS) else f"cluster_{cid}"
            fingerprint = self._fingerprint(centroids[cid], global_mean)
            catalog[str(cid)] = {
                "archetype_id": cid,
                "label": label,
                "size": sizes[cid],
                "size_fraction": round(sizes[cid] / n, 6) if n else 0.0,
                "is_target_archetype": label in _TARGET_LABELS,
                "fingerprint": fingerprint,
            }
        return catalog

    @staticmethod
    def _fingerprint(
        centroid: npt.NDArray[np.float32], global_mean: npt.NDArray[np.float32]
    ) -> dict[str, object]:
        """Compute a cluster's discriminating-dimension fingerprint.

        The top dimensions are those whose centroid coordinate deviates most from
        the global centroid mean (the cluster's distinctive directions). Returned
        as ``{dim_index: deviation}`` pairs (sorted by |deviation| desc) — a
        compact, deterministic signature for reasoning ("matches the
        product-ML-engineer archetype").
        """
        deviation = centroid - global_mean
        magnitude = np.abs(deviation)
        top = np.argsort(-magnitude)[:_TOP_DIMS]
        return {
            "top_dims": [int(i) for i in top.tolist()],
            "top_deviations": [round(float(deviation[i]), 6) for i in top.tolist()],
        }


def archetype_discovery_stage() -> ArchetypeDiscoveryStage:
    """Factory: construct the O7 archetype-discovery stage."""
    return ArchetypeDiscoveryStage()