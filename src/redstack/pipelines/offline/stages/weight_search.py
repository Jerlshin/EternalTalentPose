

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from redstack.domain.enums import ScoreComponent
from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "WeightSearchStage",
    "weight_search_stage",
    "composite_score",
)

#: Challenge composite weights (Part 8).
_W_NDCG10: Final[float] = 0.50
_W_NDCG50: Final[float] = 0.30
_W_MAP: Final[float] = 0.15
_W_P10: Final[float] = 0.05

#: Tier-A floors: the JD-critical components may not fall below this share of the
#: weight mass (Feature Layer Part 10 Tier-A; prevents the search from zeroing the
#: discriminating components on a tiny gold set).
_TIER_A_COMPONENTS: Final[frozenset[ScoreComponent]] = frozenset(
    {ScoreComponent.SEMANTIC_FIT, ScoreComponent.CAREER_FIT}
)
_TIER_A_FLOOR: Final[float] = 0.12

#: Ridge regularization strength (toward the uniform prior) — overfitting guard.
_RIDGE: Final[float] = 0.02
#: Relevance tier (0..4) above which a candidate counts as "relevant" for MAP/P@10.
_RELEVANT_TIER: Final[int] = 3
#: Default search budget if config does not pin one.
_DEFAULT_BUDGET: Final[int] = 2000


def _dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of an ordered gain sequence."""
    return float(
        sum(g / np.log2(i + 2.0) for i, g in enumerate(gains))
    )


def _ndcg_at_k(order_tiers: Sequence[int], k: int) -> float:
    """NDCG@k over tiers ordered by the candidate ranking (gain = 2^tier − 1)."""
    if not order_tiers:
        return 0.0
    gains = [float(2**t - 1) for t in order_tiers[:k]]
    ideal = sorted((float(2**t - 1) for t in order_tiers), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return _dcg(gains) / idcg


def _map_score(order_tiers: Sequence[int]) -> float:
    """Mean average precision treating tier ≥ ``_RELEVANT_TIER`` as relevant."""
    relevant_total = sum(1 for t in order_tiers if t >= _RELEVANT_TIER)
    if relevant_total == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, tier in enumerate(order_tiers):
        if tier >= _RELEVANT_TIER:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / relevant_total


def _precision_at_k(order_tiers: Sequence[int], k: int) -> float:
    """Precision@k treating tier ≥ ``_RELEVANT_TIER`` as relevant."""
    if k == 0:
        return 0.0
    top = order_tiers[:k]
    if not top:
        return 0.0
    return sum(1 for t in top if t >= _RELEVANT_TIER) / len(top)


def composite_score(order_tiers: Sequence[int]) -> float:
    """The challenge composite over a tier sequence ordered by a candidate ranking.

    ``0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`` — the single scalar the
    weight search maximizes. Pure; deterministic given the order.
    """
    return (
        _W_NDCG10 * _ndcg_at_k(order_tiers, 10)
        + _W_NDCG50 * _ndcg_at_k(order_tiers, 50)
        + _W_MAP * _map_score(order_tiers)
        + _W_P10 * _precision_at_k(order_tiers, 10)
    )


class WeightSearchStage(OfflineStage):
    """O9 — seeded regularized listwise search → locked scoring weights."""

    stage_id = "O9"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        manifest = self._load_json(ctx, "feature_manifest")
        layout_version = self._layout_version(manifest, ctx)
        component_order = tuple(ScoreComponent)
        comp_matrix, tiers, labeled_ids = self._build_component_matrix(
            ctx, manifest, component_order
        )
        train_idx, val_idx = self._split_indices(ctx, labeled_ids)

        best_weights, cv_ndcg, stability = self._search(
            ctx, comp_matrix, tiers, train_idx, val_idx, component_order
        )
        ablation = self._ablation_deltas(
            comp_matrix, tiers, val_idx, best_weights, component_order
        )

        weights_map = {
            component.value: round(float(best_weights[i]), 6)
            for i, component in enumerate(component_order)
        }
        weights_artifact = self.emit_yaml(
            ctx,
            "scoring_weights",
            {"layout_version": layout_version, "weights": weights_map},
        )
        report_artifact = self.emit_json(
            ctx,
            "calibration_report",
            {
                "cv_ndcg": round(cv_ndcg, 6),
                "weight_stability": round(stability, 6),
                "ablation_deltas": ablation,
                "labeled": len(labeled_ids),
                "train": len(train_idx),
                "val": len(val_idx),
                "tier_a_floor": _TIER_A_FLOOR,
                "ridge": _RIDGE,
            },
        )
        metrics: dict[str, object] = {
            "cv_ndcg": round(cv_ndcg, 6),
            "weight_stability": round(stability, 6),
            "layout_version": layout_version,
            "labeled": len(labeled_ids),
            "weights": weights_map,
        }
        return StageResult(
            artifacts=(weights_artifact, report_artifact), metrics=metrics
        )

    # ------------------------------------------------------------------ #
    # Input loading + component aggregation                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json(ctx: OfflinePipelineContext, key: str) -> Mapping[str, object]:
        try:
            raw = ctx.artifact_store.load_text(key)
        except Exception as exc:  # noqa: BLE001
            msg = f"cannot load {key} for weight calibration: {exc}"
            raise ArtifactContractError(msg) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"{key} is not valid JSON: {exc}"
            raise ArtifactContractError(msg) from exc
        if not isinstance(parsed, Mapping):
            raise ArtifactContractError(f"{key} must be an object")
        return parsed

    @staticmethod
    def _layout_version(
        manifest: Mapping[str, object], ctx: OfflinePipelineContext
    ) -> str:
        """Return the manifest layout_version, asserting it matches the context.

        Raises:
            ArtifactContractError: the manifest layout_version is missing or
                disagrees with the registry/feature layout (Part 8 / §8 coherence).
        """
        version = manifest.get("layout_version")
        if not isinstance(version, str) or not version:
            raise ArtifactContractError("feature_manifest missing layout_version")
        if version != ctx.layout_version:
            msg = (
                f"feature_manifest layout_version {version!r} != context "
                f"layout_version {ctx.layout_version!r}"
            )
            raise ArtifactContractError(msg)
        return version

    def _build_component_matrix(
        self,
        ctx: OfflinePipelineContext,
        manifest: Mapping[str, object],
        component_order: Sequence[ScoreComponent],
    ) -> tuple[npt.NDArray[np.float32], list[int], list[str]]:
        """Aggregate the labeled candidates' features into a ``(M, 7)`` matrix.

        Reads the O14 feature_snapshot parquet (the exact online CQV), maps each
        feature column to a ``ScoreComponent`` via the manifest group map, and sets
        each component value to the mean of its mapped columns (a deterministic,
        linear, online-consistent aggregation). Returns the matrix, the aligned
        gold tiers, and the labeled ids — in sorted-id order.
        """
        gold = self._load_json(ctx, "gold_labels")
        labels = gold.get("labels")
        if not isinstance(labels, Mapping) or not labels:
            raise ArtifactContractError("gold_labels has no labels")

        feature_ids, group_of = self._manifest_layout(manifest)
        comp_index = {c: i for i, c in enumerate(component_order)}
        col_to_component = self._map_columns(feature_ids, group_of, comp_index)

        rows, all_ids = self._read_snapshot(ctx, feature_ids)
        labeled: list[tuple[str, int]] = []
        for cid in sorted(labels):
            record = labels[cid]
            if isinstance(record, Mapping) and cid in rows:
                tier = record.get("tier")
                if isinstance(tier, int) and not isinstance(tier, bool):
                    labeled.append((cid, tier))
        if len(labeled) < 2:
            raise ArtifactContractError(
                "fewer than 2 labeled candidates present in the feature snapshot"
            )

        n_comp = len(component_order)
        matrix = np.zeros((len(labeled), n_comp), dtype=np.float32)
        counts = np.zeros(n_comp, dtype=np.float32)
        for ci, comp_idx in col_to_component.items():
            counts[comp_idx] += 1.0
        for r, (cid, _tier) in enumerate(labeled):
            feats = rows[cid]
            acc = np.zeros(n_comp, dtype=np.float32)
            for ci, comp_idx in col_to_component.items():
                acc[comp_idx] += feats[ci]
            with np.errstate(invalid="ignore", divide="ignore"):
                matrix[r] = np.where(counts > 0, acc / np.maximum(counts, 1.0), 0.0)
        tiers = [t for _cid, t in labeled]
        ids = [cid for cid, _t in labeled]
        return np.ascontiguousarray(matrix), tiers, ids

    @staticmethod
    def _manifest_layout(
        manifest: Mapping[str, object],
    ) -> tuple[tuple[str, ...], Mapping[str, str]]:
        features = manifest.get("features")
        group_of = manifest.get("group_of")
        if not isinstance(features, (list, tuple)) or not features:
            raise ArtifactContractError("feature_manifest missing 'features'")
        if not isinstance(group_of, Mapping):
            raise ArtifactContractError("feature_manifest missing 'group_of'")
        feature_ids = tuple(str(f) for f in features)
        gmap = {str(k): str(v) for k, v in group_of.items()}
        return feature_ids, gmap

    @staticmethod
    def _map_columns(
        feature_ids: Sequence[str],
        group_of: Mapping[str, str],
        comp_index: Mapping[ScoreComponent, int],
    ) -> dict[int, int]:
        """Map each feature column index → its ``ScoreComponent`` index.

        The mapping is by the feature's group name prefix → component. Groups not
        recognized as a component contribute to ``CREDIBILITY`` (a neutral catch-all
        consistent with the structural-credibility role). Deterministic.
        """
        group_to_component = {
            "retr": ScoreComponent.SEMANTIC_FIT,
            "rank": ScoreComponent.SEMANTIC_FIT,
            "ir": ScoreComponent.SEMANTIC_FIT,
            "jd": ScoreComponent.SEMANTIC_FIT,
            "skill": ScoreComponent.SKILL_MATCH,
            "career": ScoreComponent.CAREER_FIT,
            "pvs": ScoreComponent.CAREER_FIT,
            "exp": ScoreComponent.EXPERIENCE_FIT,
            "edu": ScoreComponent.EDUCATION_FIT,
            "cons": ScoreComponent.CREDIBILITY,
            "hp": ScoreComponent.CREDIBILITY,
            "bhv": ScoreComponent.CREDIBILITY,
            "arch": ScoreComponent.ARCHETYPE_FIT,
        }
        col_to_component: dict[int, int] = {}
        for col, fid in enumerate(feature_ids):
            group = group_of.get(fid, "")
            prefix = group.split(".", 1)[0] if group else fid.split(".", 1)[0]
            component = group_to_component.get(prefix, ScoreComponent.CREDIBILITY)
            col_to_component[col] = comp_index[component]
        return col_to_component

    def _read_snapshot(
        self, ctx: OfflinePipelineContext, feature_ids: Sequence[str]
    ) -> tuple[dict[str, npt.NDArray[np.float32]], list[str]]:
        """Read the feature_snapshot parquet → {id: (D,) row} + id order."""
        spec = self.registry.spec("feature_snapshot")
        path = (ctx.artifacts_root / spec.relative_path).resolve()
        if not path.is_file():
            raise ArtifactContractError(f"feature_snapshot not found at {path}")
        import pyarrow.parquet as pq

        table = pq.read_table(path)  # type: ignore[no-untyped-call]
        columns = table.column_names
        value_cols = sorted(
            (c for c in columns if c.startswith("f") and c[1:].isdigit()),
            key=lambda c: int(c[1:]),
        )
        if len(value_cols) != len(feature_ids):
            msg = (
                f"feature_snapshot has {len(value_cols)} value cols != "
                f"{len(feature_ids)} manifest features"
            )
            raise ArtifactContractError(msg)
        ids = [str(v) for v in table.column("id").to_pylist()]
        mat = np.column_stack(
            [table.column(c).to_numpy(zero_copy_only=False) for c in value_cols]
        ).astype(np.float32, copy=False)
        return {cid: mat[i] for i, cid in enumerate(ids)}, ids

    # ------------------------------------------------------------------ #
    # Split alignment                                                    #
    # ------------------------------------------------------------------ #
    def _split_indices(
        self, ctx: OfflinePipelineContext, labeled_ids: Sequence[str]
    ) -> tuple[list[int], list[int]]:
        """Map the O8 calibration split onto labeled-row indices (leakage-free)."""
        split = self._load_json(ctx, "calibration_split")
        train_ids = set(self._as_str_list(split.get("train_ids")))
        val_ids = set(self._as_str_list(split.get("val_ids")))
        train_idx = [i for i, cid in enumerate(labeled_ids) if cid in train_ids]
        val_idx = [i for i, cid in enumerate(labeled_ids) if cid in val_ids]
        if not train_idx or not val_idx:
            # Fall back to a deterministic 75/25 index split if the artifact split
            # does not intersect the snapshot-present labeled set.
            cut = max(1, int(len(labeled_ids) * 0.75))
            train_idx = list(range(cut))
            val_idx = list(range(cut, len(labeled_ids)))
            if not val_idx:
                val_idx = [len(labeled_ids) - 1]
                train_idx = list(range(len(labeled_ids) - 1))
        return train_idx, val_idx

    @staticmethod
    def _as_str_list(value: object) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if isinstance(v, str)]
        return []

    # ------------------------------------------------------------------ #
    # The search                                                         #
    # ------------------------------------------------------------------ #
    def _search(
        self,
        ctx: OfflinePipelineContext,
        comp_matrix: npt.NDArray[np.float32],
        tiers: Sequence[int],
        train_idx: Sequence[int],
        val_idx: Sequence[int],
        component_order: Sequence[ScoreComponent],
    ) -> tuple[npt.NDArray[np.float64], float, float]:
        """Seeded simplex search maximizing the regularized composite.

        Draws candidate weight vectors from a Dirichlet on the simplex (seeded via
        the entropy port), projects each onto the Tier-A floors, scores the
        train-fold composite minus the ridge penalty, and keeps the best. Weight
        stability is the cosine between the train-best and val-best weight vectors
        (overfitting check). Returns (best_weights, cv_ndcg, stability).
        """
        budget = self._budget(ctx)
        generator = ctx.entropy.numpy_generator("weight_search")
        n_comp = len(component_order)
        floor = self._floor_vector(component_order)

        def evaluate(weights: npt.NDArray[np.float64], idx: Sequence[int]) -> float:
            scores = comp_matrix[list(idx)] @ weights.astype(np.float32)
            order = np.argsort(-scores, kind="stable")
            ordered_tiers = [tiers[idx[i]] for i in order.tolist()]
            base = composite_score(ordered_tiers)
            penalty = _RIDGE * float(np.sum((weights - 1.0 / n_comp) ** 2))
            return base - penalty

        best_w = np.full(n_comp, 1.0 / n_comp, dtype=np.float64)
        best_train = evaluate(best_w, train_idx)
        best_val_w = best_w.copy()
        best_val = evaluate(best_w, val_idx)
        alpha = np.ones(n_comp)
        for _ in range(budget):
            sample = generator.dirichlet(alpha)
            weights = self._project_floor(sample, floor)
            train_obj = evaluate(weights, train_idx)
            if train_obj > best_train:
                best_train = train_obj
                best_w = weights
            val_obj = evaluate(weights, val_idx)
            if val_obj > best_val:
                best_val = val_obj
                best_val_w = weights
        cv_ndcg = (
            _ndcg_at_k(
                self._ordered_tiers(comp_matrix, tiers, val_idx, best_w), 10
            )
        )
        stability = self._cosine(best_w, best_val_w)
        return best_w, cv_ndcg, stability

    @staticmethod
    def _ordered_tiers(
        comp_matrix: npt.NDArray[np.float32],
        tiers: Sequence[int],
        idx: Sequence[int],
        weights: npt.NDArray[np.float64],
    ) -> list[int]:
        scores = comp_matrix[list(idx)] @ weights.astype(np.float32)
        order = np.argsort(-scores, kind="stable")
        return [tiers[idx[i]] for i in order.tolist()]

    def _budget(self, ctx: OfflinePipelineContext) -> int:
        offline = ctx.config.offline
        if offline is not None and offline.search_budget > 0:
            return int(offline.search_budget)
        return _DEFAULT_BUDGET

    @staticmethod
    def _floor_vector(
        component_order: Sequence[ScoreComponent],
    ) -> npt.NDArray[np.float64]:
        floor = np.zeros(len(component_order), dtype=np.float64)
        for i, component in enumerate(component_order):
            if component in _TIER_A_COMPONENTS:
                floor[i] = _TIER_A_FLOOR
        return floor

    @staticmethod
    def _project_floor(
        weights: npt.NDArray[np.float64], floor: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Project a simplex point so each component ≥ its floor, with Σ=1 exactly.

        Reserves the total floor mass ``Σfloor``, distributes the remaining
        ``1 − Σfloor`` proportionally to the (renormalized) sample, then adds the
        floors back. Each floored component ends at ``floor_i + share_i ≥ floor_i``
        and the result sums to 1 — so the Tier-A floors hold *after* normalization
        (the naive lift-then-renormalize approach can push a floored component back
        under its floor).
        """
        floor_mass = float(np.sum(floor))
        if floor_mass >= 1.0:
            # Degenerate (floors alone exhaust the simplex) — return the floors
            # renormalized; should never happen with sane Tier-A floors.
            return floor / floor_mass
        free_mass = 1.0 - floor_mass
        sample_sum = float(np.sum(weights))
        if sample_sum <= 0.0:
            base = np.full_like(weights, 1.0 / len(weights))
        else:
            base = weights / sample_sum
        return floor + base * free_mass

    @staticmethod
    def _cosine(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _ablation_deltas(
        self,
        comp_matrix: npt.NDArray[np.float32],
        tiers: Sequence[int],
        val_idx: Sequence[int],
        weights: npt.NDArray[np.float64],
        component_order: Sequence[ScoreComponent],
    ) -> dict[str, float]:
        """Per-component composite delta when its weight is zeroed (reasoning cue).

        The drop in the val composite from removing each component — a deterministic
        contribution estimate folded into the calibration report.
        """
        base = composite_score(
            self._ordered_tiers(comp_matrix, tiers, val_idx, weights)
        )
        deltas: dict[str, float] = {}
        for i, component in enumerate(component_order):
            ablated = weights.copy()
            ablated[i] = 0.0
            total = float(np.sum(ablated))
            if total > 0.0:
                ablated = ablated / total
            score = composite_score(
                self._ordered_tiers(comp_matrix, tiers, val_idx, ablated)
            )
            deltas[component.value] = round(base - score, 6)
        return deltas


def weight_search_stage() -> WeightSearchStage:
    """Factory: construct the O9 weight-search stage."""
    return WeightSearchStage()