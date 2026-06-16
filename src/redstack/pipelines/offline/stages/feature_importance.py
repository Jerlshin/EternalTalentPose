"""O10 — Feature Importance Analysis (permutation importance).

Owner stage: O10 (Offline Pipeline Part 2/§O10). Quantify each feature's
contribution to the composite under the locked O9 weights, so Reasoning (O16) can
pick which features to cite and pruning can find dead columns. Method:
**permutation importance + ablation NDCG deltas** (Part 10) — permute a feature
column across the labeled candidates and measure the drop in the composite; a
feature whose shuffling hurts the ranking is important.

Algorithm (deps O9, O14):

1. Load the locked ``scoring_weights`` + ``feature_manifest`` (layout + group map)
   + O8 ``gold_labels``, and read the O14 ``feature_snapshot`` for the labeled set.
2. Compute the baseline composite under the locked weights.
3. For each feature column: **seeded permutation** of that column's values across
   the labeled candidates (``OfflineEntropy.numpy_generator("feature_importance")``),
   recompute the component aggregation + composite, and record
   ``importance = baseline − permuted`` (averaged over a few seeded repeats for
   stability on the tiny gold set).

Output: ``feature_importance.json`` — ``{importances: {feature_id: score}, ...}``
(``_v_feature_importance``: non-empty mapping; features ⊆ layout).

Determinism: permutations are drawn from the labeled entropy substream; repeats
are fixed; importances are rounded — byte-stable for a given seed + inputs.
"""

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
from redstack.pipelines.offline.stages.weight_search import composite_score

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "FeatureImportanceStage",
    "feature_importance_stage",
)

#: Seeded permutation repeats averaged per feature (stability on tiny gold).
_PERMUTE_REPEATS: Final[int] = 5
#: Group prefix → ScoreComponent (mirrors O9's mapping for online consistency).
_GROUP_TO_COMPONENT: Final[Mapping[str, ScoreComponent]] = {
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


class FeatureImportanceStage(OfflineStage):
    """O10 — seeded permutation importance over feature columns."""

    stage_id = "O10"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        manifest = self._load_json(ctx, "feature_manifest")
        weights_doc = self._load_yaml(ctx, "scoring_weights")
        feature_ids = self._feature_ids(manifest)
        group_of = self._group_of(manifest)
        component_order = tuple(ScoreComponent)
        weight_vec = self._weight_vector(weights_doc, component_order)
        col_to_comp = self._map_columns(feature_ids, group_of, component_order)

        features, tiers = self._labeled_matrix(ctx, feature_ids)
        if features.shape[0] < 2:
            raise ArtifactContractError(
                "fewer than 2 labeled candidates present for importance analysis"
            )

        baseline = self._composite_for(features, tiers, weight_vec, col_to_comp, component_order)
        importances = self._permutation_importance(
            ctx, features, tiers, weight_vec, col_to_comp, component_order,
            feature_ids, baseline,
        )

        artifact = self.emit_json(
            ctx,
            "feature_importance",
            {
                "importances": importances,
                "baseline_composite": round(baseline, 6),
                "method": "permutation",
                "repeats": _PERMUTE_REPEATS,
                "labeled": int(features.shape[0]),
            },
        )
        top = sorted(importances.items(), key=lambda kv: -kv[1])[:5]
        metrics: dict[str, object] = {
            "features_scored": len(importances),
            "baseline_composite": round(baseline, 6),
            "top_features": [f for f, _ in top],
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json(ctx: OfflinePipelineContext, key: str) -> Mapping[str, object]:
        try:
            raw = ctx.artifact_store.load_text(key)
        except Exception as exc:  # noqa: BLE001
            msg = f"cannot load {key} for feature importance: {exc}"
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
    def _load_yaml(ctx: OfflinePipelineContext, key: str) -> Mapping[str, object]:
        import yaml

        try:
            raw = ctx.artifact_store.load_text(key)
        except Exception as exc:  # noqa: BLE001
            msg = f"cannot load {key} for feature importance: {exc}"
            raise ArtifactContractError(msg) from exc
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, Mapping):
            raise ArtifactContractError(f"{key} must be a mapping")
        return parsed

    @staticmethod
    def _feature_ids(manifest: Mapping[str, object]) -> tuple[str, ...]:
        features = manifest.get("features")
        if not isinstance(features, (list, tuple)) or not features:
            raise ArtifactContractError("feature_manifest missing 'features'")
        return tuple(str(f) for f in features)

    @staticmethod
    def _group_of(manifest: Mapping[str, object]) -> Mapping[str, str]:
        group_of = manifest.get("group_of")
        if not isinstance(group_of, Mapping):
            raise ArtifactContractError("feature_manifest missing 'group_of'")
        return {str(k): str(v) for k, v in group_of.items()}

    @staticmethod
    def _weight_vector(
        weights_doc: Mapping[str, object], component_order: Sequence[ScoreComponent]
    ) -> npt.NDArray[np.float32]:
        weights = weights_doc.get("weights")
        if not isinstance(weights, Mapping):
            raise ArtifactContractError("scoring_weights missing 'weights'")
        vec = np.zeros(len(component_order), dtype=np.float32)
        for i, component in enumerate(component_order):
            value = weights.get(component.value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                vec[i] = float(value)
        return vec

    @staticmethod
    def _map_columns(
        feature_ids: Sequence[str],
        group_of: Mapping[str, str],
        component_order: Sequence[ScoreComponent],
    ) -> dict[int, int]:
        comp_index = {c: i for i, c in enumerate(component_order)}
        out: dict[int, int] = {}
        for col, fid in enumerate(feature_ids):
            group = group_of.get(fid, "")
            prefix = group.split(".", 1)[0] if group else fid.split(".", 1)[0]
            component = _GROUP_TO_COMPONENT.get(prefix, ScoreComponent.CREDIBILITY)
            out[col] = comp_index[component]
        return out

    def _labeled_matrix(
        self, ctx: OfflinePipelineContext, feature_ids: Sequence[str]
    ) -> tuple[npt.NDArray[np.float32], list[int]]:
        """Read the feature_snapshot for labeled candidates → (M,D) matrix + tiers."""
        gold = self._load_json(ctx, "gold_labels")
        labels = gold.get("labels")
        if not isinstance(labels, Mapping) or not labels:
            raise ArtifactContractError("gold_labels has no labels")

        spec = self.registry.spec("feature_snapshot")
        path = (ctx.artifacts_root / spec.relative_path).resolve()
        if not path.is_file():
            raise ArtifactContractError(f"feature_snapshot not found at {path}")
        import pyarrow.parquet as pq

        table = pq.read_table(path)  # type: ignore[no-untyped-call]
        value_cols = sorted(
            (c for c in table.column_names if c.startswith("f") and c[1:].isdigit()),
            key=lambda c: int(c[1:]),
        )
        if len(value_cols) != len(feature_ids):
            raise ArtifactContractError("snapshot value-column count != manifest features")
        ids = [str(v) for v in table.column("id").to_pylist()]
        mat = np.column_stack(
            [table.column(c).to_numpy(zero_copy_only=False) for c in value_cols]
        ).astype(np.float32, copy=False)
        row_of = {cid: i for i, cid in enumerate(ids)}

        rows: list[int] = []
        tiers: list[int] = []
        for cid in sorted(labels):
            record = labels[cid]
            if isinstance(record, Mapping) and cid in row_of:
                tier = record.get("tier")
                if isinstance(tier, int) and not isinstance(tier, bool):
                    rows.append(row_of[cid])
                    tiers.append(tier)
        return np.ascontiguousarray(mat[rows]), tiers

    # ------------------------------------------------------------------ #
    # Composite + permutation                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aggregate(
        features: npt.NDArray[np.float32],
        col_to_comp: Mapping[int, int],
        n_comp: int,
    ) -> npt.NDArray[np.float32]:
        """Aggregate an (M,D) feature matrix into (M,7) component means."""
        m = features.shape[0]
        acc = np.zeros((m, n_comp), dtype=np.float32)
        counts = np.zeros(n_comp, dtype=np.float32)
        for col, comp in col_to_comp.items():
            acc[:, comp] += features[:, col]
            counts[comp] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, acc / np.maximum(counts, 1.0), 0.0).astype(
                np.float32
            )

    def _composite_for(
        self,
        features: npt.NDArray[np.float32],
        tiers: Sequence[int],
        weight_vec: npt.NDArray[np.float32],
        col_to_comp: Mapping[int, int],
        component_order: Sequence[ScoreComponent],
    ) -> float:
        comp = self._aggregate(features, col_to_comp, len(component_order))
        scores = comp @ weight_vec
        order = np.argsort(-scores, kind="stable")
        ordered = [tiers[i] for i in order.tolist()]
        return composite_score(ordered)

    def _permutation_importance(
        self,
        ctx: OfflinePipelineContext,
        features: npt.NDArray[np.float32],
        tiers: Sequence[int],
        weight_vec: npt.NDArray[np.float32],
        col_to_comp: Mapping[int, int],
        component_order: Sequence[ScoreComponent],
        feature_ids: Sequence[str],
        baseline: float,
    ) -> dict[str, float]:
        """Per-feature permutation importance = baseline − mean permuted composite.

        Each feature column is shuffled across candidates ``_PERMUTE_REPEATS`` times
        with the seeded ``"feature_importance"`` substream; the averaged composite
        drop is the importance. Positive ⇒ shuffling the feature hurts the ranking.
        """
        generator = ctx.entropy.numpy_generator("feature_importance")
        m = features.shape[0]
        importances: dict[str, float] = {}
        for col, fid in enumerate(feature_ids):
            drop_total = 0.0
            for _ in range(_PERMUTE_REPEATS):
                permuted = features.copy()
                perm = generator.permutation(m)
                permuted[:, col] = features[perm, col]
                permuted_composite = self._composite_for(
                    permuted, tiers, weight_vec, col_to_comp, component_order
                )
                drop_total += baseline - permuted_composite
            importances[fid] = round(drop_total / _PERMUTE_REPEATS, 6)
        return importances


def feature_importance_stage() -> FeatureImportanceStage:
    """Factory: construct the O10 feature-importance stage."""
    return FeatureImportanceStage()