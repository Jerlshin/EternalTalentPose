"""O12 — Risk Calibration.

Owner stage: O12 (Offline Pipeline Part 4/Part 8; Feature Layer Part 5/§Risk;
Engine Layer §9 ``CandidateRiskEngine`` inputs). Finalize the honeypot composite
threshold and the **confidence-shrink** parameters that ``CandidateRiskEngine``
uses to modulate ``final_score`` (shrink toward prior when confidence is low),
trading honeypot **recall** against false-positive loss — false positives drop
real candidates and cost NDCG, so the calibration is recall-first within a
bounded real-candidate-loss budget (Part 5).

Algorithm (deps O3, O14):

1. Read O3's ``integrity_thresholds.json`` (per-detector prevalence + initial
   composite threshold + hard-gate count) and ``honeypot_catalog.json`` (the
   discovered suspect cohort).
2. Read the O14 ``feature_snapshot`` metadata (row count) to express the
   real-candidate-loss budget as a fraction of the pool.
3. Choose the final composite ``honeypot_threshold`` that preserves the hard-gate
   (≥2 HARD) recall while keeping composite-gated suspects within the loss budget;
   set per-detector risk weights (HARD detectors weigh more) and the
   confidence-shrink params (how hard low confidence pulls toward the prior).

Outputs (delegated to the bound store):

* ``risk_weights.json`` — ``{weights: {detector→weight}, confidence_shrink: {...}}``
  (``_v_risk_weights``: requires ``weights`` + ``confidence_shrink``).
* ``integrity_thresholds.json`` — **finalized**: the O3 per-detector block carried
  forward + the recalibrated composite ``honeypot_threshold`` + the risk weights
  merged in (Part 10: this artifact is co-owned O3,O12). ``_v_integrity_thresholds``.

Determinism: a pure function of the O3 artifacts + the snapshot row count; no RNG,
no clock. Weights + thresholds are rounded deterministically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

from redstack.domain.enums import IntegrityFlag
from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "RiskCalibrationStage",
    "risk_calibration_stage",
)

#: Maximum fraction of the pool allowed to be composite-gated as honeypots (the
#: real-candidate-loss budget; the hard gate is exempt — it is categorical).
_MAX_COMPOSITE_LOSS: Final[float] = 0.02
#: Risk weight assigned to HARD detectors (relative to SOFT = 1.0).
_HARD_WEIGHT: Final[float] = 3.0
_SOFT_WEIGHT: Final[float] = 1.0
#: Confidence-shrink: max fraction of relevance pulled toward the prior at zero
#: confidence (bounded so risk never fully erases a real signal).
_MAX_SHRINK: Final[float] = 0.35
#: Confidence floor below which shrink saturates (avoids over-penalizing sparse).
_CONFIDENCE_FLOOR: Final[float] = 0.2

#: The HARD detector set (mirrors O3's severity assignment).
_HARD_FLAGS: Final[frozenset[str]] = frozenset(
    {
        IntegrityFlag.CURRENT_ROLE_HAS_END_DATE.value,
        IntegrityFlag.TENURE_EXCEEDS_EXPERIENCE.value,
        IntegrityFlag.EXPERT_SKILL_ZERO_USAGE.value,
        IntegrityFlag.EDUCATION_TIMELINE_IMPOSSIBLE.value,
        IntegrityFlag.ASSESSMENT_FOR_ABSENT_SKILL.value,
        IntegrityFlag.EMPLOYMENT_OVERLAP_IMPOSSIBLE.value,
        IntegrityFlag.SIGNAL_IMPOSSIBILITY.value,
        IntegrityFlag.IDENTITY_ANOMALY.value,
    }
)


class RiskCalibrationStage(OfflineStage):
    """O12 — finalize honeypot threshold + risk weights + confidence-shrink."""

    stage_id = "O12"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        thresholds = self._load_json(ctx, "integrity_thresholds")
        catalog = self._load_json(ctx, "honeypot_catalog")
        snapshot_rows = self._snapshot_row_count(ctx)

        per_detector = thresholds.get("per_detector")
        if not isinstance(per_detector, Mapping):
            raise ArtifactContractError(
                "integrity_thresholds.json missing 'per_detector' from O3"
            )
        initial_threshold = self._as_float(thresholds.get("honeypot_threshold"), 0.7)
        hard_gate_count = self._as_int(thresholds.get("hard_gate_count"), 2)

        suspects = catalog.get("suspect_ids")
        suspect_count = len(suspects) if isinstance(suspects, (list, tuple)) else 0
        composite_gated = self._count_composite_gated(catalog, initial_threshold)

        final_threshold = self._recalibrate_threshold(
            initial_threshold, composite_gated, snapshot_rows
        )
        weights = self._risk_weights(per_detector)
        confidence_shrink = self._confidence_shrink_params()

        risk_artifact = self.emit_json(
            ctx,
            "risk_weights",
            {"weights": weights, "confidence_shrink": confidence_shrink},
        )
        # Finalize the co-owned integrity_thresholds: carry O3's per-detector block,
        # overwrite the composite threshold, merge the risk weights.
        finalized: dict[str, object] = {
            "per_detector": dict(per_detector),
            "honeypot_threshold": round(final_threshold, 6),
            "hard_gate_count": hard_gate_count,
            "risk_weights": weights,
            "confidence_shrink": confidence_shrink,
            "calibrated_by": "O12",
        }
        thresholds_artifact = self.emit_json(ctx, "integrity_thresholds", finalized)

        metrics: dict[str, object] = {
            "initial_threshold": round(initial_threshold, 6),
            "final_threshold": round(final_threshold, 6),
            "suspect_count": suspect_count,
            "composite_gated": composite_gated,
            "snapshot_rows": snapshot_rows,
            "loss_budget": _MAX_COMPOSITE_LOSS,
        }
        return StageResult(
            artifacts=(risk_artifact, thresholds_artifact), metrics=metrics
        )

    # ------------------------------------------------------------------ #
    # Input loading                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json(ctx: OfflinePipelineContext, key: str) -> Mapping[str, object]:
        """Load + parse a JSON artifact from the bound store.

        Raises:
            ArtifactContractError: missing or malformed (O12 lineage requires O3).
        """
        try:
            raw = ctx.artifact_store.load_text(key)
        except Exception as exc:  # noqa: BLE001 — surface as a contract failure.
            msg = f"cannot load {key} for risk calibration: {exc}"
            raise ArtifactContractError(msg) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"{key} is not valid JSON: {exc}"
            raise ArtifactContractError(msg) from exc
        if not isinstance(parsed, Mapping):
            raise ArtifactContractError(f"{key} must be an object")
        return parsed

    def _snapshot_row_count(self, ctx: OfflinePipelineContext) -> int:
        """Read the O14 feature_snapshot row count (the loss-budget denominator).

        Reads the parquet metadata at the registry-resolved path (offline; pyarrow
        is the offline dependency). Falls back to the candidate source count if the
        snapshot is unavailable, so calibration still proceeds deterministically.
        """
        spec = self.registry.spec("feature_snapshot")
        path = (ctx.artifacts_root / spec.relative_path).resolve()
        if path.is_file():
            import pyarrow.parquet as pq

            try:
                meta = pq.read_metadata(path)  # type: ignore[no-untyped-call]
                return int(meta.num_rows)
            except Exception:  # noqa: BLE001 — fall back below.
                pass
        known = ctx.candidate_source.count()
        return int(known) if known is not None else 0

    @staticmethod
    def _count_composite_gated(
        catalog: Mapping[str, object], threshold: float
    ) -> int:
        """Count catalog exemplars gated by composite alone (no hard gate).

        These are the candidates the composite threshold is responsible for — the
        ones whose count must stay within the loss budget. Hard-gated exemplars
        (≥2 HARD) are categorical and excluded from the budget.
        """
        exemplars = catalog.get("exemplars")
        if not isinstance(exemplars, (list, tuple)):
            return 0
        count = 0
        for exemplar in exemplars:
            if not isinstance(exemplar, Mapping):
                continue
            hard = exemplar.get("hard_count")
            composite = exemplar.get("composite")
            hard_n = hard if isinstance(hard, int) and not isinstance(hard, bool) else 0
            comp = composite if isinstance(composite, (int, float)) else 0.0
            if hard_n < 2 and float(comp) >= threshold:
                count += 1
        return count

    # ------------------------------------------------------------------ #
    # Calibration                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _recalibrate_threshold(
        initial: float, composite_gated: int, snapshot_rows: int
    ) -> float:
        """Recalibrate the composite threshold to respect the loss budget.

        If the composite-gated cohort already exceeds the budget fraction of the
        pool, the threshold is raised toward 1.0 (fewer composite gates, recall
        preserved by the categorical hard gate). Otherwise the O3 threshold is
        kept. Bounded to (0, 1].
        """
        if snapshot_rows <= 0:
            return min(1.0, max(initial, 1e-3))
        loss_fraction = composite_gated / snapshot_rows
        if loss_fraction > _MAX_COMPOSITE_LOSS:
            # Raise the threshold proportionally to the overshoot, capped at 1.0.
            overshoot = loss_fraction / _MAX_COMPOSITE_LOSS
            raised = initial + (1.0 - initial) * min(1.0, (overshoot - 1.0))
            return min(1.0, max(initial, raised))
        return min(1.0, max(1e-3, initial))

    @staticmethod
    def _risk_weights(per_detector: Mapping[str, object]) -> dict[str, float]:
        """Assign per-detector risk weights (HARD > SOFT), sorted deterministically.

        Every detector present in the O3 per-detector block gets a weight; HARD
        detectors weigh :data:`_HARD_WEIGHT`, SOFT weigh :data:`_SOFT_WEIGHT`.
        Unknown codes default to SOFT (conservative).
        """
        weights: dict[str, float] = {}
        for code in sorted(per_detector):
            weights[code] = _HARD_WEIGHT if code in _HARD_FLAGS else _SOFT_WEIGHT
        return weights

    @staticmethod
    def _confidence_shrink_params() -> dict[str, float]:
        """The shrink-toward-prior parameters consumed by ``CandidateRiskEngine``.

        ``max_shrink`` is the largest fraction of relevance pulled toward the prior
        at zero confidence; ``confidence_floor`` is where shrink saturates. The
        engine computes ``shrink = max_shrink · (1 − clamp(confidence, floor, 1))``
        — bounded so risk modulates but never erases a real signal (Engine §9).
        """
        return {
            "max_shrink": _MAX_SHRINK,
            "confidence_floor": _CONFIDENCE_FLOOR,
        }

    # ------------------------------------------------------------------ #
    # Scalar coercion                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_float(value: object, default: float) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return default

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return default


def risk_calibration_stage() -> RiskCalibrationStage:
    """Factory: construct the O12 risk-calibration stage."""
    return RiskCalibrationStage()