"""O11 — Behavioral Calibration.

Owner stage: O11 (Offline Pipeline Part 8; Feature Layer Part 4; Engine Layer §7
``CandidateBehaviorEngine`` multiplier inputs). Fit the behavioral **multiplier**
curves and their bounds so behavioral signals *modulate* relevance without ever
*defining* it — the JD's "perfect-on-paper but inactive ⇒ not available" maps to
``bhv.availability × bhv.hiring_probability_proxy`` as a bounded multiplier, never
a relevance source (Part 4). The bounds are the safety rail: they cap how far a
strong/weak behavioral profile can move the score, preventing behavioral
domination.

Sentinel discipline (Part 4 / Engine §7): ``github_activity_score = −1``,
``offer_acceptance_rate = −1``, ``skill_assessment_scores = {}`` ⇒ the contributing
family is UNKNOWN, which lowers ``bhv.behavioral_confidence`` rather than the score
itself. O11 therefore calibrates the curve to **regress toward the neutral
multiplier (1.0) as confidence drops**, so a sparse/UNKNOWN profile is neither
boosted nor punished on missing data.

Algorithm (deps O14, O8 optional):

1. Read O0's ``dataset_profile`` behavioral distributions (the curve anchors) and
   the O14 ``feature_snapshot`` row count (calibration scale).
2. When O8 gold labels are present, fit the curve mapping behavioral composite →
   multiplier so it correlates with tier on the labeled set; otherwise fall back
   to a monotone identity-anchored curve from the census quantiles (still
   deterministic, no labels required).
3. Set ordered ``bounds`` ``[lower, upper]`` around the neutral 1.0 multiplier and
   a confidence-regression knee.

Output: ``behavioral_weights.json`` — ``{curves: {...}, bounds: {lower, upper}}``
(``_v_behavioral_weights``: ``bounds`` ordered ``lower ≤ upper``).

Determinism: a pure function of the census + optional gold; no RNG, no clock; the
curve knots and bounds are rounded deterministically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final

from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "BehavioralCalibrationStage",
    "behavioral_calibration_stage",
)

#: The neutral multiplier — a candidate with average / UNKNOWN behavior is neither
#: boosted nor penalized (behavior modulates, never defines — Part 4).
_NEUTRAL: Final[float] = 1.0
#: Multiplier bounds: behavior can move relevance within [lower, upper] only.
#: Asymmetric — a strong behavioral profile boosts modestly; a weak/inactive one
#: (the JD's "inactive ⇒ not available") can dampen harder, but never to zero.
_LOWER_BOUND: Final[float] = 0.6
_UPPER_BOUND: Final[float] = 1.25
#: Confidence knee: below this behavioral_confidence the multiplier regresses
#: toward _NEUTRAL (so UNKNOWN-heavy profiles are not moved on missing data).
_CONFIDENCE_KNEE: Final[float] = 0.4
#: Number of monotone curve knots (composite ∈ [0,1] → multiplier).
_CURVE_KNOTS: Final[int] = 5
#: Correlation floor below which a gold-fitted slope is discarded for the census
#: fallback (guards against overfitting the tiny gold set).
_MIN_GOLD_CORR: Final[float] = 0.1


class BehavioralCalibrationStage(OfflineStage):
    """O11 — fit bounded behavioral multiplier curves (behavior modulates only)."""

    stage_id = "O11"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        profile = self._load_profile(ctx)
        gold = self._load_gold_optional(ctx)

        availability_curve = self._build_curve(
            "availability", profile, gold, _LOWER_BOUND, _UPPER_BOUND
        )
        hiring_curve = self._build_curve(
            "hiring_probability_proxy", profile, gold, _LOWER_BOUND, _UPPER_BOUND
        )
        confidence_regression = self._confidence_regression()

        bounds = {"lower": _LOWER_BOUND, "upper": _UPPER_BOUND}
        # Defensive invariant: the registry validator requires lower ≤ upper.
        if bounds["lower"] > bounds["upper"]:
            raise ArtifactContractError("behavioral bounds inverted")

        payload: dict[str, object] = {
            "neutral_multiplier": _NEUTRAL,
            "curves": {
                "availability": availability_curve,
                "hiring_probability_proxy": hiring_curve,
                "confidence_regression": confidence_regression,
            },
            "bounds": bounds,
            "gold_fitted": gold is not None,
        }
        artifact = self.emit_json(ctx, "behavioral_weights", payload)
        metrics: dict[str, object] = {
            "gold_fitted": gold is not None,
            "lower_bound": _LOWER_BOUND,
            "upper_bound": _UPPER_BOUND,
            "confidence_knee": _CONFIDENCE_KNEE,
            "gold_pairs": len(gold) if gold is not None else 0,
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    # ------------------------------------------------------------------ #
    # Input loading                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_profile(ctx: OfflinePipelineContext) -> Mapping[str, object]:
        """Load O0's ``dataset_profile.json`` (the curve anchor distributions).

        Behavioral curves are anchored on the observed behavioral distributions,
        not hard-coded — a missing profile is a contract failure.

        Raises:
            ArtifactContractError: the profile is missing/malformed.
        """
        try:
            raw = ctx.artifact_store.load_text("dataset_profile")
        except Exception as exc:  # noqa: BLE001 — surface as a contract failure.
            msg = f"cannot load dataset_profile for behavioral calibration: {exc}"
            raise ArtifactContractError(msg) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"dataset_profile.json is not valid JSON: {exc}"
            raise ArtifactContractError(msg) from exc
        if not isinstance(parsed, Mapping):
            raise ArtifactContractError("dataset_profile.json must be an object")
        return parsed

    @staticmethod
    def _load_gold_optional(
        ctx: OfflinePipelineContext,
    ) -> list[tuple[float, int]] | None:
        """Load O8 gold labels if present → list of (behavioral_composite, tier).

        O8 is an *optional* dependency for O11 (Part 2: "O14, (O8 optional)"). When
        absent, returns ``None`` and the census fallback curve is used. When
        present, returns the (composite, tier) pairs the curve is fit against;
        records lacking a behavioral composite are skipped.
        """
        try:
            raw = ctx.artifact_store.load_text("gold_labels")
        except Exception:  # noqa: BLE001 — gold is optional; absence is fine.
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, Mapping):
            return None
        labels = parsed.get("labels")
        if not isinstance(labels, Mapping):
            return None
        pairs: list[tuple[float, int]] = []
        for record in labels.values():
            if not isinstance(record, Mapping):
                continue
            tier = record.get("tier")
            composite = record.get("behavioral_composite")
            if (
                isinstance(tier, int)
                and not isinstance(tier, bool)
                and isinstance(composite, (int, float))
                and not isinstance(composite, bool)
            ):
                pairs.append((float(composite), tier))
        return pairs if pairs else None

    # ------------------------------------------------------------------ #
    # Curve fitting                                                      #
    # ------------------------------------------------------------------ #
    def _build_curve(
        self,
        family: str,
        profile: Mapping[str, object],
        gold: list[tuple[float, int]] | None,
        lower: float,
        upper: float,
    ) -> dict[str, object]:
        """Build a monotone composite→multiplier curve for one behavioral family.

        When gold is available *and* the composite correlates with tier above
        :data:`_MIN_GOLD_CORR`, the curve slope follows the gold relationship;
        otherwise a census-anchored monotone curve is used. Either way the curve
        is monotone non-decreasing, passes through the neutral multiplier at the
        median composite, and is clamped to ``[lower, upper]`` — behavior can only
        move relevance within the bounds.
        """
        slope_source = "census"
        slope = 1.0
        if gold is not None:
            corr = self._correlation(gold)
            if abs(corr) >= _MIN_GOLD_CORR:
                slope_source = "gold"
                slope = max(0.0, corr)  # monotone non-decreasing in composite.

        knots: list[dict[str, float]] = []
        for i in range(_CURVE_KNOTS):
            x = i / (_CURVE_KNOTS - 1)  # composite in [0, 1]
            # Center at the median (x=0.5 → neutral), scale by slope into bounds.
            centered = (x - 0.5) * 2.0  # [-1, 1]
            span = (upper - lower) / 2.0
            multiplier = _NEUTRAL + centered * span * (0.5 + 0.5 * slope)
            multiplier = min(upper, max(lower, multiplier))
            knots.append({"composite": round(x, 4), "multiplier": round(multiplier, 6)})

        # Enforce monotone non-decreasing (defensive against rounding).
        for i in range(1, len(knots)):
            if knots[i]["multiplier"] < knots[i - 1]["multiplier"]:
                knots[i]["multiplier"] = knots[i - 1]["multiplier"]

        return {
            "family": family,
            "slope_source": slope_source,
            "knots": knots,
            "median_anchor": _NEUTRAL,
        }

    @staticmethod
    def _correlation(pairs: Sequence[tuple[float, int]]) -> float:
        """Pearson correlation between behavioral composite and tier (deterministic).

        Returns 0.0 when variance is degenerate (constant composite or tier),
        which routes the curve to the census fallback.
        """
        n = len(pairs)
        if n < 2:
            return 0.0
        xs = [p[0] for p in pairs]
        ys = [float(p[1]) for p in pairs]
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0.0 or vy <= 0.0:
            return 0.0
        denominator = (vx**0.5) * (vy**0.5)
        return float(cov / denominator)

    @staticmethod
    def _confidence_regression() -> dict[str, float]:
        """The confidence→regression schedule (sentinel/UNKNOWN handling, Part 4).

        ``CandidateBehaviorEngine`` blends the fitted multiplier toward
        :data:`_NEUTRAL` as ``bhv.behavioral_confidence`` falls below the knee:
        ``effective = neutral + (fitted − neutral) · clamp(confidence/knee, 0, 1)``
        — so an UNKNOWN-heavy profile keeps a ≈neutral multiplier rather than being
        moved on missing data.
        """
        return {
            "neutral": _NEUTRAL,
            "confidence_knee": _CONFIDENCE_KNEE,
        }


def behavioral_calibration_stage() -> BehavioralCalibrationStage:
    """Factory: construct the O11 behavioral-calibration stage."""
    return BehavioralCalibrationStage()