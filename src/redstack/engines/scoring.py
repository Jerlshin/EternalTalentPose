

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, final

from pydantic import BaseModel, ConfigDict

from redstack.config.schema import ScoringPolicy
from redstack.domain.candidate.eligibility import EligibilityReport
from redstack.domain.candidate.integrity import IntegrityReport
from redstack.domain.enums import ScoreComponent
from redstack.domain.errors import ScoreInvariantError
from redstack.domain.ids import CandidateId, Multiplier, Score, UnitScore
from redstack.domain.provenance import EvidenceRef
from redstack.domain.scoring import (
    GateOutcome,
    ScoreBreakdown,
    ScoredCandidate,
    ScoreComponentValue,
    ScoringWeights,
)

# Fixed reduction order (Determinism §R): the enum's declaration order.
_COMPONENT_ORDER: Final[tuple[ScoreComponent, ...]] = tuple(ScoreComponent)

# One component's projected raw value plus the evidence backing it.
ComponentRaw = tuple[UnitScore, tuple[EvidenceRef, ...]]


@final
class ScoringEngine(BaseModel):
    """Stateless, pure scoring engine; weights + policy are injected, immutable."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    weights: ScoringWeights
    policy: ScoringPolicy

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        # Weight set must equal ScoreComponent exactly (ArtifactContractError is
        # raised at load; here we defensively re-assert against silent drift).
        missing = set(_COMPONENT_ORDER) - set(self.weights.weights)
        if missing:
            raise ScoreInvariantError(
                f"ScoringWeights missing components: {sorted(c.value for c in missing)}"
            )

    # ------------------------------------------------------------------ public
    def score(
        self,
        *,
        candidate_id: CandidateId,
        components: Mapping[ScoreComponent, ComponentRaw],
        integrity: IntegrityReport,
        eligibility: EligibilityReport,
        behavioral_multiplier: Multiplier,
        logistics_multiplier: Multiplier,
        archetype_adjustment: float,
        confidence: UnitScore,
    ) -> ScoredCandidate:
        """Produce the ``ScoredCandidate`` with a fully reconstructable breakdown."""
        component_values = self._component_values(components)
        base = self._sum_weighted(component_values)

        integrity_gate = self._integrity_gate(integrity)
        eligibility_gate = self._eligibility_gate(eligibility)
        gated = not (integrity_gate.passed and eligibility_gate.passed)

        if gated:
            final = Score(float(self.policy.floor))
            beh = behavioral_multiplier
            log = logistics_multiplier
            adjustment = 0.0  # no multipliers/adjustments applied to a floored row
        else:
            combined = (
                float(base)
                * float(behavioral_multiplier)
                * float(logistics_multiplier)
                + archetype_adjustment
            )
            final = Score(self._shrink(combined, confidence))
            beh = behavioral_multiplier
            log = logistics_multiplier
            adjustment = archetype_adjustment

        self._assert_finite(float(base), float(final))

        breakdown = ScoreBreakdown(
            components=component_values,
            base_relevance=base,
            integrity_gate=integrity_gate,
            eligibility_gate=eligibility_gate,
            behavioral_multiplier=beh,
            logistics_multiplier=log,
            archetype_adjustment=adjustment,
            final_score=final,
        )
        if gated and float(final) != float(self.policy.floor):
            raise ScoreInvariantError("gated candidate not floored")

        return ScoredCandidate(
            candidate_id=candidate_id,
            final_score=final,
            breakdown=breakdown,
            tiebreak_key=candidate_id,
        )

    # --------------------------------------------------------------- internals
    def _component_values(
        self, components: Mapping[ScoreComponent, ComponentRaw]
    ) -> tuple[ScoreComponentValue, ...]:
        values: list[ScoreComponentValue] = []
        for component in _COMPONENT_ORDER:
            raw, evidence = components.get(
                component, (UnitScore(0.0), ())
            )
            weight = float(self.weights.weights[component])
            weighted = float(raw) * weight
            values.append(
                ScoreComponentValue(
                    component=component,
                    raw=raw,
                    weight=weight,
                    weighted=weighted,
                    evidence=evidence,
                )
            )
        return tuple(values)

    @staticmethod
    def _sum_weighted(values: tuple[ScoreComponentValue, ...]) -> Score:
        # Summed in ScoreComponent order (values already ordered); float32-stable.
        total = 0.0
        for value in values:
            total += value.weighted
        return Score(total)

    def _shrink(self, combined: float, confidence: UnitScore) -> float:
        """Shrink ``combined`` toward the neutral prior in proportion to confidence.

        ``shrunk = prior + confidence·(combined − prior)``: full confidence keeps the
        value, zero confidence collapses to the prior. Monotone in ``combined`` for a
        fixed confidence, so ranking order is preserved among equal-confidence rows.
        """
        prior = self.policy.neutral_prior
        return prior + float(confidence) * (combined - prior)

    @staticmethod
    def _integrity_gate(report: IntegrityReport) -> GateOutcome:
        if not report.is_honeypot:
            return GateOutcome(passed=True, reason=None)
        reason = next(
            (f.code for f in report.findings if f.severity.value == "hard"), None
        )
        if reason is None and report.findings:
            reason = report.findings[0].code
        return GateOutcome(passed=False, reason=reason)

    @staticmethod
    def _eligibility_gate(report: EligibilityReport) -> GateOutcome:
        if report.is_eligible:
            return GateOutcome(passed=True, reason=None)
        reason = report.hard_blocks[0].code if report.hard_blocks else None
        return GateOutcome(passed=False, reason=reason)

    @staticmethod
    def _assert_finite(base: float, final: float) -> None:
        for label, value in (("base_relevance", base), ("final_score", final)):
            if value != value or value in (float("inf"), float("-inf")):
                raise ScoreInvariantError(f"{label} is non-finite ({value!r})")


__all__: tuple[str, ...] = ("ComponentRaw", "ScoringEngine")