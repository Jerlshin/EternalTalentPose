
from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from redstack.domain.enums import EligibilityCode, IntegrityFlag, ScoreComponent
from redstack.domain.errors import ArtifactContractError
from redstack.domain.ids import CandidateId, Multiplier, Score, UnitScore
from redstack.domain.provenance import EvidenceRef

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)

#: Fixed sentinel a gated-out candidate's ``final_score`` is floored to (§H.2).
SCORE_FLOOR: Final[Score] = Score(0.0)
#: Declared bounds for behavioral / logistics multipliers (§H.4, §A). They may
#: only dampen relevance: an on-paper-perfect but inactive candidate is
#: down-weighted, never boosted past its earned base relevance.
MULTIPLIER_MIN: Final[float] = 0.0
MULTIPLIER_MAX: Final[float] = 1.0

#: Canonical, immutable reduction order for components (determinism, §R).
_COMPONENT_ORDER: Final[tuple[ScoreComponent, ...]] = tuple(ScoreComponent)
_ABS_TOL: Final[float] = 1e-5
_REL_TOL: Final[float] = 1e-6


@final
class ScoringWeights(BaseModel):
    """Per-component weights from ``scoring_weights.locked.yaml``.

    The weight set must equal ``ScoreComponent`` exactly, else
    ``ArtifactContractError``. Shared by reference across all candidates (§Q).
    """

    model_config = _STRICT

    weights: Mapping[ScoreComponent, float]
    schema_version: str = Field(min_length=1)

    @field_validator("weights", mode="after")
    @classmethod
    def _exact_component_set(
        cls, value: Mapping[ScoreComponent, float]
    ) -> Mapping[ScoreComponent, float]:
        if set(value.keys()) != set(ScoreComponent):
            raise ArtifactContractError(
                "ScoringWeights keys must equal the ScoreComponent set exactly"
            )
        for weight in value.values():
            if not math.isfinite(weight):
                raise ArtifactContractError("ScoringWeights values must be finite")
        return MappingProxyType(dict(value))

    @field_serializer("weights")
    def _dump_weights(
        self, value: Mapping[ScoreComponent, float]
    ) -> dict[ScoreComponent, float]:
        return dict(value)

    def weight_for(self, component: ScoreComponent) -> float:
        """Return the weight for ``component`` (always present by invariant)."""
        return self.weights[component]


@final
class ScoreComponentValue(BaseModel):
    """One base-relevance component: ``weighted == raw * weight``."""

    model_config = _STRICT

    component: ScoreComponent
    raw: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    weight: float = Field(allow_inf_nan=False)
    weighted: float = Field(allow_inf_nan=False)
    evidence: tuple[EvidenceRef, ...]

    @model_validator(mode="after")
    def _weighted_consistent(self) -> ScoreComponentValue:
        if not math.isclose(
            self.weighted, self.raw * self.weight, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
        ):
            raise ValueError("ScoreComponentValue.weighted must equal raw * weight")
        return self


@final
class GateOutcome(BaseModel):
    """A pass/fail gate verdict with an optional failure reason."""

    model_config = _STRICT

    passed: bool
    reason: EligibilityCode | IntegrityFlag | None

    @model_validator(mode="after")
    def _reason_only_on_failure(self) -> GateOutcome:
        if self.passed and self.reason is not None:
            raise ValueError("a passed GateOutcome must carry no reason")
        return self


@final
class ScoreBreakdown(BaseModel):
    """Full, auditable decomposition of a candidate's final score."""

    model_config = _STRICT

    components: tuple[ScoreComponentValue, ...]
    base_relevance: Score = Field(allow_inf_nan=False)
    integrity_gate: GateOutcome
    eligibility_gate: GateOutcome
    behavioral_multiplier: Multiplier = Field(
        ge=MULTIPLIER_MIN, le=MULTIPLIER_MAX, allow_inf_nan=False
    )
    logistics_multiplier: Multiplier = Field(
        ge=MULTIPLIER_MIN, le=MULTIPLIER_MAX, allow_inf_nan=False
    )
    archetype_adjustment: float = Field(allow_inf_nan=False)
    final_score: Score = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _scoring_contract(self) -> ScoreBreakdown:
        present = tuple(cv.component for cv in self.components)
        if present != _COMPONENT_ORDER:
            raise ValueError(
                "ScoreBreakdown.components must list every ScoreComponent "
                "exactly once, in canonical order"
            )
        running = 0.0
        for cv in self.components:
            running += cv.weighted
        if not math.isclose(
            running, self.base_relevance, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
        ):
            raise ValueError("base_relevance must equal Σ component.weighted")
        gated_out = not (self.integrity_gate.passed and self.eligibility_gate.passed)
        if gated_out and self.final_score != SCORE_FLOOR:
            raise ValueError(
                "a gated-out candidate must have final_score == SCORE_FLOOR"
            )
        if self.final_score < SCORE_FLOOR:
            raise ValueError("final_score must not fall below SCORE_FLOOR")
        return self


@final
class ScoredCandidate(BaseModel):
    """A candidate's final score + its breakdown; identity-carrying."""

    model_config = _STRICT

    candidate_id: CandidateId
    final_score: Score = Field(allow_inf_nan=False)
    breakdown: ScoreBreakdown
    tiebreak_key: CandidateId

    @model_validator(mode="after")
    def _consistency(self) -> ScoredCandidate:
        if self.final_score != self.breakdown.final_score:
            raise ValueError(
                "ScoredCandidate.final_score must equal breakdown.final_score"
            )
        if self.tiebreak_key != self.candidate_id:
            raise ValueError("tiebreak_key must equal candidate_id")
        return self


__all__: tuple[str, ...] = (
    "MULTIPLIER_MAX",
    "MULTIPLIER_MIN",
    "SCORE_FLOOR",
    "GateOutcome",
    "ScoreBreakdown",
    "ScoreComponentValue",
    "ScoredCandidate",
    "ScoringWeights",
)