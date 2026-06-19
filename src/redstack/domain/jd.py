
from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.enums import EligibilityCode
from redstack.domain.ids import AnchorId, ArchetypeId


@final
class JobDescriptionSpec(BaseModel):
    """Frozen, parsed JD: experience band, hub set, anchors, gate codes."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
    )

    role_title: str = Field(min_length=1)
    min_experience_years: float = Field(ge=0.0, le=50.0, allow_inf_nan=False)
    max_experience_years: float = Field(ge=0.0, le=50.0, allow_inf_nan=False)
    preferred_hubs: frozenset[str]
    positive_anchors: frozenset[AnchorId]
    negative_anchors: frozenset[AnchorId]
    hard_disqualifiers: frozenset[EligibilityCode]
    soft_penalties: frozenset[EligibilityCode]
    target_archetypes: frozenset[ArchetypeId]

    @model_validator(mode="after")
    def _check(self) -> JobDescriptionSpec:
        if self.min_experience_years > self.max_experience_years:
            raise ValueError("min_experience_years exceeds max_experience_years")
        if self.positive_anchors & self.negative_anchors:
            raise ValueError("positive and negative anchors must be disjoint")
        if self.hard_disqualifiers & self.soft_penalties:
            raise ValueError("hard and soft eligibility codes must be disjoint")
        return self


__all__: tuple[str, ...] = ("JobDescriptionSpec",)