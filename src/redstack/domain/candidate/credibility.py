
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from redstack.domain.enums import Proficiency
from redstack.domain.ids import Months, SkillName, UnitScore

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


@final
class SkillTrust(BaseModel):
    """Per-skill trust derived from endorsement, duration, and assessment."""

    model_config = _STRICT

    name: SkillName
    proficiency: Proficiency
    endorsements: int = Field(ge=0)
    duration_months: Months | None = Field(default=None, ge=0)
    assessment_score: UnitScore | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    trust: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    is_credible: bool


@final
class CredibilityProfile(BaseModel):
    """Trustworthiness of claimed skills/profile."""

    model_config = _STRICT

    skill_trust: Mapping[SkillName, SkillTrust]
    keyword_stuffing_score: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    claimed_vs_assessed_gap: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    title_description_coherence: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    relevant_skill_credibility: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("skill_trust", mode="after")
    @classmethod
    def _freeze_trust(
        cls, value: Mapping[SkillName, SkillTrust]
    ) -> Mapping[SkillName, SkillTrust]:
        for key, trust in value.items():
            if trust.name != key:
                raise ValueError("skill_trust key must equal SkillTrust.name")
        return MappingProxyType(dict(value))


__all__: tuple[str, ...] = ("CredibilityProfile", "SkillTrust")
