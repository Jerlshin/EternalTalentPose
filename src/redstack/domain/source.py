
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from types import MappingProxyType
from typing import Any, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)

from redstack.domain.enums import (
    CompanySize,
    InstitutionTier,
    LanguageProficiency,
    Proficiency,
    WorkMode,
)
from redstack.domain.errors import SchemaError
from redstack.domain.ids import CandidateId, LpaAmount, Months, SkillName

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


@final
class RawProfile(BaseModel):
    """Top-level profile facts."""

    model_config = _STRICT

    anonymized_name: str = Field(min_length=1)
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float = Field(ge=0.0, le=50.0, allow_inf_nan=False)
    current_title: str
    current_company: str
    current_company_size: CompanySize
    current_industry: str


@final
class RawPosition(BaseModel):
    """One career-history position, mirrored verbatim."""

    model_config = _STRICT

    company: str
    title: str
    start_date: date
    end_date: date | None
    duration_months: Months = Field(ge=0)
    is_current: bool
    industry: str
    company_size: CompanySize
    description: str


@final
class RawEducation(BaseModel):
    """One education record."""

    model_config = _STRICT

    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str | None
    tier: InstitutionTier


@final
class RawSkill(BaseModel):
    """One claimed skill."""

    model_config = _STRICT

    name: SkillName
    proficiency: Proficiency
    endorsements: int = Field(ge=0)
    duration_months: Months | None = Field(default=None, ge=0)


@final
class RawCertification(BaseModel):
    """One certification."""

    model_config = _STRICT

    name: str
    issuer: str
    year: int


@final
class RawLanguage(BaseModel):
    """One spoken language."""

    model_config = _STRICT

    language: str
    proficiency: LanguageProficiency


@final
class RawSalaryRange(BaseModel):
    """Expected salary range (INR lpa); inversion preserved, never corrected.

    Field names mirror the source JSON verbatim (``{"min": ..., "max": ...}``),
    not the ``*_lpa``-suffixed names used by downstream domain models.
    """

    model_config = _STRICT

    min: LpaAmount = Field(ge=0.0, allow_inf_nan=False)
    max: LpaAmount = Field(ge=0.0, allow_inf_nan=False)


@final
class RawSignals(BaseModel):
    """The 23 ``redrob_signals``, typed exactly; sentinels preserved as-is."""

    model_config = _STRICT

    profile_completeness_score: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: int = Field(ge=0)
    applications_submitted_30d: int = Field(ge=0)
    recruiter_response_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    avg_response_time_hours: float = Field(ge=0.0, allow_inf_nan=False)
    skill_assessment_scores: Mapping[str, float]
    connection_count: int = Field(ge=0)
    endorsements_received: int = Field(ge=0)
    notice_period_days: int = Field(ge=0, le=180)
    expected_salary_range_inr_lpa: RawSalaryRange
    preferred_work_mode: WorkMode
    willing_to_relocate: bool
    github_activity_score: float = Field(ge=-1.0, le=100.0, allow_inf_nan=False)
    search_appearance_30d: int = Field(ge=0)
    saved_by_recruiters_30d: int = Field(ge=0)
    interview_completion_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    offer_acceptance_rate: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    @field_validator("skill_assessment_scores", mode="after")
    @classmethod
    def _freeze_scores(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        for score in value.values():
            if not (0.0 <= score <= 100.0):
                raise ValueError("skill_assessment_scores out of range [0, 100]")
        return MappingProxyType(dict(value))

    @field_serializer("skill_assessment_scores")
    def _dump_scores(self, value: Mapping[str, float]) -> dict[str, float]:
        # ``MappingProxyType`` is not natively serializable; project the
        # read-only view back to a plain ``dict`` so ``RawCandidate`` round-trips
        # to JSON losslessly (provenance depends on this, §P).
        return dict(value)


@final
class RawCandidate(BaseModel):
    """Aggregate of raw facts — the canonical evidence source."""

    model_config = _STRICT

    candidate_id: CandidateId = Field(pattern=r"^CAND_[0-9]{7}$")
    profile: RawProfile
    career_history: tuple[RawPosition, ...] = Field(min_length=1, max_length=10)
    education: tuple[RawEducation, ...] = Field(max_length=5)
    skills: tuple[RawSkill, ...]
    certifications: tuple[RawCertification, ...]
    languages: tuple[RawLanguage, ...]
    redrob_signals: RawSignals

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RawCandidate:
        """Validate and narrow a raw mapping — the sole ``Any`` boundary.

        Type/shape violations are re-raised as ``SchemaError``; semantic
        contradictions are intentionally preserved for downstream detection.
        """
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise SchemaError(str(exc)) from exc


__all__: tuple[str, ...] = (
    "RawCandidate",
    "RawCertification",
    "RawEducation",
    "RawLanguage",
    "RawPosition",
    "RawProfile",
    "RawSalaryRange",
    "RawSignals",
    "RawSkill",
)