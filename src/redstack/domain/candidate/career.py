

from __future__ import annotations

from datetime import date
from typing import final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.enums import CareerTrack, CompanySize
from redstack.domain.ids import Months, UnitScore

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


@final
class PositionFact(BaseModel):
    """One structured career position."""

    model_config = _STRICT

    company: str
    title: str
    start_date: date
    end_date: date | None
    duration_months: Months = Field(ge=0)
    is_current: bool
    industry: str
    company_size: CompanySize
    is_product_company: bool
    is_consulting_firm: bool
    description_role_match: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)


@final
class TenureStats(BaseModel):
    """Aggregate tenure statistics across positions."""

    model_config = _STRICT

    position_count: int = Field(ge=0)
    mean_tenure_months: float = Field(ge=0.0, allow_inf_nan=False)
    min_tenure_months: float = Field(ge=0.0, allow_inf_nan=False)
    hop_rate: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)


@final
class CareerRecency(BaseModel):
    """Recency derivations computed against an injected ``as_of``."""

    model_config = _STRICT

    most_recent_start: date
    is_currently_employed: bool
    months_since_last_role: Months = Field(ge=0)


@final
class CareerProfile(BaseModel):
    """Career intelligence: progression, product-vs-services, recency."""

    model_config = _STRICT

    stated_experience_years: float = Field(ge=0.0, le=50.0, allow_inf_nan=False)
    derived_experience_years: float = Field(ge=0.0, le=50.0, allow_inf_nan=False)
    positions: tuple[PositionFact, ...]
    current_position: PositionFact | None
    track: CareerTrack
    tenure: TenureStats
    recency: CareerRecency
    title_consistency: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check(self) -> CareerProfile:
        starts = [p.start_date for p in self.positions]
        if starts != sorted(starts, reverse=True):
            raise ValueError("positions must be sorted by start_date descending")
        current = [p for p in self.positions if p.is_current]
        if len(current) > 1:
            raise ValueError("at most one position may be is_current")
        if self.current_position is None:
            if current:
                raise ValueError("current_position must be the unique is_current entry")
        elif not self.current_position.is_current:
            raise ValueError("current_position must itself be is_current")
        elif self.current_position not in self.positions:
            raise ValueError("current_position must be one of positions")
        return self


__all__: tuple[str, ...] = (
    "CareerProfile",
    "CareerRecency",
    "PositionFact",
    "TenureStats",
)