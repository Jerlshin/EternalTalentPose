"""``LogisticsProfile`` slice — location / notice / relocation / salary fit (R2).

Owner layer: domain.
Allowed imports: ids, enums, pydantic.

Salary inversion is preserved (not corrected); it is a logistics sanity flag,
not a honeypot flag.
"""

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.enums import LocationFit, NoticeFit, WorkMode
from redstack.domain.ids import LpaAmount, UnitScore

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


@final
class SalaryBand(BaseModel):
    """Expected salary band; ``is_inverted`` records (min > max) faithfully."""

    model_config = _STRICT

    min_lpa: LpaAmount = Field(ge=0.0, allow_inf_nan=False)
    max_lpa: LpaAmount = Field(ge=0.0, allow_inf_nan=False)
    is_inverted: bool

    @model_validator(mode="after")
    def _inversion_consistent(self) -> SalaryBand:
        if self.is_inverted != (self.min_lpa > self.max_lpa):
            raise ValueError("is_inverted must equal (min_lpa > max_lpa)")
        return self


@final
class LogisticsProfile(BaseModel):
    """Location / notice / relocation / salary fit vs the JD."""

    model_config = _STRICT

    location: str
    country: str
    location_fit: LocationFit
    willing_to_relocate: bool
    notice_period_days: int = Field(ge=0, le=180)
    notice_fit: NoticeFit
    preferred_work_mode: WorkMode
    work_mode_fit: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    salary: SalaryBand


__all__: tuple[str, ...] = ("LogisticsProfile", "SalaryBand")