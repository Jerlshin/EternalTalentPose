

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field

from redstack.domain.enums import SignalAvailability
from redstack.domain.ids import UnitScore
from redstack.domain.source import RawSignals

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)
_UNIT = Field(ge=0.0, le=1.0, allow_inf_nan=False)


@final
class BehavioralProfile(BaseModel):
    """Normalized behaviour: availability, responsiveness, engagement,
    reliability, and verification families."""

    model_config = _STRICT

    availability: UnitScore = _UNIT
    availability_status: SignalAvailability
    responsiveness: UnitScore = _UNIT
    responsiveness_status: SignalAvailability
    engagement: UnitScore = _UNIT
    engagement_status: SignalAvailability
    reliability: UnitScore = _UNIT
    reliability_status: SignalAvailability
    verification: UnitScore = _UNIT
    verification_status: SignalAvailability
    raw: RawSignals


__all__: tuple[str, ...] = ("BehavioralProfile",)