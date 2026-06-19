
from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.ids import ArchetypeId, UnitScore


@final
class ArchetypeAssignment(BaseModel):
    """Which O7 cluster the candidate belongs to."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
    )

    archetype_id: ArchetypeId
    distance: float = Field(ge=0.0, allow_inf_nan=False)
    membership_confidence: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    secondary_archetype: ArchetypeId | None
    label: str | None
    is_target_archetype: bool

    @model_validator(mode="after")
    def _secondary_distinct(self) -> ArchetypeAssignment:
        if (
            self.secondary_archetype is not None
            and self.secondary_archetype == self.archetype_id
        ):
            raise ValueError("secondary_archetype must differ from archetype_id")
        return self


__all__: tuple[str, ...] = ("ArchetypeAssignment",)