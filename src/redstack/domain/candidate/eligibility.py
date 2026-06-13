"""``EligibilityReport`` slice — JD hard blocks + soft penalties (R4).

Owner layer: domain.
Allowed imports: ids, enums, provenance, pydantic.

``is_eligible`` is true iff there are no hard blocks; soft penalties never make
a candidate ineligible — they feed scoring as down-weights.
"""

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.enums import EligibilityCode, Severity
from redstack.domain.provenance import EvidenceRef

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


@final
class EligibilityFinding(BaseModel):
    """A single eligibility predicate outcome with its evidence."""

    model_config = _STRICT

    code: EligibilityCode
    severity: Severity
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    detail: str


@final
class EligibilityReport(BaseModel):
    """JD disqualifier / penalty verdict."""

    model_config = _STRICT

    hard_blocks: tuple[EligibilityFinding, ...]
    soft_penalties: tuple[EligibilityFinding, ...]
    is_eligible: bool
    gates_passed: frozenset[EligibilityCode]

    @model_validator(mode="after")
    def _check(self) -> EligibilityReport:
        if (
            tuple(sorted(self.hard_blocks, key=lambda f: f.code.value))
            != self.hard_blocks
        ):
            raise ValueError("hard_blocks must be sorted by code")
        if (
            tuple(sorted(self.soft_penalties, key=lambda f: f.code.value))
            != self.soft_penalties
        ):
            raise ValueError("soft_penalties must be sorted by code")
        if any(f.severity is not Severity.HARD for f in self.hard_blocks):
            raise ValueError("every hard_block must have severity HARD")
        if any(f.severity is not Severity.SOFT for f in self.soft_penalties):
            raise ValueError("every soft_penalty must have severity SOFT")
        if self.is_eligible != (len(self.hard_blocks) == 0):
            raise ValueError("is_eligible must equal (no hard_blocks)")
        return self


__all__: tuple[str, ...] = ("EligibilityFinding", "EligibilityReport")
