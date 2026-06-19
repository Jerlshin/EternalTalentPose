

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
    """A single fired eligibility gate with its supporting evidence."""

    model_config = _STRICT

    code: EligibilityCode
    severity: Severity
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    detail: str


@final
class EligibilityReport(BaseModel):
    """JD-fit verdict consumed by the scoring gate and reasoning concerns."""

    model_config = _STRICT

    hard_blocks: tuple[EligibilityFinding, ...]
    soft_penalties: tuple[EligibilityFinding, ...]
    is_eligible: bool
    gates_passed: frozenset[EligibilityCode]

    @model_validator(mode="after")
    def _check(self) -> EligibilityReport:
        if any(f.severity is not Severity.HARD for f in self.hard_blocks):
            raise ValueError("hard_blocks must all carry Severity.HARD")
        if any(f.severity is not Severity.SOFT for f in self.soft_penalties):
            raise ValueError("soft_penalties must all carry Severity.SOFT")
        if tuple(sorted(self.hard_blocks, key=lambda f: f.code.value)) != self.hard_blocks:
            raise ValueError("hard_blocks must be sorted by code")
        if (
            tuple(sorted(self.soft_penalties, key=lambda f: f.code.value))
            != self.soft_penalties
        ):
            raise ValueError("soft_penalties must be sorted by code")
        if self.is_eligible != (len(self.hard_blocks) == 0):
            raise ValueError("is_eligible must equal (len(hard_blocks) == 0)")
        return self


__all__: tuple[str, ...] = ("EligibilityFinding", "EligibilityReport")
