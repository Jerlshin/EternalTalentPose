"""``IntegrityReport`` slice — honeypot / impossible-profile verdict (R4).

Owner layer: domain.
Allowed imports: ids, enums, provenance, pydantic.

Verdicts are data, not exceptions. Findings are stored pre-sorted by code for
determinism; each finding cites at least one ``EvidenceRef``.
"""

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.enums import IntegrityFlag, Severity
from redstack.domain.ids import UnitScore
from redstack.domain.provenance import EvidenceRef

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


@final
class IntegrityFinding(BaseModel):
    """A single fired integrity rule with its supporting evidence."""

    model_config = _STRICT

    code: IntegrityFlag
    severity: Severity
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    detail: str


@final
class IntegrityReport(BaseModel):
    """The honeypot verdict consumed by the scoring gate and ranking guard."""

    model_config = _STRICT

    findings: tuple[IntegrityFinding, ...]
    honeypot_score: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    is_honeypot: bool
    rules_evaluated: frozenset[IntegrityFlag]

    @model_validator(mode="after")
    def _check(self) -> IntegrityReport:
        ordered = tuple(sorted(self.findings, key=lambda f: f.code.value))
        if ordered != self.findings:
            raise ValueError("integrity findings must be sorted by code")
        if (
            any(f.severity is Severity.HARD for f in self.findings)
            and not self.is_honeypot
        ):
            raise ValueError("a HARD finding forces is_honeypot=True")
        for finding in self.findings:
            if finding.code not in self.rules_evaluated:
                raise ValueError("every fired finding must be in rules_evaluated")
        return self


__all__: tuple[str, ...] = ("IntegrityFinding", "IntegrityReport")