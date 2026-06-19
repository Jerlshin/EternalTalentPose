

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, model_validator

from redstack.domain.enums import Severity, ValidationCode

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


@final
class ValidationFinding(BaseModel):
    """A single validation outcome."""

    model_config = _STRICT

    code: ValidationCode
    severity: Severity
    message: str
    location: str | None


@final
class ValidationReport(BaseModel):
    """The validator verdict; valid iff no HARD finding."""

    model_config = _STRICT

    findings: tuple[ValidationFinding, ...]
    is_valid: bool
    checks_run: frozenset[ValidationCode]

    @model_validator(mode="after")
    def _check(self) -> ValidationReport:
        if tuple(sorted(self.findings, key=lambda f: f.code.value)) != self.findings:
            raise ValueError("validation findings must be sorted by code")
        if self.is_valid != all(f.severity is not Severity.HARD for f in self.findings):
            raise ValueError("is_valid must equal (no HARD finding)")
        for finding in self.findings:
            if finding.code not in self.checks_run:
                raise ValueError("every finding's code must be in checks_run")
        return self


__all__: tuple[str, ...] = ("ValidationFinding", "ValidationReport")