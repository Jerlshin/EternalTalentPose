"""Domain error hierarchy for invariant / programming failures.

Owner layer: domain.
Allowed imports: stdlib.

Business verdicts ("honeypot", "ineligible") are *data* (Report objects) and
are never raised. Only invariant or schema violations raise; constructing an
object that would break an invariant fails immediately.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base of every domain-raised error."""


class SchemaError(DomainError):
    """Raw input violates the candidate schema mirror (type/shape only)."""


class InvalidCandidateId(SchemaError):
    """A candidate id does not match the canonical pattern."""


class FieldSchemaViolation(SchemaError):
    """A raw field has the wrong type or shape."""


class InvariantViolation(DomainError):
    """A constructed object would break a domain invariant."""


class RepresentationStageError(InvariantViolation):
    """A slice was attached out of order or accessed before population."""


class CQVInvariantError(InvariantViolation):
    """Wrong dim, NaN/inf, out-of-range cell, or schema_version mismatch."""


class ScoreInvariantError(InvariantViolation):
    """A gating or scoring-formula contradiction."""


class RankingInvariantError(InvariantViolation):
    """Count / unique-rank / monotonicity / tie-break violation."""


class ProvenanceError(DomainError):
    """An ``EvidenceRef`` path does not resolve in the ``RawCandidate``."""


class ArtifactContractError(DomainError):
    """Feature-layout / anchor-set / weights schema_version mismatch."""


class IntegrityViolation(DomainError):
    """An engine chose to fail hard on an integrity breach (rare path)."""


class IneligibleCandidate(DomainError):
    """An engine chose to fail hard on ineligibility (rare path)."""


__all__: tuple[str, ...] = (
    "ArtifactContractError",
    "CQVInvariantError",
    "DomainError",
    "FieldSchemaViolation",
    "IneligibleCandidate",
    "IntegrityViolation",
    "InvalidCandidateId",
    "InvariantViolation",
    "ProvenanceError",
    "RankingInvariantError",
    "RepresentationStageError",
    "SchemaError",
    "ScoreInvariantError",
)