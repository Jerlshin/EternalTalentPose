"""REDSTACK Domain Layer — Domain exception hierarchy for software invariant failures."""
__all__ = ["DomainError", "SchemaError", "InvariantViolation", "ArtifactContractError"]

class DomainError(Exception): pass
class SchemaError(DomainError): pass
class InvariantViolation(DomainError): pass
class ArtifactContractError(DomainError): pass
