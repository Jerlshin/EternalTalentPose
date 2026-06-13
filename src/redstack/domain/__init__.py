"""REDSTACK domain layer — pure value objects, invariants, provenance.

Owner layer: domain.
Allowed imports: stdlib, pydantic v2, numpy (vectors only), sibling domain modules.

Import-time wiring: ``ProvenanceHandle.inline`` is a forward reference to
``RawCandidate`` (``provenance`` may not import ``source`` at runtime). The
reference is resolved here, where both concrete types are visible, so any
import of ``redstack.domain`` yields a fully-built ``ProvenanceHandle`` schema
without a circular dependency.
"""

from __future__ import annotations

from redstack.domain import provenance as _provenance
from redstack.domain.source import RawCandidate as _RawCandidate

# Expose the concrete type in the provenance module namespace, then rebuild the
# deferred pydantic schema so the forward reference resolves.
_provenance.RawCandidate = _RawCandidate  # type: ignore[attr-defined]
_provenance.ProvenanceHandle.model_rebuild()

__all__: tuple[str, ...] = ()
