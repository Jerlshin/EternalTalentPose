
from __future__ import annotations

from redstack.domain.provenance import ProvenanceHandle
from redstack.domain.source import RawCandidate

# Resolve ProvenanceHandle.inline: RawCandidate (defer_build=True on the model).
ProvenanceHandle.model_rebuild(_types_namespace={"RawCandidate": RawCandidate})

__all__: tuple[str, ...] = ()