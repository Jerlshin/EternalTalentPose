"""Domain package init — resolves deferred provenance forward references.

``ProvenanceHandle.inline`` is typed as ``RawCandidate`` but ``provenance`` must
not import ``source`` at module load (purity: ids/enums/datetime only). The
concrete ``RawCandidate`` is bound here, at package-import time, by rebuilding the
deferred model once ``source`` is importable. This is the single, intended place
the forward reference is resolved (canonical provenance design).
"""

from __future__ import annotations

from redstack.domain.provenance import ProvenanceHandle
from redstack.domain.source import RawCandidate

# Resolve ProvenanceHandle.inline: RawCandidate (defer_build=True on the model).
ProvenanceHandle.model_rebuild(_types_namespace={"RawCandidate": RawCandidate})

__all__: tuple[str, ...] = ()