
from __future__ import annotations
import re
from collections.abc import Mapping, Sequence
from datetime import date
from redstack.domain.enums import EvidenceKind
from redstack.domain.errors import ProvenanceError
from redstack.domain.provenance import EvidenceRef

#: The scalar union an EvidenceRef value may take (mirrors the canonical VO).
EvidenceValue = str | bool | int | float

_INDEX_RE = re.compile(r"^(?P<name>[^\[]+)\[(?P<idx>\d+)\]$")


def resolve_path(raw: Mapping[str, object], path: str) -> object:

    cursor: object = raw
    for segment in path.split("."):
        match = _INDEX_RE.match(segment)
        if match is not None:
            name = match.group("name")
            idx = int(match.group("idx"))
            if not isinstance(cursor, Mapping) or name not in cursor:
                raise ProvenanceError(f"path {path!r} segment {name!r} missing")
            container = cursor[name]
            if not isinstance(container, Sequence) or isinstance(container, (str, bytes)):
                raise ProvenanceError(f"path {path!r} segment {name!r} not a list")
            if idx >= len(container):
                raise ProvenanceError(f"path {path!r} index {idx} out of range")
            cursor = container[idx]
        else:
            if not isinstance(cursor, Mapping) or segment not in cursor:
                raise ProvenanceError(f"path {path!r} segment {segment!r} missing")
            cursor = cursor[segment]
    return cursor


def mint(
    raw: Mapping[str, object],
    *,
    kind: EvidenceKind,
    path: str,
    as_of: date | None = None,
) -> EvidenceRef:
    """Mint an EvidenceRef, verifying the path resolves to a JSON scalar.

    Raises:
        ProvenanceError: the path does not resolve, or resolves to a non-scalar.
    """
    value = resolve_path(raw, path)
    if isinstance(value, bool) or isinstance(value, (str, int, float)):
        scalar: EvidenceValue = value
    elif value is None:
        # A null at a resolvable path is recorded as the literal string for audit
        # (the path *exists*; its value is null), preserving the resolve guarantee.
        scalar = "null"
    else:
        raise ProvenanceError(f"path {path!r} resolves to non-scalar {type(value).__name__}")
    return EvidenceRef(kind=kind, path=path, value=scalar, as_of=as_of)