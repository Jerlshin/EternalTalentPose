"""Raw mapping -> ``RawCandidate`` ingestion, plus the feature-layer's
foundational value types (``features/parsing.py``).

Owner layer: features (pure). Allowed imports: ``domain`` + stdlib + numpy.
This module imports no ports, adapters, engines, pipelines; performs no IO,
no ML, no clock, no RNG.

Realizes ``CandidateIngestionEngine`` / O2-Validation (Repository §8): the dict
-> ``RawCandidate`` boundary (``validate``) and the canonical minting site for
every ``EvidenceRef`` (``mint_evidence``). Because it is the first extractor in
the feature build order and the documented owner of evidence provenance, it also
hosts the two value types every other extractor depends on:

* ``FeatureId`` — the ``"<group>.<name>"`` key (e.g. ``geo.hub_match``).
* ``FeatureCell`` — the ``(value, confidence, evidence)`` triple every feature
  emits (Feature Layer "feature cell model").

The ``Any`` boundary is confined to ``RawCandidate.from_mapping`` (the schema
mirror's sole narrowing point); nothing in this module surfaces an un-narrowed
``Any`` to callers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import NewType, final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from redstack.domain.enums import EvidenceKind
from redstack.domain.errors import ProvenanceError, SchemaError
from redstack.domain.ids import UnitScore
from redstack.domain.provenance import EvidenceRef
from redstack.domain.source import RawCandidate

__feature_version__ = "1.1.0"

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)

# ``FeatureId`` is the stable string key of the taxonomy; ``features.layout`` /
# ``features.registry`` bind it to a ``FeatureIndex``. Minted here (lowercased,
# dotted) so every extractor refers to features by one nominal type.
FeatureId = NewType("FeatureId", str)

_FEATURE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def feature_id(group: str, name: str) -> FeatureId:
    """Mint a validated ``"<group>.<name>"`` feature id."""
    raw = f"{group}.{name}"
    if not _FEATURE_ID_RE.match(raw):
        raise ValueError(f"malformed feature id: {raw!r}")
    return FeatureId(raw)


@final
class FeatureCell(BaseModel):
    """A single feature's output: ``(value, confidence, evidence)``.

    ``value`` is the bulk-path scalar folded into the ``(N, D)`` CQV matrix
    (must be finite — all sentinels resolved upstream; per-feature *bounds* are
    enforced by the registry, not here). ``confidence`` is the group-granularity
    trust ``UnitScore``. ``evidence`` are pre-minted ``EvidenceRef``s; their path
    resolution is guaranteed at mint time by ``mint_evidence``.
    """

    model_config = _STRICT

    value: float = Field(allow_inf_nan=False)
    confidence: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence: tuple[EvidenceRef, ...]

    @field_validator("value", "confidence", mode="after")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("feature value/confidence must be finite")
        return v


def clamp_unit(value: float) -> UnitScore:
    """Clamp an arbitrary finite float into ``[0, 1]`` and mint a ``UnitScore``.

    The sole place ``float`` is narrowed to ``UnitScore`` in the parsing module.
    """
    if not math.isfinite(value):
        raise ValueError("cannot clamp a non-finite value to UnitScore")
    if value <= 0.0:
        return UnitScore(0.0)
    if value >= 1.0:
        return UnitScore(1.0)
    return UnitScore(value)


def make_cell(
    value: float,
    confidence: float,
    evidence: tuple[EvidenceRef, ...],
) -> FeatureCell:
    """Construct a ``FeatureCell``, clamping ``confidence`` into ``[0, 1]``."""
    return FeatureCell(value=value, confidence=clamp_unit(confidence), evidence=evidence)


def validate(raw: Mapping[str, object]) -> RawCandidate:
    """Validate and narrow a raw mapping into a ``RawCandidate``.

    The dict -> model boundary used by ``CandidateIngestionEngine`` (R1 / O-read).
    Type/shape violations raise ``SchemaError``; *semantic* contradictions
    (inverted salary, expert-at-zero-months, current-with-end-date) are preserved
    faithfully for downstream honeypot detection — never silently coerced.
    """
    return RawCandidate.from_mapping(raw)


# --------------------------------------------------------------------------- #
# EvidenceRef minting: resolve a dotted/bracket path against the RawCandidate. #
# --------------------------------------------------------------------------- #

# Tokens: ``name`` | ``[12]`` | ``['key']`` | ``["key"]`` | ``[key]``
_PATH_TOKEN_RE = re.compile(
    r"""
      (?P<attr>[A-Za-z_][A-Za-z0-9_]*)        # attribute / field name
    | \[\s*(?P<index>\d+)\s*\]                 # numeric sequence index
    | \[\s*'(?P<sq>[^']*)'\s*\]                # single-quoted mapping key
    | \[\s*"(?P<dq>[^"]*)"\s*\]                # double-quoted mapping key
    | \[\s*(?P<bare>[^\]\['\"]+?)\s*\]         # bare mapping key
    """,
    re.VERBOSE,
)


def _scalarize(value: object, path: str) -> str | int | float | bool:
    """Project a resolved leaf to a JSON scalar (the ``EvidenceRef.value`` type).

    Dates serialize to ISO strings and enums to their ``.value`` (serialization
    is by value, §P/§R), so evidence round-trips stably. Non-scalar leaves are a
    provenance error: an ``EvidenceRef`` must point at an atomic fact.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        member: object = value.value
        if isinstance(member, bool):
            return member
        if isinstance(member, (str, int, float)):
            return member
    raise ProvenanceError(
        f"evidence path {path!r} resolves to a non-scalar value of type "
        f"{type(value).__name__}"
    )


def resolve_path(raw: RawCandidate, path: str) -> str | int | float | bool:
    """Resolve a dotted/bracket ``path`` against ``raw`` to its scalar value.

    Supports attribute traversal (``profile.location``), numeric sequence
    indexing (``career_history[0].title``) and mapping-key indexing
    (``redrob_signals.skill_assessment_scores['pytorch']``). Any failure to
    resolve — unknown attribute, out-of-range index, missing key, malformed
    syntax, or a non-scalar terminus — raises ``ProvenanceError``.
    """
    if not path:
        raise ProvenanceError("empty evidence path")

    pos = 0
    expect_dot = False
    current: object = raw
    length = len(path)

    while pos < length:
        char = path[pos]
        if char == ".":
            if not expect_dot:
                raise ProvenanceError(f"unexpected '.' in evidence path {path!r}")
            expect_dot = False
            pos += 1
            continue

        match = _PATH_TOKEN_RE.match(path, pos)
        if match is None or match.start() != pos:
            raise ProvenanceError(f"malformed token in evidence path {path!r} at {pos}")

        attr = match.group("attr")
        index = match.group("index")
        try:
            if attr is not None:
                if expect_dot:
                    raise ProvenanceError(f"missing '.' in evidence path {path!r}")
                current = getattr(current, attr)
                expect_dot = True
            elif index is not None:
                if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                    raise ProvenanceError(
                        f"index [{index}] applied to non-sequence in path {path!r}"
                    )
                current = current[int(index)]
                expect_dot = True
            else:
                key = match.group("sq")
                if key is None:
                    key = match.group("dq")
                if key is None:
                    key = match.group("bare")
                if not isinstance(current, Mapping):
                    raise ProvenanceError(
                        f"key access applied to non-mapping in path {path!r}"
                    )
                current = current[key]
                expect_dot = True
        except (AttributeError, IndexError, KeyError, ValueError, TypeError) as exc:
            raise ProvenanceError(
                f"evidence path {path!r} does not resolve in RawCandidate: {exc}"
            ) from exc

        pos = match.end()

    if not expect_dot:
        raise ProvenanceError(f"evidence path {path!r} ends mid-token")
    return _scalarize(current, path)


def mint_evidence(
    raw: RawCandidate,
    kind: EvidenceKind,
    path: str,
    *,
    as_of: date | None = None,
) -> EvidenceRef:
    """Mint an ``EvidenceRef`` whose ``path`` is verified to resolve in ``raw``.

    This is the *only* sanctioned constructor of ``EvidenceRef`` in the system
    (Domain §D): the value embedded in the ref is the actual scalar at ``path``,
    so a clause built from this ref cannot cite a fact the profile lacks. A
    dangling path raises ``ProvenanceError``.
    """
    value = resolve_path(raw, path)
    return EvidenceRef(kind=kind, path=path, value=value, as_of=as_of)


__all__: tuple[str, ...] = (
    "FeatureCell",
    "FeatureId",
    "SchemaError",
    "clamp_unit",
    "feature_id",
    "make_cell",
    "mint_evidence",
    "resolve_path",
    "validate",
)