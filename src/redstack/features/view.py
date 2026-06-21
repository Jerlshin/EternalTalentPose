from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from types import MappingProxyType
from typing import Final, final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from redstack.domain.enums import EvidenceKind
from redstack.domain.ids import UnitScore
from redstack.domain.provenance import EvidenceRef
from redstack.domain.source import RawCandidate
from redstack.features.parsing import resolve_path

_VO = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)

FeatureId = str
CellEmission = tuple[tuple[FeatureId, "FeatureCell"], ...]


# --------------------------------------------------------------------------- #
# Pure numeric helpers (shared normalization vocabulary).                     #
# --------------------------------------------------------------------------- #
def clamp_unit(value: float) -> float:
    """Clamp a finite float into ``[0, 1]``; a non-finite input is a bug → raise."""
    if not math.isfinite(value):
        raise ValueError(f"clamp_unit received a non-finite value: {value!r}")
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def unit(value: float) -> UnitScore:
    """Construct a ``UnitScore`` from a float, clamping into ``[0, 1]``."""
    return UnitScore(clamp_unit(value))


def bounded_log_scale(count: float, *, saturation: float) -> float:
    """Map a non-negative count onto ``[0, 1]`` with diminishing returns.

    ``log1p(count) / log1p(saturation)`` — a count equal to ``saturation`` maps
    to ~1.0; growth past it is clamped. Negative counts (sentinels) clamp to 0.
    """
    if saturation <= 0.0:
        raise ValueError("saturation must be positive")
    safe = count if count > 0.0 else 0.0
    return clamp_unit(math.log1p(safe) / math.log1p(saturation))


def inverse_bounded(value: float, *, scale: float) -> float:
    """Map a non-negative magnitude onto ``(0, 1]`` decreasing in ``value``.

    ``scale / (scale + value)`` — ``value == 0`` → 1.0, ``value == scale`` → 0.5.
    Used for "smaller is better" quantities such as response time in hours.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    safe = value if value > 0.0 else 0.0
    return clamp_unit(scale / (scale + safe))


def recency_unit(days_elapsed: float, *, half_life_days: float) -> float:
    """Exponential recency in ``[0, 1]``: 1.0 today, 0.5 at one half-life.

    A negative ``days_elapsed`` (a future date relative to ``as_of``) is treated
    as 0 days (fully recent) here; the *impossibility* of a future date is the
    honeypot layer's job, not the normalizer's.
    """
    if half_life_days <= 0.0:
        raise ValueError("half_life_days must be positive")
    safe = days_elapsed if days_elapsed > 0.0 else 0.0
    return clamp_unit(math.pow(0.5, safe / half_life_days))


def days_between(later: date, earlier: date) -> int:
    """Signed day delta ``later - earlier`` (negative if ``later`` precedes)."""
    return (later - earlier).days


def mean_of(values: tuple[float, ...]) -> float:
    """Arithmetic mean of a non-empty tuple; empty → 0.0 (neutral)."""
    if not values:
        return 0.0
    return math.fsum(values) / len(values)


def make_evidence(
    kind: EvidenceKind,
    path: str,
    value: str | int | float | bool,
    *,
    raw: RawCandidate | None = None,
) -> EvidenceRef:
    """Mint an ``EvidenceRef``; ``date`` callers pass ``.isoformat()`` strings.

    When ``raw`` is given, ``path`` is verified to resolve inside it before
    the ref is minted -- a dangling path (wrong index, renamed field) raises
    ``ProvenanceError`` immediately rather than shipping a citation nothing
    backs. ``value`` is kept as the caller supplied it (it may be a derived
    label, not the literal scalar at ``path``); only existence is checked.
    Callers citing a literal ``RawCandidate`` field must pass ``raw``.
    ``EvidenceKind.DERIVED`` evidence (and citations of fields on an
    already-validated domain profile, where no raw record exists to dangle
    against) may omit it.
    """
    if raw is not None:
        resolve_path(raw, path)
    return EvidenceRef(kind=kind, path=path, value=value)


# --------------------------------------------------------------------------- #
# Feature cell.                                                               #
# --------------------------------------------------------------------------- #
@final
class FeatureCell(BaseModel):
    """One feature's ``(value, confidence, evidence)`` output.

    ``value`` carries no range constraint here beyond finiteness — the per-index
    bounds in ``FeatureLayout`` are checked when the cell folds into the CQV.
    ``evidence`` is non-empty by construction: a feature with no evidence cannot
    be cited by Reasoning, so emitting one would be a silent hallucination risk.
    """

    model_config = _VO

    value: float = Field(allow_inf_nan=False)
    confidence: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)

    @field_validator("confidence", mode="after")
    @classmethod
    def _confidence_unit(cls, value: float) -> UnitScore:
        return UnitScore(value)


def cell(
    value: float, confidence: float, evidence: tuple[EvidenceRef, ...]
) -> FeatureCell:
    """Build a ``FeatureCell``, clamping ``confidence`` into ``[0, 1]``."""
    return FeatureCell(value=value, confidence=unit(confidence), evidence=evidence)


def group_of(feature_id: str) -> str:
    """Group prefix (text before the first dot) of a feature id."""
    return feature_id.split(".", 1)[0]


# --------------------------------------------------------------------------- #
# Read-only feature view (Part 9 — the sole engine read surface).             #
# --------------------------------------------------------------------------- #
@final
class FeatureView(BaseModel):
    """Typed, read-only accessor over one candidate's cells + group confidence.

    Engines never touch raw arrays; they resolve features by id through this
    view. Construction is via ``from_cells`` (which derives group confidence as
    the mean of each group's member-cell confidences). All three accessors are
    pure and deterministic.
    """

    model_config = _VO

    cells: Mapping[FeatureId, FeatureCell]
    group_confidences: Mapping[str, float]
    importances: Mapping[FeatureId, float]

    @field_validator("cells", mode="after")
    @classmethod
    def _freeze_cells(
        cls, value: Mapping[FeatureId, FeatureCell]
    ) -> Mapping[FeatureId, FeatureCell]:
        return MappingProxyType(dict(value))

    @field_validator("group_confidences", mode="after")
    @classmethod
    def _freeze_group_conf(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        for group, conf in value.items():
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"group_confidence for {group!r} not in [0, 1]")
        return MappingProxyType(dict(value))

    @field_validator("importances", mode="after")
    @classmethod
    def _freeze_importance(
        cls, value: Mapping[FeatureId, float]
    ) -> Mapping[FeatureId, float]:
        for feature_id, weight in value.items():
            if not math.isfinite(weight):
                raise ValueError(f"importance for {feature_id!r} is not finite")
        return MappingProxyType(dict(value))

    # -- the Part 9 contract --------------------------------------------- #
    def get(self, feature_id: FeatureId) -> FeatureCell:
        """Resolve a feature's cell. Unknown id → ``KeyError`` (programming error)."""
        return self.cells[feature_id]

    def group_confidence(self, group: str) -> UnitScore:
        """Group-granular confidence. Unknown group → ``KeyError``."""
        return UnitScore(self.group_confidences[group])

    def importance(self, feature_id: FeatureId) -> float:
        """Learned importance; a feature with no learned weight returns ``0.0``."""
        return self.importances.get(feature_id, 0.0)

    # -- convenience (still read-only) ----------------------------------- #
    def has(self, feature_id: FeatureId) -> bool:
        """Whether a cell was emitted for ``feature_id``."""
        return feature_id in self.cells

    def value_of(self, feature_id: FeatureId, default: float = 0.0) -> float:
        """The cell value, or ``default`` if the feature was not emitted."""
        found = self.cells.get(feature_id)
        return found.value if found is not None else default

    @classmethod
    def from_cells(
        cls,
        cells: Mapping[FeatureId, FeatureCell],
        *,
        importances: Mapping[FeatureId, float] | None = None,
    ) -> FeatureView:
        """Assemble a view, deriving group confidence as the per-group mean.

        Engines build the full ``{feature_id: FeatureCell}`` map from every
        extractor, then hand it here; group confidence is the deterministic mean
        of member-cell confidences (Part 7: confidence is stored at group
        granularity).
        """
        grouped: dict[str, list[float]] = {}
        for feature_id, feature_cell in cells.items():
            grouped.setdefault(group_of(feature_id), []).append(
                float(feature_cell.confidence)
            )
        group_conf = {
            group: math.fsum(confs) / len(confs) for group, confs in grouped.items()
        }
        return cls(
            cells=dict(cells),
            group_confidences=group_conf,
            importances=dict(importances) if importances is not None else {},
        )


# Saturation / scale constants shared by the extractors (documented once here).
ENDORSEMENT_SATURATION: Final[float] = 50.0
DURATION_SATURATION_MONTHS: Final[float] = 36.0
ACTIVITY_HALF_LIFE_DAYS: Final[float] = 90.0
STALE_HALF_LIFE_DAYS: Final[float] = 180.0
RESPONSE_TIME_SCALE_HOURS: Final[float] = 24.0


__all__ = (
    "ACTIVITY_HALF_LIFE_DAYS",
    "CellEmission",
    "DURATION_SATURATION_MONTHS",
    "ENDORSEMENT_SATURATION",
    "FeatureCell",
    "FeatureId",
    "FeatureView",
    "RESPONSE_TIME_SCALE_HOURS",
    "STALE_HALF_LIFE_DAYS",
    "bounded_log_scale",
    "cell",
    "clamp_unit",
    "days_between",
    "group_of",
    "inverse_bounded",
    "make_evidence",
    "mean_of",
    "recency_unit",
    "unit",
)
