"""Feature store metadata + the explainability chain (``features/store.py``).

Owner layer: features (pure). Allowed imports: ``features.layout``,
``features.registry``, ``features.parsing``, ``domain``, stdlib, numpy. No ports,
adapters, engines, pipelines, IO (the snapshot *holds* arrays; persistence is an
adapter concern), ML runtime, clock, or RNG.

Realizes Feature Layer Part 6 (store design) + Part 8 (explainability):

* ``FeatureProvenance`` — feature → ``EvidenceRef``[] → raw-field chain.
* ``FeatureLineage`` — the acyclic feature→dependencies DAG (invalidation +
  explainability); validated and topologically ordered at construction.
* ``FeatureSnapshot`` — the materialized ``(N, D)`` value matrix + ``(N, G)``
  group-confidence matrix + id index for a given input+version.
* ``FeatureAuditRecord`` — per (candidate, feature) audit row materialized for
  survivors/top-K. The "audit timestamp" is an *injected* ``recorded_at`` date
  (this layer takes no wall clock).
* ``FeatureImportance`` — per-feature contribution (O8 weights + ablation),
  read by Reasoning to pick which features to cite.
* ``FeatureContracts`` — cross-feature invariants (``*.competency`` ≤ group
  corroboration; ``hp.composite`` ≥ max single hard detector).
* ``FeatureValidation`` — run-time assertions: no NaN, in-range, dependency
  satisfaction, layout match.
* ``FeatureCache`` — content-addressed within-run memo (immutable; COW).

The chain ``ScoreBreakdown → FeatureImportance → FeatureProvenance →
EvidenceRef → RawCandidate.field`` (Part 8) is the single structure serving
reasoning, audit, and Stage-4 defence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from redstack.domain.ids import CandidateId, UnitScore
from redstack.domain.provenance import EvidenceRef
from redstack.features.layout import (
    DIM,
    FEATURE_IDS,
    GROUP_ORDER,
    LAYOUT_VERSION,
    NUM_GROUPS,
)
from redstack.features.parsing import FeatureCell, FeatureId
from redstack.features.registry import FeatureRegistry, FeatureVersion

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)
_ARRAY_STRICT = ConfigDict(
    frozen=True, extra="forbid", validate_default=True, arbitrary_types_allowed=True
)


# --------------------------------------------------------------------------- #
# Provenance + lineage (Part 8).                                              #
# --------------------------------------------------------------------------- #


@final
class FeatureProvenance(BaseModel):
    """One feature's ``EvidenceRef`` chain back to raw fields."""

    model_config = _STRICT

    feature_id: FeatureId
    evidence: tuple[EvidenceRef, ...]
    raw_field_paths: tuple[str, ...]

    @classmethod
    def from_cell(cls, feature_id_: FeatureId, cell: FeatureCell) -> FeatureProvenance:
        """Derive provenance from an emitted ``FeatureCell``'s evidence."""
        return cls(
            feature_id=feature_id_,
            evidence=cell.evidence,
            raw_field_paths=tuple(ref.path for ref in cell.evidence),
        )


@final
class FeatureLineage(BaseModel):
    """The feature→dependencies DAG + feature→raw-field bindings.

    Acyclicity is enforced at construction (Kahn's algorithm); a cycle in the
    extraction graph would break the fixed topological order determinism.
    """

    model_config = _STRICT

    feature_dependencies: Mapping[FeatureId, tuple[FeatureId, ...]]
    raw_field_bindings: Mapping[FeatureId, tuple[str, ...]]

    @model_validator(mode="after")
    def _acyclic(self) -> FeatureLineage:
        self._topological_order(self.feature_dependencies)
        return self

    @staticmethod
    def _topological_order(
        edges: Mapping[FeatureId, tuple[FeatureId, ...]],
    ) -> tuple[FeatureId, ...]:
        # Nodes = keys ∪ all referenced dependencies.
        nodes: set[str] = set()
        for feature, deps in edges.items():
            nodes.add(str(feature))
            for dep in deps:
                nodes.add(str(dep))
        indegree: dict[str, int] = {node: 0 for node in nodes}
        adjacency: dict[str, list[str]] = {node: [] for node in nodes}
        # An edge dep -> feature (dependency must be produced first).
        for feature, deps in edges.items():
            for dep in deps:
                adjacency[str(dep)].append(str(feature))
                indegree[str(feature)] += 1
        # Deterministic Kahn: process ready nodes in sorted id order.
        ready: list[str] = sorted(
            node for node, degree in indegree.items() if degree == 0
        )
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for successor in sorted(adjacency[current]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
            ready.sort()
        if len(order) != len(nodes):
            raise ValueError("FeatureLineage contains a dependency cycle")
        return tuple(FeatureId(node) for node in order)

    def topological_order(self) -> tuple[FeatureId, ...]:
        """A deterministic dependency-respecting feature order."""
        return self._topological_order(self.feature_dependencies)

    def dependencies_of(self, feature_id_: FeatureId) -> tuple[FeatureId, ...]:
        return self.feature_dependencies.get(feature_id_, ())

    @classmethod
    def from_registry(
        cls,
        registry: FeatureRegistry,
        raw_field_bindings: Mapping[FeatureId, tuple[str, ...]] | None = None,
    ) -> FeatureLineage:
        """Build lineage from the registry's declared dependency edges."""
        edges: Mapping[FeatureId, tuple[FeatureId, ...]] = MappingProxyType(
            {d.feature_id: d.dependencies for d in registry.definitions}
        )
        bindings: Mapping[FeatureId, tuple[str, ...]] = (
            MappingProxyType(dict(raw_field_bindings))
            if raw_field_bindings is not None
            else MappingProxyType({})
        )
        return cls(feature_dependencies=edges, raw_field_bindings=bindings)


# --------------------------------------------------------------------------- #
# Snapshot (Part 6) — the bulk materialization.                               #
# --------------------------------------------------------------------------- #


@final
class FeatureSnapshot(BaseModel):
    """The ``(N, D)`` value matrix + ``(N, G)`` group-confidence + id index.

    A reproducible artifact keyed by input+version (persistence handled by an
    adapter — this model only holds the arrays and asserts their shape).
    """

    model_config = _ARRAY_STRICT

    candidate_ids: tuple[CandidateId, ...] = Field(min_length=1)
    values: npt.NDArray[np.float32]
    group_confidence: npt.NDArray[np.float32]
    layout_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)

    @field_validator("values", "group_confidence", mode="before")
    @classmethod
    def _coerce_float32_2d(cls, value: object) -> npt.NDArray[np.float32]:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("snapshot matrices must be 2-D")
        if not np.all(np.isfinite(array)):
            raise ValueError("snapshot matrices must contain no NaN/inf")
        frozen = np.ascontiguousarray(array, dtype=np.float32).copy()
        frozen.setflags(write=False)
        return frozen

    @model_validator(mode="after")
    def _shapes_consistent(self) -> FeatureSnapshot:
        n = len(self.candidate_ids)
        if self.values.shape != (n, DIM):
            raise ValueError(
                f"values shape {self.values.shape} != ({n}, {DIM})"
            )
        if self.group_confidence.shape != (n, NUM_GROUPS):
            raise ValueError(
                f"group_confidence shape {self.group_confidence.shape} "
                f"!= ({n}, {NUM_GROUPS})"
            )
        return self

    @property
    def n_rows(self) -> int:
        return len(self.candidate_ids)

    def row(self, source_index: int) -> npt.NDArray[np.float32]:
        """The ``(D,)`` value row for a source index (read-only view copy)."""
        if source_index < 0 or source_index >= self.n_rows:
            raise IndexError(f"source_index {source_index} out of range")
        out: npt.NDArray[np.float32] = np.array(
            self.values[source_index], dtype=np.float32
        )
        out.setflags(write=False)
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureSnapshot):
            return NotImplemented
        return (
            self.candidate_ids == other.candidate_ids
            and self.layout_version == other.layout_version
            and self.schema_version == other.schema_version
            and np.array_equal(self.values, other.values)
            and np.array_equal(self.group_confidence, other.group_confidence)
        )

    __hash__ = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Audit + importance (Part 6 / Part 8).                                       #
# --------------------------------------------------------------------------- #


@final
class FeatureAuditRecord(BaseModel):
    """Per (candidate, feature) audit row for survivors/top-K.

    ``recorded_at`` is an *injected* date (or ``None``); this layer never reads a
    wall clock.
    """

    model_config = _STRICT

    candidate_id: CandidateId
    feature_id: FeatureId
    value: float = Field(allow_inf_nan=False)
    confidence: UnitScore = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence: tuple[EvidenceRef, ...]
    version: FeatureVersion
    recorded_at: date | None = None

    @field_validator("value", "confidence", mode="after")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("audit value/confidence must be finite")
        return v

    @classmethod
    def from_cell(
        cls,
        candidate_id_: CandidateId,
        feature_id_: FeatureId,
        cell: FeatureCell,
        version: FeatureVersion,
        recorded_at: date | None = None,
    ) -> FeatureAuditRecord:
        return cls(
            candidate_id=candidate_id_,
            feature_id=feature_id_,
            value=cell.value,
            confidence=cell.confidence,
            evidence=cell.evidence,
            version=version,
            recorded_at=recorded_at,
        )


@final
class FeatureImportance(BaseModel):
    """Per-feature contribution scores (O8 weights + ablation), read by Reasoning."""

    model_config = _STRICT

    scores: Mapping[FeatureId, float]
    layout_version: str = Field(min_length=1)

    @field_validator("scores", mode="after")
    @classmethod
    def _finite_known(cls, value: Mapping[FeatureId, float]) -> Mapping[FeatureId, float]:
        known = set(FEATURE_IDS)
        for feature_id_, score in value.items():
            if str(feature_id_) not in known:
                raise ValueError(f"importance for unknown feature {feature_id_!r}")
            if not math.isfinite(score):
                raise ValueError("importance scores must be finite")
        return MappingProxyType(dict(value))

    def importance(self, feature_id_: FeatureId) -> float:
        """Importance of ``feature_id_`` (0.0 if not scored)."""
        return self.scores.get(feature_id_, 0.0)

    def top_k(self, k: int) -> tuple[tuple[FeatureId, float], ...]:
        """The ``k`` highest-importance features, ties broken by id (ascending)."""
        if k < 0:
            raise ValueError("k must be non-negative")
        ranked = sorted(
            self.scores.items(), key=lambda item: (-item[1], str(item[0]))
        )
        return tuple(ranked[:k])


# --------------------------------------------------------------------------- #
# Cross-feature contracts (Part 6).                                           #
# --------------------------------------------------------------------------- #


class ContractKind(str, Enum):
    """The relation a cross-feature invariant asserts."""

    LE_MAX_GROUP = "le_max_group"  # target <= max(operands)
    GE_MAX_GROUP = "ge_max_group"  # target >= max(operands)
    LE_FEATURE = "le_feature"      # target <= operands[0]


@final
class CrossFeatureInvariant(BaseModel):
    """One declarative cross-feature invariant."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    kind: ContractKind
    target: FeatureId
    operands: tuple[FeatureId, ...] = Field(min_length=1)
    description: str = Field(min_length=1)


@final
class ContractViolation(BaseModel):
    """A failed cross-feature invariant for a specific candidate row."""

    model_config = _STRICT

    name: str
    target: FeatureId
    detail: str


@final
class FeatureContracts(BaseModel):
    """A set of cross-feature invariants with a pure row evaluator."""

    model_config = _STRICT

    invariants: tuple[CrossFeatureInvariant, ...]

    def evaluate(
        self, values: Mapping[FeatureId, float], tolerance: float = 1e-6
    ) -> tuple[ContractViolation, ...]:
        """Return any invariants violated by a candidate's feature values."""
        violations: list[ContractViolation] = []
        for invariant in self.invariants:
            if invariant.target not in values:
                continue
            present = [values[op] for op in invariant.operands if op in values]
            if not present:
                continue
            target_value = values[invariant.target]
            if invariant.kind is ContractKind.LE_MAX_GROUP:
                bound = max(present)
                ok = target_value <= bound + tolerance
            elif invariant.kind is ContractKind.GE_MAX_GROUP:
                bound = max(present)
                ok = target_value >= bound - tolerance
            else:  # LE_FEATURE
                bound = present[0]
                ok = target_value <= bound + tolerance
            if not ok:
                violations.append(
                    ContractViolation(
                        name=invariant.name,
                        target=invariant.target,
                        detail=(
                            f"{invariant.target}={target_value:.6f} violates "
                            f"{invariant.kind.value} bound {bound:.6f}"
                        ),
                    )
                )
        return tuple(violations)


def _competency_corroboration_invariants() -> tuple[CrossFeatureInvariant, ...]:
    groups = ("retr", "rank", "recsys", "ir", "nlp", "llm", "mle", "mlops", "eval")
    out: list[CrossFeatureInvariant] = []
    for group in groups:
        out.append(
            CrossFeatureInvariant(
                name=f"{group}.competency_le_corroboration",
                kind=ContractKind.LE_MAX_GROUP,
                target=FeatureId(f"{group}.competency"),
                operands=(
                    FeatureId(f"{group}.trust"),
                    FeatureId(f"{group}.in_career"),
                    FeatureId(f"{group}.semantic"),
                ),
                description=(
                    "fused competency cannot exceed its strongest corroborating "
                    "source (anti keyword-stuffing)"
                ),
            )
        )
    return tuple(out)


def _honeypot_composite_invariant() -> CrossFeatureInvariant:
    hard = (
        "timeline_impossible", "skill_time_contradiction", "employment_overlap",
        "title_seniority_anomaly", "education_career_anomaly",
        "experience_inflation", "keyword_stuffing", "behavioral_inconsistency",
        "signal_impossibility", "identity_anomaly",
    )
    return CrossFeatureInvariant(
        name="hp.composite_ge_max_detector",
        kind=ContractKind.GE_MAX_GROUP,
        target=FeatureId("hp.composite"),
        operands=tuple(FeatureId(f"hp.{name}") for name in hard),
        description=(
            "composite honeypot risk must dominate the strongest single hard "
            "detector"
        ),
    )


def default_contracts() -> FeatureContracts:
    """The frozen default cross-feature invariant set (Part 6 examples)."""
    return FeatureContracts(
        invariants=_competency_corroboration_invariants()
        + (_honeypot_composite_invariant(),)
    )


# --------------------------------------------------------------------------- #
# Validation (Part 6).                                                        #
# --------------------------------------------------------------------------- #


class ValidationCode(str, Enum):
    """Run-time feature-assertion codes."""

    NAN_OR_INF = "nan_or_inf"
    OUT_OF_RANGE = "out_of_range"
    MISSING_FEATURE = "missing_feature"
    LAYOUT_MISMATCH = "layout_mismatch"
    DEPENDENCY_UNSATISFIED = "dependency_unsatisfied"


@final
class FeatureValidationFinding(BaseModel):
    """One failed run-time assertion."""

    model_config = _STRICT

    code: ValidationCode
    feature_id: FeatureId | None
    detail: str


@final
class FeatureValidation(BaseModel):
    """Pure run-time validators over a feature row / manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    registry: FeatureRegistry

    def check_row(
        self, values: Mapping[FeatureId, float]
    ) -> tuple[FeatureValidationFinding, ...]:
        """No-NaN, in-range (per schema), and full layout coverage."""
        findings: list[FeatureValidationFinding] = []
        for definition in self.registry.definitions:
            if definition.feature_id not in values:
                findings.append(
                    FeatureValidationFinding(
                        code=ValidationCode.MISSING_FEATURE,
                        feature_id=definition.feature_id,
                        detail="feature absent from row",
                    )
                )
                continue
            value = values[definition.feature_id]
            if not math.isfinite(value):
                findings.append(
                    FeatureValidationFinding(
                        code=ValidationCode.NAN_OR_INF,
                        feature_id=definition.feature_id,
                        detail="value is NaN/inf",
                    )
                )
                continue
            schema = definition.schema_
            if not (schema.lower <= value <= schema.upper):
                findings.append(
                    FeatureValidationFinding(
                        code=ValidationCode.OUT_OF_RANGE,
                        feature_id=definition.feature_id,
                        detail=f"{value} not in [{schema.lower}, {schema.upper}]",
                    )
                )
        return tuple(findings)

    def check_dependencies(
        self, present: frozenset[FeatureId]
    ) -> tuple[FeatureValidationFinding, ...]:
        """Every present feature's declared dependencies must also be present."""
        findings: list[FeatureValidationFinding] = []
        for definition in self.registry.definitions:
            if definition.feature_id not in present:
                continue
            for dependency in definition.dependencies:
                if dependency not in present:
                    findings.append(
                        FeatureValidationFinding(
                            code=ValidationCode.DEPENDENCY_UNSATISFIED,
                            feature_id=definition.feature_id,
                            detail=f"missing dependency {dependency}",
                        )
                    )
        return tuple(findings)

    def check_layout_match(
        self, layout_version: str
    ) -> tuple[FeatureValidationFinding, ...]:
        """The row's layout version must match the pinned layout."""
        if layout_version == LAYOUT_VERSION:
            return ()
        return (
            FeatureValidationFinding(
                code=ValidationCode.LAYOUT_MISMATCH,
                feature_id=None,
                detail=f"layout {layout_version!r} != pinned {LAYOUT_VERSION!r}",
            ),
        )


# --------------------------------------------------------------------------- #
# Cache (Part 6) — content-addressed within-run memo (immutable / COW).       #
# --------------------------------------------------------------------------- #


@final
class FeatureCacheKey(BaseModel):
    """Content-addressed cache key: ``(candidate_hash, layout_version)``."""

    model_config = _STRICT

    candidate_hash: str = Field(min_length=1)
    layout_version: str = Field(min_length=1)


@final
class FeatureCache(BaseModel):
    """An immutable within-run feature memo.

    No global mutable state: ``with_entry`` returns a *new* cache (copy-on-write),
    preserving the layer's purity guarantee.
    """

    model_config = _STRICT

    entries: Mapping[FeatureCacheKey, Mapping[FeatureId, FeatureCell]]

    @classmethod
    def empty(cls) -> FeatureCache:
        empty: Mapping[FeatureCacheKey, Mapping[FeatureId, FeatureCell]] = (
            MappingProxyType({})
        )
        return cls(entries=empty)

    def get(self, key: FeatureCacheKey) -> Mapping[FeatureId, FeatureCell] | None:
        return self.entries.get(key)

    def has(self, key: FeatureCacheKey) -> bool:
        return key in self.entries

    def with_entry(
        self, key: FeatureCacheKey, cells: Mapping[FeatureId, FeatureCell]
    ) -> FeatureCache:
        """Return a new cache with ``key`` mapped to ``cells`` (COW)."""
        merged = dict(self.entries)
        merged[key] = MappingProxyType(dict(cells))
        return FeatureCache(entries=MappingProxyType(merged))


# The frozen default contract set shared by reference.
DEFAULT_FEATURE_CONTRACTS: FeatureContracts = default_contracts()


__all__: tuple[str, ...] = (
    "DEFAULT_FEATURE_CONTRACTS",
    "ContractKind",
    "ContractViolation",
    "CrossFeatureInvariant",
    "FeatureAuditRecord",
    "FeatureCache",
    "FeatureCacheKey",
    "FeatureContracts",
    "FeatureImportance",
    "FeatureLineage",
    "FeatureProvenance",
    "FeatureSnapshot",
    "FeatureValidation",
    "FeatureValidationFinding",
    "ValidationCode",
    "default_contracts",
)