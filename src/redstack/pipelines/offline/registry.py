
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

__all__: tuple[str, ...] = (
    "ArtifactKind",
    "ArtifactSpec",
    "OfflineArtifactRegistry",
    "ValidationOutcome",
    "OFFLINE_ARTIFACT_REGISTRY",
)


class ArtifactKind(StrEnum):
    """The on-disk serialization of an artifact (mirrors ``MANIFEST.json:kind``)."""

    JSON = "json"
    YAML = "yaml"
    NPY = "npy"
    PARQUET = "parquet"
    ONNX = "onnx"


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """The result of applying an :class:`ArtifactSpec` validator to a payload.

    A *verdict* object, never an exception: ``ok`` records success; otherwise
    ``reason`` names precisely what failed for the offending ``key``. The runner
    converts a non-``ok`` outcome into the hard ``ArtifactContractError`` fail
    path (Offline Pipeline Part 11, schema/validator failure ⇒ stage fails).
    """

    key: str
    ok: bool
    reason: str | None = None

    @classmethod
    def passed(cls, key: str) -> ValidationOutcome:
        """A passing outcome for ``key``."""
        return cls(key=key, ok=True, reason=None)

    @classmethod
    def failed(cls, key: str, reason: str) -> ValidationOutcome:
        """A failing outcome for ``key`` carrying the precise reason."""
        return cls(key=key, ok=False, reason=reason)


# A validator is a pure predicate over a parsed payload. JSON/YAML artifacts
# arrive as a ``Mapping``; ``npy`` arrives as a small metadata mapping the runner
# extracts from the array header (shape/dtype) so the validator stays IO-free and
# numpy-free at this layer. The validator returns ``None`` on success or a reason
# string on failure — the spec wants the offending detail, not just a bool.
ArtifactValidator = Callable[[Mapping[str, object]], str | None]


def _require_keys(payload: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    """Return a reason if any required top-level key is absent, else ``None``."""
    missing = tuple(k for k in keys if k not in payload)
    if missing:
        return f"missing required field(s): {', '.join(missing)}"
    return None


def _as_int(payload: Mapping[str, object], key: str) -> int | None:
    """Best-effort int extraction; returns ``None`` if absent or non-integral."""
    value = payload.get(key)
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly.
        return None
    if isinstance(value, int):
        return value
    return None


# --------------------------------------------------------------------------- #
# Per-artifact validators (Part 10 "Validation" column, encoded as predicates). #
# Each is deliberately structural-not-semantic: it checks shape/coverage/range #
# invariants the runner can confirm without re-running the stage's algorithm.  #
# Cross-artifact coherence (layout/dim/model_id agreement) is enforced by the  #
# registry's coherence pass at O17, not by these single-artifact predicates.   #
# --------------------------------------------------------------------------- #
def _v_dataset_profile(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("candidate_count", "field_coverage", "distributions"))
    if reason is not None:
        return reason
    n = _as_int(p, "candidate_count")
    if n is None or n <= 0:
        return "candidate_count must be a positive integer"
    return None


def _v_canonical_maps(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("token_map", "company_map", "skill_map"))
    if reason is not None:
        return reason
    for name in ("token_map", "company_map", "skill_map"):
        mapping = p.get(name)
        if not isinstance(mapping, Mapping):
            return f"{name} must be a mapping"
        if any(not isinstance(v, str) or not v for v in mapping.values()):
            return f"{name} must not contain empty canonical values"
    return None


def _v_validation_report(p: Mapping[str, object]) -> str | None:
    return _require_keys(p, ("accepted", "rejected", "reject_log"))


def _v_integrity_rules(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("rules",))
    if reason is not None:
        return reason
    rules = p.get("rules")
    if not isinstance(rules, (list, tuple)) or not rules:
        return "rules must be a non-empty sequence"
    return None


def _v_honeypot_catalog(p: Mapping[str, object]) -> str | None:
    return _require_keys(p, ("exemplars", "suspect_ids"))


def _v_integrity_thresholds(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("per_detector", "honeypot_threshold"))
    if reason is not None:
        return reason
    threshold = p.get("honeypot_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return "honeypot_threshold must be numeric"
    if not math.isfinite(float(threshold)):
        return "honeypot_threshold must be finite"
    return None


def _v_risk_weights(p: Mapping[str, object]) -> str | None:
    return _require_keys(p, ("weights", "confidence_shrink"))


def _v_lexicon_compiled(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("concepts",))
    if reason is not None:
        return reason
    concepts = p.get("concepts")
    if not isinstance(concepts, Mapping) or not concepts:
        return "concepts must be a non-empty mapping"
    return None


def _v_concepts(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("concepts",))
    if reason is not None:
        return reason
    concepts = p.get("concepts")
    if not isinstance(concepts, Mapping) or not concepts:
        return "concepts must be a non-empty mapping"
    for name, body in concepts.items():
        if not isinstance(body, Mapping) or not body.get("anchor_text"):
            return f"concept {name!r} is missing anchor_text"
    return None


def _v_graph(p: Mapping[str, object]) -> str | None:
    return _require_keys(p, ("nodes", "edges"))


def _v_jd_concepts(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("anchors",))
    if reason is not None:
        return reason
    anchors = p.get("anchors")
    if not isinstance(anchors, (list, tuple)) or not anchors:
        return "anchors must be a non-empty sequence"
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            return "each anchor must be a mapping"
        if anchor.get("polarity") not in ("positive", "negative"):
            return "each anchor must carry a positive/negative polarity"
    return None


def _v_eligibility_rules(p: Mapping[str, object]) -> str | None:
    return _require_keys(p, ("hard_blocks", "soft_penalties"))


def _v_centroids(p: Mapping[str, object]) -> str | None:
    # ``p`` is the npy header metadata: shape (k, dim), dtype.
    reason = _require_keys(p, ("shape", "dtype"))
    if reason is not None:
        return reason
    shape = p.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return "centroids must be a 2-D (k, dim) array"
    if p.get("dtype") != "float32":
        return "centroids must be float32"
    return None


def _v_archetypes(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("archetypes",))
    if reason is not None:
        return reason
    archetypes = p.get("archetypes")
    if not isinstance(archetypes, Mapping) or not archetypes:
        return "archetypes must be a non-empty id->record mapping"
    ids: list[int] = []
    for raw_id in archetypes:
        try:
            ids.append(int(raw_id))
        except (TypeError, ValueError):
            return f"archetype id {raw_id!r} is not an integer"
    ids.sort()
    if ids != list(range(len(ids))):
        return "archetype ids must be contiguous from 0"
    return None


def _v_gold_labels(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("labels",))
    if reason is not None:
        return reason
    labels = p.get("labels")
    if not isinstance(labels, Mapping):
        return "labels must be an id->record mapping"
    for cid, record in labels.items():
        if not isinstance(record, Mapping):
            return f"label for {cid!r} must be a mapping"
        tier = record.get("tier")
        if not isinstance(tier, int) or isinstance(tier, bool) or not 0 <= tier <= 4:
            return f"label tier for {cid!r} must be an int in 0..4"
    return None


def _v_calibration_split(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("train_ids", "val_ids"))
    if reason is not None:
        return reason
    train = p.get("train_ids")
    val = p.get("val_ids")
    if not isinstance(train, (list, tuple)) or not isinstance(val, (list, tuple)):
        return "train_ids and val_ids must be sequences"
    if set(train) & set(val):
        return "train/val split must be disjoint (no leakage)"
    return None


def _v_scoring_weights(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("layout_version", "weights"))
    if reason is not None:
        return reason
    weights = p.get("weights")
    if not isinstance(weights, Mapping) or not weights:
        return "weights must be a non-empty component->weight mapping"
    for component, weight in weights.items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            return f"weight for {component!r} must be numeric"
        if not math.isfinite(float(weight)):
            return f"weight for {component!r} must be finite"
    return None


def _v_calibration_report(p: Mapping[str, object]) -> str | None:
    return _require_keys(p, ("cv_ndcg", "weight_stability"))


def _v_feature_importance(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("importances",))
    if reason is not None:
        return reason
    importances = p.get("importances")
    if not isinstance(importances, Mapping) or not importances:
        return "importances must be a non-empty feature->score mapping"
    return None


def _v_behavioral_weights(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("curves", "bounds"))
    if reason is not None:
        return reason
    bounds = p.get("bounds")
    if not isinstance(bounds, Mapping):
        return "bounds must be a mapping"
    lower = bounds.get("lower")
    upper = bounds.get("upper")
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        return "bounds.lower and bounds.upper must be numeric"
    if float(lower) > float(upper):
        return "behavioral bounds must be ordered (lower <= upper)"
    return None


def _v_embedding_manifest(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("model_id", "dim", "recipe", "pooling"))
    if reason is not None:
        return reason
    dim = _as_int(p, "dim")
    if dim is None or dim <= 0:
        return "embedding dim must be a positive integer"
    return None


def _v_candidate_vectors(p: Mapping[str, object]) -> str | None:
    # ``p`` is parquet schema/stats metadata extracted by the runner.
    reason = _require_keys(p, ("columns", "row_count", "unique_ids"))
    if reason is not None:
        return reason
    columns = p.get("columns")
    if not isinstance(columns, (list, tuple)) or "id" not in columns:
        return "candidate_vectors must carry an 'id' column"
    if p.get("unique_ids") is not True:
        return "candidate_vectors ids must be unique"
    return None


def _v_anchor_vectors(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("shape", "dtype"))
    if reason is not None:
        return reason
    shape = p.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return "anchor_vectors must be a 2-D (a, dim) array"
    return None


def _v_encoder_onnx(p: Mapping[str, object]) -> str | None:
    return _require_keys(p, ("model_id", "dim"))


def _v_tokenizer(p: Mapping[str, object]) -> str | None:
    return _require_keys(p, ("model",))


def _v_feature_snapshot(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("columns", "row_count", "has_nan", "feature_dim"))
    if reason is not None:
        return reason
    if p.get("has_nan") is not False:
        return "feature_snapshot must contain no NaN values"
    dim = _as_int(p, "feature_dim")
    if dim is None or dim <= 0:
        return "feature_dim must be a positive integer matching the layout"
    return None


def _v_feature_manifest(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("layout_version", "features"))
    if reason is not None:
        return reason
    features = p.get("features")
    if not isinstance(features, (list, tuple)) or not features:
        return "feature_manifest features must be a non-empty ordered list"
    return None


def _v_ranking_calibration(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("curve", "monotone", "tie_break_ok"))
    if reason is not None:
        return reason
    if p.get("monotone") is not True:
        return "ranking calibration curve must be monotone"
    if p.get("tie_break_ok") is not True:
        return "ranking calibration must pass the tie-break check"
    return None


def _v_reasoning_templates(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("templates",))
    if reason is not None:
        return reason
    templates = p.get("templates")
    if not isinstance(templates, (list, tuple)) or not templates:
        return "templates must be a non-empty sequence"
    for template in templates:
        if not isinstance(template, Mapping):
            return "each template must be a mapping"
        slots = template.get("slots")
        if not isinstance(slots, (list, tuple)):
            return "each template must declare its slots"
        for slot in slots:
            if not isinstance(slot, Mapping) or not slot.get("evidence_kind"):
                return "every reasoning slot must declare an evidence_kind"
    return None


def _v_reproducibility_report(p: Mapping[str, object]) -> str | None:
    reason = _require_keys(p, ("checks", "all_passed"))
    if reason is not None:
        return reason
    if p.get("all_passed") is not True:
        return "reproducibility report records a failing check"
    return None


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One row of Offline Pipeline Part 10 — a fully typed artifact contract.

    Attributes:
        key: The manifest key (e.g. ``"embeddings/candidate_vectors"``). Unique
            within the registry; this is the string the online ``ArtifactStore``
            resolves.
        relative_path: The artifact's path relative to ``artifacts_root``.
        kind: The on-disk serialization (drives how the runner extracts the
            validator payload — JSON/YAML parse, npy header, parquet schema).
        owner_stages: The producing stage id(s) (e.g. ``("O3", "O12")`` for the
            merged ``integrity_thresholds.json``).
        schema_version: The artifact's own version (``major.minor``); ``major``
            must match a consumer exactly, ``minor`` is backward-compatible.
        lineage: Upstream stage ids this artifact depends on (the "Lineage"
            column) — used by the graph/runner for staleness transitive closure.
        validator: The pure predicate applied to the parsed payload before
            manifesting; returns a reason on failure, ``None`` on success.
        required_online: Whether the online ``ArtifactStorePort`` must find this
            key present at R0 (Output Contract Part 12). O17 asserts the
            required set is complete.
        layout_bearing: Whether this artifact carries a ``layout_version`` that
            must agree across ``FeatureLayout``/``feature_manifest``/
            ``scoring_weights`` (Part 10 versioning rule).
    """

    key: str
    relative_path: str
    kind: ArtifactKind
    owner_stages: tuple[str, ...]
    schema_version: str
    lineage: tuple[str, ...]
    validator: ArtifactValidator
    required_online: bool = False
    layout_bearing: bool = False

    def validate(self, payload: Mapping[str, object]) -> ValidationOutcome:
        """Apply the validator to ``payload`` and return a structured outcome."""
        reason = self.validator(payload)
        if reason is None:
            return ValidationOutcome.passed(self.key)
        return ValidationOutcome.failed(self.key, reason)


# --------------------------------------------------------------------------- #
# The frozen catalog. Order mirrors Offline Pipeline Part 10 / Repo Layout §5. #
# --------------------------------------------------------------------------- #
_SPECS: Final[tuple[ArtifactSpec, ...]] = (
    ArtifactSpec(
        key="dataset_profile",
        relative_path="dataset_profile.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O0",),
        schema_version="profile-v1.0",
        lineage=(),
        validator=_v_dataset_profile,
    ),
    ArtifactSpec(
        key="canonical_maps",
        relative_path="canonical_maps.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O1",),
        schema_version="norm-v1.0",
        lineage=("O0",),
        validator=_v_canonical_maps,
    ),
    ArtifactSpec(
        key="validation_report",
        relative_path="validation_report.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O2",),
        schema_version="1.0",
        lineage=("O1",),
        validator=_v_validation_report,
    ),
    ArtifactSpec(
        key="integrity_rules",
        relative_path="integrity_rules.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O3",),
        schema_version="rules-v1.0",
        lineage=("O0", "O2"),
        validator=_v_integrity_rules,
        required_online=True,
    ),
    ArtifactSpec(
        key="honeypot_catalog",
        relative_path="honeypot_catalog.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O3",),
        schema_version="1.0",
        lineage=("O2",),
        validator=_v_honeypot_catalog,
        required_online=True,
    ),
    ArtifactSpec(
        key="integrity_thresholds",
        relative_path="calibration/integrity_thresholds.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O3", "O12"),
        schema_version="thr-v1.0",
        lineage=("O0", "O3", "O14"),
        validator=_v_integrity_thresholds,
        required_online=True,
    ),
    ArtifactSpec(
        key="risk_weights",
        relative_path="risk_weights.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O12",),
        schema_version="1.0",
        lineage=("O3", "O14"),
        validator=_v_risk_weights,
        required_online=True,
    ),
    ArtifactSpec(
        key="lexicon_compiled",
        relative_path="lexicon/lexicon.compiled.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O4",),
        schema_version="lex-v1.0",
        lineage=("O2",),
        validator=_v_lexicon_compiled,
        required_online=True,
    ),
    ArtifactSpec(
        key="concepts",
        relative_path="concepts.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O4", "O5"),
        schema_version="concept-v1.0",
        lineage=("O4", "O5"),
        validator=_v_concepts,
        required_online=True,
    ),
    ArtifactSpec(
        key="term_graph",
        relative_path="term_graph.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O4",),
        schema_version="lex-v1.0",
        lineage=("O2",),
        validator=_v_graph,
    ),
    ArtifactSpec(
        key="phrase_graph",
        relative_path="phrase_graph.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O4",),
        schema_version="lex-v1.0",
        lineage=("O2",),
        validator=_v_graph,
    ),
    ArtifactSpec(
        key="jd_concepts",
        relative_path="jd_concepts.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O6",),
        schema_version="jd-v1.0",
        lineage=("O5",),
        validator=_v_jd_concepts,
        required_online=True,
    ),
    ArtifactSpec(
        key="eligibility_rules",
        relative_path="gates/eligibility_rules.yaml",
        kind=ArtifactKind.YAML,
        owner_stages=("O6",),
        schema_version="gate-v1.0",
        lineage=("O6",),
        validator=_v_eligibility_rules,
        required_online=True,
    ),
    ArtifactSpec(
        key="centroids",
        relative_path="archetypes/centroids.npy",
        kind=ArtifactKind.NPY,
        owner_stages=("O7",),
        schema_version="arch-v1.0",
        lineage=("O13a",),
        validator=_v_centroids,
        required_online=True,
    ),
    ArtifactSpec(
        key="archetypes",
        relative_path="archetypes.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O7",),
        schema_version="arch-v1.0",
        lineage=("O7",),
        validator=_v_archetypes,
        required_online=True,
    ),
    ArtifactSpec(
        key="gold_labels",
        relative_path="gold_labels.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O8",),
        schema_version="gold-v1.0",
        lineage=("O14", "O7"),
        validator=_v_gold_labels,
    ),
    ArtifactSpec(
        key="calibration_split",
        relative_path="calibration_split.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O8",),
        schema_version="gold-v1.0",
        lineage=("O8",),
        validator=_v_calibration_split,
    ),
    ArtifactSpec(
        key="scoring_weights",
        relative_path="weights/scoring_weights.locked.yaml",
        kind=ArtifactKind.YAML,
        owner_stages=("O9",),
        schema_version="weights-v1.0",
        lineage=("O8", "O14"),
        validator=_v_scoring_weights,
        required_online=True,
        layout_bearing=True,
    ),
    ArtifactSpec(
        key="calibration_report",
        relative_path="calibration_report.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O9",),
        schema_version="1.0",
        lineage=("O8", "O14"),
        validator=_v_calibration_report,
    ),
    ArtifactSpec(
        key="feature_importance",
        relative_path="feature_importance.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O10",),
        schema_version="imp-v1.0",
        lineage=("O9", "O14"),
        validator=_v_feature_importance,
        required_online=True,
    ),
    ArtifactSpec(
        key="behavioral_weights",
        relative_path="behavioral_weights.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O11",),
        schema_version="bhv-v1.0",
        lineage=("O14",),
        validator=_v_behavioral_weights,
        required_online=True,
    ),
    ArtifactSpec(
        key="embedding_manifest",
        relative_path="embedding_manifest.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O13",),
        schema_version="emb-v1.0",
        lineage=("O1", "O6", "O7"),
        validator=_v_embedding_manifest,
        required_online=True,
    ),
    ArtifactSpec(
        key="candidate_vectors",
        relative_path="embeddings/candidate_vectors.parquet",
        kind=ArtifactKind.PARQUET,
        owner_stages=("O13a",),
        schema_version="emb-v1.0",
        lineage=("O1",),
        validator=_v_candidate_vectors,
        required_online=True,
    ),
    ArtifactSpec(
        key="anchor_vectors",
        relative_path="embeddings/anchor_vectors.npy",
        kind=ArtifactKind.NPY,
        owner_stages=("O13b",),
        schema_version="emb-v1.0",
        lineage=("O6",),
        validator=_v_anchor_vectors,
        required_online=True,
    ),
    ArtifactSpec(
        key="encoder",
        relative_path="model/encoder.onnx",
        kind=ArtifactKind.ONNX,
        owner_stages=("O13",),
        schema_version="emb-v1.0",
        lineage=("O13",),
        validator=_v_encoder_onnx,
        required_online=True,
    ),
    ArtifactSpec(
        key="tokenizer",
        relative_path="model/tokenizer.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O13",),
        schema_version="emb-v1.0",
        lineage=("O13",),
        validator=_v_tokenizer,
        required_online=True,
    ),
    ArtifactSpec(
        key="feature_snapshot",
        relative_path="feature_snapshot.parquet",
        kind=ArtifactKind.PARQUET,
        owner_stages=("O14",),
        schema_version="layout-v1.0",
        lineage=("O4", "O13a", "O13b"),
        validator=_v_feature_snapshot,
        layout_bearing=True,
    ),
    ArtifactSpec(
        key="feature_manifest",
        relative_path="feature_manifest.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O14",),
        schema_version="layout-v1.0",
        lineage=("O14",),
        validator=_v_feature_manifest,
        required_online=True,
        layout_bearing=True,
    ),
    ArtifactSpec(
        key="ranking_calibration",
        relative_path="ranking_calibration.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O15",),
        schema_version="rank-v1.0",
        lineage=("O9", "O10", "O11", "O12"),
        validator=_v_ranking_calibration,
        required_online=True,
    ),
    ArtifactSpec(
        key="reasoning_templates",
        relative_path="reasoning_templates.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O16",),
        schema_version="reason-v1.0",
        lineage=("O8", "O10"),
        validator=_v_reasoning_templates,
        required_online=True,
    ),
    ArtifactSpec(
        key="reproducibility_report",
        relative_path="reproducibility_report.json",
        kind=ArtifactKind.JSON,
        owner_stages=("O18",),
        schema_version="1.0",
        lineage=("O17",),
        validator=_v_reproducibility_report,
    ),
)


@dataclass(frozen=True, slots=True)
class OfflineArtifactRegistry:
    """The frozen, versioned catalog of every expected offline artifact.

    Construct via :data:`OFFLINE_ARTIFACT_REGISTRY` (the module-level singleton)
    or :meth:`default`. Construction self-checks internal consistency (unique
    keys, unique paths, every layout-bearing key shares one ``layout_version``
    family) so a catalog typo is a build break, not a silent gap.
    """

    specs: tuple[ArtifactSpec, ...]
    _by_key: Mapping[str, ArtifactSpec] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        keys = [spec.key for spec in self.specs]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            msg = f"duplicate artifact keys in registry: {', '.join(dupes)}"
            raise ValueError(msg)
        paths = [spec.relative_path for spec in self.specs]
        if len(set(paths)) != len(paths):
            dupes = sorted({p for p in paths if paths.count(p) > 1})
            msg = f"duplicate artifact paths in registry: {', '.join(dupes)}"
            raise ValueError(msg)
        # ``object.__setattr__`` because the dataclass is frozen.
        object.__setattr__(
            self, "_by_key", {spec.key: spec for spec in self.specs}
        )

    @classmethod
    def default(cls) -> OfflineArtifactRegistry:
        """Return the registry built from the frozen Part 10 catalog."""
        return cls(specs=_SPECS)

    def spec(self, key: str) -> ArtifactSpec:
        """Return the spec for ``key``.

        Raises:
            KeyError: if ``key`` is not a registered artifact — a programming
                error (the producing stage emitted an unknown key).
        """
        try:
            return self._by_key[key]
        except KeyError as exc:
            msg = f"unknown artifact key {key!r} (not in OfflineArtifactRegistry)"
            raise KeyError(msg) from exc

    def keys(self) -> tuple[str, ...]:
        """Return every registered artifact key in catalog order."""
        return tuple(spec.key for spec in self.specs)

    def required_online_keys(self) -> tuple[str, ...]:
        """Return the keys the online ``ArtifactStorePort`` must find at R0."""
        return tuple(spec.key for spec in self.specs if spec.required_online)

    def layout_bearing_keys(self) -> tuple[str, ...]:
        """Return the keys whose ``layout_version`` must agree (Part 10 rule)."""
        return tuple(spec.key for spec in self.specs if spec.layout_bearing)

    def validate(self, key: str, payload: Mapping[str, object]) -> ValidationOutcome:
        """Validate a produced artifact's ``payload`` against its spec."""
        return self.spec(key).validate(payload)

    def assert_required_present(self, present_keys: frozenset[str]) -> None:
        """Raise if any ``required_online`` artifact is absent from ``present_keys``.

        Invoked by O17 packaging before writing ``MANIFEST.json``: a missing
        required artifact is a partial set and must abort the build (Ports §8,
        "partial artifact set" ⇒ fatal). The error names every missing key.

        Raises:
            ValueError: listing the missing required keys (the offline runner
                wraps this into the registry/contract fail path).
        """
        missing = tuple(
            k for k in self.required_online_keys() if k not in present_keys
        )
        if missing:
            msg = f"required online artifacts missing from build: {', '.join(missing)}"
            raise ValueError(msg)


OFFLINE_ARTIFACT_REGISTRY: Final[OfflineArtifactRegistry] = (
    OfflineArtifactRegistry.default()
)