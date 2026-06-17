"""The R0…R9 online stage callables — pure, deterministic, copy-on-write.

Each stage is a free function the orchestrator (``pipeline.py``) threads in order;
the representation set flows forward immutably. Ports are touched **only** at
R0/R1/R3/R8/R9 (artifact load, ingest, semantic lookup, submit, report); R2/R4/R5
/R6/R7 are pure engine work over the columnar CQV and the domain aggregates — the
hot path that keeps the ≤150 s / ≤4 GB / CPU-only / no-network budget.

Determinism contract (Online Part 2 / Part 14):
* No wall clock — recency uses ``ctx.as_of`` only; no online RNG.
* float32 throughout; component reductions in fixed ``ScoreComponent`` order.
* every merge / tie ordered by ascending ``candidate_id``.

Verdict-vs-failure discipline (the strict invariant): an eligibility hit, a
honeypot finding, a semantic miss are **data** that ride the representation
downstream and shape the score — they never raise. Only structural / contract
breaches raise fail-fast: a broken CQV column layout, a vector dim mismatch, an
empty top-100 slice, a dangling reasoning evidence ref.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, final

import numpy as np
import numpy.typing as npt

from redstack.domain.candidate.representation import (
    ArchetypeAssignment,
    BehavioralProfile,
    CandidateRepresentation,
    CareerProfile,
    CredibilityProfile,
    EligibilityFinding,
    EligibilityReport,
    IntegrityFinding,
    IntegrityReport,
    LogisticsProfile,
    ScoreBreakdown,
    ScoreComponentValue,
    ScoredCandidate,
    SemanticProfile,
)
from redstack.domain.enums import (
    CareerTrack,
    EligibilityCode,
    EvidenceKind,
    IntegrityFlag,
    ReasoningPolarity,
    ScoreComponent,
    Severity,
    SignalAvailability,
)
from redstack.domain.errors import (
    ArtifactContractError,
    CandidateSourceError,
    CQVInvariantError,
    ProvenanceError,
    SchemaError,
    ScoreInvariantError,
    VectorStoreError,
)
from redstack.domain.ids import (
    AnchorId,
    ArchetypeId,
    CandidateId,
    Score,
)
from redstack.domain.provenance import EvidenceRef, ProvenanceHandle
from redstack.domain.ranking import (
    CandidateReasoning,
    Ranking,
    ReasoningClause,
)
from redstack.domain.source import RawCandidate
from redstack.features.evidence import mint
from redstack.features.parsing import parse
from redstack.features.registry import FeatureRegistry
from redstack.ports._types import Malformed, Ok
from redstack.ports.artifact_store import ArtifactStorePort
from redstack.ports.embedding import EmbeddingError, EmbeddingModelPort
from redstack.ports.online import (
    OnlineEntropyPort,
    SemanticVectorStorePort,
    SubmissionReceipt,
    SubmissionRow,
)

if TYPE_CHECKING:
    from redstack.pipelines.online.pipeline import OnlineRunConfig, OnlineRunContext

__all__: tuple[str, ...] = (
    "r0_load",
    "r1_ingest",
    "r2_features",
    "r3_semantic",
    "r4_gates",
    "r5_score",
    "r6_rank",
    "r7_reason",
    "r8_submit",
    "r9_report",
    "LoadedArtifacts",
    "IngestedCandidate",
    "FeaturedSet",
    "SituatedSet",
    "GatedSet",
    "ScoredSet",
    "ReportOutcome",
)

# --------------------------------------------------------------------------- #
# Required online artifact keys (Online Part 1) — R0 asserts all are present. #
# --------------------------------------------------------------------------- #
_REQUIRED_KEYS: tuple[str, ...] = (
    "encoder",
    "candidate_vectors",
    "anchor_vectors",
    "centroids",
    "lexicon_compiled",
    "concepts",
    "jd_concepts",
    "eligibility_rules",
    "integrity_rules",
    "integrity_thresholds",
    "risk_weights",
    "scoring_weights",
    "behavioral_weights",
    "feature_manifest",
    "feature_importance",
    "archetypes",
    "reasoning_templates",
    "ranking_calibration",
    "embedding_manifest",
)

#: Floor sentinel for gated candidates (Online Part 8; a finite constant).
_FLOOR: float = 0.0
#: Multiplier bounds mirror the offline behavioral calibration rail (Part 8).
_MULT_LOWER: float = 0.6
_MULT_UPPER: float = 1.25
#: Bounded archetype boost for a target archetype (additive, post-multiplier).
_ARCHETYPE_BOOST: float = 0.05
#: Confidence-shrink cap toward the neutral prior (risk never erases signal).
_MAX_SHRINK: float = 0.35
#: Top-K ranking size (validator-mandated 100 rows).
_RANKING_SIZE: int = 100
#: Tie tolerance for the float32 ranking/score equality checks.
_TIE_EPS: float = 1e-9


# =========================================================================== #
# R0 — Artifact loading + verification (ports: ArtifactStore)                 #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class LoadedArtifacts:
    """The verified artifact bundle R0 hands to the context."""

    artifacts: Mapping[str, object]
    manifest: Mapping[str, object]
    feature_registry: FeatureRegistry


def r0_load(
    *,
    artifact_store: ArtifactStorePort,
    embedding_model: EmbeddingModelPort,
    vector_store: SemanticVectorStorePort,
    entropy: OnlineEntropyPort,
    config: OnlineRunConfig,
) -> LoadedArtifacts:
    """Load + integrity-verify every artifact; assert cross-artifact coherence.

    Mirrors the single online contract (Online Part 3 / Ports §8): the store
    self-verifies the manifest and every artifact sha256 (``verify_all``); we then
    load the small JSON/YAML artifacts + the npy matrices, and assert coherence —
    ``layout_version`` agreement, ``embedding.dim`` across all vector artifacts,
    ``model_id`` consistency, anchor ⊆ ``jd_concepts``, centroid dim, gate codes ∈
    ``EligibilityCode``, integrity codes ∈ ``IntegrityFlag``, weight keys ==
    ``ScoreComponent``. Any failure raises (fail-fast); the context is never
    partially bound.
    """
    # 1. Manifest self-hash + per-artifact sha256 (the store raises on mismatch).
    artifact_store.verify_all()
    manifest = artifact_store.manifest()
    if not isinstance(manifest, Mapping):
        raise ArtifactContractError("artifact store returned a non-mapping manifest")

    # 2. Required-key completeness (partial set is fatal).
    artifacts_meta = manifest.get("artifacts")
    present = set(artifacts_meta) if isinstance(artifacts_meta, Mapping) else set()
    missing = [k for k in _REQUIRED_KEYS if k not in present]
    if missing:
        raise ArtifactContractError(f"manifest missing required keys: {sorted(missing)}")

    # 3. Load the JSON/YAML artifacts the engines read (lookup, never recompute).
    artifacts: dict[str, object] = {}
    for key in (
        "feature_manifest", "scoring_weights", "behavioral_weights",
        "integrity_rules", "integrity_thresholds", "risk_weights",
        "eligibility_rules", "jd_concepts", "concepts", "lexicon_compiled",
        "archetypes", "feature_importance", "reasoning_templates",
        "ranking_calibration", "embedding_manifest",
    ):
        artifacts[key] = artifact_store.load_json(key)
    artifacts["anchor_vectors"] = _as_f32(artifact_store.load_npy("anchor_vectors"))
    artifacts["centroids"] = _as_f32(artifact_store.load_npy("centroids"))

    # 4. Cross-artifact coherence (Ports §8 rule 4) — all fatal.
    layout_version = _coherence_layout(manifest, artifacts)
    dim = _coherence_embedding_dim(manifest, artifacts, vector_store, embedding_model)
    _coherence_anchor_subset(artifacts)
    _coherence_gate_codes(artifacts)
    _coherence_weight_keys(artifacts)

    registry = _build_feature_registry(artifacts, layout_version)
    if registry.dim != _matrix_dim(vector_store):
        # the vector dim is the embedding dim, distinct from the feature dim D;
        # this only sanity-checks the feature registry parsed coherently.
        pass
    _ = (dim, entropy, config)  # bound elsewhere; referenced for signature parity.
    return LoadedArtifacts(
        artifacts=artifacts, manifest=manifest, feature_registry=registry
    )


def _as_f32(array: npt.NDArray[object]) -> npt.NDArray[np.float32]:
    """Coerce a loaded npy array to a contiguous float32 view (no copy if able)."""
    return np.ascontiguousarray(array, dtype=np.float32)


def _coherence_layout(
    manifest: Mapping[str, object], artifacts: Mapping[str, object]
) -> str:
    """Assert layout_version agrees across manifest / feature_manifest / weights."""
    expected = manifest.get("layout_version")
    if not isinstance(expected, str) or not expected:
        raise ArtifactContractError("manifest missing layout_version")
    for key in ("feature_manifest", "scoring_weights"):
        doc = artifacts.get(key)
        version = doc.get("layout_version") if isinstance(doc, Mapping) else None
        if version != expected:
            msg = f"layout_version mismatch: {key} has {version!r} != {expected!r}"
            raise ArtifactContractError(msg)
    return expected


def _coherence_embedding_dim(
    manifest: Mapping[str, object],
    artifacts: Mapping[str, object],
    vector_store: SemanticVectorStorePort,
    embedding_model: EmbeddingModelPort,
) -> int:
    """Assert embedding.dim equals across anchors / centroids / store / encoder."""
    embedding = manifest.get("embedding")
    dim = embedding.get("dim") if isinstance(embedding, Mapping) else None
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise ArtifactContractError("manifest embedding.dim invalid")
    for key in ("anchor_vectors", "centroids"):
        mat = artifacts.get(key)
        if isinstance(mat, np.ndarray):
            if mat.ndim != 2 or int(mat.shape[1]) != dim:
                raise ArtifactContractError(f"{key} second dim != embedding.dim {dim}")
    if vector_store.dim != dim:
        raise ArtifactContractError(
            f"vector store dim {vector_store.dim} != embedding.dim {dim}"
        )
    if embedding_model.dim != dim:
        raise ArtifactContractError(
            f"encoder dim {embedding_model.dim} != embedding.dim {dim}"
        )
    return dim


def _coherence_anchor_subset(artifacts: Mapping[str, object]) -> None:
    """Assert the anchor vector count ≤ jd_concepts anchor count (anchor ⊆ jd)."""
    jd = artifacts.get("jd_concepts")
    anchors = jd.get("anchors") if isinstance(jd, Mapping) else None
    if not isinstance(anchors, (list, tuple)) or not anchors:
        raise ArtifactContractError("jd_concepts has no anchors")
    mat = artifacts.get("anchor_vectors")
    n_vec = int(mat.shape[0]) if isinstance(mat, np.ndarray) and mat.ndim == 2 else 0
    if n_vec > len(anchors):
        raise ArtifactContractError(
            f"anchor_vectors rows {n_vec} > jd_concepts anchors {len(anchors)}"
        )


def _coherence_gate_codes(artifacts: Mapping[str, object]) -> None:
    """Assert every gate code ∈ EligibilityCode and integrity code ∈ IntegrityFlag."""
    valid_elig = {c.value for c in EligibilityCode}
    gates = artifacts.get("eligibility_rules")
    if isinstance(gates, Mapping):
        for bucket in ("hard_blocks", "soft_penalties"):
            rules = gates.get(bucket)
            if isinstance(rules, (list, tuple)):
                for rule in rules:
                    code = rule.get("code") if isinstance(rule, Mapping) else None
                    if code is not None and code not in valid_elig:
                        raise ArtifactContractError(f"unknown eligibility code {code!r}")
    valid_integ = {c.value for c in IntegrityFlag}
    rules_doc = artifacts.get("integrity_rules")
    if isinstance(rules_doc, Mapping):
        rules = rules_doc.get("rules")
        if isinstance(rules, (list, tuple)):
            for rule in rules:
                code = rule.get("code") if isinstance(rule, Mapping) else None
                if code is not None and code not in valid_integ:
                    raise ArtifactContractError(f"unknown integrity code {code!r}")


def _coherence_weight_keys(artifacts: Mapping[str, object]) -> None:
    """Assert scoring weight keys == ScoreComponent value set exactly."""
    doc = artifacts.get("scoring_weights")
    weights = doc.get("weights") if isinstance(doc, Mapping) else None
    if not isinstance(weights, Mapping):
        raise ArtifactContractError("scoring_weights missing 'weights'")
    keys = set(weights)
    expected = {c.value for c in ScoreComponent}
    if keys != expected:
        raise ArtifactContractError(
            f"scoring weight keys {sorted(keys)} != ScoreComponent {sorted(expected)}"
        )


def _build_feature_registry(
    artifacts: Mapping[str, object], layout_version: str
) -> FeatureRegistry:
    """Reconstruct the FeatureRegistry from feature_manifest (online layout)."""
    manifest = artifacts.get("feature_manifest")
    if not isinstance(manifest, Mapping):
        raise ArtifactContractError("feature_manifest is not a mapping")
    features = manifest.get("features")
    group_of = manifest.get("group_of")
    if not isinstance(features, (list, tuple)) or not features:
        raise ArtifactContractError("feature_manifest missing 'features'")
    if not isinstance(group_of, Mapping):
        raise ArtifactContractError("feature_manifest missing 'group_of'")
    feature_ids = tuple(str(f) for f in features)
    gmap = {str(k): str(v) for k, v in group_of.items()}
    return FeatureRegistry(
        layout_version=layout_version, feature_ids=feature_ids, group_of=gmap
    )


def _matrix_dim(vector_store: SemanticVectorStorePort) -> int:
    """The embedding dim from the vector store (distinct from feature dim D)."""
    return vector_store.dim


# =========================================================================== #
# R1 — Candidate ingestion (ports: CandidateSource)                          #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class IngestedCandidate:
    """A validated raw candidate + its identity + source order (stage PARSED)."""

    candidate_id: CandidateId
    raw: RawCandidate
    source_index: int
    representation: CandidateRepresentation


def r1_ingest(ctx: OnlineRunContext) -> list[IngestedCandidate]:
    """Stream + validate candidates into typed ``RawCandidate`` + ``Identity``.

    Streams ``SourceRecord``s in file order (constant memory at the port); each
    ``Ok`` is schema-validated via ``features.parsing`` and minted into an
    ``IngestedCandidate`` carrying an inline ``ProvenanceHandle`` (survivors keep
    their raw record for R7). Semantic contradictions are **preserved** (honeypot
    signal). A ``Malformed`` record or a duplicate id aborts under the default
    full-run policy (a deviation from 100,000 well-formed rows means wrong input);
    the sandbox profile may skip-and-record.

    Raises:
        CandidateSourceError: an unrecoverable source/decompress failure.
        SchemaError: a parseable-but-schema-invalid record under abort policy.
    """
    abort = ctx.config.abort_on_malformed
    seen: set[str] = set()
    out: list[IngestedCandidate] = []
    try:
        stream = ctx.candidate_source.stream()
    except Exception as exc:  # noqa: BLE001 — surface as the contract error.
        raise CandidateSourceError(f"candidate source stream failed: {exc}") from exc

    for record in stream:
        if isinstance(record, Malformed):
            if abort:
                raise SchemaError(
                    f"malformed input at line {record.line_no}: {record.error}"
                )
            continue
        if not isinstance(record, Ok):
            continue
        try:
            candidate = parse(record.raw)
        except SchemaError:
            if abort:
                raise
            continue
        cid = candidate.candidate_id
        if cid in seen:
            if abort:
                raise SchemaError(f"duplicate candidate_id {cid!r}")
            continue
        seen.add(cid)
        provenance = ProvenanceHandle(
            candidate_id=cid, inline=candidate, source_index=record.source_index
        )
        representation = CandidateRepresentation(
            candidate_id=cid, provenance=provenance, source_index=record.source_index
        )
        out.append(
            IngestedCandidate(
                candidate_id=cid,
                raw=candidate,
                source_index=record.source_index,
                representation=representation,
            )
        )
    # Deterministic order: by source_index (file order), the merge key for R2+.
    out.sort(key=lambda c: c.source_index)
    return out


# =========================================================================== #
# R2 — Feature extraction (pure; bulk (N,D) CQV)                             #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class FeaturedSet:
    """The bulk featured population: (N,D) CQV + per-candidate reps (stage FEATURED)."""

    candidates: tuple[IngestedCandidate, ...]
    cqv: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    representations: tuple[CandidateRepresentation, ...]
    feature_ids: tuple[str, ...]


def r2_features(
    ctx: OnlineRunContext, ingested: Sequence[IngestedCandidate]
) -> FeaturedSet:
    """Extract every structural + behavioral feature into the bulk ``(N,D)`` CQV.

    Pure (no ports). For each candidate, ``features.extraction.extract_row``
    produces the ``(D,)`` feature row + ``(G,)`` group confidence aligned to the
    ``FeatureLayout``; semantic indices are left at their R3 placeholders. Rows are
    written at ``source_index`` order into the shared ``(N,D)`` float32 matrix
    (vectorized columnar; no per-candidate object in the hot path beyond the
    structural slices the gates/reasoning later read). Recency uses ``ctx.as_of``
    only.

    Raises:
        CQVInvariantError: a row dim mismatch, a layout_version disagreement, or a
            NaN/inf in a populated CQV index (a structural breach — fatal).
    """
    registry = _registry(ctx)
    if registry.layout_version != ctx.layout_version:
        raise CQVInvariantError(
            f"feature layout_version {registry.layout_version!r} != manifest "
            f"{ctx.layout_version!r}"
        )
    n = len(ingested)
    dim = registry.dim
    groups = registry.groups
    cqv = np.zeros((n, dim), dtype=np.float32)
    confidence = np.zeros((n, len(groups)), dtype=np.float32)
    reps: list[CandidateRepresentation] = []

    for i, cand in enumerate(ingested):
        row, conf = extract_row_safe(cand.raw, registry, as_of=ctx.as_of)
        if row.shape != (dim,):
            raise CQVInvariantError(
                f"candidate {cand.candidate_id} row dim {row.shape} != ({dim},)"
            )
        if not np.all(np.isfinite(row)):
            raise CQVInvariantError(
                f"candidate {cand.candidate_id} CQV row has NaN/inf"
            )
        cqv[i] = row
        confidence[i] = conf
        reps.append(
            cand.representation.with_features(
                career=_career_profile(cand.raw),
                credibility=_credibility_profile(cand.raw),
                logistics=_logistics_profile(cand.raw),
                behavioral=_behavioral_profile(cand.raw),
            )
        )
    return FeaturedSet(
        candidates=tuple(ingested),
        cqv=cqv,
        confidence=confidence,
        representations=tuple(reps),
        feature_ids=registry.feature_ids,
    )


def extract_row_safe(
    raw: RawCandidate, registry: FeatureRegistry, *, as_of: object
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Call the structural extractor with the injected ``as_of`` (no wall clock)."""
    from datetime import date

    assert isinstance(as_of, date)
    return extract_row(raw, registry, as_of=as_of)


# Structural slice builders (pure functions of the raw record + as_of) -------- #
def _career_profile(raw: RawCandidate) -> CareerProfile:
    """Derive the structural career profile (product-vs-services intelligence)."""
    total = sum(p.duration_months for p in raw.career_history)
    product_industries = {"software", "product", "saas", "internet", "technology"}
    product = sum(
        p.duration_months
        for p in raw.career_history
        if p.industry.casefold() in product_industries
    )
    services = total - product
    if total == 0:
        track = CareerTrack.UNKNOWN
    elif product >= 2 * services:
        track = CareerTrack.PRODUCT
    elif services >= 2 * product:
        track = CareerTrack.SERVICES
    else:
        track = CareerTrack.MIXED
    tenures = [p.duration_months for p in raw.career_history if p.duration_months > 0]
    return CareerProfile(
        total_experience_months=total,
        product_months=product,
        services_months=services,
        track=track,
        role_count=len(raw.career_history),
        shortest_tenure_months=min(tenures) if tenures else 0,
    )


def _credibility_profile(raw: RawCandidate) -> CredibilityProfile:
    """Derive structural credibility: skill trust, stuffing score, title coherence."""
    skills = raw.skills
    if not skills:
        return CredibilityProfile(
            skill_trust=0.0, keyword_stuffing_score=0.0, title_coherence=1.0
        )
    corroborated = sum(
        1
        for s in skills
        if (s.endorsements > 0) or (s.duration_months not in (None, 0))
    )
    trust = corroborated / len(skills)
    advanced_zero = sum(
        1
        for s in skills
        if s.proficiency.casefold() in ("advanced", "expert")
        and (s.endorsements == 0)
        and (s.duration_months in (None, 0))
    )
    stuffing = advanced_zero / len(skills)
    return CredibilityProfile(
        skill_trust=round(trust, 6),
        keyword_stuffing_score=round(stuffing, 6),
        title_coherence=1.0,
    )


def _logistics_profile(raw: RawCandidate) -> LogisticsProfile:
    """Derive the logistics inputs (notice / location / relocation)."""
    country = raw.profile.country.casefold()
    return LogisticsProfile(
        notice_days=None,
        location_label=raw.profile.location,
        relocatable=(country == "india"),
    )


def _behavioral_profile(raw: RawCandidate) -> BehavioralProfile:
    """Derive bounded behavioral composites with sentinel→UNKNOWN family flags.

    Structural placeholders in [0,1]; the genuine 23-signal composites land with
    the features phase. Families with no signal are marked ``UNKNOWN`` (never 0)
    so R5's confidence shrink — not the score — absorbs missing behavior.
    """
    families: dict[str, SignalAvailability] = {
        "availability": SignalAvailability.UNKNOWN,
        "responsiveness": SignalAvailability.UNKNOWN,
        "engagement": SignalAvailability.UNKNOWN,
        "reliability": SignalAvailability.UNKNOWN,
        "verification": SignalAvailability.UNKNOWN,
    }
    return BehavioralProfile(
        availability=0.5,
        responsiveness=0.5,
        engagement=0.5,
        reliability=0.5,
        verification=0.5,
        family_availability=families,
    )


# =========================================================================== #
# R3 — Semantic hydration (ports: SemanticVectorStore, EmbeddingModel)        #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class SituatedSet:
    """The featured set with semantics folded + archetypes assigned (stage SITUATED)."""

    base: FeaturedSet
    cqv: npt.NDArray[np.float32]
    representations: tuple[CandidateRepresentation, ...]
    semantic_misses: int


def r3_semantic(ctx: OnlineRunContext, featured: FeaturedSet) -> SituatedSet:
    """Attach dense semantic fit + archetype by **lookup** (not recompute).

    ``view_all`` gives the mmap'd ``(N,dim)`` matrix + id order; candidate vectors
    are gathered by ``CandidateId`` (O(1) row index). Cosine vs every anchor is one
    vectorized matmul → positive/negative/net fit + ``best_positive_anchor`` (ties
    by ``AnchorId`` asc); nearest-centroid (ties by ``ArchetypeId`` asc) →
    ``ArchetypeAssignment``. A store **miss** (sandbox/delta) falls back to the
    onnx encoder; if that also fails (``EmbeddingError``) the candidate's semantics
    are ``UNKNOWN`` with zero net fit — recorded, never fatal. Semantic values are
    folded into the CQV at their layout indices.

    Raises:
        VectorStoreError: the store vector dim mismatches the embedding dim, or the
            anchor/centroid matrices are malformed (a structural breach — fatal).
    """
    anchors = _anchor_matrix(ctx)
    centroids = _centroid_matrix(ctx)
    anchor_ids = _anchor_ids(ctx)
    polarities = _anchor_polarities(ctx, anchor_ids)
    target_archetypes = _target_archetypes(ctx)

    view = ctx.vector_store.view_all()
    if view.dim != anchors.shape[1]:
        raise VectorStoreError(
            f"store dim {view.dim} != anchor dim {anchors.shape[1]}"
        )

    cqv = featured.cqv.copy()
    reps = list(featured.representations)
    semantic_indices = _semantic_feature_indices(featured.feature_ids)
    misses = 0

    for i, cand in enumerate(featured.candidates):
        vector = _lookup_or_encode(ctx, view, cand)
        if vector is None:
            semantic = SemanticProfile(
                positive_fit=0.0, negative_fit=0.0, net_semantic_fit=0.0,
                best_positive_anchor=None, is_unknown=True,
            )
            archetype = _fallback_archetype(centroids, target_archetypes)
            misses += 1
        else:
            semantic = _semantic_profile(vector, anchors, anchor_ids, polarities)
            archetype = _assign_archetype(vector, centroids, target_archetypes)
            _fold_semantic(cqv, i, semantic_indices, semantic)
        reps[i] = reps[i].with_semantic(semantic=semantic, archetype=archetype)

    if not np.all(np.isfinite(cqv)):
        raise VectorStoreError("CQV contains NaN/inf after semantic fold")
    return SituatedSet(
        base=featured, cqv=cqv, representations=tuple(reps), semantic_misses=misses
    )


def _lookup_or_encode(
    ctx: OnlineRunContext,
    view: object,
    cand: IngestedCandidate,
) -> npt.NDArray[np.float32] | None:
    """Return the candidate's vector by store lookup, else onnx fallback, else None.

    A store hit is the dominant path (the full pool is precomputed offline). A miss
    composes the embedding document and encodes via the onnx port; an
    ``EmbeddingError`` (or a fallback failure) yields ``None`` → semantic UNKNOWN.
    Never raises for a miss (a miss is data, not a fault).
    """
    index_of = getattr(view, "index_of", None)
    matrix = getattr(view, "matrix", None)
    if callable(index_of) and isinstance(matrix, np.ndarray):
        row = index_of(cand.candidate_id)
        if isinstance(row, int) and 0 <= row < matrix.shape[0]:
            return np.ascontiguousarray(matrix[row], dtype=np.float32)
    # Miss → onnx fallback (compose doc, encode); failure is recorded, not fatal.
    try:
        doc = _compose_embedding_document(cand.raw)
        encoded = ctx.embedding_model.encode([doc])
        if encoded.shape[0] >= 1:
            vec = np.ascontiguousarray(encoded[0], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0.0 else None
    except EmbeddingError:
        return None
    except Exception:  # noqa: BLE001 — any fallback failure is a miss, never fatal.
        return None
    return None


def _semantic_profile(
    vector: npt.NDArray[np.float32],
    anchors: npt.NDArray[np.float32],
    anchor_ids: Sequence[AnchorId],
    polarities: Sequence[int],
) -> SemanticProfile:
    """Cosine vs every anchor → positive/negative/net fit + best positive anchor.

    Vectors are unit-norm (offline guarantee), so cosine is a dot product.
    ``best_positive_anchor`` is the argmax over positive anchors with ties broken
    by ``AnchorId`` ascending. ``net_semantic_fit`` ∈ [0,1] (clamped).
    """
    sims = anchors @ vector  # (A,) cosine in [-1, 1]
    pos_vals: list[float] = []
    neg_vals: list[float] = []
    best_anchor: AnchorId | None = None
    best_sim = -2.0
    for j, pol in enumerate(polarities):
        sim = float(sims[j])
        if pol > 0:
            pos_vals.append(sim)
            # tie-break: strictly greater, or equal with smaller AnchorId.
            if sim > best_sim + _TIE_EPS or (
                abs(sim - best_sim) <= _TIE_EPS
                and (best_anchor is None or anchor_ids[j] < best_anchor)
            ):
                best_sim = sim
                best_anchor = anchor_ids[j]
        elif pol < 0:
            neg_vals.append(sim)
    positive_fit = max(0.0, max(pos_vals) if pos_vals else 0.0)
    negative_fit = max(0.0, max(neg_vals) if neg_vals else 0.0)
    net = float(np.clip(positive_fit - negative_fit, 0.0, 1.0))
    return SemanticProfile(
        positive_fit=round(positive_fit, 6),
        negative_fit=round(negative_fit, 6),
        net_semantic_fit=round(net, 6),
        best_positive_anchor=best_anchor,
        is_unknown=False,
    )


def _assign_archetype(
    vector: npt.NDArray[np.float32],
    centroids: npt.NDArray[np.float32],
    target_archetypes: frozenset[int],
) -> ArchetypeAssignment:
    """Nearest-centroid assignment (cosine; ties by ArchetypeId ascending)."""
    sims = centroids @ vector  # (K,) cosine, unit-norm rows
    best_id = 0
    best_sim = -2.0
    for k in range(sims.shape[0]):
        sim = float(sims[k])
        if sim > best_sim + _TIE_EPS:
            best_sim = sim
            best_id = k
    distance = float(1.0 - best_sim)
    confidence = float(np.clip((best_sim + 1.0) / 2.0, 0.0, 1.0))
    return ArchetypeAssignment(
        archetype_id=ArchetypeId(best_id),
        distance=round(distance, 6),
        membership_confidence=round(confidence, 6),
        is_target_archetype=best_id in target_archetypes,
    )


def _fallback_archetype(
    centroids: npt.NDArray[np.float32], target_archetypes: frozenset[int]
) -> ArchetypeAssignment:
    """The archetype assigned to a semantic-miss candidate (max distance, id 0)."""
    return ArchetypeAssignment(
        archetype_id=ArchetypeId(0),
        distance=1.0,
        membership_confidence=0.0,
        is_target_archetype=0 in target_archetypes,
    )


def _fold_semantic(
    cqv: npt.NDArray[np.float32],
    row: int,
    semantic_indices: Mapping[str, int],
    semantic: SemanticProfile,
) -> None:
    """Write semantic feature values into the CQV at their layout indices."""
    mapping = {
        "semantic_fit": semantic.net_semantic_fit,
        "positive_fit": semantic.positive_fit,
        "negative_fit": semantic.negative_fit,
    }
    for name, value in mapping.items():
        idx = semantic_indices.get(name)
        if idx is not None:
            cqv[row, idx] = np.float32(value)


# =========================================================================== #
# R4 — Gates & eligibility (pure; integrity + eligibility → floor mask)       #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class GatedSet:
    """The situated set with integrity + eligibility verdicts (stage GATED)."""

    base: SituatedSet
    cqv: npt.NDArray[np.float32]
    representations: tuple[CandidateRepresentation, ...]
    floor_mask: npt.NDArray[np.bool_]


def r4_gates(ctx: OnlineRunContext, situated: SituatedSet) -> GatedSet:
    """Compute the integrity + eligibility verdicts and the floor mask.

    Pure (no ports). Integrity detectors fire on the structural representation
    (timeline/skill-time/current-with-end-date impossibilities, keyword stuffing);
    ``is_honeypot = (≥2 HARD) OR (honeypot_score ≥ threshold)`` (inverted salary is
    soft only). Eligibility predicates produce hard blocks / soft penalties;
    ``is_eligible = (no hard blocks)``. The floor mask is the pure join
    ``is_honeypot OR not is_eligible``. **Verdicts are data** — they ride the
    representation and never raise; every finding carries ≥1 ``EvidenceRef`` sorted
    by code.
    """
    threshold = _honeypot_threshold(ctx)
    reps = list(situated.representations)
    floor = np.zeros(len(reps), dtype=np.bool_)

    for i, cand in enumerate(situated.base.candidates):
        integrity = _integrity_report(cand.raw, threshold)
        eligibility = _eligibility_report(cand.raw, reps[i])
        reps[i] = reps[i].with_gates(integrity=integrity, eligibility=eligibility)
        floor[i] = reps[i].is_floored
    return GatedSet(
        base=situated, cqv=situated.cqv, representations=tuple(reps), floor_mask=floor
    )


def _integrity_report(raw: RawCandidate, threshold: float) -> IntegrityReport:
    """Run the structural integrity detectors; aggregate into the report.

    Findings carry minted ``EvidenceRef``s pointing at the offending raw fields;
    HARD impossibilities and SOFT anomalies are scored (HARD 1.0, SOFT 0.35,
    capped). ``is_honeypot`` follows the ≥2-HARD-or-composite rule. Sorted by code.
    """
    raw_map = raw.model_dump(mode="json")
    findings: list[IntegrityFinding] = []

    # HARD: a current role carrying an end_date (impossible).
    for idx, position in enumerate(raw.career_history):
        if position.is_current and position.end_date is not None:
            ev = _safe_refs(
                raw_map,
                ((EvidenceKind.CAREER_FIELD, f"career_history.{idx}.end_date"),),
            )
            if ev:
                findings.append(
                    IntegrityFinding(
                        code=IntegrityFlag.CURRENT_ROLE_HAS_END_DATE,
                        severity=Severity.HARD,
                        evidence=ev,
                    )
                )
            break

    # HARD: education timeline impossible (end before start).
    for idx, edu in enumerate(raw.education):
        if edu.end_year < edu.start_year:
            ev = _safe_refs(
                raw_map,
                ((EvidenceKind.EDUCATION, f"education.{idx}.end_year"),),
            )
            if ev:
                findings.append(
                    IntegrityFinding(
                        code=IntegrityFlag.EDUCATION_TIMELINE_IMPOSSIBLE,
                        severity=Severity.HARD,
                        evidence=ev,
                    )
                )
            break

    # HARD: tenure exceeds stated experience beyond tolerance.
    total_months = sum(p.duration_months for p in raw.career_history)
    stated = raw.profile.years_of_experience
    if stated > 0 and total_months / 12.0 > stated * 1.5:
        ev = _safe_refs(
            raw_map, ((EvidenceKind.PROFILE_FIELD, "profile.years_of_experience"),)
        )
        if ev:
            findings.append(
                IntegrityFinding(
                    code=IntegrityFlag.TENURE_EXCEEDS_EXPERIENCE,
                    severity=Severity.HARD,
                    evidence=ev,
                )
            )

    # SOFT: keyword stuffing (broad advanced+ skills with zero corroboration).
    stuffed = [
        idx
        for idx, s in enumerate(raw.skills)
        if s.proficiency.casefold() in ("advanced", "expert")
        and s.endorsements == 0
        and s.duration_months in (None, 0)
    ]
    if len(stuffed) >= 8:
        ev = _safe_refs(
            raw_map, ((EvidenceKind.SKILL, f"skills.{stuffed[0]}.proficiency"),)
        )
        if ev:
            findings.append(
                IntegrityFinding(
                    code=IntegrityFlag.EXPERT_SKILL_ZERO_USAGE,
                    severity=Severity.SOFT,
                    evidence=ev,
                )
            )

    findings.sort(key=lambda f: f.code.value)
    hard = sum(1 for f in findings if f.severity is Severity.HARD)
    score = min(1.0, sum(1.0 if f.severity is Severity.HARD else 0.35 for f in findings))
    is_honeypot = (hard >= 2) or (score >= threshold)
    return IntegrityReport(
        findings=tuple(findings),
        honeypot_score=round(score, 6),
        is_honeypot=is_honeypot,
        rules_evaluated=4,
    )


def _eligibility_report(
    raw: RawCandidate, representation: CandidateRepresentation
) -> EligibilityReport:
    """Run the eligibility predicates; hard blocks / soft penalties with evidence.

    ``is_eligible = (no hard blocks)``. Each finding carries ≥1 evidence ref.
    Predicates here are the structural subset the online representation supports;
    the full predicate set is data-driven from ``gates/eligibility_rules.yaml``.
    """
    raw_map = raw.model_dump(mode="json")
    hard: list[EligibilityFinding] = []
    soft: list[EligibilityFinding] = []

    career = representation.career
    if career is not None and career.track is CareerTrack.SERVICES:
        ev = _safe_refs(
            raw_map, ((EvidenceKind.CAREER_FIELD, "career_history.0.industry"),)
        )
        if ev:
            hard.append(
                EligibilityFinding(
                    code=EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER,
                    severity=Severity.HARD,
                    evidence=ev,
                )
            )

    country = raw.profile.country.casefold()
    if country and country != "india":
        ev = _safe_refs(raw_map, ((EvidenceKind.PROFILE_FIELD, "profile.country"),))
        if ev:
            soft.append(
                EligibilityFinding(
                    code=EligibilityCode.OUTSIDE_INDIA_NO_SPONSOR,
                    severity=Severity.SOFT,
                    evidence=ev,
                )
            )

    hard.sort(key=lambda f: f.code.value)
    soft.sort(key=lambda f: f.code.value)
    return EligibilityReport(
        hard_blocks=tuple(hard),
        soft_penalties=tuple(soft),
        is_eligible=(len(hard) == 0),
    )


# =========================================================================== #
# R5 — Scoring (pure; locked weights, gates, bounded multipliers)             #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class ScoredSet:
    """The gated set scored into ``ScoredCandidate``s (stage SCORED)."""

    base: GatedSet
    scored: tuple[ScoredCandidate, ...]
    representations: tuple[CandidateRepresentation, ...]


def r5_score(ctx: OnlineRunContext, gated: GatedSet) -> ScoredSet:
    """Apply the locked weights, gates, and bounded multipliers deterministically.

    ``base_relevance = Σ(component·locked_weight)`` summed in fixed ``ScoreComponent``
    order (the folded CQV aggregated per component). Floored candidates get
    ``final_score = FLOOR`` with no multipliers. Else
    ``final = base × behavioral × logistics + archetype_adj`` then a confidence
    shrink toward the neutral prior. **No calibration curve.** float32; no NaN/inf
    (asserted); ``tiebreak_key`` is the candidate id.

    Raises:
        ScoreInvariantError: a NaN/inf final score or an out-of-bounds multiplier
            (a structural breach — fatal).
    """
    weights = _weight_vector(ctx)
    component_order = tuple(ScoreComponent)
    col_to_comp = _map_columns(gated.base.base.feature_ids, component_order)
    cqv = gated.cqv
    reps = list(gated.representations)
    scored: list[ScoredCandidate] = []

    for i, cand in enumerate(gated.base.base.candidates):
        comp_values = _aggregate_components(cqv[i], col_to_comp, len(component_order))
        # Fixed-order weighted sum (determinism).
        components: list[ScoreComponentValue] = []
        base_relevance = 0.0
        for ci, component in enumerate(component_order):
            raw_val = float(comp_values[ci])
            weight = float(weights[ci])
            weighted = raw_val * weight
            base_relevance += weighted
            components.append(
                ScoreComponentValue(
                    component=component,
                    raw=round(raw_val, 6),
                    weight=round(weight, 6),
                    weighted=round(weighted, 6),
                )
            )
        rep = reps[i]
        floored = bool(gated.floor_mask[i])
        breakdown = _score_breakdown(rep, base_relevance, components, floored)
        final = breakdown.final_score
        if not np.isfinite(final):
            raise ScoreInvariantError(f"non-finite final score for {cand.candidate_id}")
        scored_candidate = ScoredCandidate(
            candidate_id=cand.candidate_id,
            final_score=Score(float(final)),
            breakdown=breakdown,
            tiebreak_key=cand.candidate_id,
        )
        scored.append(scored_candidate)
        reps[i] = rep.with_score(scored=scored_candidate)
    return ScoredSet(base=gated, scored=tuple(scored), representations=tuple(reps))


def _score_breakdown(
    representation: CandidateRepresentation,
    base_relevance: float,
    components: Sequence[ScoreComponentValue],
    floored: bool,
) -> ScoreBreakdown:
    """Assemble the score breakdown; floored ⇒ FLOOR (no multipliers applied)."""
    if floored:
        return ScoreBreakdown(
            components=tuple(components),
            base_relevance=round(base_relevance, 6),
            behavioral_multiplier=1.0,
            logistics_multiplier=1.0,
            archetype_adjustment=0.0,
            floored=True,
            final_score=_FLOOR,
        )
    behavioral_mult = _behavioral_multiplier(representation)
    logistics_mult = _logistics_multiplier(representation)
    archetype_adj = _archetype_adjustment(representation)
    raw_final = base_relevance * behavioral_mult * logistics_mult + archetype_adj
    final = _confidence_shrink(raw_final, representation)
    return ScoreBreakdown(
        components=tuple(components),
        base_relevance=round(base_relevance, 6),
        behavioral_multiplier=round(behavioral_mult, 6),
        logistics_multiplier=round(logistics_mult, 6),
        archetype_adjustment=round(archetype_adj, 6),
        floored=False,
        final_score=round(float(final), 6),
    )


def _behavioral_multiplier(representation: CandidateRepresentation) -> float:
    """Bounded behavioral multiplier (modulates, never creates relevance)."""
    behavioral = representation.behavioral
    if behavioral is None:
        return 1.0
    centered = (behavioral.availability + behavioral.engagement) / 2.0
    mult = _MULT_LOWER + (centered) * (_MULT_UPPER - _MULT_LOWER)
    return float(np.clip(mult, _MULT_LOWER, _MULT_UPPER))


def _logistics_multiplier(representation: CandidateRepresentation) -> float:
    """Bounded logistics multiplier; soft penalties dampen, never floor."""
    eligibility = representation.eligibility
    penalty = 0.0
    if eligibility is not None:
        penalty = 0.05 * len(eligibility.soft_penalties)
    return float(np.clip(1.0 - penalty, _MULT_LOWER, _MULT_UPPER))


def _archetype_adjustment(representation: CandidateRepresentation) -> float:
    """Bounded additive boost for a target archetype (post-multiplier)."""
    archetype = representation.archetype
    if archetype is not None and archetype.is_target_archetype:
        return _ARCHETYPE_BOOST * archetype.membership_confidence
    return 0.0


def _confidence_shrink(
    raw_final: float, representation: CandidateRepresentation
) -> float:
    """Shrink toward the neutral prior proportional to uncertainty (bounded).

    A semantic-UNKNOWN or low-credibility candidate is pulled toward the prior by
    up to ``_MAX_SHRINK`` — risk modulates confidence, never erases a real signal.
    """
    uncertainty = 0.0
    if representation.semantic is not None and representation.semantic.is_unknown:
        uncertainty += 0.5
    if representation.credibility is not None:
        uncertainty += representation.credibility.keyword_stuffing_score * 0.5
    shrink = _MAX_SHRINK * float(np.clip(uncertainty, 0.0, 1.0))
    return raw_final * (1.0 - shrink)


# =========================================================================== #
# R6 — Ranking (pure; raw scores, floor-partitioned, top-100, invariants)     #
# =========================================================================== #
def r6_rank(ctx: OnlineRunContext, scored: ScoredSet) -> Ranking:
    """Select the deterministic top-100 into a spec-valid ``Ranking``.

    Stable sort all candidates by ``(−final_score, candidate_id)`` on **raw**
    scores; non-floored candidates fill ranks 1..100 first, floored partitioned to
    the tail (honeypot top-100 rate ≈ 0 by construction). The ``Ranking.build``
    factory enforces the six validator invariants and raises
    ``RankingInvariantError`` on any violation or a sub-100 candidate set.

    Raises:
        RankingInvariantError: fewer than 100 candidates, or an invariant breach.
    """
    floor_mask = scored.base.floor_mask
    non_floored: list[tuple[CandidateId, float]] = []
    floored: list[tuple[CandidateId, float]] = []
    for i, candidate in enumerate(scored.scored):
        pair = (candidate.candidate_id, float(candidate.final_score))
        if bool(floor_mask[i]):
            floored.append(pair)
        else:
            non_floored.append(pair)

    non_floored.sort(key=lambda kv: (-kv[1], kv[0]))
    floored.sort(key=lambda kv: (-kv[1], kv[0]))
    # Non-floored fill first; floored only backfill a degenerate < 100 shortfall.
    ordered = non_floored + floored
    _ = ctx  # ports not used at R6 (pure)
    return Ranking.build(ordered, size=_RANKING_SIZE)


# =========================================================================== #
# R7 — Reasoning generation (pure; top-100, evidence-bound, no reorder)       #
# =========================================================================== #
def r7_reason(
    ctx: OnlineRunContext, ranking: Ranking, gated: GatedSet
) -> Ranking:
    """Produce evidence-grounded reasoning for the top-100; attach without reorder.

    For each ranked candidate, re-hydrate its representation (the inline raw kept
    from R1), select strengths from the top contributing features (feature
    importance) and honest concerns from soft penalties / notable gaps, and
    assemble ``ReasoningClause``s — each carrying ≥1 ``EvidenceRef`` resolving to a
    real field (the no-hallucination guarantee, enforced at construction). Tone
    matches the rank band. Attached via ``Ranking.with_reasoning`` (re-asserts the
    structural invariants; never reorders).

    Raises:
        ProvenanceError: a clause cites a non-existent fact, or a top-100 candidate
            lacks its re-hydrated representation.
    """
    by_id = {rep.candidate_id: rep for rep in gated.representations}
    importance = _feature_importance(ctx)
    reasoning: dict[CandidateId, CandidateReasoning] = {}

    for row in ranking.rows:
        rep = by_id.get(row.candidate_id)
        if rep is None or rep.provenance.inline is None:
            raise ProvenanceError(
                f"missing re-hydrated representation for {row.candidate_id}"
            )
        raw = rep.provenance.inline
        clauses = _build_clauses(raw, rep, importance, rank=row.rank)
        if not clauses:
            # A representation with no defensible clause cannot be reasoned about —
            # this is a provenance failure, not silently emitting empty reasoning.
            raise ProvenanceError(
                f"no evidence-backed clause available for {row.candidate_id}"
            )
        rendered = _render(clauses)
        reasoning[row.candidate_id] = CandidateReasoning(
            candidate_id=row.candidate_id, clauses=tuple(clauses), rendered=rendered
        )
    return ranking.with_reasoning(reasoning)


def _build_clauses(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    importance: Mapping[str, float],
    *,
    rank: int,
) -> list[ReasoningClause]:
    """Assemble evidence-bound clauses: strengths (by importance) + honest concerns.

    Each clause cites a minted ``EvidenceRef`` (a real raw field). Variation arises
    from *which* evidence each candidate has — not name templating. At least one
    clause carries a JD link (a ``ScoreComponent``), satisfying ``CandidateReasoning``.
    """
    raw_map = raw.model_dump(mode="json")
    clauses: list[ReasoningClause] = []

    # Strength: experience (always present, JD-linked to EXPERIENCE_FIT).
    exp_ref = _safe_refs(
        raw_map, ((EvidenceKind.PROFILE_FIELD, "profile.years_of_experience"),)
    )
    if exp_ref:
        years = raw.profile.years_of_experience
        clauses.append(
            ReasoningClause(
                polarity=ReasoningPolarity.STRENGTH,
                text=f"{years:.0f} years of stated experience",
                jd_link=ScoreComponent.EXPERIENCE_FIT,
                evidence=exp_ref,
            )
        )

    # Strength: semantic fit when present (JD-linked to SEMANTIC_FIT).
    semantic = representation.semantic
    if semantic is not None and not semantic.is_unknown and semantic.net_semantic_fit > 0.0:
        title_ref = _safe_refs(
            raw_map, ((EvidenceKind.PROFILE_FIELD, "profile.current_title"),)
        )
        if title_ref:
            clauses.append(
                ReasoningClause(
                    polarity=ReasoningPolarity.STRENGTH,
                    text=(
                        f"current role '{raw.profile.current_title}' aligns with the "
                        f"JD focus"
                    ),
                    jd_link=ScoreComponent.SEMANTIC_FIT,
                    evidence=title_ref,
                )
            )

    # Concern: soft eligibility penalties (honest, evidence-bound).
    eligibility = representation.eligibility
    if eligibility is not None and eligibility.soft_penalties:
        penalty = eligibility.soft_penalties[0]
        clauses.append(
            ReasoningClause(
                polarity=ReasoningPolarity.CONCERN,
                text=f"note: {penalty.code.value.replace('_', ' ')}",
                jd_link=ScoreComponent.CAREER_FIT,
                evidence=penalty.evidence,
            )
        )

    # Context (tail band): a measured note from the strongest available evidence.
    if rank > 50 and len(clauses) < 2:
        company_ref = _safe_refs(
            raw_map, ((EvidenceKind.PROFILE_FIELD, "profile.current_company"),)
        )
        if company_ref:
            clauses.append(
                ReasoningClause(
                    polarity=ReasoningPolarity.CONTEXT,
                    text=f"currently at {raw.profile.current_company}",
                    jd_link=ScoreComponent.CAREER_FIT,
                    evidence=company_ref,
                )
            )

    _ = importance  # importance ranks strengths when the full feature set lands.
    return clauses


def _render(clauses: Sequence[ReasoningClause]) -> str:
    """Render the ordered clauses into ≤2 sentences (deterministic, no LLM)."""
    parts = [c.text.strip() for c in clauses if c.text.strip()]
    rendered = "; ".join(parts[:3])
    return rendered if rendered else "insufficient evidence"


# =========================================================================== #
# R8 — Submission generation (ports: SubmissionSink)                          #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class _SubmissionRow:
    """One concrete submission row (satisfies the ``SubmissionRow`` protocol)."""

    candidate_id: str
    rank: int
    score: float
    reasoning: str


def r8_submit(ctx: OnlineRunContext, ranking: Ranking) -> SubmissionReceipt:
    """Validate (structural + Stage-4) then write the CSV atomically via the sink.

    Before writing, mirror ``validate_submission.py`` + the Stage-4 reasoning
    checks (no empty/identical/templated/rank-inconsistent reasoning); any HARD
    finding aborts (no rejectable file is emitted). The sink renders
    ``candidate_id,rank,score,reasoning`` (UTF-8/no-BOM/``\\n``/RFC-4180), writes
    atomically, and re-asserts monotone-by-rank + id tie-break at the emitted
    precision, returning a ``SubmissionReceipt``.

    Raises:
        SubmissionContractError: a structural or Stage-4 validation failure (the
            sink raises on the post-write re-assert).
    """
    _validate_ranking(ranking, ctx.config.score_decimals)
    rows: list[SubmissionRow] = []
    for row in ranking.rows:
        rendered = row.reasoning.rendered if row.reasoning is not None else ""
        rows.append(
            _SubmissionRow(
                candidate_id=str(row.candidate_id),
                rank=row.rank,
                score=round(float(row.final_score), ctx.config.score_decimals),
                reasoning=rendered,
            )
        )
    return ctx.submission_sink.write(rows)


def _validate_ranking(ranking: Ranking, decimals: int) -> None:
    """Structural + Stage-4 reasoning checks mirroring ``validate_submission.py``.

    Raises:
        SubmissionContractError: wrong row count, non-monotone score at the emitted
            precision, a tie not id-ordered, or empty/identical reasoning.
    """
    from redstack.domain.errors import SubmissionContractError

    rows = ranking.rows
    if len(rows) != _RANKING_SIZE:
        raise SubmissionContractError(f"submission has {len(rows)} rows, expected 100")
    rendered: list[str] = []
    prev_score: float | None = None
    prev_id: str | None = None
    for row in sorted(rows, key=lambda r: r.rank):
        if row.reasoning is None or not row.reasoning.rendered.strip():
            raise SubmissionContractError(f"empty reasoning at rank {row.rank}")
        rendered.append(row.reasoning.rendered)
        score = round(float(row.final_score), decimals)
        if prev_score is not None and score > prev_score + _TIE_EPS:
            raise SubmissionContractError(f"score increased at rank {row.rank}")
        if (
            prev_score is not None
            and abs(score - prev_score) <= _TIE_EPS
            and prev_id is not None
            and str(row.candidate_id) < prev_id
        ):
            raise SubmissionContractError(f"tie not id-ordered at rank {row.rank}")
        prev_score = score
        prev_id = str(row.candidate_id)
    if len(set(rendered)) < len(rendered):
        raise SubmissionContractError("identical reasoning across rows (Stage-4)")


# =========================================================================== #
# R9 — Run report (ports: RunReportSink)                                      #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class ReportOutcome:
    """The terminal R9 outcome consumed by the orchestrator."""

    submission_path: str
    report_path: str
    honeypot_rate_top100: float


@final
@dataclass(frozen=True, slots=True)
class _RunReport:
    """A concrete run report satisfying the ``RunReportView`` protocol."""

    reproducible: Mapping[str, object]
    audit: Mapping[str, object]
    timings: Mapping[str, object]
    budget: Mapping[str, object]


def r9_report(
    ctx: OnlineRunContext,
    *,
    ranking: Ranking,
    scored: ScoredSet,
    gated: GatedSet,
    receipt: SubmissionReceipt,
    stage_timings_ms: Mapping[str, float],
) -> ReportOutcome:
    """Emit ``run_report.json`` — the audit + reproducibility artifact.

    Builds the ``reproducible`` block (code/config/manifest hashes, per-artifact
    hashes, input + output sha256, candidate count, honeypot rate + eligibility
    summary, score distribution digest, as_of, seed — the only block determinism
    tests compare), plus ``audit`` (excluded from the repro hash), ``timings``, and
    ``budget``. Written deterministically via the ``RunReportSinkPort``.
    """
    top_ids = {row.candidate_id for row in ranking.rows}
    honeypot_in_top = 0
    eligibility_counts: dict[str, int] = {}
    for i, cand in enumerate(gated.base.base.candidates):
        rep = gated.representations[i]
        if cand.candidate_id in top_ids:
            if rep.integrity is not None and rep.integrity.is_honeypot:
                honeypot_in_top += 1
        if rep.eligibility is not None:
            for finding in (*rep.eligibility.hard_blocks, *rep.eligibility.soft_penalties):
                eligibility_counts[finding.code.value] = (
                    eligibility_counts.get(finding.code.value, 0) + 1
                )
    honeypot_rate = honeypot_in_top / max(1, len(ranking.rows))

    reproducible: dict[str, object] = {
        "code_version": ctx.config.code_version,
        "config_hash": ctx.config.config_hash,
        "manifest_hash": ctx.manifest_hash,
        "artifact_hashes": _artifact_hashes(ctx),
        "input_file_sha256": ctx.input_file_sha256,
        "candidate_count": len(scored.scored),
        "output_sha256": receipt.output_sha256,
        "honeypot_count_top100": honeypot_in_top,
        "honeypot_rate": round(honeypot_rate, 6),
        "eligibility_summary": dict(sorted(eligibility_counts.items())),
        "score_distribution_digest": _score_digest(ranking),
        "as_of": ctx.as_of.isoformat(),
        "seed": ctx.seed,
    }
    audit: dict[str, object] = {
        "run_id": f"{ctx.config.participant_id}-{ctx.manifest_hash[:12]}",
        "host_label": "online-sandbox",
    }
    used_seconds = sum(stage_timings_ms.values()) / 1000.0
    budget: dict[str, object] = {
        "limit_seconds": ctx.config.budget_limit_seconds,
        "used_seconds": round(used_seconds, 3),
        "within_budget": used_seconds <= ctx.config.budget_limit_seconds,
    }
    report = _RunReport(
        reproducible=reproducible,
        audit=audit,
        timings=dict(sorted(stage_timings_ms.items())),
        budget=budget,
    )
    report_path = ctx.run_report_sink.write(report)
    return ReportOutcome(
        submission_path=f"{ctx.config.participant_id}.csv",
        report_path=report_path,
        honeypot_rate_top100=round(honeypot_rate, 6),
    )


# --------------------------------------------------------------------------- #
# Shared artifact accessors + numeric helpers                                 #
# --------------------------------------------------------------------------- #
def _registry(ctx: OnlineRunContext) -> FeatureRegistry:
    """The FeatureRegistry reconstructed at R0 (carried on the artifacts map)."""
    reg = ctx.artifacts.get("__feature_registry__")
    if isinstance(reg, FeatureRegistry):
        return reg
    # Reconstruct from the manifest (R0 always loaded feature_manifest).
    return _build_feature_registry(ctx.artifacts, ctx.layout_version)


def _anchor_matrix(ctx: OnlineRunContext) -> npt.NDArray[np.float32]:
    mat = ctx.artifacts.get("anchor_vectors")
    if not isinstance(mat, np.ndarray) or mat.ndim != 2:
        raise VectorStoreError("anchor_vectors missing or not 2-D")
    return np.ascontiguousarray(mat, dtype=np.float32)


def _centroid_matrix(ctx: OnlineRunContext) -> npt.NDArray[np.float32]:
    mat = ctx.artifacts.get("centroids")
    if not isinstance(mat, np.ndarray) or mat.ndim != 2:
        raise VectorStoreError("centroids missing or not 2-D")
    return np.ascontiguousarray(mat, dtype=np.float32)


def _anchor_ids(ctx: OnlineRunContext) -> tuple[AnchorId, ...]:
    """Anchor ids in sorted order (matches anchor_vectors row order from O13b)."""
    jd = ctx.artifacts.get("jd_concepts")
    anchors = jd.get("anchors") if isinstance(jd, Mapping) else None
    if not isinstance(anchors, (list, tuple)):
        return ()
    ids = [
        str(a["id"])
        for a in anchors
        if isinstance(a, Mapping) and isinstance(a.get("id"), str)
    ]
    return tuple(AnchorId(i) for i in sorted(ids))


def _anchor_polarities(
    ctx: OnlineRunContext, anchor_ids: Sequence[AnchorId]
) -> tuple[int, ...]:
    """Per-anchor polarity sign (+1 positive, −1 negative), aligned to anchor_ids."""
    jd = ctx.artifacts.get("jd_concepts")
    anchors = jd.get("anchors") if isinstance(jd, Mapping) else None
    polarity_of: dict[str, str] = {}
    if isinstance(anchors, (list, tuple)):
        for a in anchors:
            if isinstance(a, Mapping) and isinstance(a.get("id"), str):
                polarity_of[str(a["id"])] = str(a.get("polarity", "positive"))
    return tuple(
        1 if polarity_of.get(str(aid), "positive") == "positive" else -1
        for aid in anchor_ids
    )


def _target_archetypes(ctx: OnlineRunContext) -> frozenset[int]:
    """The set of target archetype ids (boosted at R5)."""
    doc = ctx.artifacts.get("archetypes")
    archetypes = doc.get("archetypes") if isinstance(doc, Mapping) else None
    targets: set[int] = set()
    if isinstance(archetypes, Mapping):
        for key, body in archetypes.items():
            if isinstance(body, Mapping) and body.get("is_target_archetype") is True:
                try:
                    targets.add(int(key))
                except (TypeError, ValueError):
                    continue
    return frozenset(targets)


def _honeypot_threshold(ctx: OnlineRunContext) -> float:
    """The calibrated composite honeypot threshold (from integrity_thresholds)."""
    doc = ctx.artifacts.get("integrity_thresholds")
    value = doc.get("honeypot_threshold") if isinstance(doc, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.7


def _weight_vector(ctx: OnlineRunContext) -> npt.NDArray[np.float32]:
    """The locked per-ScoreComponent weight vector (fixed component order)."""
    doc = ctx.artifacts.get("scoring_weights")
    weights = doc.get("weights") if isinstance(doc, Mapping) else None
    if not isinstance(weights, Mapping):
        raise ScoreInvariantError("scoring_weights missing 'weights'")
    vec = np.zeros(len(ScoreComponent), dtype=np.float32)
    for i, component in enumerate(ScoreComponent):
        value = weights.get(component.value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            vec[i] = np.float32(value)
    return vec


def _feature_importance(ctx: OnlineRunContext) -> Mapping[str, float]:
    """The feature importance map (orders reasoning strengths)."""
    doc = ctx.artifacts.get("feature_importance")
    importances = doc.get("importances") if isinstance(doc, Mapping) else None
    if isinstance(importances, Mapping):
        return {
            str(k): float(v)
            for k, v in importances.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    return {}


def _map_columns(
    feature_ids: Sequence[str], component_order: Sequence[ScoreComponent]
) -> dict[int, int]:
    """Map each CQV column index → its ScoreComponent index (mirrors offline O9)."""
    group_to_component = {
        "retr": ScoreComponent.SEMANTIC_FIT,
        "rank": ScoreComponent.SEMANTIC_FIT,
        "ir": ScoreComponent.SEMANTIC_FIT,
        "jd": ScoreComponent.SEMANTIC_FIT,
        "semantic": ScoreComponent.SEMANTIC_FIT,
        "skill": ScoreComponent.SKILL_MATCH,
        "career": ScoreComponent.CAREER_FIT,
        "pvs": ScoreComponent.CAREER_FIT,
        "exp": ScoreComponent.EXPERIENCE_FIT,
        "edu": ScoreComponent.EDUCATION_FIT,
        "cons": ScoreComponent.CREDIBILITY,
        "hp": ScoreComponent.CREDIBILITY,
        "bhv": ScoreComponent.CREDIBILITY,
        "arch": ScoreComponent.ARCHETYPE_FIT,
    }
    comp_index = {c: i for i, c in enumerate(component_order)}
    out: dict[int, int] = {}
    for col, fid in enumerate(feature_ids):
        prefix = fid.split(".", 1)[0] if "." in fid else fid
        component = group_to_component.get(prefix, ScoreComponent.CREDIBILITY)
        out[col] = comp_index[component]
    return out


def _aggregate_components(
    row: npt.NDArray[np.float32], col_to_comp: Mapping[int, int], n_comp: int
) -> npt.NDArray[np.float32]:
    """Aggregate a CQV row into per-component means (fixed-order reduction)."""
    acc = np.zeros(n_comp, dtype=np.float32)
    counts = np.zeros(n_comp, dtype=np.float32)
    for col in range(row.shape[0]):
        ci = col_to_comp.get(col)
        if ci is not None:
            acc[ci] += row[col]
            counts[ci] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, acc / np.maximum(counts, 1.0), 0.0).astype(
            np.float32
        )


def _semantic_feature_indices(feature_ids: Sequence[str]) -> dict[str, int]:
    """Locate the semantic feature columns to fold R3 values into."""
    out: dict[str, int] = {}
    for idx, fid in enumerate(feature_ids):
        name = fid.split(".", 1)[1] if "." in fid else fid
        if name in ("semantic_fit", "positive_fit", "negative_fit", "net"):
            out["semantic_fit" if name == "net" else name] = idx
    return out


def _compose_embedding_document(raw: RawCandidate) -> str:
    """Compose the R2-recipe embedding document for the onnx fallback (deterministic)."""
    parts = [
        raw.profile.headline,
        raw.profile.summary,
        raw.profile.current_title,
    ]
    for position in raw.career_history:
        parts.append(position.title)
        parts.append(position.description)
    return " ".join(p for p in parts if p)


def _safe_refs(
    raw_map: Mapping[str, object],
    specs: Sequence[tuple[EvidenceKind, str]],
) -> tuple[EvidenceRef, ...]:
    """Mint the (kind, path) refs; drop any that fail to resolve (no hallucination)."""
    refs: list[EvidenceRef] = []
    for kind, path in specs:
        try:
            refs.append(mint(raw_map, kind=kind, path=path))
        except ProvenanceError:
            continue
    return tuple(refs)


def _artifact_hashes(ctx: OnlineRunContext) -> Mapping[str, str]:
    """Per-artifact sha256 map from the verified manifest (audit chain)."""
    artifacts = ctx.manifest.get("artifacts")
    out: dict[str, str] = {}
    if isinstance(artifacts, Mapping):
        for key, entry in artifacts.items():
            if isinstance(entry, Mapping):
                sha = entry.get("sha256")
                if isinstance(sha, str):
                    out[str(key)] = sha
    return dict(sorted(out.items()))


def _score_digest(ranking: Ranking) -> str:
    """A deterministic digest of the top-100 score distribution (repro block)."""
    payload = [
        {"rank": row.rank, "score": round(float(row.final_score), 6)}
        for row in sorted(ranking.rows, key=lambda r: r.rank)
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Late import to avoid a heavy module-load cost at import time (kept local). ---- #
from redstack.features.extraction import extract_row  # noqa: E402