from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final

import numpy as np
import numpy.typing as npt
import yaml

from redstack.config.schema import (
    BehavioralPolicy,
    EligibilityRuleSet,
    IntegrityThresholds,
    LogisticsPolicy,
    ScoringPolicy,
    default_eligibility_rules,
)
from redstack.domain.candidate.archetype import ArchetypeAssignment
from redstack.domain.candidate.eligibility import EligibilityFinding, EligibilityReport
from redstack.domain.candidate.identity import Identity
from redstack.domain.candidate.integrity import IntegrityReport
from redstack.domain.candidate.representation import CandidateRepresentation
from redstack.domain.candidate.semantic import SemanticProfile
from redstack.domain.enums import (
    EligibilityCode,
    EvidenceKind,
    IntegrityFlag,
    LocationFit,
    NoticeFit,
    ScoreComponent,
    Severity,
)
from redstack.domain.errors import (
    ArtifactContractError,
    CQVInvariantError,
    ProvenanceError,
    SchemaError,
)
from redstack.domain.ids import (
    AnchorId,
    ArchetypeId,
    CandidateId,
    Similarity,
    UnitScore,
)
from redstack.domain.jd import JobDescriptionSpec
from redstack.domain.provenance import ProvenanceHandle
from redstack.domain.ranking import Ranking
from redstack.domain.scoring import ScoredCandidate, ScoringWeights
from redstack.domain.source import RawCandidate
from redstack.engines.behavioral import BehavioralEngine
from redstack.engines.eligibility import EligibilityEngine
from redstack.engines.integrity import IntegrityEngine
from redstack.engines.logistics import LogisticsEngine
from redstack.engines.ranking import RankingEngine
from redstack.engines.reasoning import ReasoningEngine
from redstack.engines.scoring import ComponentRaw, ScoringEngine
from redstack.engines.semantic import AnchorSet, ArchetypeSpace, SemanticEngine
from redstack.engines.validation import ValidationEngine
from redstack.features.extraction import (
    build_behavioral_profile,
    build_career_profile,
    build_credibility_profile,
    build_logistics_profile,
    extract_row,
    fold_semantic,
)
from redstack.features.layout import GROUP_ORDER
from redstack.features.parsing import validate as validate_raw
from redstack.features.registry import FEATURE_REGISTRY, FeatureRegistry
from redstack.features.view import FeatureCell, clamp_unit, make_evidence
from redstack.ports._types import ArtifactKey, SourceMalformed
from redstack.ports.artifact_store import ArtifactStorePort
from redstack.ports.candidate_source import CandidateSourceError
from redstack.ports.embedding import EmbeddingError, EmbeddingModelPort
from redstack.ports.online import (
    OnlineEntropyPort,
    SemanticVectorStorePort,
    SubmissionReceipt,
)
from redstack.ports.semantic_index import VectorStoreError
from redstack.ports.submission_sink import SubmissionContractError

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
)

#: The fixed hard/soft eligibility code split (mirrors ``engines.eligibility``'s
#: own ``_HARD_CODES``/``_SOFT_CODES`` partition).
_HARD_ELIGIBILITY_CODES: Final[frozenset[EligibilityCode]] = frozenset(
    (
        EligibilityCode.PURE_RESEARCH_NO_PRODUCTION,
        EligibilityCode.LANGCHAIN_OPENAI_ONLY_RECENT,
        EligibilityCode.NO_PRODUCTION_CODE_18M,
        EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER,
        EligibilityCode.PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP,
        EligibilityCode.CLOSED_SOURCE_5Y_NO_VALIDATION,
    )
)
_SOFT_ELIGIBILITY_CODES: Final[frozenset[EligibilityCode]] = frozenset(
    (
        EligibilityCode.TITLE_CHASER_SUB_18M_HOPS,
        EligibilityCode.NOTICE_OVER_30,
        EligibilityCode.OUTSIDE_INDIA_NO_SPONSOR,
        EligibilityCode.OUTSIDE_EXPERIENCE_BAND,
    )
)
#: A single, static JD spec. Re-authoring per-JD config is a separate,
#: not-yet-wired concern (``EligibilityEngine.evaluate`` accepts ``jd`` for
#: forward compatibility but its current detectors never read it).
_JD_SPEC: Final[JobDescriptionSpec] = JobDescriptionSpec(
    role_title="Applied ML Engineer",
    min_experience_years=5.0,
    max_experience_years=9.0,
    preferred_hubs=frozenset({"pune", "noida", "hyderabad", "mumbai", "delhi"}),
    positive_anchors=frozenset(),
    negative_anchors=frozenset(),
    hard_disqualifiers=_HARD_ELIGIBILITY_CODES,
    soft_penalties=_SOFT_ELIGIBILITY_CODES,
    target_archetypes=frozenset(),
)
_COMPETENCY_GROUPS: Final[tuple[str, ...]] = (
    "retr",
    "rank",
    "recsys",
    "ir",
    "nlp",
    "llm",
    "mle",
    "mlops",
    "eval",
)
_RANKING_SIZE: Final[int] = 100

# Per-group weight for the R5 confidence average that drives ``_shrink``'s pull
# toward the neutral prior. A flat ``np.mean`` over all 32 CQV groups let
# structurally-sparse optional groups (``oss``/``bhv``/``risk``/``hp``/...  —
# low-confidence for most of the population regardless of domain fit, e.g. a
# missing GitHub link or no prior offer history) dilute the handful of groups
# that actually carry domain-fit signal, collapsing genuine skill/semantic
# differences into a narrow band around 0.5 for everyone. Weights below are
# proportional to each group's role in the locked ``ScoringWeights``: the nine
# competency groups get the largest share (skill_match's 0.15, plus standing in
# for semantic_fit's 0.30 since ``SemanticProfile`` carries no confidence field
# of its own — these groups' nested ``.semantic`` cells are the closest proxy);
# career/pvs/cons/edu/exp follow their own component weights. Every other group
# (logistics/behavioral/integrity/identity) keeps a small non-zero floor so it
# still nudges confidence — never a hard zero, per the "never create or
# destroy relevance outright" multiplier discipline — but can no longer
# dominate the average.
_DOMAIN_FIT_GROUP_WEIGHT: Final[Mapping[str, float]] = MappingProxyType(
    {
        **{group: 3.0 for group in _COMPETENCY_GROUPS},
        "career": 1.5,
        "pvs": 1.5,
        "cons": 1.5,
        "edu": 1.0,
        "exp": 1.0,
    }
)
_OTHER_GROUP_CONFIDENCE_WEIGHT: Final[float] = 0.2
_GROUP_CONFIDENCE_WEIGHTS: Final[npt.NDArray[np.float64]] = np.array(
    [
        _DOMAIN_FIT_GROUP_WEIGHT.get(group, _OTHER_GROUP_CONFIDENCE_WEIGHT)
        for group in GROUP_ORDER
    ],
    dtype=np.float64,
)
# Denominator of the R5 confidence weighted-average (see ``r5_score``):
# precomputed once so the per-candidate reduction matches ``np.average``'s
# ``(a * weights).sum() / weights.sum()`` exactly, just batched over all N
# rows in one call instead of N separate ``np.average`` calls.
_GROUP_CONFIDENCE_WEIGHTS_SUM: Final[float] = float(_GROUP_CONFIDENCE_WEIGHTS.sum())

# Static feature-id groupings for R5's component aggregation (``_score_components``/
# ``_skill_match_value``). Hoisted to module scope so the hot per-candidate loop
# doesn't reconstruct the same list/tuple literal (and, for skill-match, redo the
# same f-string formatting) on every one of the 100k calls.
_SKILL_MATCH_IDS: Final[tuple[str, ...]] = tuple(
    f"{group}.competency" for group in _COMPETENCY_GROUPS
)
_CAREER_FIT_IDS: Final[tuple[str, ...]] = (
    "career.progression_quality",
    "pvs.product_density",
)
_EXPERIENCE_FIT_IDS: Final[tuple[str, ...]] = (
    "exp.in_band",
    "career.experience_authenticity",
)
_EDUCATION_FIT_IDS: Final[tuple[str, ...]] = (
    "edu.tier_score",
    "edu.field_relevance",
    "edu.timeline_valid",
)
_CREDIBILITY_IDS: Final[tuple[str, ...]] = (
    "cons.skill_role_coherence",
    "cons.title_role_coherence",
    "cons.summary_coherence",
)


# =========================================================================== #
# R0 — Artifact loading (ports: ArtifactStore, EmbeddingModel, VectorStore,   #
# Entropy)                                                                     #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class LoadedArtifacts:
    """R0's output: the typed policy/anchor/archetype objects + manifest dict."""

    artifacts: Mapping[str, object]
    manifest: Mapping[str, object]


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactContractError(
            f"expected a JSON object, got {type(value).__name__}"
        )
    return value


def _as_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ArtifactContractError(
            f"expected a JSON array, got {type(value).__name__}"
        )
    return value


def _as_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise ArtifactContractError(f"expected a number, got {type(value).__name__}")
    return float(value)


def _as_int(value: object) -> int:
    if not isinstance(value, int | float):
        raise ArtifactContractError(f"expected a number, got {type(value).__name__}")
    return int(value)


def _as_str(value: object) -> str:
    if not isinstance(value, str):
        raise ArtifactContractError(f"expected a string, got {type(value).__name__}")
    return value


def r0_load(
    *,
    artifact_store: ArtifactStorePort,
    embedding_model: EmbeddingModelPort,
    vector_store: SemanticVectorStorePort,
    entropy: OnlineEntropyPort,
    config: OnlineRunConfig,
) -> LoadedArtifacts:
    """Load + verify every artifact; bind the typed policy objects engines need.

    Raises:
        ManifestError: the manifest fails its self-hash check.
        ArtifactContractError: a required key is missing, an artifact's
            sha256 mismatches, or anchor/centroid dims disagree with the
            vector store's dim.
    """
    _ = embedding_model, entropy, config  # bound for parity with the port surface
    artifact_store.verify_all()
    manifest = artifact_store.manifest()
    manifest_dict: dict[str, object] = {
        "manifest_schema_version": manifest.manifest_schema_version,
        "layout_version": manifest.layout_version,
        "manifest_sha256": manifest.manifest_sha256,
        "embedding_model_id": manifest.embedding.model_id,
        "embedding_dim": manifest.embedding.dim,
    }

    # "scoring_weights" is registered as YAML (block-style, from O9's
    # emit_yaml), not JSON — load_text + yaml.safe_load, not load_json.
    weights_text = artifact_store.load_text(ArtifactKey("scoring_weights"))
    weights_json = _as_mapping(yaml.safe_load(weights_text))
    weights_map = _as_mapping(weights_json["weights"])
    weights = ScoringWeights(
        weights={ScoreComponent(k): _as_float(v) for k, v in weights_map.items()},
        # The real artifact has no "schema_version" key (only O9's own
        # "layout_version"); fall back to a fixed literal rather than KeyError.
        schema_version=_as_str(weights_json.get("layout_version", "1.0")),
    )
    scoring_policy = ScoringPolicy(
        floor=config.floor_sentinel,
        neutral_prior=_as_float(weights_json.get("neutral_prior", 0.5)),
    )

    # ``integrity_thresholds`` (real, from O3/O12) carries ``honeypot_threshold``
    # and ``risk_weights`` (-> flag_weights) but no per-flag severity and none of
    # the four online re-detection tolerances below — those are IntegrityEngine's
    # own re-derivation constants (it independently re-runs all seven rules per
    # candidate; it does not replay O3's bulk findings). Severity instead comes
    # from the real ``integrity_rules`` artifact, which does carry it per flag.
    thr = artifact_store.load_json(ArtifactKey("integrity_thresholds"))
    flag_weights = _as_mapping(thr.get("risk_weights", {}))
    rules_catalog = artifact_store.load_json(ArtifactKey("integrity_rules"))
    flag_severity: dict[IntegrityFlag, Severity] = {}
    for rule in _as_sequence(rules_catalog.get("rules", [])):
        rule_obj = _as_mapping(rule)
        flag_severity[IntegrityFlag(_as_str(rule_obj["code"]))] = Severity(
            _as_str(rule_obj["severity"])
        )
    integrity_thresholds = IntegrityThresholds(
        honeypot_threshold=_as_float(thr["honeypot_threshold"]),
        flag_severity=flag_severity,
        flag_weights={IntegrityFlag(k): _as_float(v) for k, v in flag_weights.items()},
        # Locked online re-detection tolerances (Repository Layout "expert
        # heuristics locked in" — no gold-labeled data exists to calibrate
        # these; chosen consistent with the offline detectors' own constants
        # where one exists, e.g. honeypot_discovery.py's expert-zero-usage
        # minimum count of 5).
        tolerance_experience_years=2.0,
        duration_date_tolerance_months=1.0,
        expert_zero_usage_min_count=5,
        experience_predates_tolerance_years=3,
    )

    # ``eligibility_rules`` (real, from O6) is YAML, and carries only human
    # descriptions per code (no calibrated numeric thresholds — O6 never
    # calibrates them, it authors rule *shape*). EligibilityEngine's thresholds
    # are therefore the locked defaults in ``config.schema.default_eligibility_rules``
    # (also used by O13a, so both paths agree by construction), not loaded from
    # the artifact; several are pinned by the EligibilityCode names themselves
    # (18-month / 5-year).
    _ = artifact_store.load_text(ArtifactKey("eligibility_rules"))  # presence check
    eligibility_rules = default_eligibility_rules()

    # ``behavioral_weights`` (real, from O11) is calibrated as monotone curves
    # keyed by {availability, hiring_probability_proxy, confidence_regression} —
    # not the 5 fixed families (availability/responsiveness/engagement/
    # reliability/verification) BehavioralEngine combines, so family_weights
    # cannot be derived from it. Bounds and the neutral base translate directly:
    # ``bounds.{lower,upper}`` are exactly ``m_min``/``m_max``; the pre-affine
    # base that reproduces the real artifact's ``neutral_multiplier`` is solved
    # from the engine's own affine map.
    beh = artifact_store.load_json(ArtifactKey("behavioral_weights"))
    bounds = _as_mapping(beh["bounds"])
    m_min = _as_float(bounds["lower"])
    # Clamped to 1.0: a multiplier only ever dampens relevance, never creates
    # it (domain.scoring.MULTIPLIER_MAX / ScoreBreakdown.behavioral_multiplier
    # both enforce <= 1.0 regardless), but the real calibration's upper bound
    # (1.25) predates that invariant and would fail it untouched.
    m_max = min(_as_float(bounds["upper"]), 1.0)
    neutral_multiplier = _as_float(beh.get("neutral_multiplier", 1.0))
    span = m_max - m_min
    unknown_neutral_base = (
        0.5 if span <= 0.0 else clamp_unit((neutral_multiplier - m_min) / span)
    )
    # Family weights are tilted toward availability/responsiveness (JD +
    # redrob_signals_doc: "a perfect-on-paper candidate who hasn't logged in
    # for 6 months and has a 5% recruiter response rate is, for hiring
    # purposes, not actually available. Down-weight them appropriately.").
    # Uniform weighting let a catastrophic response-rate/availability reading
    # get diluted by three healthy-looking secondary families before the
    # affine map ever saw it; this re-balances without zeroing any family.
    behavioral_policy = BehavioralPolicy(
        family_weights={
            "availability": 1.5,
            "responsiveness": 1.5,
            "engagement": 0.75,
            "reliability": 1.0,
            "verification": 0.5,
        },
        unknown_neutral_base=unknown_neutral_base,
        m_min=m_min,
        m_max=m_max,
    )

    # No offline stage calibrates logistics at all (no "logistics_weights"
    # artifact exists in the registry). Locked defaults below replace the
    # previous m_min=m_max=1.0 no-op, which made the affine map a no-op so
    # notice period, location, and salary inversion never affected score
    # regardless of how poorly a candidate banded. ``notice_fit_factor``
    # reuses the same calibration points as ``features.geography``'s
    # ``_NOTICE_FIT_SCORE`` for consistency. Location/salary factors are new,
    # conservative locked defaults: never above 1.0 (logistics only dampens,
    # per Domain §G.2/§G.8), and salary inversion stays a small soft dampen
    # (common in the pool, never a near-honeypot penalty).
    logistics_policy = LogisticsPolicy(
        location_fit_factor={
            LocationFit.PREFERRED_HUB: 1.0,
            LocationFit.INDIA_RELOCATABLE: 1.0,
            LocationFit.INDIA_NON_RELOCATABLE: 0.8,
            LocationFit.OUTSIDE_INDIA_NO_SPONSOR: 0.55,
        },
        location_default_factor=1.0,
        notice_fit_factor={
            NoticeFit.SUB_30_IDEAL: 1.0,
            NoticeFit.BUYOUTABLE: 0.6,
            NoticeFit.OVER_30_HIGHER_BAR: 0.25,
        },
        notice_default_factor=1.0,
        work_mode_weight=0.0,
        salary_inversion_factor=0.9,
        m_min=0.5,
        m_max=1.0,
    )

    dim = vector_store.dim
    anchor_set = _load_anchor_set(artifact_store, dim=dim)
    archetype_space = _load_archetype_space(artifact_store, dim=dim)

    artifacts: dict[str, object] = {
        "scoring_weights": weights,
        "scoring_policy": scoring_policy,
        "integrity_thresholds": integrity_thresholds,
        "eligibility_rules": eligibility_rules,
        "behavioral_policy": behavioral_policy,
        "logistics_policy": logistics_policy,
        "anchor_set": anchor_set,
        "archetype_space": archetype_space,
        "feature_registry": FEATURE_REGISTRY,
    }
    return LoadedArtifacts(artifacts=artifacts, manifest=manifest_dict)


def _load_anchor_set(artifact_store: ArtifactStorePort, *, dim: int) -> AnchorSet:
    """Build the ``AnchorSet`` from the real ``jd_concepts`` + ``anchor_vectors``.

    ``jd_concepts.json`` carries each anchor's id/polarity/text but no vector;
    the vectors live separately in ``anchor_vectors.npy``, one row per anchor in
    *global* (both polarities combined) ascending-id order — exactly the order
    O13b (``AnchorEmbeddingStage``) writes them in (``sorted(anchors)`` over all
    ids). Re-deriving that same global sort here is what lines a anchor id back
    up with its row.
    """
    jd_concepts = artifact_store.load_json(ArtifactKey("jd_concepts"))
    raw_anchors = [_as_mapping(a) for a in _as_sequence(jd_concepts.get("anchors", []))]
    by_id = {_as_str(a["id"]): _as_str(a["polarity"]) for a in raw_anchors}
    ordered_ids = sorted(by_id)

    vectors = artifact_store.load_npy(ArtifactKey("anchor_vectors"))
    if vectors.shape[0] != len(ordered_ids):
        raise ArtifactContractError(
            f"anchor_vectors row count {vectors.shape[0]} != "
            f"jd_concepts anchor count {len(ordered_ids)}"
        )
    if vectors.shape[0] and vectors.shape[1] != dim:
        raise ArtifactContractError(
            f"anchor vector dim {vectors.shape[1]} != embedding dim {dim}"
        )

    def _side(polarity: str) -> tuple[tuple[AnchorId, ...], npt.NDArray[np.float32]]:
        rows = [i for i, aid in enumerate(ordered_ids) if by_id[aid] == polarity]
        ids = tuple(AnchorId(ordered_ids[i]) for i in rows)
        matrix = (
            np.ascontiguousarray(vectors[rows])
            if rows
            else np.zeros((0, dim), dtype=np.float32)
        )
        return ids, matrix

    positive_ids, positive_matrix = _side("positive")
    negative_ids, negative_matrix = _side("negative")
    return AnchorSet(
        positive_ids=positive_ids,
        positive_matrix=positive_matrix,
        negative_ids=negative_ids,
        negative_matrix=negative_matrix,
    )


def _load_archetype_space(
    artifact_store: ArtifactStorePort, *, dim: int
) -> ArchetypeSpace:
    """Build the ``ArchetypeSpace`` from the real ``archetypes`` + ``centroids``.

    ``archetypes.json``'s ``archetypes`` map is keyed by the archetype's id as a
    string, and that key *is* its row index into ``centroids.npy`` (O7 writes
    ``centroids[order]`` and the catalog by the same ``order``-derived id).
    """
    catalog = _as_mapping(
        artifact_store.load_json(ArtifactKey("archetypes")).get("archetypes", {})
    )
    matrix = artifact_store.load_npy(ArtifactKey("centroids"))
    if matrix.shape[0] != len(catalog):
        raise ArtifactContractError(
            f"centroids row count {matrix.shape[0]} != "
            f"archetype catalog size {len(catalog)}"
        )
    if matrix.shape[0] and matrix.shape[1] != dim:
        raise ArtifactContractError(
            f"centroid vector dim {matrix.shape[1]} != embedding dim {dim}"
        )

    ids: list[ArchetypeId] = []
    target: set[ArchetypeId] = set()
    labels: dict[ArchetypeId, str] = {}
    for key in sorted(catalog, key=int):
        entry = _as_mapping(catalog[key])
        aid = ArchetypeId(_as_int(entry["archetype_id"]))
        ids.append(aid)
        labels[aid] = _as_str(entry["label"])
        if entry.get("is_target_archetype") is True:
            target.add(aid)

    return ArchetypeSpace(
        centroids=matrix,
        archetype_ids=tuple(ids),
        target_archetypes=frozenset(target),
        labels=labels,
    )


# =========================================================================== #
# R1 — Candidate ingestion (port: CandidateSource)                            #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class IngestedCandidate:
    """One PARSED-stage candidate: its raw record + initial representation."""

    candidate_id: CandidateId
    raw: RawCandidate
    representation: CandidateRepresentation


def r1_ingest(ctx: OnlineRunContext) -> list[IngestedCandidate]:
    """Stream + schema-validate every candidate; mint identity + provenance.

    Every candidate's ``RawCandidate`` is inlined (a deliberate simplification
    for test-scale pools, see the module docstring). Per the default abort
    policy, a malformed/schema-invalid record aborts the run.

    Raises:
        CandidateSourceError: the source cannot be opened, or a record is
            malformed/schema-invalid under the abort policy.
    """
    out: list[IngestedCandidate] = []
    seen: set[CandidateId] = set()
    for record in ctx.candidate_source.stream():
        if isinstance(record, SourceMalformed):
            if ctx.config.abort_on_malformed:
                raise CandidateSourceError(
                    f"malformed record at line {record.line_no}: {record.error}"
                )
            continue
        try:
            raw = validate_raw(record.raw)
        except SchemaError:
            if ctx.config.abort_on_malformed:
                raise
            continue
        if raw.candidate_id in seen:
            if ctx.config.abort_on_malformed:
                raise CandidateSourceError(
                    f"duplicate candidate_id: {raw.candidate_id}"
                )
            continue
        seen.add(raw.candidate_id)
        identity = Identity(
            candidate_id=raw.candidate_id,
            anonymized_name=raw.profile.anonymized_name,
            provenance=ProvenanceHandle(
                candidate_id=raw.candidate_id, inline=raw, source_index=None
            ),
        )
        out.append(
            IngestedCandidate(
                candidate_id=raw.candidate_id,
                raw=raw,
                representation=CandidateRepresentation.parsed(identity),
            )
        )
    return out


# =========================================================================== #
# R2 — Feature extraction (pure)                                              #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class FeaturedSet:
    """The bulk featured population: (N,D) CQV + per-candidate reps (FEATURED)."""

    candidates: tuple[IngestedCandidate, ...]
    cqv: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    representations: tuple[CandidateRepresentation, ...]
    #: ids already certain to be hard-blocked on career/credibility alone
    #: (``EligibilityEngine.evaluate_structural``) -> their fired findings.
    #: R3 reads this to skip the vector-store lookup/onnx fallback for them;
    #: R4 reads it to build their ``EligibilityReport`` without a semantic
    #: profile. Absent entries simply weren't structurally blocked yet.
    structural_findings: Mapping[CandidateId, tuple[EligibilityFinding, ...]]


def r2_features(
    ctx: OnlineRunContext, ingested: Sequence[IngestedCandidate]
) -> FeaturedSet:
    """Extract every structural + behavioral feature into the bulk ``(N,D)`` CQV.

    Raises:
        CQVInvariantError: a registry/extractor contract breach (fatal).
    """
    registry = _registry(ctx)
    eligibility_engine = EligibilityEngine(rules=_eligibility_rules(ctx))
    n = len(ingested)
    cqv = np.zeros((n, registry.dim), dtype=np.float32)
    confidence = np.zeros((n, len(registry.groups)), dtype=np.float32)
    reps: list[CandidateRepresentation] = []
    structural_findings: dict[CandidateId, tuple[EligibilityFinding, ...]] = {}

    for i, cand in enumerate(ingested):
        # ``extract_row`` builds the placeholder (semantic={}) cell map itself
        # internally — R3's ``fold_semantic`` is what needs the *resolved*
        # cell map, and it builds + returns that one. Building a placeholder
        # cell map here too would just be the same ``build_cells`` call run
        # twice with identical arguments, discarded either way.
        row, conf_row = extract_row(cand.raw, registry, as_of=ctx.as_of)
        cqv[i] = row
        confidence[i] = conf_row
        career = build_career_profile(cand.raw, as_of=ctx.as_of)
        credibility = build_credibility_profile(cand.raw)
        findings = eligibility_engine.evaluate_structural(
            career=career, credibility=credibility
        )
        if findings:
            structural_findings[cand.candidate_id] = findings
        reps.append(
            cand.representation.with_features(
                career=career,
                credibility=credibility,
                behavioral=build_behavioral_profile(cand.raw, as_of=ctx.as_of),
                logistics=build_logistics_profile(cand.raw),
            )
        )
    return FeaturedSet(
        candidates=tuple(ingested),
        cqv=cqv,
        confidence=confidence,
        representations=tuple(reps),
        structural_findings=structural_findings,
    )


# =========================================================================== #
# R3 — Semantic hydration (ports: SemanticVectorStore, EmbeddingModel)        #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class SituatedSet:
    """The bulk situated population: semantic-folded CQV + reps (SITUATED)."""

    candidates: tuple[IngestedCandidate, ...]
    cqv: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    representations: tuple[CandidateRepresentation, ...]
    cells: tuple[Mapping[str, FeatureCell], ...]
    #: carried forward from ``FeaturedSet`` (see its docstring); R4 reads it.
    structural_findings: Mapping[CandidateId, tuple[EligibilityFinding, ...]]


def r3_semantic(ctx: OnlineRunContext, featured: FeaturedSet) -> SituatedSet:
    """Hydrate semantics by lookup (onnx fallback on a store miss); fold into CQV.

    Candidates already hard-blocked by ``featured.structural_findings`` (career/
    credibility alone) never reach the vector store or the onnx fallback —
    asking about them would be a guaranteed miss for no decision-relevant
    gain, since their eligibility is already settled. They get a deterministic
    all-zero placeholder vector instead, run through the same
    ``SemanticEngine.profile_for`` as everyone else purely to satisfy the
    representation's FEATURED->SITUATED stage invariant (Domain
    ``CandidateRepresentation``); R4 ignores this placeholder profile for them
    and builds their ``EligibilityReport`` from the structural findings alone.
    """
    registry = _registry(ctx)
    anchor_set = _anchor_set(ctx)
    archetype_space = _archetype_space(ctx)
    engine = SemanticEngine(
        store=ctx.vector_store,
        embedder=ctx.embedding_model,
        anchors=anchor_set,
        archetypes=archetype_space,
    )

    skip_ids = frozenset(featured.structural_findings)
    embed_candidates = [
        cand for cand in featured.candidates if cand.candidate_id not in skip_ids
    ]
    ids = [cand.candidate_id for cand in embed_candidates]
    documents = {
        cand.candidate_id: _compose_fallback_document(cand.raw)
        for cand in embed_candidates
    }
    try:
        matrix, _misses = (
            engine.resolve_vectors(ids, documents)
            if ids
            else (np.zeros((0, ctx.vector_store.dim), dtype=np.float32), ())
        )
    except EmbeddingError as exc:
        raise VectorStoreError(f"semantic fallback encode failed: {exc}") from exc
    vector_by_id: dict[CandidateId, npt.NDArray[np.float32]] = dict(
        zip(ids, matrix, strict=True)
    )
    placeholder_vector = np.zeros(ctx.vector_store.dim, dtype=np.float32)

    cqv = featured.cqv.copy()
    confidence = featured.confidence.copy()
    reps: list[CandidateRepresentation] = []
    cells_list: list[Mapping[str, FeatureCell]] = []

    for i, cand in enumerate(featured.candidates):
        vector = vector_by_id.get(cand.candidate_id, placeholder_vector)
        semantic, archetype = engine.profile_for(vector, row_index=i)
        similarity_map: Mapping[str, Similarity] = {
            str(anchor_id): sim
            for anchor_id, sim in semantic.anchor_similarities.items()
        }
        row = cqv[i]
        conf_row = confidence[i]
        # ``fold_semantic`` already builds the resolved cell map internally to
        # fold it into the row/confidence arrays — reuse that return instead
        # of calling ``build_cells`` a second time with the same arguments.
        full_cells = fold_semantic(
            row, conf_row, cand.raw, registry, as_of=ctx.as_of, semantic=similarity_map
        )
        cells_list.append(full_cells)
        reps.append(
            featured.representations[i].with_semantic(
                semantic=semantic, archetype=archetype
            )
        )

    if not np.all(np.isfinite(cqv)):
        raise CQVInvariantError("semantic fold produced a non-finite CQV cell")

    return SituatedSet(
        candidates=featured.candidates,
        cqv=cqv,
        confidence=confidence,
        representations=tuple(reps),
        cells=tuple(cells_list),
        structural_findings=featured.structural_findings,
    )


def _compose_fallback_document(raw: RawCandidate) -> str:
    """A simple, deterministic text blob for the onnx encode-fallback path."""
    parts = [
        raw.profile.headline,
        raw.profile.summary,
        raw.profile.current_title,
        raw.profile.current_company,
        " ".join(skill.name for skill in raw.skills),
    ]
    return "\n".join(part for part in parts if part)


# =========================================================================== #
# R4 — Gates & eligibility (pure; verdicts are data)                          #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class GatedSet:
    """The bulk gated population: gate reports attached + floor mask (GATED)."""

    candidates: tuple[IngestedCandidate, ...]
    cqv: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    representations: tuple[CandidateRepresentation, ...]
    cells: tuple[Mapping[str, FeatureCell], ...]
    floor_mask: npt.NDArray[np.bool_]


def r4_gates(ctx: OnlineRunContext, situated: SituatedSet) -> GatedSet:
    """Run integrity + eligibility gates; build the floor mask (data, never raises).

    Candidates already hard-blocked at R2/R3 on career/credibility alone get
    their ``EligibilityReport`` built directly from those structural findings
    instead of the full ``evaluate()`` -- their R3 semantic profile is a
    zero-vector placeholder (no real document was ever embedded for them), so
    asking the semantic-dependent detectors about it would report a finding
    "evidenced" by a fact that isn't true of the candidate.
    """
    integrity_engine = IntegrityEngine(thresholds=_integrity_thresholds(ctx))
    eligibility_engine = EligibilityEngine(rules=_eligibility_rules(ctx))

    reps: list[CandidateRepresentation] = []
    floor = np.zeros(len(situated.candidates), dtype=np.bool_)

    for i, cand in enumerate(situated.candidates):
        rep = situated.representations[i]
        career = rep.require_career()
        credibility = rep.require_credibility()
        logistics = rep.require_logistics()

        integrity_report: IntegrityReport = integrity_engine.evaluate(career, cand.raw)
        structural = situated.structural_findings.get(cand.candidate_id)
        eligibility_report: EligibilityReport
        if structural is not None:
            eligibility_report = EligibilityReport(
                hard_blocks=structural,
                soft_penalties=(),
                is_eligible=False,
                gates_passed=frozenset(),
            )
        else:
            semantic = rep.require_semantic()
            eligibility_report = eligibility_engine.evaluate(
                career=career,
                credibility=credibility,
                semantic=semantic,
                logistics=logistics,
                jd=_JD_SPEC,
                skill_match=_skill_match_value(
                    situated.cells[i],
                    breadth_exponent=_SKILL_MATCH_GATE_BREADTH_EXPONENT,
                ),
            )
        floor[i] = integrity_report.is_honeypot or not eligibility_report.is_eligible
        reps.append(
            rep.with_gates(integrity=integrity_report, eligibility=eligibility_report)
        )

    return GatedSet(
        candidates=situated.candidates,
        cqv=situated.cqv,
        confidence=situated.confidence,
        representations=tuple(reps),
        cells=situated.cells,
        floor_mask=floor,
    )


# =========================================================================== #
# R5 — Scoring (locked weights, gates, bounded multipliers; no calibration)   #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class ScoredSet:
    """R5's output: every candidate's ``ScoredCandidate`` + the carried-forward
    gated population (for R6 ranking and R7 reasoning)."""

    scored: tuple[ScoredCandidate, ...]
    base: GatedSet


def r5_score(ctx: OnlineRunContext, gated: GatedSet) -> ScoredSet:
    """Score every candidate via the real ``ScoringEngine`` (locked weights/policy)."""
    weights = _scoring_weights(ctx)
    policy = _scoring_policy(ctx)
    behavioral_policy = _behavioral_policy(ctx)
    logistics_policy = _logistics_policy(ctx)

    scoring_engine = ScoringEngine(weights=weights, policy=policy)
    behavioral_engine = BehavioralEngine(policy=behavioral_policy)
    logistics_engine = LogisticsEngine(policy=logistics_policy)

    # Batched once over the full (N, G) confidence matrix instead of N separate
    # ``np.average`` calls in the loop below -- same reduction
    # (``(row * weights).sum() / weights.sum()``), just amortized over the whole
    # population in a single vectorized pass.
    confidence_weighted_avg = (
        gated.confidence.astype(np.float64) * _GROUP_CONFIDENCE_WEIGHTS
    ).sum(axis=1) / _GROUP_CONFIDENCE_WEIGHTS_SUM

    scored: list[ScoredCandidate] = []
    for i, cand in enumerate(gated.candidates):
        rep = gated.representations[i]
        cells = gated.cells[i]
        components = _score_components(cells, rep.semantic, rep.archetype)
        integrity = rep.require_integrity()
        eligibility = rep.require_eligibility()

        behavioral_multiplier = behavioral_engine.multiplier(
            rep.require_behavioral(), as_of=ctx.as_of
        )
        logistics_multiplier = logistics_engine.multiplier(rep.require_logistics())
        archetype_adjustment = _archetype_adjustment(rep.archetype)
        confidence = UnitScore(float(confidence_weighted_avg[i]))

        scored.append(
            scoring_engine.score(
                candidate_id=cand.candidate_id,
                components=components,
                integrity=integrity,
                eligibility=eligibility,
                behavioral_multiplier=behavioral_multiplier,
                logistics_multiplier=logistics_multiplier,
                archetype_adjustment=archetype_adjustment,
                confidence=confidence,
            )
        )
    return ScoredSet(scored=tuple(scored), base=gated)


_SKILL_MATCH_TOP_K: Final[int] = 3
# Floor below which a top-K competency value is treated as dataset noise
# rather than a genuine corroborated area -- see ``_skill_match_value``.
_SKILL_MATCH_BREADTH_FLOOR: Final[float] = 0.02
# Breadth-fraction exponent for the *eligibility gate* call only (the score
# component keeps the linear default below). The gate answers a binary
# plausibility question ("any genuine domain relevance at all?") where a
# *two*-of-three-corroborated profile -- one real spike plus one secondary,
# usually-weaker hit riding the same single injected sentence/self-reported
# skill -- should not pass: a fabricated "Mechanical Engineer" and "Graphic
# Designer" profile in a full-pool rerun, both carrying an "MLOps"-family
# skill plus the dataset's one reused AI-buzzword sentence, landed at
# 0.0505-0.0509 against the 0.05 floor under linear breadth. Squaring leaves
# fully-corroborated (3-of-3) candidates exactly as before (1.0**2 == 1.0)
# while sharply discounting partial corroboration (2-of-3: 0.667 -> 0.44;
# 1-of-3: 0.33 -> 0.11) for the gate decision specifically.
#
# The score component must NOT use this exponent: squaring there punished
# candidates who were never eligibility-risks to begin with -- e.g. an "AI
# Research Engineer" with nlp=0.477 (a strong, clearly-corroborated signal)
# but a legitimately-zero third competency slot saw their SKILL_MATCH
# component crushed from 0.196 to 0.087 and fell out of the Top 100, purely
# for being a narrow specialist rather than for any plausibility concern.
# Linear breadth already fixed the gate's worst leaks (the literal QA/Civil/
# HR-Manager-titled cases); only the gate's last-mile edge cases need the
# stricter exponent.
_SKILL_MATCH_GATE_BREADTH_EXPONENT: Final[float] = 2.0


def _skill_match_value(
    cells: Mapping[str, FeatureCell], *, breadth_exponent: float = 1.0
) -> float:
    """Breadth-weighted mean of the candidate's top-``_SKILL_MATCH_TOP_K``
    (of nine) ``{group}.competency`` cells (Scoring's SKILL_MATCH raw).

    Shared by ``_score_components`` (the scored component, default linear
    breadth) and ``r4_gates`` (the eligibility gate input, which passes
    ``_SKILL_MATCH_GATE_BREADTH_EXPONENT``) -- both read the identical
    underlying top-K selection and noise floor, diverging only in how
    sharply partial breadth is discounted, since the two callers ask
    different questions (see ``_SKILL_MATCH_GATE_BREADTH_EXPONENT``'s
    comment).

    A flat mean over all nine retr/rank/recsys/ir/nlp/llm/mle/mlops/eval
    groups requires breadth no real specialist has: a full-pool audit of
    ``feature_snapshot.parquet`` found 98.5% of candidates have a nonzero
    competency cell in at most *one* of the nine groups, so the flat mean
    collapsed to ~0 for nearly everyone (max across the entire 100k pool was
    0.19) -- which made the eligibility gate's 0.05 floor hard-block 99.6% of
    the pool. Top-``_SKILL_MATCH_TOP_K`` rewards genuine corroborated depth in
    a candidate's strongest 2-3 areas -- matching the JD's "we don't care
    which retrieval/ranking tech, we care that you've operated it in
    production" framing -- without requiring unrealistic breadth across all
    nine specialties at once.

    Plain top-K averaging still has a gap the JD calls out by name: "a
    candidate who has all the AI keywords listed as skills but whose title
    is 'Marketing Manager' is not a fit." A spot audit of out-of-domain
    titles surfacing in the Top 100 (QA Engineer, Civil Engineer, HR
    Manager, ...) found each one's entire ``skill_match`` traced to a
    *single* one of the nine groups -- one self-reported buzzword skill
    (e.g. a bare "Embeddings"/"MLOps" entry with no role ever describing
    that work) or one injected AI-buzzword sentence riding inside an
    otherwise unrelated role description, with the other top-K slots at or
    near zero. Dividing that lone spike by ``_SKILL_MATCH_TOP_K`` dilutes it,
    but a single ~0.15-0.22 spike still clears the eligibility gate's 0.05
    floor on dilution alone. Genuinely domain-relevant titles never show
    this shape: their top-K slots are consistently *all* populated (real
    job-description text naturally touches several adjacent competency
    groups via shared vocabulary -- "embeddings" alone corroborates both
    ``ir`` and ``retr``), so weighting the mean by how many of the top-K
    slots clear a small noise floor leaves them untouched while collapsing
    the single-spike/partial-breadth pattern below the gate.
    """
    values = sorted(
        (cells[fid].value for fid in _SKILL_MATCH_IDS if fid in cells), reverse=True
    )
    if not values:
        return 0.0
    top = values[:_SKILL_MATCH_TOP_K]
    mean = math.fsum(top) / len(top)
    corroborated = sum(1 for v in top if v > _SKILL_MATCH_BREADTH_FLOOR)
    breadth = corroborated / len(top)
    return clamp_unit(mean * breadth**breadth_exponent)


def _agg(cells: Mapping[str, FeatureCell], ids: Sequence[str]) -> ComponentRaw:
    found = [cells[fid] for fid in ids if fid in cells]
    if not found:
        return (UnitScore(0.0), ())
    value = UnitScore(clamp_unit(math.fsum(c.value for c in found) / len(found)))
    evidence = tuple(c.evidence[0] for c in found)
    return (value, evidence)


def _score_components(
    cells: Mapping[str, FeatureCell],
    semantic: SemanticProfile | None,
    archetype: ArchetypeAssignment | None,
) -> dict[ScoreComponent, ComponentRaw]:
    """Project the CQV cells into the seven ``ScoreComponent`` raw values."""
    skill_match_value = _skill_match_value(cells)
    skill_match_evidence = tuple(
        cells[fid].evidence[0] for fid in _SKILL_MATCH_IDS if fid in cells
    )
    skill_match: ComponentRaw = (UnitScore(skill_match_value), skill_match_evidence)
    career_fit = _agg(cells, _CAREER_FIT_IDS)
    experience_fit = _agg(cells, _EXPERIENCE_FIT_IDS)
    education_fit = _agg(cells, _EDUCATION_FIT_IDS)
    credibility = _agg(cells, _CREDIBILITY_IDS)

    if semantic is not None:
        semantic_fit: ComponentRaw = (
            semantic.net_semantic_fit,
            (
                make_evidence(
                    EvidenceKind.DERIVED,
                    "semantic.net_semantic_fit",
                    float(semantic.net_semantic_fit),
                ),
            ),
        )
    else:
        semantic_fit = (UnitScore(0.0), ())

    if archetype is not None and archetype.is_target_archetype:
        archetype_fit: ComponentRaw = (
            archetype.membership_confidence,
            (
                make_evidence(
                    EvidenceKind.DERIVED,
                    "archetype.membership_confidence",
                    float(archetype.membership_confidence),
                ),
            ),
        )
    else:
        archetype_fit = (UnitScore(0.0), ())

    return {
        ScoreComponent.SKILL_MATCH: skill_match,
        ScoreComponent.SEMANTIC_FIT: semantic_fit,
        ScoreComponent.CAREER_FIT: career_fit,
        ScoreComponent.EXPERIENCE_FIT: experience_fit,
        ScoreComponent.EDUCATION_FIT: education_fit,
        ScoreComponent.CREDIBILITY: credibility,
        ScoreComponent.ARCHETYPE_FIT: archetype_fit,
    }


def _archetype_adjustment(archetype: ArchetypeAssignment | None) -> float:
    if archetype is None or not archetype.is_target_archetype:
        return 0.0
    return 0.05 * float(archetype.membership_confidence)


# =========================================================================== #
# R6 — Ranking (pure; the real RankingEngine)                                 #
# =========================================================================== #
def r6_rank(ctx: OnlineRunContext, scored: ScoredSet) -> Ranking:
    """Deterministic top-100 selection via the real ``RankingEngine``.

    Raises:
        RankingInvariantError: fewer than the ranking size's worth of
            candidates, or an invariant breach.
    """
    _ = ctx  # pure stage; no ports
    engine = RankingEngine(size=_RANKING_SIZE)
    return engine.rank(scored.scored)


# =========================================================================== #
# R7 — Reasoning (pure; the real ReasoningEngine; top-100, no reorder)        #
# =========================================================================== #
def r7_reason(ctx: OnlineRunContext, ranking: Ranking, gated: GatedSet) -> Ranking:
    """Attach evidence-grounded reasoning to every ranked candidate.

    Raises:
        ProvenanceError: a ranked candidate's representation is not hydrated.
    """
    _ = ctx
    by_id: dict[CandidateId, CandidateRepresentation] = {
        rep.candidate_id: rep for rep in gated.representations
    }
    missing = [
        ranked.candidate_id
        for ranked in ranking.ordered
        if ranked.candidate_id not in by_id
    ]
    if missing:
        raise ProvenanceError(f"representation not hydrated for: {missing[:5]}")
    engine = ReasoningEngine()
    return engine.explain(ranking, by_id)


# =========================================================================== #
# R8 — Submission generation (port: SubmissionSink)                           #
# =========================================================================== #
def r8_submit(ctx: OnlineRunContext, ranking: Ranking) -> SubmissionReceipt:
    """Validate (structural + Stage-4) then write the CSV via the real sink.

    Raises:
        SubmissionContractError: the ``ValidationEngine`` reports a HARD
            finding (no rejectable file is written).
    """
    validation_engine = ValidationEngine(expected_size=ranking.size)
    report = validation_engine.validate_ranking(ranking)
    if not report.is_valid:
        hard = [f for f in report.findings if f.severity is Severity.HARD]
        raise SubmissionContractError(
            f"validation failed with {len(hard)} hard finding(s): "
            f"{[f.code.value for f in hard][:5]}"
        )
    return ctx.submission_sink.write(ranking)


# =========================================================================== #
# R9 — Run report generation (port: RunReportSink)                            #
# =========================================================================== #
@final
@dataclass(frozen=True, slots=True)
class ReportOutcome:
    """R9's terminal result: where the submission/report landed + the metric
    the honeypot gate (§10) reads."""

    submission_path: str
    report_path: str
    honeypot_rate_top100: float


@final
@dataclass(frozen=True, slots=True)
class _Reproducible:
    code_version: str
    config_hash: str
    manifest_hash: str
    artifact_hashes: Mapping[str, str]
    input_file_sha256: str
    candidate_count: int
    output_sha256: str
    honeypot_count_top100: int
    honeypot_rate: float
    eligibility_summary: Mapping[str, int]
    score_distribution_digest: str


@final
@dataclass(frozen=True, slots=True)
class _Audit:
    run_id: str
    started_at: str
    ended_at: str
    host_label: str


@final
@dataclass(frozen=True, slots=True)
class _Budget:
    limit_seconds: float
    used_seconds: float
    within_budget: bool
    peak_rss_mb: float


@final
@dataclass(frozen=True, slots=True)
class _RunReport:
    reproducible: _Reproducible
    audit: _Audit
    timings: Mapping[str, float]
    budget: _Budget


def r9_report(
    ctx: OnlineRunContext,
    *,
    ranking: Ranking,
    scored: ScoredSet,
    gated: GatedSet,
    receipt: SubmissionReceipt,
    stage_timings_ms: Mapping[str, float],
) -> ReportOutcome:
    """Assemble + persist the deterministic run report; return the terminal paths."""
    honeypot_top100 = sum(
        1
        for ranked in ranking.ordered
        if not ranked.scored.breakdown.integrity_gate.passed
    )
    honeypot_rate = honeypot_top100 / ranking.size if ranking.size else 0.0

    eligibility_summary: dict[str, int] = {}
    for rep in gated.representations:
        eligibility = rep.eligibility
        if eligibility is None:
            continue
        for finding in (*eligibility.hard_blocks, *eligibility.soft_penalties):
            eligibility_summary[finding.code.value] = (
                eligibility_summary.get(finding.code.value, 0) + 1
            )

    scores_blob = json.dumps(
        sorted(round(float(sc.final_score), 6) for sc in scored.scored)
    ).encode("utf-8")
    score_digest = hashlib.sha256(scores_blob).hexdigest()

    used_seconds = sum(stage_timings_ms.values()) / 1000.0
    report = _RunReport(
        reproducible=_Reproducible(
            code_version="redstack-1.1-test",
            config_hash=hashlib.sha256(
                json.dumps(
                    {
                        "seed": ctx.config.seed,
                        "participant_id": ctx.config.participant_id,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            manifest_hash=ctx.manifest_hash,
            artifact_hashes=_manifest_string_fields(ctx.manifest),
            input_file_sha256=ctx.input_file_sha256,
            candidate_count=len(gated.candidates),
            output_sha256=receipt.output_sha256,
            honeypot_count_top100=honeypot_top100,
            honeypot_rate=honeypot_rate,
            eligibility_summary=eligibility_summary,
            score_distribution_digest=score_digest,
        ),
        audit=_Audit(
            run_id=hashlib.sha256(receipt.output_sha256.encode("utf-8")).hexdigest()[
                :12
            ],
            started_at="",
            ended_at="",
            host_label="test",
        ),
        timings=dict(stage_timings_ms),
        budget=_Budget(
            limit_seconds=ctx.config.budget_limit_seconds,
            used_seconds=used_seconds,
            within_budget=used_seconds <= ctx.config.budget_limit_seconds,
            peak_rss_mb=0.0,
        ),
    )
    report_receipt = ctx.run_report_sink.write(report)
    _ = report_receipt
    return ReportOutcome(
        submission_path=ctx.config.participant_id + ".csv",
        report_path="run_report.json",
        honeypot_rate_top100=honeypot_rate,
    )


def _manifest_string_fields(manifest: Mapping[str, object]) -> dict[str, str]:
    """Project the manifest dict's string-valued entries for ``artifact_hashes``."""
    return {k: str(v) for k, v in manifest.items() if isinstance(v, str)}


# --------------------------------------------------------------------------- #
# Shared artifact accessors (typed views over ``ctx.artifacts``).             #
# --------------------------------------------------------------------------- #
def _registry(ctx: OnlineRunContext) -> FeatureRegistry:
    registry = ctx.artifacts.get("feature_registry")
    if not isinstance(registry, FeatureRegistry):
        raise ArtifactContractError("artifacts missing 'feature_registry'")
    return registry


def _anchor_set(ctx: OnlineRunContext) -> AnchorSet:
    value = ctx.artifacts.get("anchor_set")
    if not isinstance(value, AnchorSet):
        raise ArtifactContractError("artifacts missing 'anchor_set'")
    return value


def _archetype_space(ctx: OnlineRunContext) -> ArchetypeSpace:
    value = ctx.artifacts.get("archetype_space")
    if not isinstance(value, ArchetypeSpace):
        raise ArtifactContractError("artifacts missing 'archetype_space'")
    return value


def _integrity_thresholds(ctx: OnlineRunContext) -> IntegrityThresholds:
    value = ctx.artifacts.get("integrity_thresholds")
    if not isinstance(value, IntegrityThresholds):
        raise ArtifactContractError("artifacts missing 'integrity_thresholds'")
    return value


def _eligibility_rules(ctx: OnlineRunContext) -> EligibilityRuleSet:
    value = ctx.artifacts.get("eligibility_rules")
    if not isinstance(value, EligibilityRuleSet):
        raise ArtifactContractError("artifacts missing 'eligibility_rules'")
    return value


def _behavioral_policy(ctx: OnlineRunContext) -> BehavioralPolicy:
    value = ctx.artifacts.get("behavioral_policy")
    if not isinstance(value, BehavioralPolicy):
        raise ArtifactContractError("artifacts missing 'behavioral_policy'")
    return value


def _logistics_policy(ctx: OnlineRunContext) -> LogisticsPolicy:
    value = ctx.artifacts.get("logistics_policy")
    if not isinstance(value, LogisticsPolicy):
        raise ArtifactContractError("artifacts missing 'logistics_policy'")
    return value


def _scoring_policy(ctx: OnlineRunContext) -> ScoringPolicy:
    value = ctx.artifacts.get("scoring_policy")
    if not isinstance(value, ScoringPolicy):
        raise ArtifactContractError("artifacts missing 'scoring_policy'")
    return value


def _scoring_weights(ctx: OnlineRunContext) -> ScoringWeights:
    value = ctx.artifacts.get("scoring_weights")
    if not isinstance(value, ScoringWeights):
        raise ArtifactContractError("artifacts missing 'scoring_weights'")
    return value
