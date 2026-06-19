
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType

from redstack.domain.candidate.quality import FeatureLayout, FeatureLayoutEntry
from redstack.domain.ids import FeatureIndex

# Major.minor.patch — order change ⇒ major bump.
LAYOUT_VERSION = "1.1.0"
# The CQV ``schema_version`` is pinned to the layout version (one shared pin,
# Repository §0/§6): a CQV produced under one layout cannot be scored under
# another.
SCHEMA_VERSION = LAYOUT_VERSION


class SourceSlice(str, Enum):
    """The ``CandidateRepresentation`` slice (or raw origin) a feature derives
    from. Recorded per entry for traceability; serialized by value."""

    IDENTITY = "identity"
    LOGISTICS = "logistics"
    CAREER = "career"
    EDUCATION = "education"
    CREDIBILITY = "credibility"
    SEMANTIC = "semantic"
    BEHAVIORAL = "behavioral"
    INTEGRITY = "integrity"
    DERIVED = "derived"


# Competency groups (Feature Layer groups 8–16) and their fixed suffix order.
_COMPETENCY_GROUPS: tuple[str, ...] = (
    "retr", "rank", "recsys", "ir", "nlp", "llm", "mle", "mlops", "eval",
)
# (suffix, source_slice) — ``.semantic`` comes from the semantic slice, the rest
# from the credibility (trust) slice.
_COMPETENCY_SUFFIXES: tuple[tuple[str, SourceSlice], ...] = (
    ("claimed", SourceSlice.CREDIBILITY),
    ("trust", SourceSlice.CREDIBILITY),
    ("in_career", SourceSlice.CREDIBILITY),
    ("semantic", SourceSlice.SEMANTIC),
    ("competency", SourceSlice.CREDIBILITY),
)

_BHV_NAMES: tuple[str, ...] = (
    "availability", "recruitability", "response_reliability", "interview_reliability",
    "market_demand", "market_momentum", "engagement_velocity", "candidate_temperature",
    "recruiter_attractiveness", "hiring_probability_proxy", "freshness", "trust",
    "signal_consistency", "behavioral_confidence", "behavioral_risk",
)

_CAREER_NAMES: tuple[str, ...] = (
    "progression_quality", "stability", "promotion_velocity", "title_inflation",
    "role_consistency", "experience_authenticity", "company_progression",
    "product_company_density", "consulting_density", "technical_depth",
    "hands_on_engineering", "research_only", "management_only", "production_exposure",
)

_HP_NAMES: tuple[str, ...] = (
    "timeline_impossible", "skill_time_contradiction", "employment_overlap",
    "title_seniority_anomaly", "education_career_anomaly", "salary_anomaly",
    "experience_inflation", "keyword_stuffing", "behavioral_inconsistency",
    "signal_impossibility", "identity_anomaly", "composite",
)

_JD_NAMES: tuple[str, ...] = (
    "retrieval_ranking", "production_ml", "product_company", "shipping_mentality",
    "eval_framework", "hybrid_retrieval", "keyword_only", "consulting_only",
    "title_chaser", "pure_researcher", "framework_enthusiast", "inactive",
)

_UNIT = (0.0, 1.0)
_YEARS = (0.0, 50.0)


def _build_spec() -> tuple[tuple[str, str, SourceSlice, float, float], ...]:
    """Build the master ordered spec: ``(group, name, source_slice, lo, hi)``.

    Ordering follows the dependency graph (Feature Layer Part 7): primitives →
    competency → supporting → logistics → engagement → behavioral → derived
    career → consistency/risk → honeypot → latents. Latents/composites trail
    their constituents so the fixed extraction order is topological.
    """
    rows: list[tuple[str, str, SourceSlice, float, float]] = []

    # 1. Identity (metadata; structural, no scoring weight but holds a slot).
    rows.append(("id", "is_valid_id", SourceSlice.IDENTITY, *_UNIT))
    # 2. Geography.
    for name in ("hub_match", "india_relocatable", "outside_india_no_sponsor"):
        rows.append(("geo", name, SourceSlice.LOGISTICS, *_UNIT))
    # 3. Experience.
    rows.append(("exp", "years", SourceSlice.CAREER, *_YEARS))
    rows.append(("exp", "in_band", SourceSlice.CAREER, *_UNIT))
    rows.append(("exp", "derived_vs_stated_gap", SourceSlice.CAREER, *_UNIT))
    # 4. Seniority.
    rows.append(("sen", "level", SourceSlice.CAREER, *_UNIT))
    rows.append(("sen", "title_vs_scope_gap", SourceSlice.CAREER, *_UNIT))
    # 5. Education.
    for name in ("tier_score", "field_relevance", "timeline_valid"):
        rows.append(("edu", name, SourceSlice.EDUCATION, *_UNIT))
    # 6. Company.
    for name in ("scale_progression", "industry_relevance"):
        rows.append(("co", name, SourceSlice.CAREER, *_UNIT))
    # 7. Product-vs-Service.
    for name in ("product_density", "consulting_density", "product_recent"):
        rows.append(("pvs", name, SourceSlice.CAREER, *_UNIT))
    # 8–16. Competency groups.
    for group in _COMPETENCY_GROUPS:
        for suffix, slice_ in _COMPETENCY_SUFFIXES:
            rows.append((group, suffix, slice_, *_UNIT))
    # 17. Open source.
    for name in ("activity", "has_external_validation"):
        rows.append(("oss", name, SourceSlice.CREDIBILITY, *_UNIT))
    # 18. Leadership.
    for name in ("scope", "management_only"):
        rows.append(("lead", name, SourceSlice.CAREER, *_UNIT))
    # 19. Startup fit.
    for name in ("small_co_experience", "shipping_signal"):
        rows.append(("startup", name, SourceSlice.CAREER, *_UNIT))
    # 20. Founding engineer.
    for name in ("ownership", "breadth"):
        rows.append(("found", name, SourceSlice.CAREER, *_UNIT))
    # 21. Availability.
    for name in ("open", "recency", "available"):
        rows.append(("avail", name, SourceSlice.BEHAVIORAL, *_UNIT))
    # 22. Engagement.
    for name in ("passive", "active", "network", "velocity"):
        rows.append(("eng", name, SourceSlice.BEHAVIORAL, *_UNIT))
    # 23. Responsiveness.
    for name in ("rate", "speed", "reliable"):
        rows.append(("resp", name, SourceSlice.BEHAVIORAL, *_UNIT))
    # 24. Salary alignment.
    for name in ("fit", "is_inverted"):
        rows.append(("sal", name, SourceSlice.LOGISTICS, *_UNIT))
    # 25. Relocation.
    for name in ("willing", "needed"):
        rows.append(("reloc", name, SourceSlice.LOGISTICS, *_UNIT))
    # 26. Notice period.
    for name in ("fit", "over_30"):
        rows.append(("notice", name, SourceSlice.LOGISTICS, *_UNIT))
    # 27. Behavioral composites.
    for name in _BHV_NAMES:
        rows.append(("bhv", name, SourceSlice.BEHAVIORAL, *_UNIT))
    # Part 3. Career intelligence.
    for name in _CAREER_NAMES:
        rows.append(("career", name, SourceSlice.CAREER, *_UNIT))
    # 29. Consistency.
    for name in ("title_role_coherence", "skill_role_coherence", "summary_coherence"):
        rows.append(("cons", name, SourceSlice.CREDIBILITY, *_UNIT))
    # 28. Risk.
    for name in ("uncertainty", "contradiction", "confidence"):
        rows.append(("risk", name, SourceSlice.INTEGRITY, *_UNIT))
    # 30. Honeypot detectors + composite.
    for name in _HP_NAMES:
        rows.append(("hp", name, SourceSlice.INTEGRITY, *_UNIT))
    # Part 2. JD latents (composite — trail every constituent).
    for name in _JD_NAMES:
        rows.append(("jd", name, SourceSlice.DERIVED, *_UNIT))

    return tuple(rows)


_SPEC = _build_spec()


def _build_layout() -> FeatureLayout:
    entries = tuple(
        FeatureLayoutEntry(
            name=f"{group}.{name}",
            index=FeatureIndex(position),
            source_slice=slice_.value,
            lower=lower,
            upper=upper,
        )
        for position, (group, name, slice_, lower, upper) in enumerate(_SPEC)
    )
    return FeatureLayout(entries=entries, layout_version=LAYOUT_VERSION)


# The frozen, validated layout object (contiguity / uniqueness enforced by the
# domain ``FeatureLayout`` validator at import time).
FEATURE_LAYOUT: FeatureLayout = _build_layout()

# Dimensionality of the CQV value vector.
DIM: int = FEATURE_LAYOUT.dim

# Ordered feature-id tuple, aligned 1:1 with the ``(N, D)`` value columns.
FEATURE_IDS: tuple[str, ...] = tuple(entry.name for entry in FEATURE_LAYOUT.entries)

# id → index (the binding the registry exposes; pinned here).
INDEX_OF: Mapping[str, FeatureIndex] = MappingProxyType(
    {entry.name: entry.index for entry in FEATURE_LAYOUT.entries}
)
ID_AT: Mapping[FeatureIndex, str] = MappingProxyType(
    {entry.index: entry.name for entry in FEATURE_LAYOUT.entries}
)


def _group_of(feature_id: str) -> str:
    return feature_id.split(".", 1)[0]


# Group order = first-appearance order in the value layout; the column order of
# the ``(N, num_groups)`` confidence matrix.
def _build_group_order() -> tuple[str, ...]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for feature_id in FEATURE_IDS:
        group = _group_of(feature_id)
        if group not in seen_set:
            seen_set.add(group)
            seen.append(group)
    return tuple(seen)


GROUP_ORDER: tuple[str, ...] = _build_group_order()
NUM_GROUPS: int = len(GROUP_ORDER)
GROUP_COLUMN: Mapping[str, int] = MappingProxyType(
    {group: column for column, group in enumerate(GROUP_ORDER)}
)


def _build_group_members() -> Mapping[str, tuple[str, ...]]:
    members: dict[str, list[str]] = {group: [] for group in GROUP_ORDER}
    for feature_id in FEATURE_IDS:
        members[_group_of(feature_id)].append(feature_id)
    return MappingProxyType({g: tuple(ids) for g, ids in members.items()})


# group → ordered feature ids in that group.
GROUP_MEMBERS: Mapping[str, tuple[str, ...]] = _build_group_members()


def index_of(feature_id: str) -> FeatureIndex:
    """Return the CQV index of ``feature_id``; raise ``KeyError`` if unknown."""
    return INDEX_OF[feature_id]


def group_of(feature_id: str) -> str:
    """Return the group prefix of a ``"<group>.<name>"`` feature id."""
    return _group_of(feature_id)


def group_column(group: str) -> int:
    """Return the confidence-matrix column index of ``group``."""
    return GROUP_COLUMN[group]


__all__: tuple[str, ...] = (
    "DIM",
    "FEATURE_IDS",
    "FEATURE_LAYOUT",
    "GROUP_COLUMN",
    "GROUP_MEMBERS",
    "GROUP_ORDER",
    "ID_AT",
    "INDEX_OF",
    "LAYOUT_VERSION",
    "NUM_GROUPS",
    "SCHEMA_VERSION",
    "SourceSlice",
    "group_column",
    "group_of",
    "index_of",
)