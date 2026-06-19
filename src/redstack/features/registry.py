
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.errors import ArtifactContractError
from redstack.domain.ids import FeatureIndex
from redstack.features.layout import (
    FEATURE_IDS,
    FEATURE_LAYOUT,
    GROUP_MEMBERS,
    GROUP_ORDER,
    LAYOUT_VERSION,
    SourceSlice,
)
from redstack.features.parsing import FeatureId, feature_id

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


class UnknownFeatureId(KeyError):
    """Lookup of a feature id absent from the registry/layout."""


class FeatureTier(str, Enum):
    """Priority tier (Part 10): A mission-critical … D nice-to-have."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class FeaturePolarity(str, Enum):
    """Whether a higher value is desirable, undesirable, or neutral."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FeatureDType(str, Enum):
    """CQV cell dtype. The value vector is uniformly float32 (flags are 0.0/1.0)."""

    FLOAT32 = "float32"


@final
class FeatureVersion(BaseModel):
    """Semantic version of one feature's extractor (Part 6 bump rules).

    Value-changing edit ⇒ minor bump; a layout/order change is a *global*
    ``layout_version`` major bump (owned by ``features.layout``).
    """

    model_config = _STRICT

    major: int = Field(ge=0)
    minor: int = Field(ge=0)
    patch: int = Field(ge=0)

    @property
    def semver(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)


@final
class FeatureSchema(BaseModel):
    """dtype / range / nullability contract per feature; enforced at extraction.

    The CQV value vector carries no nulls — ``UNKNOWN`` is encoded as a neutral
    value plus low confidence — so ``nullable`` is ``False`` for every value
    feature.
    """

    model_config = _STRICT

    feature_id: FeatureId
    dtype: FeatureDType
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    nullable: bool

    @model_validator(mode="after")
    def _bounds_ordered(self) -> FeatureSchema:
        if self.lower > self.upper:
            raise ValueError("FeatureSchema lower bound exceeds upper bound")
        return self


@final
class FeatureMetadata(BaseModel):
    """Human-facing metadata for reviewers + the O9 validation battery."""

    model_config = _STRICT

    doc: str = Field(min_length=1)
    expected_distribution: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    tier: FeatureTier
    polarity: FeaturePolarity


@final
class FeatureDefinition(BaseModel):
    """The unit of the taxonomy: a fully-specified feature contract."""

    model_config = _STRICT

    feature_id: FeatureId
    group: str = Field(min_length=1)
    index: FeatureIndex = Field(ge=0)
    source_slice: SourceSlice
    schema_: FeatureSchema = Field(alias="schema")
    dependencies: tuple[FeatureId, ...]
    extractor_ref: str = Field(min_length=1)
    version: FeatureVersion
    tier: FeatureTier
    polarity: FeaturePolarity
    metadata: FeatureMetadata

    @model_validator(mode="after")
    def _consistent(self) -> FeatureDefinition:
        if self.schema_.feature_id != self.feature_id:
            raise ValueError("FeatureDefinition.schema feature_id mismatch")
        if self.metadata.tier is not self.tier:
            raise ValueError("FeatureDefinition tier/metadata.tier mismatch")
        if self.metadata.polarity is not self.polarity:
            raise ValueError("FeatureDefinition polarity/metadata.polarity mismatch")
        if not self.feature_id.startswith(f"{self.group}."):
            raise ValueError("FeatureDefinition group does not prefix feature_id")
        return self


@final
class FeatureManifest(BaseModel):
    """Per-run manifest: ``layout_version`` + ordered ids + group map + hash."""

    model_config = _STRICT

    layout_version: str = Field(min_length=1)
    feature_ids: tuple[FeatureId, ...] = Field(min_length=1)
    groups: Mapping[str, tuple[FeatureId, ...]]
    registry_hash: str = Field(min_length=1)

    def matches(self, layout_version: str) -> bool:
        """True iff this manifest was built for ``layout_version``."""
        return self.layout_version == layout_version

    def require_match(self, layout_version: str) -> None:
        """Raise ``ArtifactContractError`` unless versions agree (Scoring/O8)."""
        if not self.matches(layout_version):
            raise ArtifactContractError(
                f"layout_version mismatch: manifest={self.layout_version!r} "
                f"expected={layout_version!r}"
            )


# --------------------------------------------------------------------------- #
# Metadata priors: tier (Part 10), polarity (Part 1–5), extractor refs (§8).   #
# --------------------------------------------------------------------------- #

_EXTRACTOR_REF: Mapping[str, str] = MappingProxyType(
    {
        "id": "features.parsing",
        "geo": "features.geography.extract_geography",
        "reloc": "features.geography.extract_geography",
        "notice": "features.geography.extract_geography",
        "sal": "features.geography.extract_geography",
        "edu": "features.education.extract_education",
        "exp": "features.career",
        "sen": "features.career",
        "co": "features.career",
        "pvs": "features.career",
        "career": "features.career",
        "lead": "features.career",
        "startup": "features.career",
        "found": "features.career",
        "retr": "features.skills",
        "rank": "features.skills",
        "recsys": "features.skills",
        "ir": "features.skills",
        "nlp": "features.skills",
        "llm": "features.skills",
        "mle": "features.skills",
        "mlops": "features.skills",
        "eval": "features.skills",
        "oss": "features.skills",
        "cons": "features.skills",
        "avail": "features.signals",
        "eng": "features.signals",
        "resp": "features.signals",
        "bhv": "features.signals",
        "risk": "features.signals",
        "hp": "features.honeypot",
        "jd": "features.latents",
    }
)

# Features whose higher value is undesirable.
_NEGATIVE_IDS: frozenset[str] = frozenset(
    {
        "geo.outside_india_no_sponsor",
        "exp.derived_vs_stated_gap",
        "sen.title_vs_scope_gap",
        "pvs.consulting_density",
        "lead.management_only",
        "notice.over_30",
        "sal.is_inverted",
        "career.title_inflation",
        "career.consulting_density",
        "career.research_only",
        "career.management_only",
        "risk.uncertainty",
        "risk.contradiction",
        "bhv.behavioral_risk",
        "jd.keyword_only",
        "jd.consulting_only",
        "jd.title_chaser",
        "jd.pure_researcher",
        "jd.framework_enthusiast",
        "jd.inactive",
    }
    | {f"hp.{name}" for name in (
        "timeline_impossible", "skill_time_contradiction", "employment_overlap",
        "title_seniority_anomaly", "education_career_anomaly", "salary_anomaly",
        "experience_inflation", "keyword_stuffing", "behavioral_inconsistency",
        "signal_impossibility", "identity_anomaly", "composite",
    )}
)

# Features that are contextual / not directionally scored.
_NEUTRAL_IDS: frozenset[str] = frozenset(
    {"id.is_valid_id", "reloc.needed"}
    | {f"{group}.claimed" for group in (
        "retr", "rank", "recsys", "ir", "nlp", "llm", "mle", "mlops", "eval",
    )}
)

# Default tier by group; per-id overrides applied on top (Part 10).
_GROUP_TIER: Mapping[str, FeatureTier] = MappingProxyType(
    {
        "id": FeatureTier.C, "geo": FeatureTier.C, "exp": FeatureTier.C,
        "sen": FeatureTier.C, "edu": FeatureTier.C, "co": FeatureTier.C,
        "pvs": FeatureTier.A, "oss": FeatureTier.D, "lead": FeatureTier.D,
        "startup": FeatureTier.D, "found": FeatureTier.D, "avail": FeatureTier.C,
        "eng": FeatureTier.C, "resp": FeatureTier.C, "sal": FeatureTier.C,
        "reloc": FeatureTier.C, "notice": FeatureTier.C, "bhv": FeatureTier.C,
        "career": FeatureTier.C, "cons": FeatureTier.C, "risk": FeatureTier.C,
        "hp": FeatureTier.A, "jd": FeatureTier.A,
        "retr": FeatureTier.B, "rank": FeatureTier.B, "recsys": FeatureTier.B,
        "ir": FeatureTier.B, "nlp": FeatureTier.B, "llm": FeatureTier.B,
        "mle": FeatureTier.B, "mlops": FeatureTier.B, "eval": FeatureTier.B,
    }
)

_TIER_OVERRIDE: Mapping[str, FeatureTier] = MappingProxyType(
    {
        # pvs
        "pvs.product_density": FeatureTier.A,
        "pvs.product_recent": FeatureTier.A,
        "pvs.consulting_density": FeatureTier.B,
        # experience
        "exp.in_band": FeatureTier.B,
        # career intelligence (Part 10 Tier B + the JD-critical density)
        "career.product_company_density": FeatureTier.A,
        "career.production_exposure": FeatureTier.B,
        "career.technical_depth": FeatureTier.B,
        "career.hands_on_engineering": FeatureTier.B,
        "career.title_inflation": FeatureTier.B,
        "career.consulting_density": FeatureTier.B,
        # behavioral multiplier inputs
        "bhv.availability": FeatureTier.B,
        "bhv.hiring_probability_proxy": FeatureTier.B,
        "bhv.market_momentum": FeatureTier.D,
        "bhv.engagement_velocity": FeatureTier.D,
        # honeypot soft / nuance
        "hp.salary_anomaly": FeatureTier.C,
        "hp.behavioral_inconsistency": FeatureTier.B,
        # jd negatives (Part 10 Tier B)
        "jd.consulting_only": FeatureTier.B,
        "jd.pure_researcher": FeatureTier.B,
        "jd.framework_enthusiast": FeatureTier.B,
        "jd.title_chaser": FeatureTier.B,
        "jd.inactive": FeatureTier.B,
        "jd.shipping_mentality": FeatureTier.B,
    }
)

# Competency suffix → tier (anti-stuffer trust/competency are mission-critical;
# raw claimed is near-zero weight).
_COMPETENCY_SUFFIX_TIER: Mapping[str, FeatureTier] = MappingProxyType(
    {
        "claimed": FeatureTier.D,
        "trust": FeatureTier.A,
        "in_career": FeatureTier.B,
        "semantic": FeatureTier.B,
        "competency": FeatureTier.A,
    }
)

_COMPETENCY_GROUPS: frozenset[str] = frozenset(
    {"retr", "rank", "recsys", "ir", "nlp", "llm", "mle", "mlops", "eval"}
)

_GROUP_DISTRIBUTION: Mapping[str, str] = MappingProxyType(
    {
        "career": "right-skewed-low (synthetic-noisy pool)",
        "hp": "spike-at-0, thin impossible tail",
        "pvs": "bimodal (product vs services)",
        "jd": "regressed-to-prior under low confidence",
    }
)


def _tier_for(group: str, name: str, fid: str) -> FeatureTier:
    if fid in _TIER_OVERRIDE:
        return _TIER_OVERRIDE[fid]
    if group in _COMPETENCY_GROUPS:
        return _COMPETENCY_SUFFIX_TIER[name]
    return _GROUP_TIER[group]


def _polarity_for(fid: str) -> FeaturePolarity:
    if fid in _NEGATIVE_IDS:
        return FeaturePolarity.NEGATIVE
    if fid in _NEUTRAL_IDS:
        return FeaturePolarity.NEUTRAL
    return FeaturePolarity.POSITIVE


def _dependencies_for(fid: str) -> tuple[FeatureId, ...]:
    return _DEPENDENCIES.get(fid, ())


def _competency(group: str, suffix: str) -> FeatureId:
    return feature_id(group, suffix)


# Latent / composite dependency wiring (Part 2 constituents; hp.composite over
# its detectors). Primitives carry no dependencies.
_DEPENDENCIES: Mapping[str, tuple[FeatureId, ...]] = MappingProxyType(
    {
        "jd.retrieval_ranking": (
            _competency("retr", "competency"),
            _competency("rank", "competency"),
            _competency("ir", "competency"),
        ),
        "jd.production_ml": (
            _competency("mle", "competency"),
            _competency("mlops", "competency"),
            feature_id("pvs", "product_density"),
        ),
        "jd.product_company": (
            feature_id("pvs", "product_density"),
            feature_id("pvs", "product_recent"),
        ),
        "jd.shipping_mentality": (
            feature_id("startup", "shipping_signal"),
            feature_id("found", "ownership"),
        ),
        "jd.eval_framework": (_competency("eval", "competency"),),
        "jd.hybrid_retrieval": (
            _competency("ir", "competency"),
            _competency("ir", "trust"),
        ),
        "jd.keyword_only": tuple(
            _competency(group, suffix)
            for group in ("retr", "rank", "recsys", "ir", "nlp", "llm", "mle", "mlops", "eval")
            for suffix in ("claimed", "trust", "in_career", "semantic")
        ),
        "jd.consulting_only": (
            feature_id("pvs", "consulting_density"),
            feature_id("career", "consulting_density"),
        ),
        "jd.title_chaser": (
            feature_id("career", "title_inflation"),
            feature_id("career", "stability"),
        ),
        "jd.pure_researcher": (
            feature_id("career", "research_only"),
            feature_id("career", "production_exposure"),
        ),
        "jd.framework_enthusiast": (
            _competency("llm", "competency"),
            feature_id("career", "hands_on_engineering"),
        ),
        "jd.inactive": (
            feature_id("avail", "available"),
            feature_id("resp", "reliable"),
            feature_id("bhv", "availability"),
        ),
        "hp.composite": tuple(
            feature_id("hp", name)
            for name in (
                "timeline_impossible", "skill_time_contradiction", "employment_overlap",
                "title_seniority_anomaly", "education_career_anomaly", "salary_anomaly",
                "experience_inflation", "keyword_stuffing", "behavioral_inconsistency",
                "signal_impossibility", "identity_anomaly",
            )
        ),
    }
)


def _build_definition(entry_name: str) -> FeatureDefinition:
    entry = next(e for e in FEATURE_LAYOUT.entries if e.name == entry_name)
    fid = FeatureId(entry.name)
    group, name = entry.name.split(".", 1)
    tier = _tier_for(group, name, entry.name)
    polarity = _polarity_for(entry.name)
    schema = FeatureSchema(
        feature_id=fid,
        dtype=FeatureDType.FLOAT32,
        lower=entry.lower,
        upper=entry.upper,
        nullable=False,
    )
    metadata = FeatureMetadata(
        doc=f"{entry.name}: {group}-group feature (source slice {entry.source_slice}).",
        expected_distribution=_GROUP_DISTRIBUTION.get(group, "unit[0,1]"),
        owner=_EXTRACTOR_REF[group],
        tier=tier,
        polarity=polarity,
    )
    return FeatureDefinition(
        feature_id=fid,
        group=group,
        index=entry.index,
        source_slice=SourceSlice(entry.source_slice),
        schema=schema,
        dependencies=_dependencies_for(entry.name),
        extractor_ref=_EXTRACTOR_REF[group],
        version=FeatureVersion(major=1, minor=1, patch=0),
        tier=tier,
        polarity=polarity,
        metadata=metadata,
    )


def _canonical_blob(definitions: tuple[FeatureDefinition, ...]) -> str:
    """A deterministic, order-stable serialization for the registry hash."""
    rows = [
        {
            "i": int(d.index),
            "id": str(d.feature_id),
            "group": d.group,
            "slice": d.source_slice.value,
            "lo": d.schema_.lower,
            "hi": d.schema_.upper,
            "dtype": d.schema_.dtype.value,
            "nullable": d.schema_.nullable,
            "tier": d.tier.value,
            "polarity": d.polarity.value,
            "ver": d.version.semver,
            "deps": [str(dep) for dep in d.dependencies],
        }
        for d in definitions
    ]
    return json.dumps(
        {"layout_version": LAYOUT_VERSION, "features": rows},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@final
class FeatureRegistry(BaseModel):
    """The frozen set of all ``FeatureDefinition``s; id↔index source of truth."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    definitions: tuple[FeatureDefinition, ...] = Field(min_length=1)
    layout_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _matches_layout(self) -> FeatureRegistry:
        if self.layout_version != LAYOUT_VERSION:
            raise ArtifactContractError(
                f"registry layout_version {self.layout_version!r} != "
                f"layout {LAYOUT_VERSION!r}"
            )
        if len(self.definitions) != len(FEATURE_IDS):
            raise ArtifactContractError("registry size != layout dim")
        seen: set[str] = set()
        for position, definition in enumerate(self.definitions):
            if int(definition.index) != position:
                raise ArtifactContractError("registry definitions out of index order")
            if str(definition.feature_id) != FEATURE_IDS[position]:
                raise ArtifactContractError(
                    f"registry id {definition.feature_id!r} != layout "
                    f"{FEATURE_IDS[position]!r} at index {position}"
                )
            if str(definition.feature_id) in seen:
                raise ArtifactContractError("duplicate feature id in registry")
            seen.add(str(definition.feature_id))
        # Every declared dependency must itself be a known feature id.
        for definition in self.definitions:
            for dep in definition.dependencies:
                if str(dep) not in seen:
                    raise ArtifactContractError(
                        f"dependency {dep!r} of {definition.feature_id!r} is unknown"
                    )
        return self

    @property
    def dim(self) -> int:
        return len(self.definitions)

    def get(self, feature_id_: FeatureId) -> FeatureDefinition:
        """Return the definition for ``feature_id_``; raise on unknown id."""
        index = self._index_of.get(str(feature_id_))
        if index is None:
            raise UnknownFeatureId(str(feature_id_))
        return self.definitions[index]

    def has(self, feature_id_: FeatureId) -> bool:
        return str(feature_id_) in self._index_of

    def index(self, feature_id_: FeatureId) -> FeatureIndex:
        """Return the CQV index of ``feature_id_``; raise on unknown id."""
        return self.get(feature_id_).index

    def feature_id_at(self, index: FeatureIndex) -> FeatureId:
        """Return the feature id occupying ``index``."""
        position = int(index)
        if position < 0 or position >= len(self.definitions):
            raise UnknownFeatureId(f"index {position} out of range")
        return self.definitions[position].feature_id

    def by_group(self, group: str) -> tuple[FeatureDefinition, ...]:
        """All definitions in ``group`` in layout order."""
        return tuple(d for d in self.definitions if d.group == group)

    @property
    def groups(self) -> tuple[str, ...]:
        return GROUP_ORDER

    @property
    def registry_hash(self) -> str:
        """Stable sha256 of the canonical definition serialization."""
        blob = _canonical_blob(self.definitions)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def manifest(self) -> FeatureManifest:
        """Build the per-run ``FeatureManifest`` (ids, group map, hash)."""
        groups: Mapping[str, tuple[FeatureId, ...]] = MappingProxyType(
            {
                group: tuple(FeatureId(fid) for fid in GROUP_MEMBERS[group])
                for group in GROUP_ORDER
            }
        )
        return FeatureManifest(
            layout_version=self.layout_version,
            feature_ids=tuple(FeatureId(fid) for fid in FEATURE_IDS),
            groups=groups,
            registry_hash=self.registry_hash,
        )

    @property
    def _index_of(self) -> Mapping[str, int]:
        # Recomputed cheaply (frozen model; small, no cached mutable state).
        return MappingProxyType(
            {str(d.feature_id): int(d.index) for d in self.definitions}
        )


def _build_registry() -> FeatureRegistry:
    definitions = tuple(_build_definition(name) for name in FEATURE_IDS)
    return FeatureRegistry(definitions=definitions, layout_version=LAYOUT_VERSION)


# The single, frozen registry instance shared by reference (memory §Q).
FEATURE_REGISTRY: FeatureRegistry = _build_registry()
FEATURE_MANIFEST: FeatureManifest = FEATURE_REGISTRY.manifest()


__all__: tuple[str, ...] = (
    "FEATURE_MANIFEST",
    "FEATURE_REGISTRY",
    "FeatureDType",
    "FeatureDefinition",
    "FeatureManifest",
    "FeatureMetadata",
    "FeaturePolarity",
    "FeatureRegistry",
    "FeatureSchema",
    "FeatureTier",
    "FeatureVersion",
    "UnknownFeatureId",
)