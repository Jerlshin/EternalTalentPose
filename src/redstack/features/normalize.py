"""Deterministic canonicalization + composed embedding document
(``features/normalize.py``).

Owner layer: features (pure). Allowed imports: ``domain`` + ``config.schema``
+ stdlib + numpy. No ports, IO, ML, clock, or RNG.

Realizes ``CandidateNormalizationEngine`` (Engine §2): lowercase / strip /
collapse whitespace, Unicode **NFC**, lexicon canonical skill-token mapping,
company / industry canonicalization, and the **composed embedding document**
whose field order is pinned by the ``embedding_manifest`` recipe. The recipe and
canonical maps arrive already-resolved (the pipeline sources them from
artifacts); this layer never reads them from disk.

The composed document is the single determinism-critical output: offline (O13a)
and online (R3 fallback) MUST compose byte-identical text for the same input, so
composition is a pure function of ``(RawCandidate, CanonicalMaps, recipe)`` with
a fixed field order and fixed separator. Unknown skill tokens map to themselves
(never dropped); unparseable inputs that survived schema validation raise
``SchemaError`` defensively.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import final

from pydantic import BaseModel, ConfigDict, Field

from redstack.domain.enums import CompanySize, InstitutionTier, Proficiency
from redstack.domain.ids import CandidateId, Months, SkillName
from redstack.domain.source import RawCandidate

__feature_version__ = "1.1.0"

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)


def normalize_text(text: str) -> str:
    """NFC-normalize, casefold-lower, strip, and collapse internal whitespace.

    Pure and idempotent: ``normalize_text(normalize_text(x)) == normalize_text(x)``.
    """
    folded = unicodedata.normalize("NFC", text)
    return " ".join(folded.lower().split())


def _canonical(token: str, table: Mapping[str, str]) -> str:
    """Map a normalized token through a canonical table; unknown -> itself."""
    normalized = normalize_text(token)
    return table.get(normalized, normalized)


@final
class CanonicalMaps(BaseModel):
    """Resolved canonical lookup tables (from O1 ``canonical_maps.json``).

    Keys are already ``normalize_text``-normalized; values are the canonical
    surface forms. Passed in by the pipeline — this layer performs no IO.
    """

    model_config = _STRICT

    skills: Mapping[str, str]
    companies: Mapping[str, str]
    industries: Mapping[str, str]


class EmbeddingField(str, Enum):
    """The closed vocabulary of fields the embedding recipe may order."""

    HEADLINE = "headline"
    SUMMARY = "summary"
    CURRENT_TITLE = "current_title"
    CURRENT_COMPANY = "current_company"
    CURRENT_INDUSTRY = "current_industry"
    SKILLS = "skills"
    POSITION_TITLES = "position_titles"
    POSITION_DESCRIPTIONS = "position_descriptions"
    EDUCATION = "education"


# Separators are *structural*: they must survive verbatim, so this model
# deliberately omits ``str_strip_whitespace`` (which would collapse "\n" -> "").
_RECIPE_STRICT = ConfigDict(frozen=True, extra="forbid", validate_default=True)


@final
class EmbeddingDocRecipe(BaseModel):
    """The field order + separators pinned by ``embedding_manifest``.

    ``fields`` is ordered and the order is load-bearing (embedding determinism).
    ``separator`` joins field blocks; ``intra_separator`` joins multi-valued
    blocks (skills, per-position text). Separators are preserved verbatim.
    """

    model_config = _RECIPE_STRICT

    fields: tuple[EmbeddingField, ...] = Field(min_length=1)
    separator: str = "\n"
    intra_separator: str = " "


@final
class NormalizedPosition(BaseModel):
    """A career position with canonicalized text, dates retained as ``date``."""

    model_config = _STRICT

    company: str
    title: str
    description: str
    industry: str
    company_size: CompanySize
    start_date: date
    end_date: date | None
    duration_months: Months = Field(ge=0)
    is_current: bool


@final
class NormalizedSkill(BaseModel):
    """A skill with a lexicon-canonical, lowercased ``SkillName`` token."""

    model_config = _STRICT

    name: SkillName
    proficiency: Proficiency
    endorsements: int = Field(ge=0)
    duration_months: Months | None = Field(default=None, ge=0)


@final
class NormalizedEducation(BaseModel):
    """An education record with canonicalized text + retained tier/years."""

    model_config = _STRICT

    institution: str
    degree: str
    field_of_study: str
    tier: InstitutionTier
    start_year: int
    end_year: int


@final
class NormalizedCandidate(BaseModel):
    """The intermediate carrier consumed by Feature / Behavior / Retrieval.

    Not a persisted domain slice — a deterministic, fully-typed projection of the
    ``RawCandidate`` with canonical text and the composed ``embedding_document``.
    """

    model_config = _STRICT

    candidate_id: CandidateId
    headline: str
    summary: str
    location: str
    country: str
    current_title: str
    current_company: str
    current_industry: str
    positions: tuple[NormalizedPosition, ...]
    skills: tuple[NormalizedSkill, ...]
    education: tuple[NormalizedEducation, ...]
    embedding_document: str


def _canonical_skill(name: SkillName, maps: CanonicalMaps) -> SkillName:
    return SkillName(_canonical(str(name), maps.skills))


def _field_text(
    field: EmbeddingField,
    norm: NormalizedCandidateParts,
    recipe: EmbeddingDocRecipe,
) -> str:
    """Render one recipe field to its deterministic text block."""
    if field is EmbeddingField.HEADLINE:
        return norm.headline
    if field is EmbeddingField.SUMMARY:
        return norm.summary
    if field is EmbeddingField.CURRENT_TITLE:
        return norm.current_title
    if field is EmbeddingField.CURRENT_COMPANY:
        return norm.current_company
    if field is EmbeddingField.CURRENT_INDUSTRY:
        return norm.current_industry
    if field is EmbeddingField.SKILLS:
        return recipe.intra_separator.join(str(s.name) for s in norm.skills)
    if field is EmbeddingField.POSITION_TITLES:
        return recipe.intra_separator.join(p.title for p in norm.positions)
    if field is EmbeddingField.POSITION_DESCRIPTIONS:
        return recipe.intra_separator.join(p.description for p in norm.positions)
    # EmbeddingField.EDUCATION
    return recipe.intra_separator.join(
        f"{e.degree} {e.field_of_study} {e.institution}".strip() for e in norm.education
    )


@final
class NormalizedCandidateParts(BaseModel):
    """Internal carrier of normalized parts prior to document composition."""

    model_config = _STRICT

    headline: str
    summary: str
    current_title: str
    current_company: str
    current_industry: str
    positions: tuple[NormalizedPosition, ...]
    skills: tuple[NormalizedSkill, ...]
    education: tuple[NormalizedEducation, ...]


def compose_embedding_document(
    parts: NormalizedCandidateParts,
    recipe: EmbeddingDocRecipe,
) -> str:
    """Compose the embedding document in the recipe's fixed field order.

    Deterministic by construction: identical ``(parts, recipe)`` -> identical
    bytes. Empty blocks are preserved as empty strings so the field count (and
    thus the separator structure) is stable across candidates.
    """
    blocks = tuple(_field_text(field, parts, recipe) for field in recipe.fields)
    return recipe.separator.join(blocks)


def normalize_candidate(
    raw: RawCandidate,
    maps: CanonicalMaps,
    recipe: EmbeddingDocRecipe,
) -> NormalizedCandidate:
    """Project a ``RawCandidate`` into a ``NormalizedCandidate``.

    Pure function: NFC + casefold text normalization, lexicon-canonical skill
    tokens, canonical company/industry surfaces, dates retained as ``date``, and
    the recipe-ordered composed embedding document.
    """
    positions = tuple(
        NormalizedPosition(
            company=_canonical(pos.company, maps.companies),
            title=normalize_text(pos.title),
            description=normalize_text(pos.description),
            industry=_canonical(pos.industry, maps.industries),
            company_size=pos.company_size,
            start_date=pos.start_date,
            end_date=pos.end_date,
            duration_months=pos.duration_months,
            is_current=pos.is_current,
        )
        for pos in raw.career_history
    )
    skills = tuple(
        NormalizedSkill(
            name=_canonical_skill(skill.name, maps),
            proficiency=skill.proficiency,
            endorsements=skill.endorsements,
            duration_months=skill.duration_months,
        )
        for skill in raw.skills
    )
    education = tuple(
        NormalizedEducation(
            institution=normalize_text(edu.institution),
            degree=normalize_text(edu.degree),
            field_of_study=normalize_text(edu.field_of_study),
            tier=edu.tier,
            start_year=edu.start_year,
            end_year=edu.end_year,
        )
        for edu in raw.education
    )

    parts = NormalizedCandidateParts(
        headline=normalize_text(raw.profile.headline),
        summary=normalize_text(raw.profile.summary),
        current_title=normalize_text(raw.profile.current_title),
        current_company=_canonical(raw.profile.current_company, maps.companies),
        current_industry=_canonical(raw.profile.current_industry, maps.industries),
        positions=positions,
        skills=skills,
        education=education,
    )
    document = compose_embedding_document(parts, recipe)

    return NormalizedCandidate(
        candidate_id=raw.candidate_id,
        headline=parts.headline,
        summary=parts.summary,
        location=normalize_text(raw.profile.location),
        country=normalize_text(raw.profile.country),
        current_title=parts.current_title,
        current_company=parts.current_company,
        current_industry=parts.current_industry,
        positions=positions,
        skills=skills,
        education=education,
        embedding_document=document,
    )


def empty_canonical_maps() -> CanonicalMaps:
    """An identity-mapping ``CanonicalMaps`` (every token maps to itself)."""
    empty: Mapping[str, str] = MappingProxyType({})
    return CanonicalMaps(skills=empty, companies=empty, industries=empty)


__all__: tuple[str, ...] = (
    "CanonicalMaps",
    "EmbeddingDocRecipe",
    "EmbeddingField",
    "NormalizedCandidate",
    "NormalizedCandidateParts",
    "NormalizedEducation",
    "NormalizedPosition",
    "NormalizedSkill",
    "compose_embedding_document",
    "empty_canonical_maps",
    "normalize_candidate",
    "normalize_text",
)