from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from redstack.domain.enums import EvidenceKind
from redstack.domain.provenance import EvidenceRef
from redstack.domain.ids import Similarity
from redstack.domain.source import RawCandidate, RawSkill
from redstack.features.view import (
    DURATION_SATURATION_MONTHS,
    ENDORSEMENT_SATURATION,
    CellEmission,
    FeatureCell,
    cell,
    clamp_unit,
    make_evidence,
    mean_of,
)

_VO = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)

_SKILL = EvidenceKind.SKILL
_CAREER = EvidenceKind.CAREER_FIELD
_PROFILE = EvidenceKind.PROFILE_FIELD
_SEMANTIC = EvidenceKind.DERIVED
_DERIVED = EvidenceKind.DERIVED

# The nine competency groups in fixed layout order.
_GROUPS: Final[tuple[str, ...]] = (
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

# Competency fusion weights (convex) and stuffing-penalty strength.
#
# Full-pool audit (data/raw/candidates.jsonl, all 100k) found a concrete
# trap-candidate archetype the original 0.45/0.30/0.25 split let through at
# scale: a non-ML title (Cloud/DevOps/QA/Full-stack Engineer) whose actual
# job-description text has zero overlap with the concept's lexicon tokens
# (``in_career`` == 0) but who lists a trendy AI skill with inflated
# endorsements/duration (``trust`` high) and whose profile *summary* mentions
# the same buzzwords in hedged, self-disclosed-hobbyist language ("I've been
# keeping up with AI/ML at a self-learner level ... haven't done it in a
# professional capacity yet") that the full-resume embedding still reads as
# topically similar (``semantic`` high). With trust+semantic carrying 70% of
# the fused weight, this pattern alone made up 62/100 of one post-fix top-100
# ranking. ``in_career`` is the one source anchored to job-description text
# rather than self-reported metadata or hedge-blind cosine similarity --
# exactly the JD's own instruction ("if their career history shows they
# built X, they're a fit; a candidate with all the keywords but the wrong
# title is not"). Rebalanced so it dominates rather than trailing both
# gameable sources combined.
_W_TRUST: Final[float] = 0.25
_W_IN_CAREER: Final[float] = 0.50
_W_SEMANTIC: Final[float] = 0.25
_W_STUFFING: Final[float] = 0.5
_CLAIM_SATURATION: Final[float] = 4.0
_MAX_SKILL_EVIDENCE: Final[int] = 3

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9+#.]+")
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {"and", "the", "for", "with", "of", "to", "in", "on", "a", "an", "at", "by"}
)


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase alphanumeric tokens (length ≥ 2), stopwords removed."""
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= 2 and token not in _STOPWORDS
    )


# --------------------------------------------------------------------------- #
# Lexicon input (resolved O5 artifact shape; passed in, never loaded here).   #
# --------------------------------------------------------------------------- #
@final
class CompetencyConcept(BaseModel):
    """One competency concept: its canonical tokens and its JD anchor id."""

    model_config = _VO

    tokens: frozenset[str] = Field(min_length=1)
    anchor_id: str = Field(min_length=1)


@final
class CompetencyLexicon(BaseModel):
    """Concept → ``CompetencyConcept`` map covering the nine competency groups."""

    model_config = _VO

    concepts: Mapping[str, CompetencyConcept]

    @field_validator("concepts", mode="after")
    @classmethod
    def _covers_groups(
        cls, value: Mapping[str, CompetencyConcept]
    ) -> Mapping[str, CompetencyConcept]:
        missing = [g for g in _GROUPS if g not in value]
        if missing:
            raise ValueError(f"competency lexicon missing concepts: {missing}")
        return MappingProxyType(dict(value))


# --------------------------------------------------------------------------- #
# Per-skill trust.                                                            #
# --------------------------------------------------------------------------- #
def _skill_trust(skill: RawSkill, assessment: float | None) -> float:
    """Endorsement × duration × assessment-coherence trust in ``[0, 1]``.

    A skill with endorsements but no duration and no assessment stays modest; a
    skill corroborated on all three is trusted. Missing duration / assessment is
    treated as absent evidence (0 contribution), never as a negative.
    """
    from redstack.features.view import bounded_log_scale

    e_norm = bounded_log_scale(
        float(skill.endorsements), saturation=ENDORSEMENT_SATURATION
    )
    if skill.duration_months is None:
        d_norm = 0.0
    else:
        d_norm = bounded_log_scale(
            float(skill.duration_months), saturation=DURATION_SATURATION_MONTHS
        )
    a_norm = 0.0 if assessment is None else clamp_unit(assessment / 100.0)
    return clamp_unit(0.4 * e_norm + 0.3 * d_norm + 0.3 * a_norm)


def _noisy_or(values: tuple[float, ...]) -> float:
    """Soft-OR: ``1 - Π(1 - v)`` — any one strongly corroborated skill suffices."""
    product = 1.0
    for value in values:
        product *= 1.0 - clamp_unit(value)
    return clamp_unit(1.0 - product)


# --------------------------------------------------------------------------- #
# One competency group.                                                       #
# --------------------------------------------------------------------------- #
def _competency_group(
    group: str,
    concept: CompetencyConcept,
    raw: RawCandidate,
    description_tokens: frozenset[str],
    semantic: Mapping[str, Similarity],
) -> list[tuple[str, FeatureCell]]:
    tokens = concept.tokens

    matched: list[tuple[int, RawSkill, float]] = []
    for skill_index, skill in enumerate(raw.skills):
        if _tokenize(skill.name) & tokens:
            assessment = raw.redrob_signals.skill_assessment_scores.get(skill.name)
            matched.append((skill_index, skill, _skill_trust(skill, assessment)))

    # claimed: raw keyword presence, saturating (deliberately weak).
    from redstack.features.view import bounded_log_scale

    claimed = bounded_log_scale(float(len(matched)), saturation=_CLAIM_SATURATION)

    # trust: noisy-OR over matched per-skill trust.
    trust = _noisy_or(tuple(t for (_, _, t) in matched))

    # in_career: fraction of concept tokens that appear in role descriptions.
    in_career = clamp_unit(len(tokens & description_tokens) / float(len(tokens)))

    # semantic: resolved anchor cosine mapped from [-1, 1] to [0, 1].
    sim = semantic.get(concept.anchor_id)
    semantic_present = sim is not None
    semantic_value = clamp_unit((float(sim) + 1.0) / 2.0) if sim is not None else 0.0

    corroboration = mean_of((trust, in_career, semantic_value))
    weighted = (
        _W_TRUST * trust + _W_IN_CAREER * in_career + _W_SEMANTIC * semantic_value
    )
    stuffing_penalty = clamp_unit(claimed - corroboration)
    # Cap at the corroboration mean ⇒ satisfies the competency-LE contract.
    competency = clamp_unit(
        min(weighted, corroboration) - _W_STUFFING * stuffing_penalty
    )

    sources_present = (
        (1 if trust > 0.0 else 0)
        + (1 if in_career > 0.0 else 0)
        + (1 if semantic_present else 0)
    )
    corroboration_conf = clamp_unit(0.25 + 0.75 * sources_present / 3.0)

    # Evidence: matched skills (capped), else a derived concept marker.
    if matched:
        skill_ev = tuple(
            make_evidence(_SKILL, f"skills[{idx}].name", skill.name, raw=raw)
            for (idx, skill, _) in matched[:_MAX_SKILL_EVIDENCE]
        )
    else:
        skill_ev = (make_evidence(_DERIVED, f"lexicon.{group}", 0.0),)

    semantic_ev = (
        make_evidence(_SEMANTIC, f"semantic.{concept.anchor_id}", semantic_value),
    )
    career_ev: tuple[EvidenceRef, ...] = (
        (make_evidence(_DERIVED, f"{group}.in_career", in_career),)
        if in_career > 0.0
        else (make_evidence(_DERIVED, f"lexicon.{group}", 0.0),)
    )

    return [
        (f"{group}.claimed", cell(claimed, 0.6, skill_ev)),
        (f"{group}.trust", cell(trust, corroboration_conf, skill_ev)),
        (f"{group}.in_career", cell(in_career, corroboration_conf, career_ev)),
        (
            f"{group}.semantic",
            cell(semantic_value, 0.85 if semantic_present else 0.2, semantic_ev),
        ),
        (
            f"{group}.competency",
            cell(
                competency,
                corroboration_conf,
                (make_evidence(_DERIVED, f"{group}.competency", competency),),
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Consistency group (``cons.*``).                                             #
# --------------------------------------------------------------------------- #
def _consistency(
    raw: RawCandidate, description_tokens: frozenset[str]
) -> list[tuple[str, FeatureCell]]:
    profile = raw.profile
    title_tokens = _tokenize(profile.current_title)
    current_desc_tokens: frozenset[str] = frozenset()
    current_desc_path = "career_history[0].description"
    for position_index, position in enumerate(raw.career_history):
        if position.is_current:
            current_desc_tokens = _tokenize(position.description)
            current_desc_path = f"career_history[{position_index}].description"
            break
    else:
        if raw.career_history:
            current_desc_tokens = _tokenize(raw.career_history[0].description)

    def _overlap(left: frozenset[str], right: frozenset[str]) -> float:
        if not left:
            return 0.0
        return clamp_unit(len(left & right) / float(len(left)))

    title_role = _overlap(title_tokens, current_desc_tokens)

    skill_tokens: frozenset[str] = frozenset()
    for skill in raw.skills:
        skill_tokens |= _tokenize(skill.name)
    skill_role = _overlap(skill_tokens, description_tokens)

    summary_tokens = _tokenize(profile.summary) | _tokenize(profile.headline)
    summary_coherence = _overlap(summary_tokens, description_tokens | title_tokens)

    return [
        (
            "cons.title_role_coherence",
            cell(
                title_role,
                0.7,
                (
                    make_evidence(
                        _PROFILE,
                        "profile.current_title",
                        profile.current_title,
                        raw=raw,
                    ),
                    make_evidence(
                        _CAREER, current_desc_path, "role_description", raw=raw
                    ),
                ),
            ),
        ),
        (
            "cons.skill_role_coherence",
            cell(
                skill_role,
                0.7,
                (make_evidence(_DERIVED, "cons.skill_role_coherence", skill_role),),
            ),
        ),
        (
            "cons.summary_coherence",
            cell(
                summary_coherence,
                0.6,
                (
                    make_evidence(
                        _PROFILE,
                        "profile.summary",
                        profile.summary[:64] or "summary",
                        raw=raw,
                    ),
                ),
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Public extractor.                                                           #
# --------------------------------------------------------------------------- #
def extract(
    raw: RawCandidate,
    *,
    semantic: Mapping[str, Similarity],
    lexicon: CompetencyLexicon,
) -> CellEmission:
    """Emit the competency (8–16) and consistency (``cons.*``) feature cells.

    Pure function of ``(RawCandidate, resolved anchor similarities, lexicon)``.
    """
    description_tokens: frozenset[str] = frozenset()
    for position in raw.career_history:
        description_tokens |= _tokenize(position.description)

    rows: list[tuple[str, FeatureCell]] = []
    for group in _GROUPS:
        concept = lexicon.concepts[group]
        rows.extend(
            _competency_group(group, concept, raw, description_tokens, semantic)
        )
    rows.extend(_consistency(raw, description_tokens))
    return tuple(rows)


__all__ = ("CompetencyConcept", "CompetencyLexicon", "extract")
