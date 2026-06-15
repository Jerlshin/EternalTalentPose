"""Education feature extraction — ``edu.*`` (``features/education.py``).

Owner layer: features (pure). Allowed imports: ``domain`` + stdlib + numpy.
No ports, IO, ML, clock, or RNG — graduation-plausibility uses the injected
``as_of`` year, never a wall clock.

Group 5 (Feature Layer Part 1): tier / field-relevance / timeline-sanity, low JD
weight, structured ⇒ high confidence. Emits three ``UnitScore`` cells:

* ``edu.tier_score`` — best institution tier mapped to ``[0, 1]``.
* ``edu.field_relevance`` — best field-of-study relevance to the JD's ML / CS /
  IR stack.
* ``edu.timeline_valid`` — ``1.0`` when every record's years are coherent and
  plausible; ``0.0`` on an impossible timeline. This **cross-links** the
  honeypot layer (``hp.education_career_anomaly`` /
  ``EDUCATION_TIMELINE_IMPOSSIBLE``) but never raises: an impossible timeline is
  *data* the Integrity Engine consumes, not an exception.

Aggregation is best-of across records (a candidate is credited for their
strongest qualification), with a structural confidence that drops only when the
profile carries no education at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from types import MappingProxyType

from redstack.domain.enums import EvidenceKind, InstitutionTier
from redstack.domain.ids import UnitScore
from redstack.domain.source import RawCandidate, RawEducation
from redstack.features.normalize import normalize_text
from redstack.features.parsing import (
    FeatureCell,
    FeatureId,
    clamp_unit,
    feature_id,
    make_cell,
    mint_evidence,
)

__feature_version__ = "1.1.0"

_GROUP = "edu"

EDU_TIER_SCORE: FeatureId = feature_id(_GROUP, "tier_score")
EDU_FIELD_RELEVANCE: FeatureId = feature_id(_GROUP, "field_relevance")
EDU_TIMELINE_VALID: FeatureId = feature_id(_GROUP, "timeline_valid")

# Tier -> score. tier_1 strongest; UNKNOWN is a neutral-low prior, never 0
# (an unrecognized institution must not be penalised as if it were tier-4).
_TIER_SCORE: Mapping[InstitutionTier, float] = MappingProxyType(
    {
        InstitutionTier.TIER_1: 1.0,
        InstitutionTier.TIER_2: 0.75,
        InstitutionTier.TIER_3: 0.5,
        InstitutionTier.TIER_4: 0.25,
        InstitutionTier.UNKNOWN: 0.4,
    }
)

# Field relevance lexicon: normalized substrings -> relevance weight. The JD's
# stack is retrieval / ranking / ML / IR; adjacent quantitative fields earn
# partial credit, unrelated fields stay low.
_FIELD_RELEVANCE: Mapping[str, float] = MappingProxyType(
    {
        "machine learning": 1.0,
        "artificial intelligence": 1.0,
        "information retrieval": 1.0,
        "natural language processing": 1.0,
        "computer science": 0.9,
        "data science": 0.9,
        "computational": 0.8,
        "statistics": 0.7,
        "applied mathematics": 0.65,
        "mathematics": 0.6,
        "information systems": 0.6,
        "software engineering": 0.6,
        "electrical engineering": 0.5,
        "electronics": 0.45,
        "physics": 0.45,
        "engineering": 0.4,
    }
)

# Earliest plausible matriculation year; guards against absurd start years while
# tolerating non-linear paths. Upper bound is the injected ``as_of`` year.
_MIN_PLAUSIBLE_START_YEAR = 1950
# A single degree spanning more than this many years is implausible.
_MAX_DEGREE_SPAN_YEARS = 15


def _field_relevance(field_of_study: str) -> float:
    """Best substring match of a normalized field against the relevance lexicon."""
    normalized = normalize_text(field_of_study)
    best = 0.0
    for needle, weight in _FIELD_RELEVANCE.items():
        if needle in normalized and weight > best:
            best = weight
    return best


def _timeline_ok(edu: RawEducation, as_of_year: int) -> bool:
    """True when one record's years are internally and externally plausible."""
    if edu.end_year < edu.start_year:
        return False
    if edu.start_year < _MIN_PLAUSIBLE_START_YEAR:
        return False
    if edu.end_year > as_of_year:
        return False
    if (edu.end_year - edu.start_year) > _MAX_DEGREE_SPAN_YEARS:
        return False
    return True


def extract_education(
    raw: RawCandidate,
    as_of: date,
) -> Mapping[FeatureId, FeatureCell]:
    """Extract the ``edu.*`` cells for one candidate.

    Deterministic and total: a candidate with no education records yields
    neutral-low values at reduced confidence (never an error); an impossible
    timeline drives ``edu.timeline_valid`` to ``0.0`` for the Integrity Engine.
    """
    records = raw.education
    cells: dict[FeatureId, FeatureCell] = {}

    if not records:
        # No structured education: neutral-low priors, low confidence. Evidence
        # is the (empty) education arity, anchored on a field that always exists.
        no_edu = mint_evidence(
            raw, EvidenceKind.PROFILE_FIELD, "profile.years_of_experience"
        )
        cells[EDU_TIER_SCORE] = make_cell(0.0, 0.3, (no_edu,))
        cells[EDU_FIELD_RELEVANCE] = make_cell(0.0, 0.3, (no_edu,))
        cells[EDU_TIMELINE_VALID] = make_cell(1.0, 0.3, (no_edu,))
        return MappingProxyType(cells)

    as_of_year = as_of.year

    # --- edu.tier_score: best tier across records --------------------------- #
    best_tier_idx = 0
    best_tier_value = -1.0
    for idx, edu in enumerate(records):
        score = _TIER_SCORE[edu.tier]
        if score > best_tier_value:
            best_tier_value = score
            best_tier_idx = idx
    tier_evidence = mint_evidence(
        raw, EvidenceKind.EDUCATION, f"education[{best_tier_idx}].tier"
    )
    cells[EDU_TIER_SCORE] = make_cell(
        best_tier_value, clamp_unit(0.95), (tier_evidence,)
    )

    # --- edu.field_relevance: best field across records --------------------- #
    best_field_idx = 0
    best_field_value = -1.0
    for idx, edu in enumerate(records):
        relevance = _field_relevance(edu.field_of_study)
        if relevance > best_field_value:
            best_field_value = relevance
            best_field_idx = idx
    field_evidence = mint_evidence(
        raw, EvidenceKind.EDUCATION, f"education[{best_field_idx}].field_of_study"
    )
    # Confidence is high when we matched the lexicon, lower for an unrecognized
    # field (structured but semantically opaque).
    field_confidence = 0.9 if best_field_value > 0.0 else 0.5
    cells[EDU_FIELD_RELEVANCE] = make_cell(
        max(best_field_value, 0.0), field_confidence, (field_evidence,)
    )

    # --- edu.timeline_valid: every record must be plausible ----------------- #
    timeline_valid = True
    first_bad_idx = 0
    for idx, edu in enumerate(records):
        if not _timeline_ok(edu, as_of_year):
            timeline_valid = False
            first_bad_idx = idx
            break
    anchor_idx = first_bad_idx if not timeline_valid else 0
    timeline_evidence = (
        mint_evidence(raw, EvidenceKind.EDUCATION, f"education[{anchor_idx}].start_year"),
        mint_evidence(
            raw, EvidenceKind.EDUCATION, f"education[{anchor_idx}].end_year",
            as_of=as_of,
        ),
    )
    cells[EDU_TIMELINE_VALID] = make_cell(
        1.0 if timeline_valid else 0.0,
        clamp_unit(0.98),
        timeline_evidence,
    )

    return MappingProxyType(cells)


__all__: tuple[str, ...] = (
    "EDU_FIELD_RELEVANCE",
    "EDU_TIER_SCORE",
    "EDU_TIMELINE_VALID",
    "extract_education",
)