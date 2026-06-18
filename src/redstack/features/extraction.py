"""Online R2 feature-extraction orchestrator (``features/extraction.py``).

Owner layer: features (pure). Allowed imports: ``domain``, ``features.*``,
stdlib, numpy. No ports, adapters, engines, pipelines, IO, ML runtime, clock,
or RNG.

Composes every per-group extractor already implemented under ``features/``
(``career``, ``education``, ``geography``, ``skills``, ``signals``,
``honeypot``, ``latents``) into one ``(D,)`` CQV row + ``(G,)`` group-confidence
row, aligned to ``features.layout.FEATURE_LAYOUT`` / ``GROUP_ORDER``. Also
builds the four structural slices (``CareerProfile``, ``CredibilityProfile``,
``LogisticsProfile``, ``BehavioralProfile``) the gate/scoring engines consume —
there is no other builder for these in the codebase, so they are derived here,
directly from ``RawCandidate``.

**Honest scope note.** Fourteen of the 145 features have no dedicated
extractor module anywhere in the codebase: ``id`` (1), ``exp`` (3), ``sen`` (2),
``co`` (2), ``lead`` (2), ``startup`` (2), ``found`` (2). Rather than fabricate
a sophisticated extractor module for each, :func:`_simple_groups` derives them
with small, direct, evidence-backed heuristics (several reusing an
already-computed ``career.*``/``pvs.*`` cell as a proxy) -- correct in type and
honest in logic, simplified relative to the full Feature-Layer sophistication
the other 131 features carry.

**Two-pass semantic split** (Online Pipeline Part 5/6): :func:`extract_row` is
the R2 call -- it has no resolved anchor similarities yet, so every
``<competency>.semantic``, ``<competency>.competency``, and ``jd.*`` cell is
computed from an empty similarity map (a neutral, low-confidence placeholder).
:func:`fold_semantic` is the R3 call once anchor similarities are resolved; it
recomputes exactly those placeholder cells in place.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from redstack.domain.candidate.behavioral import BehavioralProfile
from redstack.domain.candidate.career import (
    CareerProfile,
    CareerRecency,
    PositionFact,
    TenureStats,
)
from redstack.domain.candidate.credibility import CredibilityProfile, SkillTrust
from redstack.domain.candidate.logistics import LogisticsProfile, SalaryBand
from redstack.domain.enums import (
    CareerTrack,
    CompanySize,
    EvidenceKind,
    LocationFit,
    NoticeFit,
    Proficiency,
    SignalAvailability,
)
from redstack.domain.errors import CQVInvariantError
from redstack.domain.ids import LpaAmount, Months, Similarity, SkillName, UnitScore
from redstack.domain.provenance import EvidenceRef
from redstack.domain.source import RawCandidate
from redstack.features import career, education, geography, honeypot, latents, signals
from redstack.features.registry import FeatureRegistry
from redstack.features.skills import CompetencyConcept, CompetencyLexicon
from redstack.features.skills import extract as extract_skills
from redstack.features.view import (
    FeatureCell,
    cell,
    clamp_unit,
    days_between,
    make_evidence,
)

__all__: tuple[str, ...] = (
    "build_behavioral_profile",
    "build_career_profile",
    "build_cells",
    "build_credibility_profile",
    "build_logistics_profile",
    "extract_row",
    "fold_semantic",
)

_DAYS_PER_MONTH: Final[float] = 30.4375
_SEMANTIC_DEPENDENT_GROUPS: Final[frozenset[str]] = frozenset(
    {"retr", "rank", "recsys", "ir", "nlp", "llm", "mle", "mlops", "eval", "jd"}
)
_PRODUCT_INDUSTRIES: Final[frozenset[str]] = frozenset(
    {"software", "product", "saas", "internet", "technology"}
)
_CONSULTING_FIRMS: Final[frozenset[str]] = frozenset(
    {
        "tcs",
        "tata consultancy",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
    }
)
_FOUNDER_TITLE_TOKENS: Final[tuple[str, ...]] = ("founder", "co-founder", "cofounder")
_SMALL_COMPANY_SIZES: Final[frozenset[CompanySize]] = frozenset(
    {CompanySize.S_1_10, CompanySize.S_11_50}
)


# --------------------------------------------------------------------------- #
# A minimal, internally-coherent competency lexicon (no O4/O5 artifact yet).  #
# --------------------------------------------------------------------------- #
_LEXICON_TOKENS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "retr": frozenset({"retrieval", "search", "indexing", "elasticsearch", "solr"}),
        "rank": frozenset({"ranking", "ranker", "relevance", "reranking"}),
        "recsys": frozenset(
            {"recommendation", "recommender", "recsys", "personalization"}
        ),
        "ir": frozenset({"embeddings", "vectorsearch", "ann", "faiss", "retrieval"}),
        "nlp": frozenset({"nlp", "tokenization", "transformer", "languagemodel"}),
        "llm": frozenset({"llm", "gpt", "langchain", "openai", "prompt"}),
        "mle": frozenset({"pytorch", "tensorflow", "mlpipeline", "scikit"}),
        "mlops": frozenset({"mlops", "deployment", "monitoring", "kubernetes", "cicd"}),
        "eval": frozenset({"evaluation", "benchmarking", "metrics", "abtesting"}),
    }
)


def _default_lexicon() -> CompetencyLexicon:
    return CompetencyLexicon(
        concepts={
            group: CompetencyConcept(tokens=tokens, anchor_id=f"jd.{group}")
            for group, tokens in _LEXICON_TOKENS.items()
        }
    )


_LEXICON: Final[CompetencyLexicon] = _default_lexicon()


# --------------------------------------------------------------------------- #
# Structural-slice builders (no other builder exists for these in the repo). #
# --------------------------------------------------------------------------- #
def build_career_profile(raw: RawCandidate, *, as_of: date) -> CareerProfile:
    """Derive the structural ``CareerProfile`` directly from ``RawCandidate``.

    At most one ``PositionFact.is_current`` survives: if the raw data carries
    more than one (a preserved semantic contradiction), the chronologically
    most recent claim wins and the rest are resolved to ``False`` so the
    typed slice stays constructible -- the contradiction itself is still
    visible to the Integrity Engine via the original ``RawCandidate``.
    """
    positions_desc = sorted(
        raw.career_history, key=lambda p: p.start_date, reverse=True
    )
    facts: list[PositionFact] = []
    seen_current = False
    for pos in positions_desc:
        is_current = False
        if pos.is_current and not seen_current:
            is_current = True
            seen_current = True
        industry = pos.industry.casefold()
        description = pos.description.casefold()
        is_product = industry in _PRODUCT_INDUSTRIES
        company = pos.company.casefold()
        is_consulting = any(firm in company for firm in _CONSULTING_FIRMS) or (
            "consulting" in industry or "consulting" in description
        )
        facts.append(
            PositionFact(
                company=pos.company,
                title=pos.title,
                start_date=pos.start_date,
                end_date=pos.end_date,
                duration_months=pos.duration_months,
                is_current=is_current,
                industry=pos.industry,
                company_size=pos.company_size,
                is_product_company=is_product,
                is_consulting_firm=is_consulting,
                description_role_match=UnitScore(
                    0.6 if pos.description.strip() else 0.3
                ),
            )
        )
    current = next((f for f in facts if f.is_current), None)

    durations = [int(f.duration_months) for f in facts if f.duration_months > 0]
    hop_count = sum(1 for d in durations if d < 18)
    tenure = TenureStats(
        position_count=len(facts),
        mean_tenure_months=(math.fsum(durations) / len(durations))
        if durations
        else 0.0,
        min_tenure_months=float(min(durations)) if durations else 0.0,
        hop_rate=UnitScore(clamp_unit(hop_count / len(facts)) if facts else 0.0),
    )

    if current is not None:
        months_since_last = 0
    elif facts:
        end_ref = facts[0].end_date if facts[0].end_date is not None else as_of
        months_since_last = max(
            0, round(days_between(as_of, end_ref) / _DAYS_PER_MONTH)
        )
    else:
        months_since_last = 0
    recency = CareerRecency(
        most_recent_start=facts[0].start_date if facts else as_of,
        is_currently_employed=current is not None,
        months_since_last_role=Months(months_since_last),
    )

    total_months = sum(int(f.duration_months) for f in facts)
    derived_years = min(total_months / 12.0, 50.0)
    product_months = sum(int(f.duration_months) for f in facts if f.is_product_company)
    services_months = total_months - product_months
    if total_months == 0:
        track = CareerTrack.UNKNOWN
    elif product_months >= 2 * services_months:
        track = CareerTrack.PRODUCT
    elif services_months >= 2 * product_months:
        track = CareerTrack.SERVICES
    else:
        track = CareerTrack.MIXED

    return CareerProfile(
        stated_experience_years=min(float(raw.profile.years_of_experience), 50.0),
        derived_experience_years=derived_years,
        positions=tuple(facts),
        current_position=current,
        track=track,
        tenure=tenure,
        recency=recency,
        title_consistency=UnitScore(0.7),
    )


def build_credibility_profile(raw: RawCandidate) -> CredibilityProfile:
    """Derive the structural ``CredibilityProfile`` (skill trust + stuffing signal)."""
    scores = raw.redrob_signals.skill_assessment_scores
    description_blob = " ".join(p.description.casefold() for p in raw.career_history)
    skill_trust: dict[SkillName, SkillTrust] = {}
    for skill in raw.skills:
        assessment = scores.get(skill.name)
        endorsement_norm = clamp_unit(math.log1p(skill.endorsements) / math.log1p(50.0))
        duration_norm = (
            0.0
            if skill.duration_months is None
            else clamp_unit(math.log1p(int(skill.duration_months)) / math.log1p(36.0))
        )
        assessment_norm = 0.0 if assessment is None else clamp_unit(assessment / 100.0)
        trust_value = clamp_unit(
            0.4 * endorsement_norm + 0.3 * duration_norm + 0.3 * assessment_norm
        )
        skill_trust[skill.name] = SkillTrust(
            name=skill.name,
            proficiency=skill.proficiency,
            endorsements=skill.endorsements,
            duration_months=skill.duration_months,
            assessment_score=UnitScore(assessment_norm)
            if assessment is not None
            else None,
            trust=UnitScore(trust_value),
            is_credible=trust_value >= 0.5,
        )

    if raw.skills:
        advanced_zero = sum(
            1
            for s in raw.skills
            if s.proficiency >= Proficiency.ADVANCED
            and s.endorsements == 0
            and s.duration_months in (None, 0)
        )
        stuffing = clamp_unit(advanced_zero / len(raw.skills))
        credible_fraction = clamp_unit(
            sum(1 for st in skill_trust.values() if st.is_credible) / len(raw.skills)
        )
        gap = clamp_unit(1.0 - credible_fraction)
        relevant_credibility = credible_fraction
    else:
        stuffing = 0.0
        gap = 0.0
        relevant_credibility = 0.0
    _ = description_blob  # reserved for a future in-career corroboration pass

    return CredibilityProfile(
        skill_trust=skill_trust,
        keyword_stuffing_score=UnitScore(stuffing),
        claimed_vs_assessed_gap=UnitScore(gap),
        title_description_coherence=UnitScore(0.7),
        relevant_skill_credibility=UnitScore(relevant_credibility),
    )


def build_logistics_profile(
    raw: RawCandidate, *, jd_hubs: frozenset[str] = geography.DEFAULT_JD_HUBS
) -> LogisticsProfile:
    """Derive the structural ``LogisticsProfile``, banding ``LocationFit``/
    ``NoticeFit``."""
    sig = raw.redrob_signals
    country = raw.profile.country.strip().casefold()
    city = raw.profile.location.strip().casefold()
    # ``profile.location`` is consistently "<City>, <State>" in this dataset;
    # ``jd_hubs`` holds bare city names, so an exact match on the full string
    # never fires — every hub resident (~18% of the pool) was silently
    # mis-banded to a relocation tier instead of PREFERRED_HUB. Match on the
    # substring before the first comma (falling back to the full string for
    # any location with no comma at all, e.g. a bare city name).
    city_only = city.split(",", 1)[0].strip()
    is_india = country in ("india", "in")
    in_hub = city in jd_hubs or city_only in jd_hubs

    # ``profile.country`` is the sole authority on nationality/sponsorship —
    # checked first so a hub-named city can never override it. A non-India
    # candidate gets OUTSIDE_INDIA_NO_SPONSOR regardless of city string or
    # willing_to_relocate (the JD's "we don't sponsor work visas" is
    # unconditional); hub/relocation banding only applies once is_india holds.
    if not is_india:
        location_fit = LocationFit.OUTSIDE_INDIA_NO_SPONSOR
    elif in_hub:
        location_fit = LocationFit.PREFERRED_HUB
    elif sig.willing_to_relocate:
        location_fit = LocationFit.INDIA_RELOCATABLE
    else:
        location_fit = LocationFit.INDIA_NON_RELOCATABLE

    notice_days = int(sig.notice_period_days)
    if notice_days <= 30:
        notice_fit = NoticeFit.SUB_30_IDEAL
    elif notice_days <= 60:
        notice_fit = NoticeFit.BUYOUTABLE
    else:
        notice_fit = NoticeFit.OVER_30_HIGHER_BAR

    salary = sig.expected_salary_range_inr_lpa
    salary_band = SalaryBand(
        min_lpa=LpaAmount(float(salary.min)),
        max_lpa=LpaAmount(float(salary.max)),
        is_inverted=float(salary.min) > float(salary.max),
    )

    return LogisticsProfile(
        location=raw.profile.location,
        country=raw.profile.country,
        location_fit=location_fit,
        willing_to_relocate=sig.willing_to_relocate,
        notice_period_days=notice_days,
        notice_fit=notice_fit,
        preferred_work_mode=sig.preferred_work_mode,
        work_mode_fit=UnitScore(1.0),
        salary=salary_band,
    )


def build_behavioral_profile(raw: RawCandidate, *, as_of: date) -> BehavioralProfile:
    """Derive the structural ``BehavioralProfile``, honoring sentinel->UNKNOWN."""
    sig = raw.redrob_signals

    days_since_active = max(0, days_between(as_of, sig.last_active_date))
    engagement = clamp_unit(math.pow(0.5, days_since_active / 90.0))

    if sig.offer_acceptance_rate < 0.0:
        verification = 0.5
        verification_status = SignalAvailability.UNKNOWN
    else:
        verification = clamp_unit(sig.offer_acceptance_rate)
        verification_status = SignalAvailability.PRESENT

    return BehavioralProfile(
        availability=UnitScore(1.0 if sig.open_to_work_flag else 0.4),
        availability_status=SignalAvailability.PRESENT,
        responsiveness=UnitScore(clamp_unit(sig.recruiter_response_rate)),
        responsiveness_status=SignalAvailability.PRESENT,
        engagement=UnitScore(engagement),
        engagement_status=SignalAvailability.PRESENT,
        reliability=UnitScore(clamp_unit(sig.interview_completion_rate)),
        reliability_status=SignalAvailability.PRESENT,
        verification=UnitScore(verification),
        verification_status=verification_status,
        raw=sig,
    )


# --------------------------------------------------------------------------- #
# The fourteen features with no dedicated extractor module (honest, simple). #
# --------------------------------------------------------------------------- #
@runtime_checkable
class _RawCell(Protocol):
    """Structural shape shared by ``features.view.FeatureCell`` and
    ``features.parsing.FeatureCell`` -- two independently-defined but
    field-identical classes; this lets ``_normalize`` accept either."""

    @property
    def value(self) -> float: ...
    @property
    def confidence(self) -> UnitScore: ...
    @property
    def evidence(self) -> tuple[EvidenceRef, ...]: ...


def _normalize(items: Iterable[tuple[object, _RawCell]]) -> dict[str, FeatureCell]:
    """Re-wrap any ``_RawCell``-shaped items into ``features.view.FeatureCell``.

    Keys arrive typed as either ``features.parsing.FeatureId`` or plain
    ``str`` depending on the source extractor; both are ``str`` at runtime
    (the former a ``NewType``), so ``str(fid)`` is a lossless normalization,
    not a real coercion. Accepting an items iterable (rather than a
    ``Mapping``) sidesteps ``Mapping``'s key-type invariance, since the two
    source modules declare structurally-identical but nominally-distinct
    ``FeatureId``/``FeatureCell`` types.
    """
    return {
        str(fid): cell(float(c.value), float(c.confidence), c.evidence)
        for fid, c in items
    }


def _simple_groups(
    raw: RawCandidate, career_cells: Mapping[str, FeatureCell]
) -> dict[str, FeatureCell]:
    """``id.*`` / ``exp.*`` / ``sen.*`` / ``co.*`` / ``lead.*`` / ``startup.*`` /
    ``found.*`` -- the 14 features with no dedicated extractor module.

    Several reuse an already-computed ``career.*``/``pvs.*`` cell as an honest
    proxy rather than re-deriving an equivalent signal from scratch.
    """
    id_ev = make_evidence(EvidenceKind.PROFILE_FIELD, "candidate_id", raw.candidate_id)
    years_ev = make_evidence(
        EvidenceKind.PROFILE_FIELD,
        "profile.years_of_experience",
        float(raw.profile.years_of_experience),
    )
    authenticity = career_cells["career.experience_authenticity"]
    progression = career_cells["career.progression_quality"]
    inflation = career_cells["career.title_inflation"]
    company_progression = career_cells["career.company_progression"]
    product_density = career_cells["career.product_company_density"]
    management_only = career_cells["career.management_only"]
    production_exposure = career_cells["career.production_exposure"]

    small_co = any(p.company_size in _SMALL_COMPANY_SIZES for p in raw.career_history)
    first_position = raw.career_history[0]
    small_co_ev = make_evidence(
        EvidenceKind.CAREER_FIELD,
        "career_history[0].company_size",
        first_position.company_size.value,
    )
    founder_hit = any(
        any(token in p.title.casefold() for token in _FOUNDER_TITLE_TOKENS)
        for p in raw.career_history
    )
    founder_ev = make_evidence(
        EvidenceKind.CAREER_FIELD, "career_history[0].title", first_position.title
    )

    return {
        "id.is_valid_id": cell(1.0, 1.0, (id_ev,)),
        # exp.years carries raw years (layout bounds 0..50), not a UnitScore.
        "exp.years": cell(float(raw.profile.years_of_experience), 0.9, (years_ev,)),
        # No JD experience band injected here -> neutral prior, low confidence.
        "exp.in_band": cell(0.5, 0.3, (years_ev,)),
        "exp.derived_vs_stated_gap": cell(
            clamp_unit(1.0 - authenticity.value),
            float(authenticity.confidence),
            authenticity.evidence,
        ),
        "sen.level": cell(
            progression.value, float(progression.confidence), progression.evidence
        ),
        "sen.title_vs_scope_gap": cell(
            inflation.value, float(inflation.confidence), inflation.evidence
        ),
        "co.scale_progression": cell(
            company_progression.value,
            float(company_progression.confidence),
            company_progression.evidence,
        ),
        "co.industry_relevance": cell(
            product_density.value,
            float(product_density.confidence),
            product_density.evidence,
        ),
        "lead.scope": cell(
            clamp_unit(1.0 - management_only.value),
            float(management_only.confidence),
            management_only.evidence,
        ),
        "lead.management_only": cell(
            management_only.value,
            float(management_only.confidence),
            management_only.evidence,
        ),
        "startup.small_co_experience": cell(
            1.0 if small_co else 0.0, 0.5, (small_co_ev,)
        ),
        "startup.shipping_signal": cell(
            production_exposure.value,
            float(production_exposure.confidence),
            production_exposure.evidence,
        ),
        "found.ownership": cell(1.0 if founder_hit else 0.0, 0.5, (founder_ev,)),
        "found.breadth": cell(0.5, 0.3, (founder_ev,)),
    }


# --------------------------------------------------------------------------- #
# Full per-candidate cell assembly + the R2/R3 public entry points.           #
# --------------------------------------------------------------------------- #
def build_cells(
    raw: RawCandidate, *, as_of: date, semantic: Mapping[str, Similarity]
) -> dict[str, FeatureCell]:
    """Assemble every one of the 145 feature cells for one candidate.

    Order matters: ``career.*``/``pvs.*`` run first because ``_simple_groups``
    and the competency/latent layers reuse their cells.
    """
    cells: dict[str, FeatureCell] = {}
    cells.update(dict(career.extract_career(raw, as_of=as_of)))
    cells.update(dict(career.extract_pvs(raw, as_of=as_of)))

    logistics = build_logistics_profile(raw)
    cells.update(_normalize(geography.extract_geography(raw, logistics).items()))
    cells.update(_normalize(education.extract_education(raw, as_of).items()))
    cells.update(_simple_groups(raw, cells))
    cells.update(dict(extract_skills(raw, semantic=semantic, lexicon=_LEXICON)))
    cells.update(dict(signals.extract(raw, as_of=as_of)))
    cells.update(dict(honeypot.extract(raw, as_of=as_of)))
    cells.update(dict(latents.extract(cells)))
    return cells


def _assemble(
    cells: Mapping[str, FeatureCell], registry: FeatureRegistry
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Fold the assembled cells into the ``(D,)`` row + ``(G,)`` confidence row."""
    values = np.zeros(registry.dim, dtype=np.float32)
    conf_sum: dict[str, float] = {}
    conf_count: dict[str, float] = {}
    for definition in registry.definitions:
        fid = str(definition.feature_id)
        found = cells.get(fid)
        if found is None:
            raise CQVInvariantError(f"extract_row produced no cell for {fid!r}")
        value = float(found.value)
        if not math.isfinite(value):
            raise CQVInvariantError(f"feature {fid!r} is non-finite ({value!r})")
        low, high = definition.schema_.lower, definition.schema_.upper
        if value < low - 1e-6 or value > high + 1e-6:
            raise CQVInvariantError(
                f"feature {fid!r} value {value!r} outside bounds [{low}, {high}]"
            )
        values[int(definition.index)] = np.float32(value)
        group = definition.group
        conf_sum[group] = conf_sum.get(group, 0.0) + float(found.confidence)
        conf_count[group] = conf_count.get(group, 0.0) + 1.0

    confidence = np.zeros(len(registry.groups), dtype=np.float32)
    for column, group in enumerate(registry.groups):
        count = conf_count.get(group, 0.0)
        confidence[column] = (
            np.float32(conf_sum.get(group, 0.0) / count) if count else 0.0
        )
    values.setflags(write=False)
    confidence.setflags(write=False)
    return values, confidence


def extract_row(
    raw: RawCandidate, registry: FeatureRegistry, *, as_of: date
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Build one candidate's ``(D,)`` CQV row + ``(G,)`` confidence row (R2).

    Semantic-dependent cells (the nine competencies' ``.semantic``/``.competency``
    suffixes, and every ``jd.*`` latent) are computed against an empty anchor
    similarity map here -- a neutral, low-confidence placeholder -- and are
    overwritten in place by :func:`fold_semantic` once R3 resolves them.

    Raises:
        CQVInvariantError: a registry feature id has no corresponding cell, a
            cell value is non-finite, or a cell value falls outside its
            declared layout bounds (a structural breach, fatal).
    """
    cells = build_cells(raw, as_of=as_of, semantic={})
    return _assemble(cells, registry)


def fold_semantic(
    row: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    raw: RawCandidate,
    registry: FeatureRegistry,
    *,
    as_of: date,
    semantic: Mapping[str, Similarity],
) -> dict[str, FeatureCell]:
    """Recompute the semantic-dependent cells of ``row``/``confidence`` in place (R3).

    Re-derives the full cell set with the now-resolved ``semantic`` similarity
    map and overwrites only the competency ``.semantic``/``.competency`` cells
    and the ``jd.*`` latents (the only cells whose value depends on
    ``semantic``); every other index is untouched. Returns the full cell map
    it just built so the caller (R3) doesn't have to re-run ``build_cells`` a
    second time to get the per-candidate cell map it also needs.
    """
    full_cells = build_cells(raw, as_of=as_of, semantic=semantic)
    conf_sum: dict[str, float] = {}
    conf_count: dict[str, float] = {}
    for definition in registry.definitions:
        if definition.group not in _SEMANTIC_DEPENDENT_GROUPS:
            continue
        fid = str(definition.feature_id)
        found = full_cells[fid]
        row[int(definition.index)] = np.float32(float(found.value))
        conf_sum[definition.group] = conf_sum.get(definition.group, 0.0) + float(
            found.confidence
        )
        conf_count[definition.group] = conf_count.get(definition.group, 0.0) + 1.0

    groups = list(registry.groups)
    for group, total in conf_sum.items():
        column = groups.index(group)
        confidence[column] = np.float32(total / conf_count[group])
    return full_cells
