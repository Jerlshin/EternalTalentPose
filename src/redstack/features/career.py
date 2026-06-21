from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

from redstack.domain.enums import EvidenceKind
from redstack.domain.provenance import EvidenceRef
from redstack.domain.source import RawCandidate, RawPosition
from redstack.features.view import (
    DURATION_SATURATION_MONTHS,
    STALE_HALF_LIFE_DAYS,
    CellEmission,
    FeatureCell,
    FeatureId,
    bounded_log_scale,
    cell,
    clamp_unit,
    days_between,
    make_evidence,
    recency_unit,
)

__feature_version__ = "1.1.0"

_CAREER = EvidenceKind.CAREER_FIELD
_PROFILE = EvidenceKind.PROFILE_FIELD

# --------------------------------------------------------------------------- #
# Lexicons (lowercased substring matches over NFC-normalized text).           #
# --------------------------------------------------------------------------- #

# Title seniority ladder: (rank, tokens). Highest matching rank wins; an IC
# title with no ladder token defaults to rank 2 ("mid").
_TITLE_LADDER: Final[tuple[tuple[int, tuple[str, ...]], ...]] = (
    (0, ("intern", "trainee", "apprentice")),
    (1, ("junior", "jr ", "jr.", "associate", "entry", "graduate")),
    (2, ("engineer", "developer", "analyst", "scientist", "programmer", "sde")),
    (3, ("senior", "sr ", "sr.", "lead", "specialist", "ii", "iii")),
    (4, ("staff", "principal", "architect", "manager", "head", "director")),
    (
        5,
        (
            "vp",
            "vice president",
            "chief",
            "cto",
            "ceo",
            "founder",
            "co-founder",
            "partner",
        ),
    ),
)
_DEFAULT_TITLE_RANK: Final[int] = 2
_MAX_RANK: Final[float] = 5.0

# Scope cues found in *descriptions* (the corroborating evidence).
_SCOPE_CUES: Final[tuple[tuple[int, tuple[str, ...]], ...]] = (
    (5, ("founded", "co-founded", "started the company", "p&l", "org of")),
    (
        4,
        (
            "managed a team",
            "led a team",
            "led the team",
            "people management",
            "direct reports",
            "headcount",
            "hired",
            "owned the",
            "set the strategy",
            "roadmap ownership",
            "architected",
            "org-wide",
        ),
    ),
    (
        3,
        (
            "led",
            "owned",
            "drove",
            "mentored",
            "designed",
            "spearheaded",
            "end-to-end",
            "cross-functional",
        ),
    ),
    (
        2,
        (
            "built",
            "implemented",
            "developed",
            "shipped",
            "wrote",
            "coded",
            "delivered",
            "contributed",
        ),
    ),
)
_DEFAULT_SCOPE_RANK: Final[int] = 1

_CONSULTING_FIRMS: Final[tuple[str, ...]] = (
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "deloitte",
    "hcl",
    "tech mahindra",
    "mindtree",
    "mphasis",
    "ibm global services",
    "ltimindtree",
    "persistent systems",
)
_CONSULTING_CUES: Final[tuple[str, ...]] = (
    "consulting",
    "consultancy",
    "client",
    "clients",
    "staff augmentation",
    "outsourc",
    "managed services",
    "system integrat",
    "body shop",
    "billable",
    "engagement",
    "sow",
    "client-facing",
)
_PRODUCT_CUES: Final[tuple[str, ...]] = (
    "product",
    "saas",
    "platform",
    "our app",
    "our users",
    "feature flag",
    "a/b test",
    "growth",
    "consumer",
    "in-house",
    "proprietary product",
    "product-led",
    "user base",
    "mau",
    "dau",
)

_CODING_CUES: Final[tuple[str, ...]] = (
    "python",
    "java",
    "c++",
    "golang",
    "rust",
    "scala",
    "typescript",
    "pytorch",
    "tensorflow",
    "spark",
    "kubernetes",
    "sql",
    "wrote",
    "coded",
    "implemented",
    "built",
    "developed",
    "refactored",
    "debugged",
    "unit test",
    "pull request",
    "code review",
    "commit",
)
_ARCH_ONLY_CUES: Final[tuple[str, ...]] = (
    "oversaw",
    "strategy",
    "roadmap",
    "stakeholder",
    "governance",
    "steering",
    "high-level design",
    "review boards",
    "vendor management",
    "budget",
    "presentation",
)
_RESEARCH_CUES: Final[tuple[str, ...]] = (
    "research",
    "published",
    "publication",
    "paper",
    "novel approach",
    "phd",
    "thesis",
    "state-of-the-art",
    "neurips",
    "icml",
    "acl",
    "cvpr",
    "prototype only",
    "proof of concept",
    "experimental study",
)
_MGMT_CUES: Final[tuple[str, ...]] = (
    "managed a team",
    "people management",
    "direct reports",
    "headcount",
    "hiring",
    "performance reviews",
    "1:1s",
    "team of",
    "line manager",
    "managed engineers",
    "delegated",
)
_PRODUCTION_CUES: Final[tuple[str, ...]] = (
    "production",
    "in prod",
    "deployed",
    "serving",
    "live traffic",
    "at scale",
    "latency",
    "throughput",
    "uptime",
    "sla",
    "ci/cd",
    "monitoring",
    "on-call",
    "rollout",
    "millions of",
    "qps",
    "p99",
)
_TECH_BREADTH_TOKENS: Final[tuple[str, ...]] = (
    "python",
    "java",
    "c++",
    "golang",
    "rust",
    "scala",
    "typescript",
    "pytorch",
    "tensorflow",
    "spark",
    "kubernetes",
    "sql",
    "kafka",
    "airflow",
    "ray",
    "onnx",
)


def _text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).lower().split())


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    return any(token in text for token in tokens)


def _count_any(text: str, tokens: Sequence[str]) -> int:
    return sum(1 for token in tokens if token in text)


def _ladder_rank(
    text: str, ladder: tuple[tuple[int, tuple[str, ...]], ...], default: int
) -> int:
    best = default
    for rank, tokens in ladder:
        if _contains_any(text, tokens):
            best = max(best, rank)
    return best


def _saturating(count: int, *, saturation: float) -> float:
    """Diminishing-returns presence score for a token hit-count."""
    return bounded_log_scale(float(count), saturation=saturation)


# --------------------------------------------------------------------------- #
# Per-position derived view (immutable).                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _PosView:
    index: int
    duration_months: float
    tenure_weight: float
    recency_weight: float
    combined_weight: float
    title_rank: int
    scope_rank: int
    level: float
    size_ordinal: int
    description_present: bool
    product_score: float
    consulting_score: float
    coding_score: float
    arch_score: float
    research_score: float
    mgmt_score: float
    production_score: float
    hands_on_score: float
    tech_tokens: frozenset[str]
    start_date: date
    end_ref: date


def _build_pos_view(index: int, pos: RawPosition, as_of: date) -> _PosView:
    title_text = _text(pos.title)
    desc_text = _text(pos.description)
    company_text = _text(pos.company)
    industry_text = _text(pos.industry)
    blob = f"{desc_text} {industry_text} {company_text}"
    description_present = bool(desc_text)

    title_rank = _ladder_rank(title_text, _TITLE_LADDER, _DEFAULT_TITLE_RANK)
    scope_rank = (
        _ladder_rank(desc_text, _SCOPE_CUES, _DEFAULT_SCOPE_RANK)
        if description_present
        else _DEFAULT_SCOPE_RANK
    )
    # Corroborated level: description-dominant. Uncorroborated titles are
    # discounted (only 50% of the claimed rank counts) so a fabricated
    # "Principal" with an empty description cannot lift trajectory metrics.
    if description_present:
        level = 0.6 * float(scope_rank) + 0.4 * float(title_rank)
    else:
        level = 0.5 * float(title_rank)

    # Classification scores (description/industry dominant; company name is a
    # strong consulting tell via the known-firm list).
    consulting_hits = _count_any(blob, _CONSULTING_CUES)
    is_known_consulting = _contains_any(company_text, _CONSULTING_FIRMS)
    product_hits = _count_any(blob, _PRODUCT_CUES)
    consulting_score = clamp_unit(
        max(
            _saturating(consulting_hits, saturation=2.0),
            1.0 if is_known_consulting else 0.0,
        )
    )
    product_score = clamp_unit(
        _saturating(product_hits, saturation=2.0) * (1.0 - 0.5 * consulting_score)
    )

    coding_score = _saturating(_count_any(blob, _CODING_CUES), saturation=3.0)
    arch_score = _saturating(_count_any(blob, _ARCH_ONLY_CUES), saturation=2.0)
    research_score = _saturating(_count_any(blob, _RESEARCH_CUES), saturation=2.0)
    mgmt_score = _saturating(_count_any(blob, _MGMT_CUES), saturation=2.0)
    production_score = _saturating(_count_any(blob, _PRODUCTION_CUES), saturation=2.0)
    # Hands-on engineering: coding presence net of architecture-only signalling.
    hands_on_score = clamp_unit(coding_score - 0.5 * arch_score)
    tech_tokens = frozenset(t for t in _TECH_BREADTH_TOKENS if t in blob)

    duration_months = float(pos.duration_months)
    tenure_weight = duration_months if duration_months > 0.0 else 0.0
    # Recency: days since the role ended (current → 0 → fully recent).
    end_ref = as_of if (pos.is_current or pos.end_date is None) else pos.end_date
    days_since = float(days_between(as_of, end_ref))
    recency_weight = recency_unit(days_since, half_life_days=STALE_HALF_LIFE_DAYS)
    combined_weight = tenure_weight * recency_weight

    return _PosView(
        index=index,
        duration_months=duration_months,
        tenure_weight=tenure_weight,
        recency_weight=recency_weight,
        combined_weight=combined_weight,
        title_rank=title_rank,
        scope_rank=scope_rank,
        level=level,
        size_ordinal=pos.company_size.ordinal,
        description_present=description_present,
        product_score=product_score,
        consulting_score=consulting_score,
        coding_score=coding_score,
        arch_score=arch_score,
        research_score=research_score,
        mgmt_score=mgmt_score,
        production_score=production_score,
        hands_on_score=hands_on_score,
        tech_tokens=tech_tokens,
        start_date=pos.start_date,
        end_ref=end_ref,
    )


# --------------------------------------------------------------------------- #
# Weighting + small numeric utilities.                                        #
# --------------------------------------------------------------------------- #


def _weighted(values: tuple[float, ...], weights: tuple[float, ...]) -> float:
    """Weighted mean; falls back to the unweighted mean if all weights are 0."""
    total = math.fsum(weights)
    if total <= 0.0:
        if not values:
            return 0.0
        return math.fsum(values) / len(values)
    return math.fsum(v * w for v, w in zip(values, weights)) / total


def _argmax_contribution(
    views: tuple[_PosView, ...], scores: tuple[float, ...], weights: tuple[float, ...]
) -> int:
    """Index of the position contributing most (score×weight); ties → lowest index."""
    best_index = views[0].index
    best_value = -1.0
    for view, score, weight in zip(views, scores, weights):
        contribution = score * (weight if weight > 0.0 else 1.0)
        if contribution > best_value:
            best_value = contribution
            best_index = view.index
    return best_index


def _union_months(views: tuple[_PosView, ...]) -> float:
    """Total non-overlapping employed months across positions (years×12)."""
    intervals: list[tuple[date, date]] = []
    for view in views:
        if view.end_ref > view.start_date:
            intervals.append((view.start_date, view.end_ref))
    if not intervals:
        return 0.0
    intervals.sort(key=lambda pair: pair[0])
    total_days = 0
    cursor_start, cursor_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cursor_end:
            if end > cursor_end:
                cursor_end = end
        else:
            total_days += (cursor_end - cursor_start).days
            cursor_start, cursor_end = start, end
    total_days += (cursor_end - cursor_start).days
    return float(total_days) / 365.25 * 12.0


# --------------------------------------------------------------------------- #
# Evidence minting from real RawCandidate paths.                              #
# --------------------------------------------------------------------------- #


def _position_scalar(pos: RawPosition, field: str) -> str | int | float | bool:
    if field == "title":
        return pos.title
    if field == "description":
        return pos.description
    if field == "company":
        return pos.company
    if field == "industry":
        return pos.industry
    if field == "company_size":
        return pos.company_size.value
    if field == "duration_months":
        return int(pos.duration_months)
    if field == "is_current":
        return pos.is_current
    if field == "start_date":
        return pos.start_date.isoformat()
    raise KeyError(f"unsupported career evidence field: {field!r}")


def _ev_position(raw: RawCandidate, index: int, field: str) -> EvidenceRef:
    pos = raw.career_history[index]
    return make_evidence(
        _CAREER,
        f"career_history[{index}].{field}",
        _position_scalar(pos, field),
        raw=raw,
    )


def _ev_years(raw: RawCandidate) -> EvidenceRef:
    return make_evidence(
        _PROFILE,
        "profile.years_of_experience",
        float(raw.profile.years_of_experience),
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Career group ids (layout order).                                            #
# --------------------------------------------------------------------------- #

_C_PROGRESSION = "career.progression_quality"
_C_STABILITY = "career.stability"
_C_PROMOTION = "career.promotion_velocity"
_C_TITLE_INFLATION = "career.title_inflation"
_C_ROLE_CONSISTENCY = "career.role_consistency"
_C_AUTHENTICITY = "career.experience_authenticity"
_C_COMPANY_PROGRESSION = "career.company_progression"
_C_PRODUCT_DENSITY = "career.product_company_density"
_C_CONSULTING_DENSITY = "career.consulting_density"
_C_TECHNICAL_DEPTH = "career.technical_depth"
_C_HANDS_ON = "career.hands_on_engineering"
_C_RESEARCH_ONLY = "career.research_only"
_C_MANAGEMENT_ONLY = "career.management_only"
_C_PRODUCTION_EXPOSURE = "career.production_exposure"

_PVS_PRODUCT_DENSITY = "pvs.product_density"
_PVS_CONSULTING_DENSITY = "pvs.consulting_density"
_PVS_PRODUCT_RECENT = "pvs.product_recent"


def _confidences(views: tuple[_PosView, ...]) -> tuple[float, float]:
    """(structural_confidence, textual_confidence) for the candidate."""
    count = len(views)
    count_factor = bounded_log_scale(float(count), saturation=4.0)
    described = sum(1 for v in views if v.description_present)
    desc_coverage = described / count if count else 0.0
    conf_struct = clamp_unit(0.55 + 0.40 * count_factor)
    conf_text = clamp_unit(0.30 + 0.55 * desc_coverage + 0.10 * count_factor)
    return conf_struct, conf_text


def extract_career(raw: RawCandidate, *, as_of: date) -> CellEmission:
    """Emit the 14 ``career.*`` cells in layout order.

    Pure and total: ``career_history`` has ≥1 position (schema), so single-role
    candidates yield neutral trajectory values at reduced confidence rather than
    errors. All weighting is against ``as_of``; no wall clock is read.
    """
    views = tuple(
        _build_pos_view(i, pos, as_of) for i, pos in enumerate(raw.career_history)
    )
    chronological = tuple(sorted(views, key=lambda v: v.start_date))
    n = len(views)

    tenure_w = tuple(v.tenure_weight for v in views)
    combined_w = tuple(v.combined_weight for v in views)
    recency_w = tuple(v.recency_weight for v in views)

    conf_struct, conf_text = _confidences(views)
    conf_traj = clamp_unit(0.5 * (conf_struct + conf_text))
    conf_mixed = clamp_unit(max(conf_text, 0.5 * conf_struct))

    emissions: list[tuple[FeatureId, FeatureCell]] = []

    # 1. progression_quality — corroborated-level trajectory (tenure-weighted).
    if n == 1:
        progression = 0.5
    else:
        scores: list[float] = []
        weights: list[float] = []
        for a, b in zip(chronological, chronological[1:]):
            delta = b.level - a.level
            step = 1.0 if delta > 0.25 else (0.6 if delta >= -0.25 else 0.2)
            scores.append(step)
            weights.append(a.tenure_weight + b.tenure_weight)
        progression = _weighted(tuple(scores), tuple(weights))
    emissions.append(
        (
            _C_PROGRESSION,
            cell(
                clamp_unit(progression),
                conf_traj,
                (
                    _ev_position(raw, chronological[0].index, "title"),
                    _ev_position(raw, chronological[-1].index, "title"),
                ),
            ),
        )
    )

    # 2. stability — normalized mean tenure blended with inverse sub-18m hop rate.
    completed = tuple(v for v in views if v.duration_months > 0.0)
    mean_tenure = (
        math.fsum(v.duration_months for v in completed) / len(completed)
        if completed
        else 0.0
    )
    tenure_norm = bounded_log_scale(mean_tenure, saturation=DURATION_SATURATION_MONTHS)
    hoppy = tuple(v for v in views if 0.0 < v.duration_months < 18.0)
    hop_rate = len(hoppy) / n if n else 0.0
    stability = clamp_unit(0.5 * tenure_norm + 0.5 * (1.0 - hop_rate))
    shortest = min(views, key=lambda v: (v.duration_months, v.index))
    emissions.append(
        (
            _C_STABILITY,
            cell(
                stability,
                conf_struct,
                (
                    _ev_position(raw, shortest.index, "duration_months"),
                    _ev_position(raw, shortest.index, "title"),
                ),
            ),
        )
    )

    # 3. promotion_velocity — level gained per year (chronological).
    if n == 1:
        velocity = 0.3
    else:
        gain = chronological[-1].level - chronological[0].level
        span_days = max(
            (chronological[-1].start_date - chronological[0].start_date).days, 0
        )
        span_years = max(span_days / 365.25, 0.5)
        rate_per_year = gain / span_years
        velocity = clamp_unit(rate_per_year / 0.5)  # 0.5 level/yr ⇒ saturates
    emissions.append(
        (
            _C_PROMOTION,
            cell(
                velocity,
                conf_struct,
                (
                    _ev_position(raw, chronological[0].index, "title"),
                    _ev_position(raw, chronological[-1].index, "title"),
                ),
            ),
        )
    )

    # 4. title_inflation — claimed title above corroborated scope (tenure-weighted).
    inflation_scores = tuple(
        clamp_unit(max(0, v.title_rank - v.scope_rank) / 3.0)
        if v.description_present
        else clamp_unit(0.6 * v.title_rank / _MAX_RANK)
        for v in views
    )
    title_inflation = clamp_unit(_weighted(inflation_scores, tenure_w))
    infl_idx = _argmax_contribution(views, inflation_scores, tenure_w)
    emissions.append(
        (
            _C_TITLE_INFLATION,
            cell(
                title_inflation,
                conf_text,
                (
                    _ev_position(raw, infl_idx, "title"),
                    _ev_position(raw, infl_idx, "description"),
                ),
            ),
        )
    )

    # 5. role_consistency — low level dispersion + track homogeneity.
    if n == 1:
        role_consistency = 0.7
    else:
        levels = tuple(v.level for v in views)
        dispersion = (max(levels) - min(levels)) / _MAX_RANK
        product_majority = sum(1 for v in views if v.product_score >= 0.5)
        consulting_majority = sum(1 for v in views if v.consulting_score >= 0.5)
        homogeneity = (
            max(
                product_majority,
                consulting_majority,
                n - product_majority - consulting_majority,
            )
            / n
        )
        role_consistency = clamp_unit(0.6 * (1.0 - dispersion) + 0.4 * homogeneity)
    emissions.append(
        (
            _C_ROLE_CONSISTENCY,
            cell(
                role_consistency,
                conf_mixed,
                (
                    _ev_position(raw, chronological[0].index, "title"),
                    _ev_position(raw, chronological[-1].index, "title"),
                ),
            ),
        )
    )

    # 6. experience_authenticity — derived (union) tenure vs stated years.
    derived_years = min(_union_months(views) / 12.0, 50.0)
    stated_years = float(raw.profile.years_of_experience)
    gap = abs(stated_years - derived_years) / max(stated_years, derived_years, 1.0)
    authenticity = clamp_unit(1.0 - gap)
    longest = max(views, key=lambda v: (v.duration_months, -v.index))
    emissions.append(
        (
            _C_AUTHENTICITY,
            cell(
                authenticity,
                conf_struct,
                (_ev_years(raw), _ev_position(raw, longest.index, "duration_months")),
            ),
        )
    )

    # 7. company_progression — company-size trajectory (tenure-weighted).
    if n == 1:
        company_progression = 0.5
    else:
        c_scores: list[float] = []
        c_weights: list[float] = []
        for a, b in zip(chronological, chronological[1:]):
            delta = b.size_ordinal - a.size_ordinal
            step = 1.0 if delta > 0 else (0.6 if delta == 0 else 0.3)
            c_scores.append(step)
            c_weights.append(a.tenure_weight + b.tenure_weight)
        company_progression = _weighted(tuple(c_scores), tuple(c_weights))
    emissions.append(
        (
            _C_COMPANY_PROGRESSION,
            cell(
                clamp_unit(company_progression),
                conf_struct,
                (
                    _ev_position(raw, chronological[0].index, "company_size"),
                    _ev_position(raw, chronological[-1].index, "company_size"),
                ),
            ),
        )
    )

    # 8. product_company_density — tenure×recency weighted product fraction.
    product_scores = tuple(v.product_score for v in views)
    product_density = clamp_unit(_weighted(product_scores, combined_w))
    prod_idx = _argmax_contribution(views, product_scores, combined_w)
    emissions.append(
        (
            _C_PRODUCT_DENSITY,
            cell(
                product_density,
                conf_mixed,
                (
                    _ev_position(raw, prod_idx, "company"),
                    _ev_position(raw, prod_idx, "description"),
                ),
            ),
        )
    )

    # 9. consulting_density — tenure×recency weighted consulting fraction.
    consulting_scores = tuple(v.consulting_score for v in views)
    consulting_density = clamp_unit(_weighted(consulting_scores, combined_w))
    cons_idx = _argmax_contribution(views, consulting_scores, combined_w)
    emissions.append(
        (
            _C_CONSULTING_DENSITY,
            cell(
                consulting_density,
                conf_mixed,
                (
                    _ev_position(raw, cons_idx, "company"),
                    _ev_position(raw, cons_idx, "description"),
                ),
            ),
        )
    )

    # 10. technical_depth — coding density + distinct-tech breadth.
    coding_scores = tuple(v.coding_score for v in views)
    coding_density = _weighted(coding_scores, tenure_w)
    distinct_tech = (
        len(frozenset().union(*(v.tech_tokens for v in views))) if views else 0
    )
    tech_breadth = bounded_log_scale(float(distinct_tech), saturation=8.0)
    technical_depth = clamp_unit(0.6 * coding_density + 0.4 * tech_breadth)
    depth_idx = _argmax_contribution(views, coding_scores, tenure_w)
    emissions.append(
        (
            _C_TECHNICAL_DEPTH,
            cell(
                technical_depth,
                conf_text,
                (
                    _ev_position(raw, depth_idx, "description"),
                    _ev_position(raw, depth_idx, "title"),
                ),
            ),
        )
    )

    # 11. hands_on_engineering — recency-weighted hands-on (coding net of arch-only).
    hands_on_scores = tuple(v.hands_on_score for v in views)
    hands_on = clamp_unit(_weighted(hands_on_scores, recency_w))
    hands_idx = _argmax_contribution(views, hands_on_scores, recency_w)
    emissions.append(
        (
            _C_HANDS_ON,
            cell(hands_on, conf_text, (_ev_position(raw, hands_idx, "description"),)),
        )
    )

    # 12. research_only — research density gated by absence of production.
    research_scores = tuple(v.research_score for v in views)
    research_density = _weighted(research_scores, tenure_w)
    production_scores = tuple(v.production_score for v in views)
    production_density = _weighted(production_scores, combined_w)
    research_only = clamp_unit(research_density * (1.0 - production_density))
    res_idx = _argmax_contribution(views, research_scores, tenure_w)
    emissions.append(
        (
            _C_RESEARCH_ONLY,
            cell(
                research_only, conf_text, (_ev_position(raw, res_idx, "description"),)
            ),
        )
    )

    # 13. management_only — management density gated by absence of hands-on IC work.
    mgmt_scores = tuple(v.mgmt_score for v in views)
    mgmt_density = _weighted(mgmt_scores, tenure_w)
    management_only = clamp_unit(mgmt_density * (1.0 - max(coding_density, hands_on)))
    mgmt_idx = _argmax_contribution(views, mgmt_scores, tenure_w)
    emissions.append(
        (
            _C_MANAGEMENT_ONLY,
            cell(
                management_only,
                conf_text,
                (
                    _ev_position(raw, mgmt_idx, "description"),
                    _ev_position(raw, mgmt_idx, "title"),
                ),
            ),
        )
    )

    # 14. production_exposure — tenure×recency weighted production density.
    production_exposure = clamp_unit(production_density)
    prodexp_idx = _argmax_contribution(views, production_scores, combined_w)
    emissions.append(
        (
            _C_PRODUCTION_EXPOSURE,
            cell(
                production_exposure,
                conf_mixed,
                (_ev_position(raw, prodexp_idx, "description"),),
            ),
        )
    )

    return tuple(emissions)


def extract_pvs(raw: RawCandidate, *, as_of: date) -> CellEmission:
    """Emit the Group 7 ``pvs.*`` cells (product-vs-service densities).

    ``pvs.product_density`` is tenure×recency weighted; ``pvs.product_recent``
    re-weights by recency alone (emphasising the latest roles); both pair with
    ``pvs.consulting_density``. Shares the Part-3 classification so the two
    groups never disagree on what counts as a product company.
    """
    views = tuple(
        _build_pos_view(i, pos, as_of) for i, pos in enumerate(raw.career_history)
    )
    combined_w = tuple(v.combined_weight for v in views)
    recency_w = tuple(v.recency_weight for v in views)
    conf_struct, conf_text = _confidences(views)
    conf_mixed = clamp_unit(max(conf_text, 0.5 * conf_struct))

    product_scores = tuple(v.product_score for v in views)
    consulting_scores = tuple(v.consulting_score for v in views)

    product_density = clamp_unit(_weighted(product_scores, combined_w))
    product_recent = clamp_unit(_weighted(product_scores, recency_w))
    consulting_density = clamp_unit(_weighted(consulting_scores, combined_w))

    prod_idx = _argmax_contribution(views, product_scores, combined_w)
    recent_idx = _argmax_contribution(views, product_scores, recency_w)
    cons_idx = _argmax_contribution(views, consulting_scores, combined_w)

    return (
        (
            _PVS_PRODUCT_DENSITY,
            cell(
                product_density,
                conf_mixed,
                (
                    _ev_position(raw, prod_idx, "company"),
                    _ev_position(raw, prod_idx, "description"),
                ),
            ),
        ),
        (
            _PVS_CONSULTING_DENSITY,
            cell(
                consulting_density,
                conf_mixed,
                (
                    _ev_position(raw, cons_idx, "company"),
                    _ev_position(raw, cons_idx, "description"),
                ),
            ),
        ),
        (
            _PVS_PRODUCT_RECENT,
            cell(
                product_recent,
                conf_mixed,
                (
                    _ev_position(raw, recent_idx, "company"),
                    _ev_position(raw, recent_idx, "description"),
                ),
            ),
        ),
    )


__all__: tuple[str, ...] = ("extract_career", "extract_pvs")
