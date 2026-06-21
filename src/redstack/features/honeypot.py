from __future__ import annotations

from datetime import date
from typing import Final

from redstack.domain.enums import EvidenceKind, Proficiency
from redstack.domain.provenance import EvidenceRef
from redstack.domain.source import RawCandidate, RawEducation, RawPosition
from redstack.features.view import (
    CellEmission,
    FeatureCell,
    bounded_log_scale,
    cell,
    clamp_unit,
    make_evidence,
    mean_of,
)

_CAREER = EvidenceKind.CAREER_FIELD
_EDU = EvidenceKind.EDUCATION
_SKILL = EvidenceKind.SKILL
_SIGNAL = EvidenceKind.SIGNAL
_PROFILE = EvidenceKind.PROFILE_FIELD
_DERIVED = EvidenceKind.DERIVED

# Detectors whose firing is a *categorical* impossibility (vs merely unusual).
_HARD_DETECTORS: Final[frozenset[str]] = frozenset(
    {
        "hp.timeline_impossible",
        "hp.education_career_anomaly",
        "hp.signal_impossibility",
        "hp.identity_anomaly",
    }
)

_OVERLAP_GRACE_MONTHS: Final[float] = 2.0
_EXPERIENCE_TOLERANCE_YEARS: Final[float] = 1.5
_SKILL_TIME_MIN_COUNT: Final[int] = 2
_STUFFING_MIN_COUNT: Final[int] = 4
_SENIORITY_CUES: Final[tuple[str, ...]] = (
    "principal",
    "staff",
    "director",
    "vp",
    "vice president",
    "head",
    "chief",
    "cto",
    "architect",
    "distinguished",
)
_SENIORITY_MIN_YEARS: Final[float] = 4.0


def _fallback(
    evidence: list[EvidenceRef], feature_id: str, value: float
) -> tuple[EvidenceRef, ...]:
    """Guarantee ≥1 evidence ref: a derived self-ref when nothing else fired."""
    if evidence:
        return tuple(evidence)
    return (make_evidence(_DERIVED, feature_id, value),)


def _end_or(position: RawPosition, as_of: date) -> date:
    """A position's effective end: its ``end_date`` or ``as_of`` if current/open."""
    return position.end_date if position.end_date is not None else as_of


def _overlap_months(a: RawPosition, b: RawPosition, as_of: date) -> float:
    """Months of date overlap between two positions (0 if disjoint)."""
    start = max(a.start_date, b.start_date)
    end = min(_end_or(a, as_of), _end_or(b, as_of))
    if end <= start:
        return 0.0
    return (end - start).days / 30.0


# --------------------------------------------------------------------------- #
# Detectors.                                                                  #
# --------------------------------------------------------------------------- #
def _timeline_impossible(raw: RawCandidate, as_of: date) -> FeatureCell:
    evidence: list[EvidenceRef] = []
    contradictions = 0
    for i, position in enumerate(raw.career_history):
        if position.end_date is not None and position.end_date < position.start_date:
            contradictions += 1
            evidence.append(
                make_evidence(
                    _CAREER,
                    f"career_history[{i}].end_date",
                    position.end_date.isoformat(),
                    raw=raw,
                )
            )
        if position.is_current and position.end_date is not None:
            contradictions += 1
            evidence.append(
                make_evidence(_CAREER, f"career_history[{i}].is_current", True, raw=raw)
            )
    for j, edu in enumerate(raw.education):
        if edu.end_year < edu.start_year:
            contradictions += 1
            evidence.append(
                make_evidence(_EDU, f"education[{j}].end_year", edu.end_year, raw=raw)
            )
    value = 1.0 if contradictions >= 1 else 0.0
    confidence = 0.97 if contradictions >= 1 else 0.9
    return cell(value, confidence, _fallback(evidence, "hp.timeline_impossible", value))


def _skill_time_contradiction(raw: RawCandidate) -> FeatureCell:
    evidence: list[EvidenceRef] = []
    count = 0
    for i, skill in enumerate(raw.skills):
        if skill.proficiency >= Proficiency.ADVANCED and (
            skill.duration_months is None or skill.duration_months == 0
        ):
            count += 1
            if len(evidence) < 4:
                evidence.append(
                    make_evidence(
                        _SKILL,
                        f"skills[{i}].duration_months",
                        skill.duration_months
                        if skill.duration_months is not None
                        else "null",
                        raw=raw,
                    )
                )
    # Require corroborating breadth: a single advanced-zero-usage skill is noise.
    if count < _SKILL_TIME_MIN_COUNT:
        value = 0.0
    else:
        value = bounded_log_scale(float(count), saturation=6.0)
    confidence = clamp_unit(0.4 + 0.1 * count)
    return cell(
        value, confidence, _fallback(evidence, "hp.skill_time_contradiction", value)
    )


def _employment_overlap(raw: RawCandidate, as_of: date) -> FeatureCell:
    evidence: list[EvidenceRef] = []
    egregious_months = 0.0
    positions = raw.career_history
    for i in range(len(positions)):
        for k in range(i + 1, len(positions)):
            overlap = _overlap_months(positions[i], positions[k], as_of)
            if overlap > _OVERLAP_GRACE_MONTHS:
                egregious_months += overlap - _OVERLAP_GRACE_MONTHS
                if len(evidence) < 4:
                    evidence.append(
                        make_evidence(
                            _CAREER,
                            f"career_history[{i}].start_date",
                            positions[i].start_date.isoformat(),
                            raw=raw,
                        )
                    )
                    evidence.append(
                        make_evidence(
                            _CAREER,
                            f"career_history[{k}].start_date",
                            positions[k].start_date.isoformat(),
                            raw=raw,
                        )
                    )
    value = bounded_log_scale(egregious_months, saturation=24.0)
    return cell(value, 0.6, _fallback(evidence, "hp.employment_overlap", value))


def _title_seniority_anomaly(raw: RawCandidate) -> FeatureCell:
    title = raw.profile.current_title.lower()
    years = raw.profile.years_of_experience
    has_cue = any(cue in title for cue in _SENIORITY_CUES)
    if has_cue and years < _SENIORITY_MIN_YEARS:
        gap = (_SENIORITY_MIN_YEARS - years) / _SENIORITY_MIN_YEARS
        value = clamp_unit(gap)
    else:
        value = 0.0
    evidence = [
        make_evidence(
            _PROFILE, "profile.current_title", raw.profile.current_title, raw=raw
        ),
        make_evidence(_PROFILE, "profile.years_of_experience", years, raw=raw),
    ]
    return cell(value, 0.55, tuple(evidence))


def _education_career_anomaly(raw: RawCandidate) -> FeatureCell:
    evidence: list[EvidenceRef] = []
    value = 0.0
    if raw.education:
        edu_idx = min(
            range(len(raw.education)), key=lambda i: raw.education[i].start_year
        )
        earliest_edu_start = raw.education[edu_idx].start_year
        # career_history is schema-guaranteed non-empty (min_length=1).
        career_idx = min(
            range(len(raw.career_history)),
            key=lambda i: raw.career_history[i].start_date,
        )
        earliest_career = raw.career_history[career_idx].start_date.year
        # Career beginning well before any education start is implausible.
        if earliest_career < earliest_edu_start - 1:
            value = 1.0
            evidence.append(
                make_evidence(
                    _CAREER,
                    f"career_history[{career_idx}].start_date",
                    earliest_career,
                    raw=raw,
                )
            )
            evidence.append(
                make_evidence(
                    _EDU,
                    f"education[{edu_idx}].start_year",
                    earliest_edu_start,
                    raw=raw,
                )
            )
        # A degree that ends absurdly far in the future relative to its start.
        for j, edu in enumerate(raw.education):
            if edu.end_year - edu.start_year > 15:
                value = max(value, 0.8)
                evidence.append(
                    make_evidence(
                        _EDU, f"education[{j}].end_year", edu.end_year, raw=raw
                    )
                )
    confidence = 0.95 if value >= 0.8 else 0.85
    return cell(
        value, confidence, _fallback(evidence, "hp.education_career_anomaly", value)
    )


def _salary_anomaly(raw: RawCandidate) -> FeatureCell:
    salary = raw.redrob_signals.expected_salary_range_inr_lpa
    inverted = salary.min > salary.max
    # Soft only: inversion is common in the pool — cap the contribution low.
    value = 0.25 if inverted else 0.0
    evidence = [
        make_evidence(
            _SIGNAL,
            "redrob_signals.expected_salary_range_inr_lpa.min",
            salary.min,
            raw=raw,
        ),
        make_evidence(
            _SIGNAL,
            "redrob_signals.expected_salary_range_inr_lpa.max",
            salary.max,
            raw=raw,
        ),
    ]
    return cell(value, 0.4, tuple(evidence))


def _experience_inflation(raw: RawCandidate) -> FeatureCell:
    summed_years = sum(int(p.duration_months) for p in raw.career_history) / 12.0
    stated = raw.profile.years_of_experience
    gap = summed_years - stated
    if gap > _EXPERIENCE_TOLERANCE_YEARS:
        value = bounded_log_scale(gap - _EXPERIENCE_TOLERANCE_YEARS, saturation=8.0)
    else:
        value = 0.0
    evidence = [
        make_evidence(_PROFILE, "profile.years_of_experience", stated, raw=raw),
        make_evidence(
            _DERIVED, "career_history.summed_duration_years", round(summed_years, 2)
        ),
    ]
    return cell(value, clamp_unit(0.5 + 0.1 * gap), tuple(evidence))


def _keyword_stuffing(
    raw: RawCandidate, description_tokens: frozenset[str]
) -> FeatureCell:
    evidence: list[EvidenceRef] = []
    count = 0
    for i, skill in enumerate(raw.skills):
        zero_corroboration = (
            skill.endorsements == 0
            and (skill.duration_months is None or skill.duration_months == 0)
            and not (set(skill.name.lower().split()) & description_tokens)
        )
        if skill.proficiency >= Proficiency.ADVANCED and zero_corroboration:
            count += 1
            if len(evidence) < 4:
                evidence.append(
                    make_evidence(_SKILL, f"skills[{i}].name", skill.name, raw=raw)
                )
    # Require breadth: a couple of uncorroborated skills is not stuffing.
    if count < _STUFFING_MIN_COUNT:
        value = 0.0
    else:
        value = bounded_log_scale(float(count), saturation=12.0)
    return cell(
        value,
        clamp_unit(0.4 + 0.05 * count),
        _fallback(evidence, "hp.keyword_stuffing", value),
    )


def _behavioral_inconsistency(raw: RawCandidate) -> FeatureCell:
    sig = raw.redrob_signals
    evidence: list[EvidenceRef] = []
    impossible = 0
    if sig.saved_by_recruiters_30d > 0 and sig.profile_views_received_30d == 0:
        impossible += 1
        evidence.append(
            make_evidence(
                _SIGNAL,
                "redrob_signals.saved_by_recruiters_30d",
                sig.saved_by_recruiters_30d,
                raw=raw,
            )
        )
    if (
        sig.search_appearance_30d > 0
        and sig.profile_views_received_30d == 0
        and sig.saved_by_recruiters_30d > 0
    ):
        impossible += 1
        evidence.append(
            make_evidence(
                _SIGNAL,
                "redrob_signals.search_appearance_30d",
                sig.search_appearance_30d,
                raw=raw,
            )
        )
    value = bounded_log_scale(float(impossible), saturation=2.0) if impossible else 0.0
    return cell(value, 0.6, _fallback(evidence, "hp.behavioral_inconsistency", value))


def _signal_impossibility(raw: RawCandidate, as_of: date) -> FeatureCell:
    sig = raw.redrob_signals
    evidence: list[EvidenceRef] = []
    value = 0.0
    if sig.last_active_date < sig.signup_date:
        value = 1.0
        evidence.append(
            make_evidence(
                _SIGNAL,
                "redrob_signals.last_active_date",
                sig.last_active_date.isoformat(),
                raw=raw,
            )
        )
        evidence.append(
            make_evidence(
                _SIGNAL,
                "redrob_signals.signup_date",
                sig.signup_date.isoformat(),
                raw=raw,
            )
        )
    if sig.last_active_date > as_of or sig.signup_date > as_of:
        value = 1.0
        evidence.append(
            make_evidence(
                _SIGNAL,
                "redrob_signals.last_active_date",
                sig.last_active_date.isoformat(),
                raw=raw,
            )
        )
    return cell(value, 0.95, _fallback(evidence, "hp.signal_impossibility", value))


def _identity_anomaly(raw: RawCandidate) -> FeatureCell:
    evidence: list[EvidenceRef] = []
    value = 0.0
    current_idx, current = next(
        ((i, p) for i, p in enumerate(raw.career_history) if p.is_current), (None, None)
    )
    if (
        current is not None
        and current.company.strip().lower()
        != raw.profile.current_company.strip().lower()
    ):
        value = 0.4
        evidence.append(
            make_evidence(
                _PROFILE,
                "profile.current_company",
                raw.profile.current_company,
                raw=raw,
            )
        )
        evidence.append(
            make_evidence(
                _CAREER,
                f"career_history[{current_idx}].company",
                current.company,
                raw=raw,
            )
        )
    return cell(value, 0.7, _fallback(evidence, "hp.identity_anomaly", value))


# --------------------------------------------------------------------------- #
# Risk group (non-honeypot uncertainty / contradiction).                      #
# --------------------------------------------------------------------------- #
def _risk(
    raw: RawCandidate, detectors: dict[str, FeatureCell]
) -> list[tuple[str, FeatureCell]]:
    sig = raw.redrob_signals
    none_duration = sum(1 for s in raw.skills if s.duration_months is None)
    skill_count = max(1, len(raw.skills))
    uncertainty = clamp_unit(
        mean_of(
            (
                1.0 if sig.github_activity_score == -1.0 else 0.0,
                1.0 if sig.offer_acceptance_rate == -1.0 else 0.0,
                1.0 if len(sig.skill_assessment_scores) == 0 else 0.0,
                none_duration / float(skill_count),
            )
        )
    )
    contradiction = clamp_unit(
        mean_of(
            (
                detectors["hp.salary_anomaly"].value,
                0.5 * detectors["hp.title_seniority_anomaly"].value,
                0.5 * detectors["hp.behavioral_inconsistency"].value,
            )
        )
    )
    confidence = clamp_unit(1.0 - uncertainty)
    return [
        (
            "risk.uncertainty",
            cell(
                uncertainty,
                0.7,
                (make_evidence(_DERIVED, "risk.uncertainty", uncertainty),),
            ),
        ),
        (
            "risk.contradiction",
            cell(
                contradiction,
                0.7,
                (make_evidence(_DERIVED, "risk.contradiction", contradiction),),
            ),
        ),
        (
            "risk.confidence",
            cell(
                confidence,
                0.8,
                (make_evidence(_DERIVED, "risk.confidence", confidence),),
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Composite + public extractor.                                               #
# --------------------------------------------------------------------------- #
def _composite(detectors: dict[str, FeatureCell]) -> FeatureCell:
    values = tuple(c.value for c in detectors.values())
    max_value = max(values)
    hard_fires = sum(
        1 for fid, c in detectors.items() if fid in _HARD_DETECTORS and c.value >= 0.5
    )
    soft_fires = sum(1 for c in detectors.values() if c.value >= 0.3)
    extra = clamp_unit(0.15 * hard_fires + 0.05 * soft_fires)
    # max lifted upward by a bounded count term ⇒ always ≥ max(detectors) and ≤ 1.
    composite = clamp_unit(max_value + (1.0 - max_value) * extra)
    confidence = max(c.confidence for c in detectors.values())
    evidence = (make_evidence(_DERIVED, "hp.composite", composite),)
    return cell(composite, float(confidence), evidence)


def extract(raw: RawCandidate, *, as_of: date) -> CellEmission:
    """Emit the twelve ``hp.*`` cells and the three ``risk.*`` cells.

    Pure deterministic function of ``(RawCandidate, as_of)``. The hard-gate
    honeypot *decision* (``≥2`` corroborating impossibilities or composite over
    the O3 threshold) is finalized by the Risk engine; here we emit graded,
    evidence-backed severities only.
    """
    description_tokens: frozenset[str] = frozenset()
    for position in raw.career_history:
        description_tokens |= frozenset(position.description.lower().split())

    detectors: dict[str, FeatureCell] = {
        "hp.timeline_impossible": _timeline_impossible(raw, as_of),
        "hp.skill_time_contradiction": _skill_time_contradiction(raw),
        "hp.employment_overlap": _employment_overlap(raw, as_of),
        "hp.title_seniority_anomaly": _title_seniority_anomaly(raw),
        "hp.education_career_anomaly": _education_career_anomaly(raw),
        "hp.salary_anomaly": _salary_anomaly(raw),
        "hp.experience_inflation": _experience_inflation(raw),
        "hp.keyword_stuffing": _keyword_stuffing(raw, description_tokens),
        "hp.behavioral_inconsistency": _behavioral_inconsistency(raw),
        "hp.signal_impossibility": _signal_impossibility(raw, as_of),
        "hp.identity_anomaly": _identity_anomaly(raw),
    }

    rows: list[tuple[str, FeatureCell]] = list(detectors.items())
    rows.append(("hp.composite", _composite(detectors)))
    rows.extend(_risk(raw, detectors))
    return tuple(rows)


__all__ = ("extract",)
