"""Behavioral signal extractor (``features/signals.py``).

Owner layer: features. Allowed imports: ``domain``, ``features.view``, stdlib.

Implements Part 4 (23 ``redrob_signals`` → 15 ``bhv.*`` composites) plus the
recruiter-availability / engagement / responsiveness / open-source primitive
groups (taxonomy 17, 21–23). Single-ownership routing (Engine §7) is honored:
this module reads signals 3–8, 10, 16–23 and the OSS description cues; signals
9, 11 belong to Credibility and 12–15 to Logistics and are read elsewhere.

**Sentinel discipline is an invariant, not a nicety.** ``github_activity_score
== -1``, ``offer_acceptance_rate == -1`` and ``skill_assessment_scores == {}``
mean *UNKNOWN*, never 0: an unknown family lowers ``bhv.behavioral_confidence``
rather than the score itself, so an engineer who simply never linked GitHub is
not mis-ranked against one with a genuinely bad signal. Recency is computed only
against the injected ``as_of`` — no wall clock.
"""

from __future__ import annotations

from datetime import date
from typing import Final

from redstack.domain.enums import EvidenceKind
from redstack.domain.provenance import EvidenceRef
from redstack.domain.ids import UnitScore
from redstack.domain.source import RawCandidate, RawSignals
from redstack.features.view import (
    ACTIVITY_HALF_LIFE_DAYS,
    STALE_HALF_LIFE_DAYS,
    RESPONSE_TIME_SCALE_HOURS,
    CellEmission,
    FeatureCell,
    bounded_log_scale,
    cell,
    clamp_unit,
    days_between,
    inverse_bounded,
    make_evidence,
    mean_of,
    recency_unit,
)

_SIGNAL = EvidenceKind.SIGNAL
_CAREER = EvidenceKind.CAREER_FIELD
_DERIVED = EvidenceKind.DERIVED

_GITHUB_SENTINEL: Final[float] = -1.0
_OFFER_SENTINEL: Final[float] = -1.0
_VIEW_SATURATION: Final[float] = 200.0
_APPLICATION_SATURATION: Final[float] = 30.0
_CONNECTION_SATURATION: Final[float] = 1500.0
_SEARCH_SATURATION: Final[float] = 100.0
_SAVES_SATURATION: Final[float] = 40.0
_SIGNUP_HALF_LIFE_DAYS: Final[float] = 365.0
_HIGH_CONF: Final[float] = 0.95
_OSS_CUES: Final[tuple[str, ...]] = (
    "open source",
    "open-source",
    "github",
    "oss",
    "contributor",
    "maintainer",
    "published",
    "patent",
    "paper",
)


def _sig_ev(field: str, value: str | int | float | bool) -> tuple[EvidenceRef, ...]:
    """One ``EvidenceRef`` tuple anchored at ``redrob_signals.<field>``."""
    return (make_evidence(_SIGNAL, f"redrob_signals.{field}", value),)


# --------------------------------------------------------------------------- #
# Open source (group 17).                                                     #
# --------------------------------------------------------------------------- #
def _oss(raw: RawCandidate) -> list[tuple[str, FeatureCell]]:
    sig = raw.redrob_signals
    github = sig.github_activity_score
    github_known = github != _GITHUB_SENTINEL

    if github_known:
        activity = clamp_unit(github / 100.0)
        activity_conf = _HIGH_CONF
    else:
        # -1 ⇒ UNKNOWN: zero value but explicitly low confidence, never a penalty.
        activity = 0.0
        activity_conf = 0.15

    # OSS cues in role descriptions corroborate external validation.
    cue_hits = 0
    cue_path = ""
    for position_index, position in enumerate(raw.career_history):
        lowered = position.description.lower()
        if any(cue in lowered for cue in _OSS_CUES):
            cue_hits += 1
            if not cue_path:
                cue_path = f"career_history[{position_index}].description"
    has_validation = clamp_unit(
        (0.6 if github_known and github > 20.0 else 0.0)
        + bounded_log_scale(float(cue_hits), saturation=3.0) * 0.6
    )
    validation_ev: tuple[EvidenceRef, ...] = _sig_ev("github_activity_score", github)
    if cue_path:
        validation_ev = (
            *validation_ev,
            make_evidence(_CAREER, cue_path, "oss_cue"),
        )

    return [
        (
            "oss.activity",
            cell(activity, activity_conf, _sig_ev("github_activity_score", github)),
        ),
        (
            "oss.has_external_validation",
            cell(has_validation, 0.6 if github_known else 0.4, validation_ev),
        ),
    ]


# --------------------------------------------------------------------------- #
# Availability (group 21).                                                    #
# --------------------------------------------------------------------------- #
def _availability(raw: RawCandidate, as_of: date) -> tuple[float, float, list[tuple[str, FeatureCell]]]:
    sig = raw.redrob_signals
    is_open = 1.0 if sig.open_to_work_flag else 0.0
    elapsed = days_between(as_of, sig.last_active_date)
    recency = recency_unit(float(elapsed), half_life_days=STALE_HALF_LIFE_DAYS)
    # Available only if reachable *and* recently active; a stale "open" flag is weak.
    available = clamp_unit(recency * (0.5 + 0.5 * is_open))
    rows: list[tuple[str, FeatureCell]] = [
        (
            "avail.open",
            cell(is_open, _HIGH_CONF, _sig_ev("open_to_work_flag", sig.open_to_work_flag)),
        ),
        (
            "avail.recency",
            cell(
                recency,
                _HIGH_CONF,
                _sig_ev("last_active_date", sig.last_active_date.isoformat()),
            ),
        ),
        (
            "avail.available",
            cell(
                available,
                _HIGH_CONF,
                (
                    make_evidence(
                        _DERIVED, "avail.available", available
                    ),
                ),
            ),
        ),
    ]
    return available, recency, rows


# --------------------------------------------------------------------------- #
# Engagement (group 22).                                                      #
# --------------------------------------------------------------------------- #
def _engagement(sig: RawSignals) -> tuple[float, float, float, list[tuple[str, FeatureCell]]]:
    views = bounded_log_scale(float(sig.profile_views_received_30d), saturation=_VIEW_SATURATION)
    searches = bounded_log_scale(float(sig.search_appearance_30d), saturation=_SEARCH_SATURATION)
    saves = bounded_log_scale(float(sig.saved_by_recruiters_30d), saturation=_SAVES_SATURATION)
    passive = mean_of((views, searches, saves))
    active = bounded_log_scale(
        float(sig.applications_submitted_30d), saturation=_APPLICATION_SATURATION
    )
    network = bounded_log_scale(float(sig.connection_count), saturation=_CONNECTION_SATURATION)
    velocity = clamp_unit(0.5 * active + 0.5 * passive)
    rows: list[tuple[str, FeatureCell]] = [
        (
            "eng.passive",
            cell(
                passive,
                _HIGH_CONF,
                (
                    make_evidence(_SIGNAL, "redrob_signals.profile_views_received_30d", sig.profile_views_received_30d),
                    make_evidence(_SIGNAL, "redrob_signals.saved_by_recruiters_30d", sig.saved_by_recruiters_30d),
                ),
            ),
        ),
        (
            "eng.active",
            cell(active, _HIGH_CONF, _sig_ev("applications_submitted_30d", sig.applications_submitted_30d)),
        ),
        (
            "eng.network",
            cell(network, _HIGH_CONF, _sig_ev("connection_count", sig.connection_count)),
        ),
        (
            "eng.velocity",
            cell(velocity, _HIGH_CONF, (make_evidence(_DERIVED, "eng.velocity", velocity),)),
        ),
    ]
    return passive, active, saves, rows


# --------------------------------------------------------------------------- #
# Responsiveness (group 23).                                                  #
# --------------------------------------------------------------------------- #
def _responsiveness(sig: RawSignals) -> tuple[float, list[tuple[str, FeatureCell]]]:
    rate = clamp_unit(sig.recruiter_response_rate)
    speed = inverse_bounded(sig.avg_response_time_hours, scale=RESPONSE_TIME_SCALE_HOURS)
    reliable = mean_of((rate, speed))
    rows: list[tuple[str, FeatureCell]] = [
        ("resp.rate", cell(rate, _HIGH_CONF, _sig_ev("recruiter_response_rate", sig.recruiter_response_rate))),
        ("resp.speed", cell(speed, _HIGH_CONF, _sig_ev("avg_response_time_hours", sig.avg_response_time_hours))),
        ("resp.reliable", cell(reliable, _HIGH_CONF, (make_evidence(_DERIVED, "resp.reliable", reliable),))),
    ]
    return reliable, rows


# --------------------------------------------------------------------------- #
# Reliability / verification families (feed bhv.* only).                      #
# --------------------------------------------------------------------------- #
def _reliability(sig: RawSignals) -> tuple[float, float, bool]:
    """Return ``(reliability, interview_rate, offer_known)``."""
    interview = clamp_unit(sig.interview_completion_rate)
    offer_known = sig.offer_acceptance_rate != _OFFER_SENTINEL
    if offer_known:
        offer = clamp_unit(sig.offer_acceptance_rate)
        reliability = mean_of((interview, offer))
    else:
        # -1 ⇒ UNKNOWN: reliability rests on interview rate alone, flagged unknown.
        reliability = interview
    return reliability, interview, offer_known


def _verification(sig: RawSignals) -> float:
    completeness = clamp_unit(sig.profile_completeness_score / 100.0)
    flags = (
        1.0 if sig.verified_email else 0.0,
        1.0 if sig.verified_phone else 0.0,
        1.0 if sig.linkedin_connected else 0.0,
    )
    return clamp_unit(0.4 * completeness + 0.6 * mean_of(flags))


def _signal_consistency(sig: RawSignals) -> float:
    """Penalize categorically incoherent signal pairs (not merely unusual)."""
    flags = 0
    if sig.saved_by_recruiters_30d > 0 and sig.profile_views_received_30d == 0:
        flags += 1  # saved but never surfaced
    if sig.applications_submitted_30d > 0 and sig.open_to_work_flag is False and sig.recruiter_response_rate == 0.0:
        flags += 0  # not impossible — applications can precede openness
    if sig.connection_count == 0 and sig.profile_views_received_30d > 50:
        flags += 1  # heavily viewed yet zero network
    return clamp_unit(1.0 - 0.5 * float(flags))


def _unknown_density(sig: RawSignals) -> float:
    unknowns = (
        1.0 if sig.github_activity_score == _GITHUB_SENTINEL else 0.0,
        1.0 if sig.offer_acceptance_rate == _OFFER_SENTINEL else 0.0,
        1.0 if len(sig.skill_assessment_scores) == 0 else 0.0,
    )
    return mean_of(unknowns)


# --------------------------------------------------------------------------- #
# Behavioral composites (Part 4 — 15 cells).                                  #
# --------------------------------------------------------------------------- #
def _behavioral(
    raw: RawCandidate,
    as_of: date,
    *,
    availability: float,
    recency: float,
    passive: float,
    active: float,
    saves: float,
    responsiveness: float,
) -> list[tuple[str, FeatureCell]]:
    sig = raw.redrob_signals
    reliability, interview, offer_known = _reliability(sig)
    verification = _verification(sig)
    consistency = _signal_consistency(sig)
    unknown_density = _unknown_density(sig)
    is_open = 1.0 if sig.open_to_work_flag else 0.0
    saves_norm = saves

    signup_recency = recency_unit(
        float(days_between(as_of, sig.signup_date)), half_life_days=_SIGNUP_HALF_LIFE_DAYS
    )
    last_active_recency = recency_unit(
        float(days_between(as_of, sig.last_active_date)), half_life_days=ACTIVITY_HALF_LIFE_DAYS
    )

    recruitability = clamp_unit(mean_of((is_open, responsiveness, saves_norm)))
    market_demand = clamp_unit(mean_of((passive, saves_norm)))
    market_momentum = clamp_unit(market_demand * last_active_recency)
    engagement_velocity = clamp_unit(0.6 * active + 0.4 * passive)
    candidate_temperature = clamp_unit(availability * recency * (0.5 + 0.5 * active))
    recruiter_attractiveness = market_demand
    hiring_probability_proxy = clamp_unit(availability * responsiveness * reliability)
    freshness = clamp_unit(0.7 * last_active_recency + 0.3 * signup_recency)
    behavioral_confidence = clamp_unit(0.5 * (1.0 - unknown_density) + 0.5 * consistency)
    behavioral_risk = clamp_unit(
        mean_of((1.0 - availability, 1.0 - reliability, 1.0 - consistency))
    )

    # Confidence is lower for families fed by a sentinel-driven UNKNOWN.
    reliability_conf = 0.9 if offer_known else 0.5

    def d(feature_id: str, value: float) -> tuple[EvidenceRef, ...]:
        return (make_evidence(_DERIVED, feature_id, value),)

    return [
        ("bhv.availability", cell(availability, _HIGH_CONF, d("bhv.availability", availability))),
        ("bhv.recruitability", cell(recruitability, 0.85, d("bhv.recruitability", recruitability))),
        ("bhv.response_reliability", cell(responsiveness, _HIGH_CONF, d("bhv.response_reliability", responsiveness))),
        (
            "bhv.interview_reliability",
            cell(interview, 0.9, _sig_ev("interview_completion_rate", sig.interview_completion_rate)),
        ),
        ("bhv.market_demand", cell(market_demand, _HIGH_CONF, d("bhv.market_demand", market_demand))),
        ("bhv.market_momentum", cell(market_momentum, 0.7, d("bhv.market_momentum", market_momentum))),
        ("bhv.engagement_velocity", cell(engagement_velocity, 0.8, d("bhv.engagement_velocity", engagement_velocity))),
        ("bhv.candidate_temperature", cell(candidate_temperature, 0.8, d("bhv.candidate_temperature", candidate_temperature))),
        ("bhv.recruiter_attractiveness", cell(recruiter_attractiveness, _HIGH_CONF, d("bhv.recruiter_attractiveness", recruiter_attractiveness))),
        ("bhv.hiring_probability_proxy", cell(hiring_probability_proxy, reliability_conf, d("bhv.hiring_probability_proxy", hiring_probability_proxy))),
        ("bhv.freshness", cell(freshness, _HIGH_CONF, d("bhv.freshness", freshness))),
        ("bhv.trust", cell(verification, 0.9, _sig_ev("profile_completeness_score", sig.profile_completeness_score))),
        ("bhv.signal_consistency", cell(consistency, 0.8, d("bhv.signal_consistency", consistency))),
        ("bhv.behavioral_confidence", cell(behavioral_confidence, _HIGH_CONF, d("bhv.behavioral_confidence", behavioral_confidence))),
        ("bhv.behavioral_risk", cell(behavioral_risk, 0.8, d("bhv.behavioral_risk", behavioral_risk))),
    ]


# --------------------------------------------------------------------------- #
# Public extractor.                                                           #
# --------------------------------------------------------------------------- #
def extract(raw: RawCandidate, *, as_of: date) -> CellEmission:
    """Emit the behavioral / engagement / availability / OSS feature cells.

    Pure function of ``(RawCandidate, as_of)``. The final behavioral *multiplier*
    is the engine's job; this layer emits only the normalized, bounded inputs.
    """
    sig = raw.redrob_signals
    rows: list[tuple[str, FeatureCell]] = []

    rows.extend(_oss(raw))

    available, recency, avail_rows = _availability(raw, as_of)
    rows.extend(avail_rows)

    passive, active, saves, eng_rows = _engagement(sig)
    rows.extend(eng_rows)

    responsiveness, resp_rows = _responsiveness(sig)
    rows.extend(resp_rows)

    rows.extend(
        _behavioral(
            raw,
            as_of,
            availability=available,
            recency=recency,
            passive=passive,
            active=active,
            saves=saves,
            responsiveness=responsiveness,
        )
    )
    return tuple(rows)


__all__ = ("extract",)