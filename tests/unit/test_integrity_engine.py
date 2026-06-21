from __future__ import annotations

from datetime import date

from redstack.config.schema import IntegrityThresholds
from redstack.domain.enums import IntegrityFlag
from redstack.domain.source import RawCandidate
from redstack.engines.integrity import IntegrityEngine
from redstack.features.extraction import build_career_profile

_THRESHOLDS = IntegrityThresholds(
    honeypot_threshold=0.5,
    tolerance_experience_years=2.0,
    duration_date_tolerance_months=2.0,
    expert_zero_usage_min_count=2,
    experience_predates_tolerance_years=2.0,
)
_ENGINE = IntegrityEngine(thresholds=_THRESHOLDS)

_BASE_PROFILE = {
    "anonymized_name": "Test User",
    "headline": "x",
    "summary": "x",
    "location": "Bangalore, KA",
    "country": "India",
    "years_of_experience": 5.0,
    "current_title": "Engineer",
    "current_company": "Acme",
    "current_company_size": "11-50",
    "current_industry": "software",
}
_BASE_SIGNALS = {
    "profile_completeness_score": 50.0,
    "signup_date": "2020-01-01",
    "last_active_date": "2024-01-01",
    "open_to_work_flag": True,
    "profile_views_received_30d": 0,
    "applications_submitted_30d": 0,
    "recruiter_response_rate": 0.0,
    "avg_response_time_hours": 1.0,
    "skill_assessment_scores": {},
    "connection_count": 0,
    "endorsements_received": 0,
    "notice_period_days": 30,
    "expected_salary_range_inr_lpa": {"min": 10, "max": 20},
    "preferred_work_mode": "remote",
    "willing_to_relocate": True,
    "github_activity_score": -1.0,
    "search_appearance_30d": 0,
    "saved_by_recruiters_30d": 0,
    "interview_completion_rate": 0.0,
    "offer_acceptance_rate": -1.0,
    "verified_email": True,
    "verified_phone": True,
    "linkedin_connected": True,
}


def _raw(
    career_history: list[dict[str, object]],
    *,
    education: list[dict[str, object]] | None = None,
) -> RawCandidate:
    return RawCandidate.model_validate(
        {
            "candidate_id": "CAND_0000001",
            "profile": _BASE_PROFILE,
            "career_history": career_history,
            "education": education if education is not None else [],
            "skills": [],
            "certifications": [],
            "languages": [],
            "redrob_signals": _BASE_SIGNALS,
        }
    )


def _education(
    *, degree: str, start_year: int, end_year: int
) -> dict[str, object]:
    return {
        "institution": "Test University",
        "degree": degree,
        "field_of_study": "Computer Science",
        "start_year": start_year,
        "end_year": end_year,
        "grade": None,
        "tier": "tier_2",
    }


def test_role_duration_mismatch_evidence_cites_the_flagged_raw_position() -> None:
    """Regression: career.positions is re-sorted reverse-chronologically by
    build_career_profile, so its index does not generally match raw.career_history's
    index. The engine listed raw.career_history positions oldest-first (idx 0 =
    OldCo, idx 1 = NewCo); career.positions[0] (the chronologically-newest, the one
    that actually has the date/duration contradiction) is NewCo. The evidence must
    cite career_history[1] (NewCo), not career_history[0] (OldCo, an unrelated,
    contradiction-free position).
    """
    raw = _raw(
        [
            {
                "company": "OldCo",
                "title": "Junior Dev",
                "start_date": "2015-01-01",
                "end_date": "2017-01-01",
                "duration_months": 24,
                "is_current": False,
                "industry": "software",
                "company_size": "11-50",
                "description": "old job",
            },
            {
                "company": "NewCo",
                "title": "Senior Dev",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "duration_months": 999,
                "is_current": False,
                "industry": "software",
                "company_size": "11-50",
                "description": "new job, duration mismatch",
            },
        ]
    )
    career = build_career_profile(raw, as_of=date(2024, 6, 1))
    assert career.positions[0].company == "NewCo"
    assert raw.career_history[0].company == "OldCo"

    report = _ENGINE.evaluate(career, raw)
    findings = [
        f
        for f in report.findings
        if f.code is IntegrityFlag.ROLE_DURATION_DATE_MISMATCH
    ]
    assert len(findings) == 1
    paths = {e.path for e in findings[0].evidence}
    assert all(path.startswith("career_history[1]") for path in paths), paths


def test_current_role_has_end_date_evidence_cites_the_flagged_raw_position() -> None:
    """Same misattribution risk for rule 3 (current role carrying an end_date)."""
    raw = _raw(
        [
            {
                "company": "OldCo",
                "title": "Junior Dev",
                "start_date": "2015-01-01",
                "end_date": "2017-01-01",
                "duration_months": 24,
                "is_current": False,
                "industry": "software",
                "company_size": "11-50",
                "description": "old job",
            },
            {
                "company": "NewCo",
                "title": "Senior Dev",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "duration_months": 12,
                "is_current": True,
                "industry": "software",
                "company_size": "11-50",
                "description": "current job that also carries an end_date",
            },
        ]
    )
    career = build_career_profile(raw, as_of=date(2024, 6, 1))
    assert career.positions[0].company == "NewCo"
    assert raw.career_history[0].company == "OldCo"

    report = _ENGINE.evaluate(career, raw)
    findings = [
        f for f in report.findings if f.code is IntegrityFlag.CURRENT_ROLE_HAS_END_DATE
    ]
    assert len(findings) == 1
    paths = {e.path for e in findings[0].evidence}
    assert all(path.startswith("career_history[1]") for path in paths), paths


_ONE_POSITION = [
    {
        "company": "Acme",
        "title": "Engineer",
        "start_date": "2022-01-01",
        "end_date": None,
        "duration_months": 24,
        "is_current": True,
        "industry": "software",
        "company_size": "11-50",
        "description": "current job",
    }
]


def test_degree_rank_backwards_fires_for_bachelors_after_masters() -> None:
    """Regression: a Master's degree (2003-2008) followed by a Bachelor's
    degree (2011-2016) is an impossible academic sequence -- you don't enroll
    in undergrad after finishing a graduate degree -- but neither individual
    entry is internally backwards (each has end_year >= start_year), so rule 5
    (_education_timeline_impossible) alone never caught it.
    """
    raw = _raw(
        _ONE_POSITION,
        education=[
            _education(degree="M.E.", start_year=2003, end_year=2008),
            _education(degree="B.Tech", start_year=2011, end_year=2016),
        ],
    )
    career = build_career_profile(raw, as_of=date(2024, 6, 1))
    report = _ENGINE.evaluate(career, raw)
    findings = [
        f for f in report.findings if f.code is IntegrityFlag.EDUCATION_TIMELINE_IMPOSSIBLE
    ]
    assert len(findings) == 1
    paths = {e.path for e in findings[0].evidence}
    assert any(path.startswith("education[1]") for path in paths), paths


def test_degree_rank_forwards_does_not_fire() -> None:
    """A normal Bachelor's-then-Master's sequence must not be flagged."""
    raw = _raw(
        _ONE_POSITION,
        education=[
            _education(degree="B.Tech", start_year=2011, end_year=2015),
            _education(degree="M.Tech", start_year=2015, end_year=2017),
        ],
    )
    career = build_career_profile(raw, as_of=date(2024, 6, 1))
    report = _ENGINE.evaluate(career, raw)
    findings = [
        f for f in report.findings if f.code is IntegrityFlag.EDUCATION_TIMELINE_IMPOSSIBLE
    ]
    assert findings == []


def test_degree_rank_same_rank_does_not_fire() -> None:
    """Two same-rank degrees (e.g. a second Bachelor's) are never "higher
    rank than each other" regardless of order, so must never fire."""
    raw = _raw(
        _ONE_POSITION,
        education=[
            _education(degree="B.Tech", start_year=2011, end_year=2015),
            _education(degree="B.Sc", start_year=2016, end_year=2019),
        ],
    )
    career = build_career_profile(raw, as_of=date(2024, 6, 1))
    report = _ENGINE.evaluate(career, raw)
    findings = [
        f for f in report.findings if f.code is IntegrityFlag.EDUCATION_TIMELINE_IMPOSSIBLE
    ]
    assert findings == []
