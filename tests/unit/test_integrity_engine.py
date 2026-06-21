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


def _raw(career_history: list[dict[str, object]]) -> RawCandidate:
    return RawCandidate.model_validate(
        {
            "candidate_id": "CAND_0000001",
            "profile": _BASE_PROFILE,
            "career_history": career_history,
            "education": [],
            "skills": [],
            "certifications": [],
            "languages": [],
            "redrob_signals": _BASE_SIGNALS,
        }
    )


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
