

from __future__ import annotations

from datetime import date

from redstack.config.schema import default_eligibility_rules
from redstack.domain.candidate.career import (
    CareerProfile,
    CareerRecency,
    PositionFact,
    TenureStats,
)
from redstack.domain.candidate.credibility import CredibilityProfile
from redstack.domain.enums import CareerTrack, CompanySize, EligibilityCode
from redstack.engines.eligibility import EligibilityEngine

_RULES = default_eligibility_rules()
_ENGINE = EligibilityEngine(rules=_RULES)


def _position(
    *,
    company: str = "Acme Tech",
    is_product_company: bool,
    is_consulting_firm: bool,
    is_current: bool = True,
    duration_months: int = 24,
) -> PositionFact:
    return PositionFact(
        company=company,
        title="ML Engineer",
        start_date=date(2022, 1, 1),
        end_date=None if is_current else date(2023, 1, 1),
        duration_months=duration_months,
        is_current=is_current,
        industry="software",
        company_size=CompanySize.S_201_500,
        is_product_company=is_product_company,
        is_consulting_firm=is_consulting_firm,
        description_role_match=0.5,
    )


def _career(
    *,
    positions: tuple[PositionFact, ...],
    track: CareerTrack = CareerTrack.PRODUCT,
    months_since_last_role: int = 0,
) -> CareerProfile:
    return CareerProfile(
        stated_experience_years=5.0,
        derived_experience_years=5.0,
        positions=positions,
        current_position=next((p for p in positions if p.is_current), None),
        track=track,
        tenure=TenureStats(
            position_count=len(positions),
            mean_tenure_months=24.0,
            min_tenure_months=12.0,
            hop_rate=0.1,
        ),
        recency=CareerRecency(
            most_recent_start=date(2022, 1, 1),
            is_currently_employed=any(p.is_current for p in positions),
            months_since_last_role=months_since_last_role,
        ),
        title_consistency=0.5,
    )


def _credibility(
    *, keyword_stuffing_score: float = 0.1, claimed_vs_assessed_gap: float = 0.1
) -> CredibilityProfile:
    return CredibilityProfile(
        skill_trust={},
        keyword_stuffing_score=keyword_stuffing_score,
        claimed_vs_assessed_gap=claimed_vs_assessed_gap,
        title_description_coherence=0.8,
        relevant_skill_credibility=0.8,
    )


def test_consulting_only_career_fires() -> None:
    career = _career(
        positions=(
            _position(is_product_company=False, is_consulting_firm=True),
            _position(
                company="TCS",
                is_product_company=False,
                is_consulting_firm=True,
                is_current=False,
            ),
        )
    )
    findings = _ENGINE.evaluate_structural(career=career, credibility=_credibility())
    codes = {f.code for f in findings}
    assert EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER in codes


def test_consulting_with_prior_product_tenure_passes() -> None:
    """Currently at a consulting firm but with prior product-company tenure
    must NOT fire consulting-only -- the JD's explicit carve-out."""
    career = _career(
        positions=(
            _position(is_product_company=False, is_consulting_firm=True),
            _position(
                company="Acme Product Co",
                is_product_company=True,
                is_consulting_firm=False,
                is_current=False,
            ),
        )
    )
    findings = _ENGINE.evaluate_structural(career=career, credibility=_credibility())
    codes = {f.code for f in findings}
    assert EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER not in codes


def test_no_production_code_18m_fires() -> None:
    career = _career(
        positions=(
            _position(
                is_product_company=False, is_consulting_firm=False, is_current=False
            ),
        ),
        months_since_last_role=24,
    )
    findings = _ENGINE.evaluate_structural(career=career, credibility=_credibility())
    codes = {f.code for f in findings}
    assert EligibilityCode.NO_PRODUCTION_CODE_18M in codes


def test_no_production_code_18m_passes_when_currently_at_product_company() -> None:
    career = _career(
        positions=(_position(is_product_company=True, is_consulting_firm=False),),
        months_since_last_role=0,
    )
    findings = _ENGINE.evaluate_structural(career=career, credibility=_credibility())
    codes = {f.code for f in findings}
    assert EligibilityCode.NO_PRODUCTION_CODE_18M not in codes


def test_langchain_openai_only_recent_fires() -> None:
    career = _career(
        positions=(_position(is_product_company=True, is_consulting_firm=False),)
    )
    credibility = _credibility(
        keyword_stuffing_score=_RULES.framework_only_stuffing_min + 0.1,
        claimed_vs_assessed_gap=_RULES.framework_only_gap_min + 0.1,
    )
    findings = _ENGINE.evaluate_structural(career=career, credibility=credibility)
    codes = {f.code for f in findings}
    assert EligibilityCode.LANGCHAIN_OPENAI_ONLY_RECENT in codes


def test_clean_candidate_has_no_structural_findings() -> None:
    career = _career(
        positions=(_position(is_product_company=True, is_consulting_firm=False),)
    )
    findings = _ENGINE.evaluate_structural(career=career, credibility=_credibility())
    assert findings == ()


def test_only_structural_codes_can_appear() -> None:
    """Excludes PURE_RESEARCH_NO_PRODUCTION/PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP --
    they need a SemanticProfile this method never receives."""
    career = _career(
        positions=(_position(is_product_company=False, is_consulting_firm=True),),
        months_since_last_role=24,
    )
    credibility = _credibility(
        keyword_stuffing_score=_RULES.framework_only_stuffing_min + 0.1,
        claimed_vs_assessed_gap=_RULES.framework_only_gap_min + 0.1,
    )
    findings = _ENGINE.evaluate_structural(career=career, credibility=credibility)
    allowed = {
        EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER,
        EligibilityCode.LANGCHAIN_OPENAI_ONLY_RECENT,
        EligibilityCode.NO_PRODUCTION_CODE_18M,
    }
    codes = [f.code for f in findings]
    assert codes, "expected multiple structural findings to fire in this fixture"
    assert set(codes) <= allowed
    assert codes == sorted(codes, key=lambda c: c.value)
