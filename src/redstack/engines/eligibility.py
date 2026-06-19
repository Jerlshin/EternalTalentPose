"""``EligibilityEngine`` — JD hard disqualifiers + soft penalties (``engines/eligibility.py``).

Owner layer: engines. Allowed imports: ``domain``, ``features``, ``config.schema``.
Forbidden: ``adapters``, ``pipelines``, ``observability``, sibling engines, clock,
online RNG.

Logical→physical (Repo §9): physical home of the ``CandidateFitEngine`` eligibility
half. Evaluates the JD's explicit "do-NOT-want" list as authored predicate data
(``EligibilityRuleSet`` from ``configs/gates/eligibility_rules.yaml``, surfaced via
``config.schema``) against the structural + semantic representation slices, and
emits the ``EligibilityReport``.

Contract (Domain §G.3):

    is_eligible == (len(hard_blocks) == 0)

Soft penalties never set ``is_eligible = False``; they feed Scoring as bounded
down-weights and Reasoning as honest concerns. Every finding carries >=1
``EvidenceRef`` and the report's tuples are sorted by code (determinism).
"""

from __future__ import annotations

from typing import Final, final

from pydantic import BaseModel, ConfigDict

from redstack.config.schema import EligibilityRuleSet
from redstack.domain.candidate.career import CareerProfile
from redstack.domain.candidate.credibility import CredibilityProfile
from redstack.domain.candidate.eligibility import (
    EligibilityFinding,
    EligibilityReport,
)
from redstack.domain.candidate.logistics import LogisticsProfile
from redstack.domain.candidate.semantic import SemanticProfile
from redstack.domain.enums import (
    CareerTrack,
    EligibilityCode,
    EvidenceKind,
    LocationFit,
    NoticeFit,
    Severity,
)
from redstack.domain.jd import JobDescriptionSpec
from redstack.domain.provenance import EvidenceRef
from redstack.features.view import make_evidence

_HARD_CODES: Final[frozenset[EligibilityCode]] = frozenset(
    (
        EligibilityCode.PURE_RESEARCH_NO_PRODUCTION,
        EligibilityCode.LANGCHAIN_OPENAI_ONLY_RECENT,
        EligibilityCode.NO_PRODUCTION_CODE_18M,
        EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER,
        EligibilityCode.PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP,
        EligibilityCode.CLOSED_SOURCE_5Y_NO_VALIDATION,
    )
)
_SOFT_CODES: Final[frozenset[EligibilityCode]] = frozenset(
    (
        EligibilityCode.TITLE_CHASER_SUB_18M_HOPS,
        EligibilityCode.NOTICE_OVER_30,
        EligibilityCode.OUTSIDE_INDIA_NO_SPONSOR,
        EligibilityCode.OUTSIDE_EXPERIENCE_BAND,
    )
)
_ALL_CODES: Final[frozenset[EligibilityCode]] = _HARD_CODES | _SOFT_CODES
#: Hard codes decidable from CareerProfile + CredibilityProfile alone --
#: PURE_RESEARCH_NO_PRODUCTION and PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP need
#: SemanticProfile/skill_match (only available once the candidate is embedded)
#: and are deliberately excluded. Lets a caller pre-gate before the expensive
#: embedding step for the subset that doesn't need its output.
_STRUCTURAL_HARD_CODES: Final[frozenset[EligibilityCode]] = frozenset(
    (
        EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER,
        EligibilityCode.LANGCHAIN_OPENAI_ONLY_RECENT,
        EligibilityCode.NO_PRODUCTION_CODE_18M,
    )
)


@final
class EligibilityEngine(BaseModel):
    """Stateless, pure JD-fit gating engine.

    Detector parameters live in the injected ``EligibilityRuleSet`` (authored data,
    not code), so re-targeting the JD never edits this module.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    rules: EligibilityRuleSet

    # ------------------------------------------------------------------ public
    def evaluate(
        self,
        *,
        career: CareerProfile,
        credibility: CredibilityProfile,
        semantic: SemanticProfile,
        logistics: LogisticsProfile,
        jd: JobDescriptionSpec,
        skill_match: float = 1.0,
    ) -> EligibilityReport:
        """Evaluate every gate; partition into hard blocks and soft penalties.

        ``skill_match`` is the mean of the nine retr/rank/recsys/ir/nlp/llm/
        mle/mlops/eval ``.competency`` cells (the same aggregate Scoring's
        ``SKILL_MATCH`` component reads) — the one component diagnostic
        evidence showed reliably tracks ML/AI domain relevance. Defaults to
        ``1.0`` (never blocks) so existing call sites that don't pass it keep
        their prior behavior.
        """
        candidates: list[EligibilityFinding | None] = [
            self._pure_research_no_production(career, semantic),
            self._langchain_openai_only_recent(credibility),
            self._no_production_code_18m(career),
            self._consulting_firms_only_career(career),
            self._primary_cv_speech_robotics_no_nlp(credibility, semantic, skill_match),
            self._closed_source_5y_no_validation(career, credibility),
            self._title_chaser_sub_18m_hops(career),
            self._notice_over_30(logistics),
            self._outside_india_no_sponsor(logistics),
            self._outside_experience_band(career),
        ]
        fired = [f for f in candidates if f is not None]

        hard_blocks = tuple(
            sorted(
                (f for f in fired if f.code in _HARD_CODES), key=lambda f: f.code.value
            )
        )
        soft_penalties = tuple(
            sorted(
                (f for f in fired if f.code in _SOFT_CODES), key=lambda f: f.code.value
            )
        )
        fired_codes = {f.code for f in fired}
        gates_passed = frozenset(_ALL_CODES - fired_codes)
        return EligibilityReport(
            hard_blocks=hard_blocks,
            soft_penalties=soft_penalties,
            is_eligible=(len(hard_blocks) == 0),
            gates_passed=gates_passed,
        )

    def evaluate_structural(
        self, *, career: CareerProfile, credibility: CredibilityProfile
    ) -> tuple[EligibilityFinding, ...]:
        """Pre-embedding hard-gate subset (``_STRUCTURAL_HARD_CODES`` only).

        For callers that need to know whether a candidate is *already*
        certain to be hard-blocked before the embedding step runs (so they
        can skip it): consulting-only career, langchain/openai-only-recent,
        and no-production-code-18m all read only ``career``/``credibility``.
        Findings are sorted by code, matching :meth:`evaluate`'s convention.
        Does not run the soft detectors or the two semantic-dependent hard
        detectors -- a candidate with no findings here may still be blocked
        once a real :class:`SemanticProfile` exists.
        """
        candidates: list[EligibilityFinding | None] = [
            self._consulting_firms_only_career(career),
            self._langchain_openai_only_recent(credibility),
            self._no_production_code_18m(career),
        ]
        return tuple(
            sorted((f for f in candidates if f is not None), key=lambda f: f.code.value)
        )

    # --------------------------------------------------------------- builders
    @staticmethod
    def _finding(
        code: EligibilityCode,
        severity: Severity,
        evidence: tuple[EvidenceRef, ...],
        detail: str,
    ) -> EligibilityFinding:
        return EligibilityFinding(
            code=code, severity=severity, evidence=evidence, detail=detail
        )

    @staticmethod
    def _has_production_company(career: CareerProfile) -> bool:
        return any(p.is_product_company for p in career.positions)

    # --------------------------------------------------------- HARD detectors
    def _pure_research_no_production(
        self, career: CareerProfile, semantic: SemanticProfile
    ) -> EligibilityFinding | None:
        if self._has_production_company(career):
            return None
        if float(semantic.net_semantic_fit) < self.rules.research_min_semantic_fit:
            return None
        if career.track not in (CareerTrack.SERVICES, CareerTrack.UNKNOWN):
            return None
        return self._finding(
            EligibilityCode.PURE_RESEARCH_NO_PRODUCTION,
            Severity.HARD,
            (
                make_evidence(EvidenceKind.DERIVED, "career.track", career.track.value),
                make_evidence(
                    EvidenceKind.DERIVED,
                    "semantic.net_semantic_fit",
                    float(semantic.net_semantic_fit),
                ),
            ),
            "research-aligned profile with no production-company tenure",
        )

    def _langchain_openai_only_recent(
        self, credibility: CredibilityProfile
    ) -> EligibilityFinding | None:
        stuffing = float(credibility.keyword_stuffing_score)
        gap = float(credibility.claimed_vs_assessed_gap)
        if (
            stuffing < self.rules.framework_only_stuffing_min
            or gap < self.rules.framework_only_gap_min
        ):
            return None
        return self._finding(
            EligibilityCode.LANGCHAIN_OPENAI_ONLY_RECENT,
            Severity.HARD,
            (
                make_evidence(
                    EvidenceKind.DERIVED, "credibility.keyword_stuffing_score", stuffing
                ),
                make_evidence(
                    EvidenceKind.DERIVED, "credibility.claimed_vs_assessed_gap", gap
                ),
            ),
            "recent framework-name claims without corroborated depth",
        )

    def _no_production_code_18m(
        self, career: CareerProfile
    ) -> EligibilityFinding | None:
        months = int(career.recency.months_since_last_role)
        produced_recently = any(
            p.is_product_company and p.is_current for p in career.positions
        )
        if produced_recently or months <= self.rules.production_recency_max_months:
            return None
        return self._finding(
            EligibilityCode.NO_PRODUCTION_CODE_18M,
            Severity.HARD,
            (
                make_evidence(
                    EvidenceKind.DERIVED,
                    "career.recency.months_since_last_role",
                    months,
                ),
            ),
            f"no production-company role within {months}m",
        )

    def _consulting_firms_only_career(
        self, career: CareerProfile
    ) -> EligibilityFinding | None:
        if not career.positions:
            return None
        if not all(p.is_consulting_firm for p in career.positions):
            return None
        return self._finding(
            EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER,
            Severity.HARD,
            (
                make_evidence(EvidenceKind.DERIVED, "career.track", career.track.value),
                make_evidence(
                    EvidenceKind.DERIVED,
                    "career.consulting_only",
                    True,
                ),
            ),
            "entire career at consulting firms; no product-side delivery",
        )

    def _primary_cv_speech_robotics_no_nlp(
        self,
        credibility: CredibilityProfile,
        semantic: SemanticProfile,
        skill_match: float,
    ) -> EligibilityFinding | None:
        """Gate on demonstrated relevant-skill credibility OR skill_match.

        Originally required ``negative_fit`` to also clear a threshold (i.e.
        the candidate had to resemble a *specific* adjacent archetype —
        vision/speech/robotics). Diagnostic evidence across 27 real candidates
        showed ``negative_fit`` clustering at 0.21-0.36 for both clearly
        relevant (ML/Applied-Science/RecSys, ``relevant_skill_credibility``
        0.80-0.94) and clearly irrelevant (HR Manager, Civil Engineer,
        Accountant, ..., 0.11-0.50) candidates alike — it carries no domain
        signal here, so AND-ing it in only suppressed the gate.

        ``relevant_skill_credibility`` alone (fraction of *all* claimed skills
        that are corroborated) still let through a second failure mode: a
        candidate whose few listed skills are all non-ML but well-corroborated
        (e.g. a Marketing Manager with credibly-endorsed marketing skills)
        scores high on it despite zero ML/AI overlap. ``skill_match`` (mean of
        the nine ML-specific competency cells) catches that case directly —
        every sampled non-ML title scored at or near 0.0 there, regardless of
        how well-corroborated their (non-ML) skills were. Both bars must clear
        to pass — failing *either* is sufficient grounds to block.
        """
        rel = float(credibility.relevant_skill_credibility)
        neg = float(semantic.negative_fit)
        if (
            rel >= self.rules.adjacent_domain_min_relevant_credibility
            and skill_match >= self.rules.adjacent_domain_min_skill_match
        ):
            return None
        return self._finding(
            EligibilityCode.PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP,
            Severity.HARD,
            (
                make_evidence(
                    EvidenceKind.DERIVED,
                    "credibility.relevant_skill_credibility",
                    rel,
                ),
                make_evidence(
                    EvidenceKind.DERIVED, "semantic.negative_fit", neg
                ),
                make_evidence(
                    EvidenceKind.DERIVED, "derived.skill_match", skill_match
                ),
            ),
            "no demonstrated relevant ML/AI skill credibility for this role",
        )

    def _closed_source_5y_no_validation(
        self, career: CareerProfile, credibility: CredibilityProfile
    ) -> EligibilityFinding | None:
        credible_skills = sum(
            1 for trust in credibility.skill_trust.values() if trust.is_credible
        )
        long_tenure = (
            float(career.derived_experience_years) >= self.rules.closed_source_min_years
        )
        no_validation = credible_skills <= self.rules.closed_source_max_credible_skills
        if not (long_tenure and no_validation):
            return None
        return self._finding(
            EligibilityCode.CLOSED_SOURCE_5Y_NO_VALIDATION,
            Severity.HARD,
            (
                make_evidence(
                    EvidenceKind.DERIVED,
                    "career.derived_experience_years",
                    float(career.derived_experience_years),
                ),
                make_evidence(
                    EvidenceKind.DERIVED, "credibility.credible_skill_count", credible_skills
                ),
            ),
            "long closed-source tenure with no externally validated competency",
        )

    # --------------------------------------------------------- SOFT detectors
    def _title_chaser_sub_18m_hops(
        self, career: CareerProfile
    ) -> EligibilityFinding | None:
        hop_rate = float(career.tenure.hop_rate)
        if hop_rate < self.rules.title_chaser_min_hop_rate:
            return None
        return self._finding(
            EligibilityCode.TITLE_CHASER_SUB_18M_HOPS,
            Severity.SOFT,
            (
                make_evidence(EvidenceKind.DERIVED, "career.tenure.hop_rate", hop_rate),
                make_evidence(
                    EvidenceKind.DERIVED,
                    "career.tenure.mean_tenure_months",
                    int(career.tenure.mean_tenure_months),
                ),
            ),
            f"hop rate {hop_rate:.2f}: many sub-18-month tenures",
        )

    def _notice_over_30(self, logistics: LogisticsProfile) -> EligibilityFinding | None:
        if logistics.notice_fit is not NoticeFit.OVER_30_HIGHER_BAR:
            return None
        return self._finding(
            EligibilityCode.NOTICE_OVER_30,
            Severity.SOFT,
            (
                make_evidence(
                    EvidenceKind.SIGNAL,
                    "redrob_signals.notice_period_days",
                    int(logistics.notice_period_days),
                ),
            ),
            f"notice period {int(logistics.notice_period_days)}d exceeds 30",
        )

    def _outside_india_no_sponsor(
        self, logistics: LogisticsProfile
    ) -> EligibilityFinding | None:
        if logistics.location_fit is not LocationFit.OUTSIDE_INDIA_NO_SPONSOR:
            return None
        return self._finding(
            EligibilityCode.OUTSIDE_INDIA_NO_SPONSOR,
            Severity.SOFT,
            (
                make_evidence(
                    EvidenceKind.PROFILE_FIELD, "profile.country", logistics.country
                ),
            ),
            "located outside India without sponsorship path",
        )

    def _outside_experience_band(
        self, career: CareerProfile
    ) -> EligibilityFinding | None:
        years = float(career.derived_experience_years)
        lo = self.rules.experience_band_min_years
        hi = self.rules.experience_band_max_years
        if lo <= years <= hi:
            return None
        return self._finding(
            EligibilityCode.OUTSIDE_EXPERIENCE_BAND,
            Severity.SOFT,
            (
                make_evidence(
                    EvidenceKind.DERIVED, "career.derived_experience_years", years
                ),
            ),
            f"derived experience {years:.1f}y outside band [{lo:.1f}, {hi:.1f}]",
        )


__all__: tuple[str, ...] = ("EligibilityEngine",)