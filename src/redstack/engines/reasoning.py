"""``ReasoningEngine`` — evidence-grounded explanation (``engines/reasoning.py``).

Owner layer: engines. Allowed imports: ``domain``. Forbidden: ``adapters``,
``pipelines``, ``observability``, sibling engines, ports, clock, online RNG, and
**any LLM / network** (Stage-4 reasoning is 100% local, deterministic, non-
hallucinatory).

Logical→physical (Repo §9): the ``CandidateReasoningEngine``. For each ranked
candidate it assembles ``ReasoningClause``s strictly from structural evidence that
already exists in the ``ScoreBreakdown`` (per-component ``EvidenceRef``s) and the
``EligibilityReport`` soft penalties — every clause therefore cites ≥1 resolvable
``EvidenceRef`` (the no-hallucination guarantee is mechanical at construction).

Designed against the six Stage-4 checks (Domain §J):
1. specific facts — clauses carry the component/finding's real evidence;
2. JD connection — ≥1 clause carries a non-null ``jd_link`` (``ScoreComponent`` /
   ``EligibilityCode``);
3. honest concerns — a ``CONCERN`` clause is emitted whenever a soft penalty exists;
4. rank consistency — tone follows ``rank_band`` (``top`` net-positive, ``tail``
   measured); ``rendered`` ≤ 2 sentences;
5/6. determinism without templating — ``rendered`` is a pure function of the
   *ordered clause set*; variation arises from *which* components/penalties are
   present, never from randomness or name-insertion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, final

from pydantic import BaseModel, ConfigDict

from redstack.domain.candidate.eligibility import EligibilityFinding
from redstack.domain.candidate.representation import CandidateRepresentation
from redstack.domain.enums import EligibilityCode, ReasoningPolarity, ScoreComponent
from redstack.domain.errors import ProvenanceError
from redstack.domain.ids import CandidateId
from redstack.domain.provenance import EvidenceRef
from redstack.domain.ranking import RankedCandidate, Ranking
from redstack.domain.reasoning import CandidateReasoning, ReasoningClause
from redstack.domain.scoring import ScoreComponentValue

RankBand = Literal["top", "mid", "tail"]

# Per-component strength phrasings: distinct content per component so that two
# candidates with different dominant components render different skeletons. The
# concrete value comes from the cited evidence (Stage-4 "specific facts").
_STRENGTH_PHRASING: Final[Mapping[ScoreComponent, str]] = {
    ScoreComponent.SKILL_MATCH: "corroborated skill match ({v:.2f}) to the role's core competencies",
    ScoreComponent.SEMANTIC_FIT: "strong semantic fit ({v:.2f}) against the JD anchors",
    ScoreComponent.CAREER_FIT: "career trajectory aligned with the role ({v:.2f})",
    ScoreComponent.EXPERIENCE_FIT: "experience squarely in the target band ({v:.2f})",
    ScoreComponent.EDUCATION_FIT: "relevant educational background ({v:.2f})",
    ScoreComponent.CREDIBILITY: "high evidence credibility ({v:.2f}), low keyword-stuffing risk",
    ScoreComponent.ARCHETYPE_FIT: "matches a target candidate archetype ({v:.2f})",
}

_CONCERN_PHRASING: Final[Mapping[EligibilityCode, str]] = {
    EligibilityCode.TITLE_CHASER_SUB_18M_HOPS: "frequent sub-18-month role changes",
    EligibilityCode.NOTICE_OVER_30: "notice period exceeds 30 days",
    EligibilityCode.OUTSIDE_INDIA_NO_SPONSOR: "based outside India without a sponsorship path",
    EligibilityCode.OUTSIDE_EXPERIENCE_BAND: "experience falls outside the target band",
    EligibilityCode.PURE_RESEARCH_NO_PRODUCTION: "research focus with limited production delivery",
    EligibilityCode.LANGCHAIN_OPENAI_ONLY_RECENT: "framework-level exposure without corroborated depth",
    EligibilityCode.NO_PRODUCTION_CODE_18M: "no recent production engineering",
    EligibilityCode.CONSULTING_FIRMS_ONLY_CAREER: "exclusively consulting-firm experience",
    EligibilityCode.PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP: "primary expertise in an adjacent domain",
    EligibilityCode.CLOSED_SOURCE_5Y_NO_VALIDATION: "long closed-source tenure without external validation",
}

_MAX_STRENGTHS: Final[int] = 2
_MAX_CONCERNS: Final[int] = 2


@final
class ReasoningEngine(BaseModel):
    """Stateless, pure, local reasoning engine over the ranked top-K."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ public
    def explain(
        self,
        ranking: Ranking,
        representations: Mapping[CandidateId, CandidateRepresentation],
    ) -> Ranking:
        """Build reasoning for every ranked candidate; attach via copy-on-write."""
        reasoning_by_id: dict[CandidateId, CandidateReasoning] = {}
        for ranked in ranking.ordered:
            rep = representations.get(ranked.candidate_id)
            if rep is None:
                raise ProvenanceError(
                    f"top-K representation for {ranked.candidate_id} not hydrated"
                )
            reasoning_by_id[ranked.candidate_id] = self.reason_for(ranked, rep)
        return ranking.with_reasoning(reasoning_by_id)

    def reason_for(
        self, ranked: RankedCandidate, representation: CandidateRepresentation
    ) -> CandidateReasoning:
        """Assemble one candidate's evidence-grounded reasoning."""
        band = self._rank_band(ranked.rank, ranked_size=self._size_of(ranked))
        strengths = self._strength_clauses(ranked)
        concerns = self._concern_clauses(representation)

        clauses: tuple[ReasoningClause, ...] = (*strengths, *concerns)
        if not clauses:
            # Defence: a ranked candidate always has a positive contributor.
            clauses = (self._fallback_strength(ranked),)

        return CandidateReasoning.assemble(
            candidate_id=ranked.candidate_id, clauses=clauses, rank_band=band
        )

    # --------------------------------------------------------------- internals
    @staticmethod
    def _size_of(ranked: RankedCandidate) -> int:
        # Rank band uses the canonical submission size; rank itself is 1-based.
        return 100

    @staticmethod
    def _rank_band(rank: int, *, ranked_size: int) -> RankBand:
        top_cut = max(1, round(ranked_size * 0.10))
        mid_cut = max(top_cut + 1, round(ranked_size * 0.50))
        if rank <= top_cut:
            return "top"
        if rank <= mid_cut:
            return "mid"
        return "tail"

    def _strength_clauses(
        self, ranked: RankedCandidate
    ) -> tuple[ReasoningClause, ...]:
        components = ranked.scored.breakdown.components
        # Strongest by weighted contribution; only those with citable evidence.
        ranked_components = sorted(
            (c for c in components if float(c.raw) > 0.0 and c.evidence),
            key=lambda c: (-float(c.weighted), c.component.value),
        )
        clauses: list[ReasoningClause] = []
        for component_value in ranked_components[:_MAX_STRENGTHS]:
            clauses.append(self._strength_clause(component_value))
        return tuple(clauses)

    @staticmethod
    def _strength_clause(component_value: ScoreComponentValue) -> ReasoningClause:
        phrasing = _STRENGTH_PHRASING[component_value.component]
        fragment = phrasing.format(v=float(component_value.raw))
        return ReasoningClause(
            polarity=ReasoningPolarity.STRENGTH,
            fragment=fragment,
            evidence=component_value.evidence,
            jd_link=component_value.component,
        )

    def _concern_clauses(
        self, representation: CandidateRepresentation
    ) -> tuple[ReasoningClause, ...]:
        if representation.eligibility is None:
            return ()
        penalties = representation.eligibility.soft_penalties
        clauses: list[ReasoningClause] = []
        for finding in penalties[:_MAX_CONCERNS]:
            clauses.append(self._concern_clause(finding))
        return tuple(clauses)

    @staticmethod
    def _concern_clause(finding: EligibilityFinding) -> ReasoningClause:
        fragment = _CONCERN_PHRASING.get(finding.code, finding.detail)
        evidence: tuple[EvidenceRef, ...] = finding.evidence
        return ReasoningClause(
            polarity=ReasoningPolarity.CONCERN,
            fragment=fragment,
            evidence=evidence,
            jd_link=finding.code,
        )

    @staticmethod
    def _fallback_strength(ranked: RankedCandidate) -> ReasoningClause:
        # Cite the highest-weighted component regardless of raw, guaranteeing
        # a citable evidence ref; never fabricates text without a backing fact.
        components = ranked.scored.breakdown.components
        best = max(
            (c for c in components if c.evidence),
            key=lambda c: float(c.weighted),
            default=None,
        )
        if best is None:
            raise ProvenanceError(
                f"no citable component evidence for {ranked.candidate_id}"
            )
        phrasing = _STRENGTH_PHRASING[best.component]
        return ReasoningClause(
            polarity=ReasoningPolarity.STRENGTH,
            fragment=phrasing.format(v=float(best.raw)),
            evidence=best.evidence,
            jd_link=best.component,
        )


__all__: tuple[str, ...] = ("RankBand", "ReasoningEngine")