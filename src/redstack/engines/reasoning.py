from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Final, Literal, final

from pydantic import BaseModel, ConfigDict

from redstack.domain.candidate.eligibility import EligibilityFinding
from redstack.domain.candidate.representation import CandidateRepresentation
from redstack.domain.enums import (
    EligibilityCode,
    EvidenceKind,
    ReasoningPolarity,
    ScoreComponent,
)
from redstack.domain.errors import ProvenanceError
from redstack.domain.ids import CandidateId
from redstack.domain.provenance import EvidenceRef
from redstack.domain.ranking import RankedCandidate, Ranking
from redstack.domain.reasoning import CandidateReasoning, ReasoningClause
from redstack.domain.scoring import ScoreComponentValue
from redstack.domain.source import RawCandidate
from redstack.features.evidence import mint
from redstack.features.view import make_evidence

RankBand = Literal["top", "mid", "tail"]

_MAX_STRENGTHS: Final[int] = 3
_MAX_CONCERNS: Final[int] = 3

_StrengthBuilder = Callable[
    [RawCandidate, CandidateRepresentation, ScoreComponentValue, str],
    tuple[str, tuple[EvidenceRef, ...]],
]
_ConcernBuilder = Callable[
    [RawCandidate, CandidateRepresentation, EligibilityFinding, str],
    tuple[str, tuple[EvidenceRef, ...]],
]


# --------------------------------------------------------------------------- #
# Deterministic, seed-derived phrase selection (no RNG, no clock; §2 / §J.5). #
# --------------------------------------------------------------------------- #
def _pick_index(seed: str, n: int) -> int:
    """Stable index in ``[0, n)`` derived purely from ``seed`` via SHA-256.

    Two distinct candidates (or two distinct components for one candidate)
    almost never land on the same index, so a small fixed-size phrase pool
    still produces wide variation across a 100-row submission -- without any
    ``random`` import, shared RNG state, or wall-clock dependency.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def _pick(seed: str, options: tuple[str, ...]) -> str:
    return options[_pick_index(seed, len(options))]


_HIGH_WORDS: Final[tuple[str, ...]] = (
    "top-tier",
    "standout",
    "best-in-class",
    "outstanding",
    "exceptional",
)
_GOOD_WORDS: Final[tuple[str, ...]] = ("strong", "solid", "well-rounded")
_FAIR_WORDS: Final[tuple[str, ...]] = ("reasonable", "moderate", "workable")
_LOW_WORDS: Final[tuple[str, ...]] = ("thin", "modest", "limited")

_VOWEL_LEADING: Final[frozenset[str]] = frozenset("aeiouAEIOU")


def _bucket_word(seed: str, value: float) -> str:
    if value >= 0.75:
        pool = _HIGH_WORDS
    elif value >= 0.5:
        pool = _GOOD_WORDS
    elif value >= 0.3:
        pool = _FAIR_WORDS
    else:
        pool = _LOW_WORDS
    return _pick(seed, pool)


def _article(word: str) -> str:
    """The indefinite article that grammatically precedes ``word``.

    Words are pulled dynamically from the intensity pools above (which keep
    growing), so this is computed from the actual word at render time rather
    than hand-curating the pools to dodge vowel-led adjectives.
    """
    return "an" if word[:1] in _VOWEL_LEADING else "a"


def _pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _position_index(
    raw: RawCandidate, *, company: str, title: str, start_date: object
) -> int | None:
    for idx, position in enumerate(raw.career_history):
        key = (position.company, position.title, position.start_date)
        if key == (company, title, start_date):
            return idx
    return None


def _skill_index(raw: RawCandidate, name: str) -> int | None:
    for idx, skill in enumerate(raw.skills):
        if skill.name == name:
            return idx
    return None


def _dump(raw: RawCandidate) -> Mapping[str, object]:
    """The plain JSON mapping ``mint`` resolves evidence paths against."""
    return raw.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Strength clause builders -- one per ScoreComponent, each citing concrete,   #
# named facts from the candidate's own profile rather than a bare float.     #
# --------------------------------------------------------------------------- #
def _skill_match_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    cv: ScoreComponentValue,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    credibility = representation.require_credibility()
    credible = sorted(
        (trust for trust in credibility.skill_trust.values() if trust.is_credible),
        key=lambda trust: (-float(trust.trust), trust.name),
    )
    evidence: list[EvidenceRef] = list(cv.evidence)
    word = _bucket_word(seed, float(cv.raw))
    article = _article(word)
    templates: tuple[str, ...]
    if credible:
        top = credible[0]
        idx = _skill_index(raw, top.name)
        path_name = EvidenceKind.SKILL
        dump = _dump(raw)
        if idx is not None:
            evidence.append(mint(dump, kind=path_name, path=f"skills[{idx}].name"))
            evidence.append(
                mint(dump, kind=path_name, path=f"skills[{idx}].endorsements")
            )
        names: str = str(top.name)
        if len(credible) > 1:
            names = f"{top.name} and {credible[1].name}"
            second_idx = _skill_index(raw, credible[1].name)
            if second_idx is not None:
                evidence.append(
                    mint(dump, kind=path_name, path=f"skills[{second_idx}].name")
                )
        templates = (
            f"{names} show up as independently corroborated rather than just listed "
            f"-- {top.endorsements} endorsements behind {top.name} alone -- which "
            f"gives the skill-match read {article} {word} footing",
            f"the skill-match signal leans on {names}, both of which clear our "
            f"credibility bar instead of sitting as bare keyword entries",
            f"of everything claimed, {names} are the ones actually corroborated by "
            f"endorsements and assessment, which is what makes this {article} {word} "
            f"skill-match case rather than a buzzword one",
            f"{names} carry real endorsement and tenure weight, the kind of "
            f"corroboration that separates this profile from a resume that just "
            f"lists the JD's keywords back",
        )
    else:
        templates = (
            f"the claimed skill set overlaps with the role on paper, but none of "
            f"it clears our endorsement/assessment bar yet -- {article} {word} but "
            f"largely unverified match",
            f"skill overlap with the JD is there in name, though it rests on "
            f"self-reported entries rather than corroborated ones, so the match "
            f"reads as {word} at best",
        )
    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


def _semantic_fit_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    cv: ScoreComponentValue,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = raw
    semantic = representation.require_semantic()
    word = _bucket_word(seed, float(cv.raw))
    article = _article(word)
    pct = _pct(float(cv.raw))
    evidence = list(cv.evidence)
    anchor = semantic.best_positive_anchor
    templates: tuple[str, ...]
    if anchor is not None:
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED, "semantic.best_positive_anchor", str(anchor)
            )
        )
        anchor_label = str(anchor).rsplit(".", maxsplit=1)[-1].replace("_", " ")
        templates = (
            f"the profile's own language reads closest to our '{anchor_label}' "
            f"anchor, {article} {word} semantic match ({pct}) against how this JD "
            f"is actually phrased",
            f"semantically this clusters tightest with the '{anchor_label}' "
            f"anchor built from the JD text itself -- {pct} fit, {article} {word} "
            f"signal on its own",
            f"beyond keyword overlap, the wording of the profile tracks the "
            f"'{anchor_label}' anchor most closely ({pct}), independent of "
            f"which exact skills are listed",
        )
    else:
        templates = (
            f"the profile's language shows {article} {word} ({pct}) cosine fit "
            f"against the JD anchors overall, without one anchor clearly dominating",
        )
    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


def _career_fit_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    cv: ScoreComponentValue,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    career = representation.require_career()
    word = _bucket_word(seed, float(cv.raw))
    article = _article(word)
    evidence = list(cv.evidence)
    position = career.current_position
    templates: tuple[str, ...]
    if position is not None:
        idx = _position_index(
            raw,
            company=position.company,
            title=position.title,
            start_date=position.start_date,
        )
        career_kind = EvidenceKind.CAREER_FIELD
        if idx is not None:
            dump = _dump(raw)
            title_path = f"career_history[{idx}].title"
            company_path = f"career_history[{idx}].company"
            evidence.append(mint(dump, kind=career_kind, path=title_path))
            evidence.append(mint(dump, kind=career_kind, path=company_path))
        org_kind = (
            "a product company" if position.is_product_company else "a services shop"
        )
        templates = (
            f"currently {position.title} at {position.company} ({org_kind}), a "
            f"trajectory the model reads as {article} {word} match for this "
            f"role's career-fit bar",
            f"the path into the current {position.title} role at "
            f"{position.company} tracks the seniority and domain this JD is "
            f"hiring for -- {article} {word} career-fit case",
            f"{position.company} -- where they hold the {position.title} title "
            f"today -- sits squarely in the kind of org this JD targets, which "
            f"drives {article} {word} career-fit score",
        )
    else:
        templates = (
            f"career trajectory overall reads as {article} {word} fit for the "
            f"role, even without a clearly current position on file",
        )
    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


def _experience_fit_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    cv: ScoreComponentValue,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    career = representation.require_career()
    years = float(career.derived_experience_years)
    stated = float(raw.profile.years_of_experience)
    word = _bucket_word(seed, float(cv.raw))
    article = _article(word)
    evidence = list(cv.evidence)
    evidence.append(
        mint(
            _dump(raw),
            kind=EvidenceKind.PROFILE_FIELD,
            path="profile.years_of_experience",
        )
    )
    templates = (
        f"{stated:.1f} years of claimed experience ({years:.1f}y derived) lands "
        f"{word} inside the band this role is targeting",
        f"at roughly {years:.1f} years of derived experience, seniority is "
        f"{article} {word} match for what the JD is asking for",
        f"experience-wise, {stated:.1f} stated years ({years:.1f} derived) puts "
        f"this squarely where the role's experience band wants someone",
    )
    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


def _education_fit_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    cv: ScoreComponentValue,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = representation
    word = _bucket_word(seed, float(cv.raw))
    article = _article(word)
    evidence = list(cv.evidence)
    templates: tuple[str, ...]
    if raw.education:
        best = min(raw.education, key=lambda edu: edu.tier.ordinal)
        idx = raw.education.index(best)
        dump = _dump(raw)
        edu_kind = EvidenceKind.EDUCATION
        evidence.append(mint(dump, kind=edu_kind, path=f"education[{idx}].institution"))
        evidence.append(mint(dump, kind=edu_kind, path=f"education[{idx}].degree"))
        tier_label = best.tier.value.replace("_", " ")
        templates = (
            f"{best.degree} from {best.institution} ({tier_label}) gives the "
            f"education-fit component {article} {word} basis",
            f"educationally, a {best.degree} from {best.institution} is "
            f"{article} {word} match for this role's bar -- not the deciding "
            f"factor, but it doesn't hurt",
            f"{best.institution}'s {tier_label} pedigree ({best.degree}) backs "
            f"up {article} {word} education-fit read",
        )
    else:
        templates = (
            f"no education record is on file, so this component leans on the "
            f"rest of the profile -- still {article} {word} read overall",
        )
    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


def _credibility_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    cv: ScoreComponentValue,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = raw
    credibility = representation.require_credibility()
    credible_count = sum(
        1 for trust in credibility.skill_trust.values() if trust.is_credible
    )
    total = len(credibility.skill_trust)
    gap = float(credibility.claimed_vs_assessed_gap)
    word = _bucket_word(seed, float(cv.raw))
    article = _article(word)
    evidence = list(cv.evidence)
    evidence.append(
        make_evidence(EvidenceKind.DERIVED, "credibility.claimed_vs_assessed_gap", gap)
    )
    templates = (
        f"{credible_count} of {total} claimed skills clear our corroboration bar "
        f"(endorsements + assessment), {article} {word} credibility signal with "
        f"a low claimed-vs-assessed gap ({gap:.2f})",
        f"the claimed-vs-assessed gap sits at {gap:.2f}, with "
        f"{credible_count}/{total} skills independently corroborated -- {word}, "
        f"not a profile that's all self-reported keywords",
        f"credibility checks out: {credible_count} of {total} listed skills are "
        f"backed by endorsements or assessment scores rather than self-reported "
        f"alone ({word})",
    )
    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


def _archetype_fit_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    cv: ScoreComponentValue,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = raw
    word = _bucket_word(seed, float(cv.raw))
    article = _article(word)
    evidence = list(cv.evidence)
    archetype = representation.archetype
    label = archetype.label if archetype is not None and archetype.label else None
    label_text = label.replace("_", " ") if label else "a target candidate"
    templates = (
        f"clusters into the '{label_text}' archetype we discovered as one of "
        f"the target profiles for this JD -- {article} {word} membership fit",
        f"profile shape matches the '{label_text}' archetype cluster ({word} "
        f"membership confidence), one of the patterns we went looking for",
    )
    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


def _generic_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    cv: ScoreComponentValue,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = (raw, representation)
    word = _bucket_word(seed, float(cv.raw))
    label = cv.component.value.replace("_", " ")
    fragment = f"{label} comes in {word} at {_pct(float(cv.raw))}"
    return fragment, cv.evidence


_STRENGTH_BUILDERS: Final[Mapping[ScoreComponent, _StrengthBuilder]] = {
    ScoreComponent.SKILL_MATCH: _skill_match_strength,
    ScoreComponent.SEMANTIC_FIT: _semantic_fit_strength,
    ScoreComponent.CAREER_FIT: _career_fit_strength,
    ScoreComponent.EXPERIENCE_FIT: _experience_fit_strength,
    ScoreComponent.EDUCATION_FIT: _education_fit_strength,
    ScoreComponent.CREDIBILITY: _credibility_strength,
    ScoreComponent.ARCHETYPE_FIT: _archetype_fit_strength,
}


# --------------------------------------------------------------------------- #
# Concern clause builders -- one per *soft* EligibilityCode (the only         #
# severity that ever reaches reasoning; hard blocks are floored out before   #
# ranking, see ``EligibilityEngine._SOFT_CODES``).                           #
# --------------------------------------------------------------------------- #
def _title_chaser_concern(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    finding: EligibilityFinding,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = raw
    career = representation.require_career()
    hop_rate = float(career.tenure.hop_rate)
    mean_tenure = career.tenure.mean_tenure_months
    templates = (
        f"job-hopping is a real concern -- average tenure runs around "
        f"{mean_tenure:.0f} months per role, a {_pct(hop_rate)} hop rate",
        f"tenure has been short (avg ~{mean_tenure:.0f} months/role, "
        f"{_pct(hop_rate)} hop rate); worth probing in an interview rather "
        f"than waving it away",
        f"{_pct(hop_rate)} of recent roles ran under 18 months -- a retention "
        f"risk the strengths above don't erase on their own",
    )
    fragment = _pick(seed, templates)
    return fragment, finding.evidence


def _notice_over_30_concern(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    finding: EligibilityFinding,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = raw
    logistics = representation.require_logistics()
    days = int(logistics.notice_period_days)
    templates = (
        f"notice period runs {days} days, well past the 30-day window the "
        f"role would prefer -- a real but manageable logistics gap",
        f"a {days}-day notice period is the main practical friction here; "
        f"the rest of the fit holds up regardless",
        f"the {days}-day notice is longer than ideal and would need "
        f"sign-off from whoever owns the start-date timeline",
    )
    fragment = _pick(seed, templates)
    return fragment, finding.evidence


def _outside_india_concern(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    finding: EligibilityFinding,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = raw
    logistics = representation.require_logistics()
    country = logistics.country
    templates = (
        f"based in {country}, outside India, with no sponsorship path on "
        f"file -- a logistics blocker independent of skill fit",
        f"location is the sticking point: {country}-based, and there's no "
        f"record of a sponsorship route into India",
        f"otherwise solid, but a {country} base with no India sponsorship "
        f"path is a real constraint on actually closing this hire",
    )
    fragment = _pick(seed, templates)
    return fragment, finding.evidence


def _outside_experience_band_concern(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    finding: EligibilityFinding,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = raw
    career = representation.require_career()
    years = float(career.derived_experience_years)
    templates = (
        f"at {years:.1f} years derived experience, seniority sits outside "
        f"the band this role was scoped for -- worth a level-set "
        f"conversation before moving forward",
        f"{years:.1f} years of experience falls outside the target band; "
        f"could read as over- or under-leveled for the opening as written",
        f"experience level ({years:.1f}y) doesn't line up cleanly with the "
        f"band the JD targets, independent of how the skills themselves look",
    )
    fragment = _pick(seed, templates)
    return fragment, finding.evidence


def _generic_concern(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    finding: EligibilityFinding,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    _ = (raw, representation, seed)
    return finding.detail, finding.evidence


_CONCERN_BUILDERS: Final[Mapping[EligibilityCode, _ConcernBuilder]] = {
    EligibilityCode.TITLE_CHASER_SUB_18M_HOPS: _title_chaser_concern,
    EligibilityCode.NOTICE_OVER_30: _notice_over_30_concern,
    EligibilityCode.OUTSIDE_INDIA_NO_SPONSOR: _outside_india_concern,
    EligibilityCode.OUTSIDE_EXPERIENCE_BAND: _outside_experience_band_concern,
}


@final
class ReasoningEngine(BaseModel):
    """Stateless, pure, local reasoning engine over the ranked top-K.

    Every clause is assembled from named facts on the candidate's own raw
    profile (current employer, named skills, institutions, tenure, notice
    period, ...) rather than a bare component float; phrasing is selected
    deterministically per ``(candidate_id, component_or_code)`` from a small
    pool of structurally distinct sentence templates, so two candidates who
    share a dominant component still render materially different text.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=False
    )

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
        raw = representation.require_raw()
        band = self._rank_band(ranked.rank, ranked_size=self._size_of(ranked))
        strengths = self._strength_clauses(raw, representation, ranked)
        concerns = self._concern_clauses(raw, representation)

        clauses: tuple[ReasoningClause, ...] = (*strengths, *concerns)
        if not clauses:
            # Defence: a ranked candidate always has a positive contributor.
            clauses = (self._fallback_strength(raw, representation, ranked),)

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
        self,
        raw: RawCandidate,
        representation: CandidateRepresentation,
        ranked: RankedCandidate,
    ) -> tuple[ReasoningClause, ...]:
        components = ranked.scored.breakdown.components
        # Strongest by weighted contribution; only those with citable evidence.
        ranked_components = sorted(
            (c for c in components if float(c.raw) > 0.0 and c.evidence),
            key=lambda c: (-float(c.weighted), c.component.value),
        )
        clauses: list[ReasoningClause] = []
        for component_value in ranked_components[:_MAX_STRENGTHS]:
            clauses.append(
                self._strength_clause(
                    raw, representation, ranked.candidate_id, component_value
                )
            )
        return tuple(clauses)

    @staticmethod
    def _strength_clause(
        raw: RawCandidate,
        representation: CandidateRepresentation,
        candidate_id: CandidateId,
        component_value: ScoreComponentValue,
    ) -> ReasoningClause:
        seed = f"{candidate_id}:{component_value.component.value}:strength"
        builder = _STRENGTH_BUILDERS.get(component_value.component, _generic_strength)
        fragment, evidence = builder(raw, representation, component_value, seed)
        return ReasoningClause(
            polarity=ReasoningPolarity.STRENGTH,
            fragment=fragment,
            evidence=evidence,
            jd_link=component_value.component,
        )

    def _concern_clauses(
        self, raw: RawCandidate, representation: CandidateRepresentation
    ) -> tuple[ReasoningClause, ...]:
        if representation.eligibility is None:
            return ()
        penalties = representation.eligibility.soft_penalties
        clauses: list[ReasoningClause] = []
        for finding in penalties[:_MAX_CONCERNS]:
            clauses.append(self._concern_clause(raw, representation, finding))
        return tuple(clauses)

    @staticmethod
    def _concern_clause(
        raw: RawCandidate,
        representation: CandidateRepresentation,
        finding: EligibilityFinding,
    ) -> ReasoningClause:
        seed = f"{representation.candidate_id}:{finding.code.value}:concern"
        builder = _CONCERN_BUILDERS.get(finding.code, _generic_concern)
        fragment, evidence = builder(raw, representation, finding, seed)
        return ReasoningClause(
            polarity=ReasoningPolarity.CONCERN,
            fragment=fragment,
            evidence=evidence,
            jd_link=finding.code,
        )

    @staticmethod
    def _fallback_strength(
        raw: RawCandidate,
        representation: CandidateRepresentation,
        ranked: RankedCandidate,
    ) -> ReasoningClause:
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
        seed = f"{ranked.candidate_id}:{best.component.value}:fallback"
        builder = _STRENGTH_BUILDERS.get(best.component, _generic_strength)
        fragment, evidence = builder(raw, representation, best, seed)
        return ReasoningClause(
            polarity=ReasoningPolarity.STRENGTH,
            fragment=fragment,
            evidence=evidence,
            jd_link=best.component,
        )


__all__: tuple[str, ...] = ("RankBand", "ReasoningEngine")
