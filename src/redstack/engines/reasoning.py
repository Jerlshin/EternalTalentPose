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


# --------------------------------------------------------------------------- #
# Shared intensity vocabulary -- quadrupled pools of recruiting-register     #
# adjectives. ``{word}`` is interpolated into nearly every builder template, #
# so widening these four pools alone multiplies effective phrase variety     #
# across the entire engine, not just within one component.                  #
# --------------------------------------------------------------------------- #
_HIGH_WORDS: Final[tuple[str, ...]] = (
    "top-tier",
    "standout",
    "best-in-class",
    "outstanding",
    "exceptional",
    "elite",
    "marquee",
    "blue-chip",
    "top-decile",
    "gold-standard",
    "premier",
    "top-shelf",
    "first-rate",
    "top-bracket",
    "high-conviction",
    "flagship",
    "best-of-pool",
    "top-percentile",
    "superlative",
    "A-grade",
)
_GOOD_WORDS: Final[tuple[str, ...]] = (
    "strong",
    "solid",
    "well-rounded",
    "dependable",
    "credible",
    "above-average",
    "competitive",
    "capable",
    "sound",
    "robust",
    "well-grounded",
    "respectable",
)
_FAIR_WORDS: Final[tuple[str, ...]] = (
    "reasonable",
    "moderate",
    "workable",
    "adequate",
    "passable",
    "middling",
    "fair",
    "acceptable",
    "even-keeled",
    "serviceable",
    "middle-of-the-road",
    "tolerable",
)
_LOW_WORDS: Final[tuple[str, ...]] = (
    "thin",
    "modest",
    "limited",
    "light",
    "sparse",
    "slim",
    "underdeveloped",
    "nascent",
    "marginal",
    "soft",
    "shallow",
    "embryonic",
)

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
            evidence.append(
                mint(dump, kind=path_name, path=f"skills[{idx}].proficiency")
            )
        names: str = str(top.name)
        if len(credible) > 1:
            names = f"{top.name} and {credible[1].name}"
            second_idx = _skill_index(raw, credible[1].name)
            if second_idx is not None:
                evidence.append(
                    mint(dump, kind=path_name, path=f"skills[{second_idx}].name")
                )
        proficiency = top.proficiency.value
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
            f"{proficiency}-level command of {top.name} is backed by "
            f"{top.endorsements} peer endorsements rather than a self-assessed "
            f"label, putting the skill-match component on {article} {word} footing",
            f"strip away the self-reported layer of this profile and what survives "
            f"is {names} -- battle-tested entries with real endorsement weight "
            f"behind them, not keyword padding",
            f"the standout signal here is {top.name}: {top.endorsements} "
            f"endorsements at {proficiency} proficiency push it well past the "
            f"self-reported tier, anchoring {article} {word} skill-match read",
            f"{names} read as production-proven rather than aspirational -- the "
            f"endorsement trail behind them is exactly what a credible "
            f"skill-match case is built on",
            f"on a profile full of claimed tooling, {names} stand out as the "
            f"entries with actual third-party corroboration, which is the bar "
            f"this skill-match score is keyed to",
            f"{top.name} clears the credibility threshold with {top.endorsements} "
            f"endorsements logged against {proficiency}-level claimed proficiency, "
            f"giving the skill-match component {article} {word} and verifiable basis",
            f"cutting through the keyword list, {names} are the claims that "
            f"actually survive endorsement and assessment scrutiny, which is the "
            f"substance behind {article} {word} skill-match score",
            f"{names} are corroborated, not just claimed -- a distinction that "
            f"matters more than raw keyword overlap and one this profile clears "
            f"with room to spare",
            f"the endorsement record behind {top.name} ({top.endorsements} logged, "
            f"{proficiency} proficiency) is what elevates this from a resume "
            f"keyword match to {article} {word}, evidence-backed skill case",
            f"rather than taking the skills section at face value, the "
            f"endorsement and assessment record behind {names} is what actually "
            f"earns {article} {word} skill-match read here",
            f"{names} show the kind of cross-validated depth -- endorsements plus "
            f"assessment signal together -- that a JD-keyword scan alone would "
            f"never surface",
            f"what tips this skill-match case from plausible to credible is "
            f"{names}: corroborated entries with real endorsement weight, not "
            f"just terms lifted from the job description",
            f"{_article(proficiency)} {proficiency}-tier claim on {top.name} is "
            f"one thing -- {top.endorsements} independent endorsements behind it "
            f"is what actually substantiates {article} {word} skill-match score",
            f"{names} hold up under scrutiny -- endorsed, assessed, and "
            f"distinguishable from the rest of the skills list, which is exactly "
            f"what earns {article} {word} read on skill match",
        )
    else:
        templates = (
            f"the claimed skill set overlaps with the role on paper, but none of "
            f"it clears our endorsement/assessment bar yet -- {article} {word} but "
            f"largely unverified match",
            f"skill overlap with the JD is there in name, though it rests on "
            f"self-reported entries rather than corroborated ones, so the match "
            f"reads as {word} at best",
            f"the listed tooling lines up with the JD vocabulary, but without "
            f"endorsements or assessment scores behind any of it, this stays "
            f"{article} {word}, keyword-level match rather than a verified one",
            f"on paper the skill set fits -- in practice nothing here has cleared "
            f"endorsement or assessment corroboration yet, so the read stays "
            f"{word} until that changes",
            f"this profile's skills section echoes the JD closely, but echoes "
            f"are not corroboration -- absent endorsements or assessment data, "
            f"the match is {word} at the keyword level only",
            f"there's overlap with what the role is asking for, though it's "
            f"self-reported overlap, {article} {word} starting point that an "
            f"interview loop would still need to substantiate",
            f"the terminology checks out against the JD, but the skill-match "
            f"score stays {word} until endorsements or assessment scores give "
            f"it independent backing",
            f"claimed proficiency tracks the role's requirements, yet none of "
            f"it is corroborated by a third party -- {article} {word}, "
            f"unverified match worth probing further before weighting it heavily",
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
            f"vector-space alignment puts this profile nearest the "
            f"'{anchor_label}' anchor at {pct} cosine similarity -- a read on "
            f"how the candidate writes about their own work, not which "
            f"buzzwords happen to appear in it",
            f"the embedding geometry here is unambiguous: closest by a clear "
            f"margin to '{anchor_label}' at {pct}, {article} {word} semantic "
            f"signal that keyword matching alone would have missed entirely",
            f"strip the explicit skills list away and the underlying language "
            f"still gravitates toward '{anchor_label}' ({pct} fit) -- "
            f"{article} {word} sign the role fit isn't just surface-level echo",
            f"{pct} cosine alignment with the '{anchor_label}' anchor marks "
            f"{article} {word} semantic case, the kind of fit that shows up in "
            f"how a candidate frames their own experience rather than in a "
            f"skills checklist",
            f"the JD's own phrasing, captured as the '{anchor_label}' anchor, "
            f"is what this profile's language sits closest to ({pct}) -- "
            f"{article} {word} read on substantive rather than superficial "
            f"alignment",
            f"this profile doesn't just echo JD keywords -- its underlying "
            f"phrasing converges on the '{anchor_label}' anchor at {pct}, "
            f"{article} {word} signal in its own right",
            f"semantic clustering places this candidate's narrative closest "
            f"to '{anchor_label}' ({pct}), a vector-level corroboration that "
            f"sits independently of the explicit skills section",
            f"{article} {word} {pct} cosine read against the '{anchor_label}' "
            f"anchor suggests the fit runs deeper than shared terminology -- "
            f"it's there in how the role itself gets described",
            f"the closest semantic neighbor to this profile's language is "
            f"'{anchor_label}', at {pct} -- {article} {word} signal that "
            f"complements, rather than duplicates, the skill-match score",
            f"independent of the skills list, this profile's prose lands "
            f"nearest '{anchor_label}' in vector space ({pct} similarity), "
            f"{article} {word} corroborating signal for role fit",
        )
    else:
        templates = (
            f"the profile's language shows {article} {word} ({pct}) cosine fit "
            f"against the JD anchors overall, without one anchor clearly dominating",
            f"no single anchor dominates, but the aggregate cosine read across "
            f"the full JD anchor set still comes in {article} {word} ({pct})",
            f"semantic alignment here is diffuse rather than concentrated -- "
            f"{article} {word} {pct} fit spread across the anchor set rather "
            f"than anchored to one dominant theme",
            f"the embedding signal is {word} on aggregate ({pct}) without a "
            f"single anchor pulling clearly ahead of the rest",
            f"taken as a whole, the profile's language sits at {article} "
            f"{word} {pct} semantic fit against the JD's anchor set, evenly "
            f"distributed rather than concentrated in one theme",
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
            evidence.append(
                mint(dump, kind=career_kind, path=f"career_history[{idx}].industry")
            )
            evidence.append(
                mint(
                    dump,
                    kind=career_kind,
                    path=f"career_history[{idx}].company_size",
                )
            )
        evidence.append(
            make_evidence(EvidenceKind.DERIVED, "career.track", str(career.track))
        )
        org_kind = (
            "a product company" if position.is_product_company else "a services shop"
        )
        org_adj = "product-company" if position.is_product_company else "services-shop"
        track_label = career.track.value
        track_article = _article(track_label)
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
            f"{position.title} at {position.company}, a "
            f"{position.company_size.value}-headcount organization in "
            f"{position.industry}, anchors {article} {word} career-fit read "
            f"for a role pegged to exactly this org profile",
            f"{track_article} {track_label}-track career culminating in the "
            f"current {position.title} seat at {position.company} lines up "
            f"with {article} {word} read on this role's career-fit bar",
            f"the throughline from prior roles into {position.title} at "
            f"{position.company} reads as deliberate progression rather than "
            f"lateral drift, supporting {article} {word} career-fit score",
            f"holding {position.title} at {position.company} today, inside "
            f"{org_kind} in {position.industry}, is the kind of trajectory "
            f"this JD's career-fit bar was written for",
            f"{position.company}'s {org_adj} profile in {position.industry}, "
            f"paired with the current {position.title} title, gives the "
            f"career-fit component {article} {word} and well-evidenced basis",
            f"this reads as {track_article} {track_label}-track profile "
            f"through and through, currently {position.title} at "
            f"{position.company} -- exactly the shape of career this role's "
            f"bar was calibrated against",
            f"the seniority implied by {position.title} at {position.company} "
            f"({org_kind}) reads as {article} {word} fit against where this "
            f"JD is pitched, on trajectory alone",
            f"career progression into {position.title} at {position.company} "
            f"-- {org_kind}, {position.industry} -- is the kind of arc that "
            f"earns {article} {word} career-fit score rather than a borderline one",
            f"sitting today as {position.title} at {position.company}, "
            f"{track_article} {track_label}-track organization in "
            f"{position.industry}, this candidate's trajectory clears "
            f"{article} {word} bar for career fit",
            f"{position.company} ({org_kind}) currently employs them as "
            f"{position.title}, and that combination of seniority and org type "
            f"is precisely {article} {word} match for this JD's career-fit "
            f"criteria",
        )
    else:
        templates = (
            f"career trajectory overall reads as {article} {word} fit for the "
            f"role, even without a clearly current position on file",
            f"absent a flagged current role, the broader arc of this career "
            f"still reads as {article} {word} fit against what this position "
            f"is asking for",
            f"there's no single current position to anchor on here, but the "
            f"trajectory across the career history as a whole still supports "
            f"{article} {word} career-fit read",
            f"without a current-role marker to point to, the career-fit score "
            f"rests on the shape of the history overall -- which still comes "
            f"in {article} {word}",
            f"the lack of an explicit current-role flag doesn't erase the "
            f"underlying trajectory, which still clears {article} {word} bar "
            f"for this role's career-fit criteria",
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
    position_count = career.tenure.position_count
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
    evidence.append(
        make_evidence(
            EvidenceKind.DERIVED, "career.tenure.position_count", position_count
        )
    )
    templates = (
        f"{stated:.1f} years of claimed experience ({years:.1f}y derived) lands "
        f"{word} inside the band this role is targeting",
        f"at roughly {years:.1f} years of derived experience, seniority is "
        f"{article} {word} match for what the JD is asking for",
        f"experience-wise, {stated:.1f} stated years ({years:.1f} derived) puts "
        f"this squarely where the role's experience band wants someone",
        f"across {position_count} tracked roles totalling {years:.1f} derived "
        f"years, the seniority band this candidate sits in reads as "
        f"{article} {word} match for the role's experience target",
        f"the stated-vs-derived gap here is small ({stated:.1f} vs {years:.1f} "
        f"years), and both land {word} inside the band this JD is hiring for",
        f"{years:.1f} years of derived tenure across {position_count} roles "
        f"gives the experience-fit component {article} {word} and "
        f"internally-consistent basis",
        f"seniority math checks out: {stated:.1f} claimed years reconciles "
        f"closely with {years:.1f} derived, putting this {article} {word} "
        f"fit for the role's experience band",
        f"this isn't a borderline read on experience -- {years:.1f} derived "
        f"years across {position_count} roles sits {article} {word} distance "
        f"inside the band the JD is targeting",
        f"the career-history math derives {years:.1f} years against a "
        f"{stated:.1f}-year claim, a {word} reconciliation that anchors the "
        f"experience-fit score with real tenure data rather than a self-report "
        f"alone",
        f"{position_count} roles deep and {years:.1f} years in by the derived "
        f"count, the seniority profile here reads as {article} {word} match "
        f"for what this opening is scoped for",
        f"tenure data across the full career history derives {years:.1f} "
        f"years of experience, a figure that lands {word} against the band "
        f"this role's experience-fit bar was set at",
        f"{stated:.1f} self-reported years and {years:.1f} independently "
        f"derived years tell largely the same story here -- {article} {word} "
        f"experience-fit case with little daylight between claim and record",
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
        evidence.append(
            mint(dump, kind=edu_kind, path=f"education[{idx}].field_of_study")
        )
        tier_label = best.tier.value.replace("_", " ")
        tier_article = _article(tier_label)
        field_article = _article(best.field_of_study)
        templates = (
            f"{best.degree} from {best.institution} ({tier_label}) gives the "
            f"education-fit component {article} {word} basis",
            f"educationally, a {best.degree} from {best.institution} is "
            f"{article} {word} match for this role's bar -- not the deciding "
            f"factor, but it doesn't hurt",
            f"{best.institution}'s {tier_label} pedigree ({best.degree}) backs "
            f"up {article} {word} education-fit read",
            f"{tier_article} {tier_label} {best.degree} in "
            f"{best.field_of_study} from {best.institution} gives the "
            f"education-fit component {article} {word} and specifically "
            f"relevant foundation",
            f"{best.field_of_study} training at {best.institution} ({tier_label}, "
            f"{best.degree}) is {article} {word} academic basis for this role, "
            f"on top of whatever experience has layered on since",
            f"the academic record here -- {best.degree}, {best.field_of_study}, "
            f"{best.institution} ({tier_label}) -- reads as {article} {word} "
            f"credential match rather than a generic one",
            f"{best.institution}'s {tier_label} standing, paired with "
            f"{field_article} {best.field_of_study}-focused {best.degree}, "
            f"supports {article} {word} education-fit score for a role in "
            f"this domain",
            f"pedigree alone rarely decides a hire, but {tier_article} "
            f"{tier_label} {best.degree} in {best.field_of_study} from "
            f"{best.institution} still earns {article} {word} education-fit "
            f"read here",
            f"formal training in {best.field_of_study} ({best.degree}, "
            f"{best.institution}, {tier_label}) lines up well enough with the "
            f"role to count as {article} {word} education-fit signal",
            f"{best.degree} ({best.field_of_study}) from {tier_article} "
            f"{tier_label} institution like {best.institution} clears "
            f"{article} {word} bar for the education-fit component, "
            f"independent of experience",
            f"on the academic side, {best.institution} ({tier_label}) and "
            f"{field_article} {best.field_of_study}-aligned {best.degree} "
            f"make for {article} {word} foundation underneath the rest of "
            f"this profile",
            f"{best.institution}'s {tier_label} tier plus a directly relevant "
            f"{best.field_of_study} {best.degree} together support {article} "
            f"{word} education-fit case, not just a checkbox pedigree match",
        )
    else:
        templates = (
            f"no education record is on file, so this component leans on the "
            f"rest of the profile -- still {article} {word} read overall",
            f"education is unrecorded here, which leaves the education-fit "
            f"component resting on the rest of the evidence -- {article} "
            f"{word} read on balance",
            f"with no academic history to cite, the education-fit score "
            f"falls back to the neutral default -- the rest of the profile "
            f"still reads {article} {word}",
            f"absent a recorded degree or institution, this component can't "
            f"add much either way -- the overall read stays {article} {word}, "
            f"driven by the other components instead",
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
    relevant = float(credibility.relevant_skill_credibility)
    word = _bucket_word(seed, float(cv.raw))
    article = _article(word)
    evidence = list(cv.evidence)
    evidence.append(
        make_evidence(EvidenceKind.DERIVED, "credibility.claimed_vs_assessed_gap", gap)
    )
    evidence.append(
        make_evidence(
            EvidenceKind.DERIVED,
            "credibility.relevant_skill_credibility",
            relevant,
        )
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
        f"relevant-skill credibility comes in at {_pct(relevant)}, with "
        f"{credible_count} of {total} claims independently corroborated -- "
        f"{article} {word} signal that this isn't a keyword-stuffed profile",
        f"a {gap:.2f} claimed-vs-assessed gap, combined with {credible_count}/"
        f"{total} skills clearing corroboration, gives the credibility "
        f"component {article} {word} and well-supported basis",
        f"{_pct(relevant)} of the skills most relevant to this role carry "
        f"independent credibility backing, putting the overall read "
        f"{article} {word}",
        f"this profile's self-reports and its assessed reality are close "
        f"({gap:.2f} gap), and {credible_count} of {total} claims are "
        f"corroborated outright -- {article} {word} credibility case",
        f"rather than inflated self-reporting, {credible_count}/{total} "
        f"skills here carry endorsement or assessment backing, with "
        f"relevant-skill credibility at {_pct(relevant)} -- {word} overall",
        f"rated against assessment data, the gap between what's claimed and "
        f"what's verified stays tight ({gap:.2f}), supporting {article} {word} "
        f"credibility read on the skills section as a whole",
        f"{credible_count} corroborated claims out of {total} total, plus a "
        f"{_pct(relevant)} relevant-skill credibility figure, together make "
        f"the case for {article} {word} credibility component",
        f"the gap between self-reported and assessed proficiency is "
        f"{gap:.2f} -- small enough, alongside {credible_count}/{total} "
        f"corroborated skills, to call this {article} {word} credibility profile",
        f"credibility here isn't assumed, it's measured: {credible_count} of "
        f"{total} skills independently verified, {_pct(relevant)} relevant-"
        f"skill credibility, both pointing to {article} {word} read",
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
    if archetype is not None:
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED, "archetype.distance", float(archetype.distance)
            )
        )
    templates = (
        f"clusters into the '{label_text}' archetype we discovered as one of "
        f"the target profiles for this JD -- {article} {word} membership fit",
        f"profile shape matches the '{label_text}' archetype cluster ({word} "
        f"membership confidence), one of the patterns we went looking for",
        f"this profile sits close to the '{label_text}' archetype centroid -- "
        f"{article} {word} cluster-membership read that corroborates the "
        f"component-level scores rather than just restating them",
        f"unsupervised clustering independently placed this candidate inside "
        f"the '{label_text}' archetype, one of the shapes we pre-identified "
        f"as a target profile, with {article} {word} membership confidence",
        f"beyond the individual component scores, this profile's overall "
        f"shape lands inside the '{label_text}' archetype -- {article} {word} "
        f"structural fit, not just a sum of independent signals",
        f"the '{label_text}' archetype this candidate clusters into was "
        f"flagged as a target pattern before any individual profile was "
        f"scored against it, and the membership fit here reads {article} {word}",
        f"pattern-matching against the discovered archetype space places this "
        f"candidate inside '{label_text}' with {article} {word} confidence, "
        f"reinforcing rather than duplicating the other component reads",
        f"as a holistic shape rather than a checklist, this profile reads "
        f"closest to '{label_text}' -- one of the archetypes this JD was "
        f"calibrated against -- with {article} {word} cluster fit",
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
    article = _article(word)
    label = cv.component.value.replace("_", " ")
    pct = _pct(float(cv.raw))
    templates = (
        f"{label} comes in {word} at {pct}",
        f"on {label}, this profile reads {article} {word} ({pct})",
        f"the {label} component lands {article} {word}, scoring {pct}",
        f"{pct} on {label} puts this {article} {word} distance from the top "
        f"of that component's range",
    )
    fragment = _pick(seed, templates)
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
    min_tenure = career.tenure.min_tenure_months
    templates = (
        f"job-hopping is a real concern -- average tenure runs around "
        f"{mean_tenure:.0f} months per role, a {_pct(hop_rate)} hop rate",
        f"tenure has been short (avg ~{mean_tenure:.0f} months/role, "
        f"{_pct(hop_rate)} hop rate) -- worth probing in an interview rather "
        f"than waving it away",
        f"{_pct(hop_rate)} of recent roles ran under 18 months -- a retention "
        f"risk the strengths above don't erase on their own",
        f"the shortest stint on record ran just {min_tenure:.0f} months, and "
        f"with a {_pct(hop_rate)} hop rate overall, retention is a fair "
        f"question to raise before extending an offer",
        f"a {mean_tenure:.0f}-month average tenure (low end: {min_tenure:.0f} "
        f"months) puts the hop rate at {_pct(hop_rate)} -- not disqualifying, "
        f"but worth a direct conversation about what's driving the moves",
        f"tenure stability is the soft spot here: {_pct(hop_rate)} of roles "
        f"under 18 months, averaging {mean_tenure:.0f} months apiece, which "
        f"tempers an otherwise strong-looking case",
        f"the pattern across this career history -- {_pct(hop_rate)} hop "
        f"rate, {mean_tenure:.0f}-month average stay -- reads as title-"
        f"chasing risk rather than settled progression",
        f"retention risk shows up clearly in the tenure data: shortest role "
        f"at {min_tenure:.0f} months, average at {mean_tenure:.0f}, hop rate "
        f"at {_pct(hop_rate)} -- all worth surfacing before a final call",
        f"{mean_tenure:.0f} months is the average stay here, and at a "
        f"{_pct(hop_rate)} hop rate this looks more like a pattern than a "
        f"one-off job change",
        f"a candidate this strong on paper with a {_pct(hop_rate)} hop rate "
        f"is exactly the profile worth a direct retention conversation with, "
        f"rather than assuming the next stop is a long one",
        f"history shows {_pct(hop_rate)} of roles closing inside 18 months "
        f"(averaging {mean_tenure:.0f} months) -- a pattern that deserves a "
        f"straight question in the loop, not a quiet pass",
        f"the tenure curve here -- {min_tenure:.0f} months at the shortest, "
        f"{mean_tenure:.0f} on average -- puts the hop rate at {_pct(hop_rate)} "
        f"and is the main thing standing between a strong profile and an easy yes",
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
    notice_label = logistics.notice_fit.value.replace("_", " ")
    templates = (
        f"notice period runs {days} days, well past the 30-day window the "
        f"role would prefer -- a real but manageable logistics gap",
        f"a {days}-day notice period is the main practical friction here -- "
        f"the rest of the fit holds up regardless",
        f"the {days}-day notice is longer than ideal and would need "
        f"sign-off from whoever owns the start-date timeline",
        f"at {days} days, notice runs squarely into '{notice_label}' "
        f"territory -- not a blocker, but a start-date conversation that "
        f"needs to happen early rather than after an offer is out",
        f"{days} days of notice is the one logistics line item here that "
        f"isn't clean -- everything else about the timeline is workable",
        f"the start-date math is the friction point: {days} days of notice "
        f"classifies as '{notice_label}', which is worth flagging to "
        f"whoever is planning the onboarding calendar",
        f"{days} days out from a signed offer is longer than the role's "
        f"30-day preference, though it's a scheduling problem rather than a "
        f"fit problem",
        f"notice period ({days} days, '{notice_label}') is the practical "
        f"catch here -- everything upstream of the offer stage looks clean",
        f"a {days}-day runway between offer and start is on the long side -- "
        f"buyout or an extended start-date window would likely be needed",
        f"this is a logistics flag, not a fit flag: {days} days of notice "
        f"sits in the '{notice_label}' band and would shift the onboarding "
        f"timeline accordingly",
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
    location = logistics.location
    relocate_note = (
        "though they've flagged willingness to relocate"
        if logistics.willing_to_relocate
        else "with no stated willingness to relocate on file"
    )
    templates = (
        f"based in {country}, outside India, with no sponsorship path on "
        f"file -- a logistics blocker independent of skill fit",
        f"location is the sticking point: {country}-based, and there's no "
        f"record of a sponsorship route into India",
        f"otherwise solid, but a {country} base with no India sponsorship "
        f"path is a real constraint on actually closing this hire",
        f"currently in {location}, {country} -- outside the India footprint "
        f"this role needs, {relocate_note}",
        f"the {country} location is the practical blocker here, {relocate_note} "
        f"-- sponsorship logistics would need to be solved before this "
        f"becomes closable",
        f"geography is the constraint, not capability: {location}, {country} "
        f"sits outside the sponsorship-supported footprint, {relocate_note}",
        f"a strong profile is undercut by location alone -- {country}-based "
        f"with no sponsorship route into India on record, {relocate_note}",
        f"this would need an immigration/sponsorship path that doesn't "
        f"currently exist on file, given the {country} base out of {location}",
        f"{location}, {country} is outside the role's supported geography -- "
        f"{relocate_note}, which changes how blocking this constraint actually is",
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
    stated = float(career.stated_experience_years)
    templates = (
        f"at {years:.1f} years derived experience, seniority sits outside "
        f"the band this role was scoped for -- worth a level-set "
        f"conversation before moving forward",
        f"{years:.1f} years of experience falls outside the target band -- "
        f"could read as over- or under-leveled for the opening as written",
        f"experience level ({years:.1f}y) doesn't line up cleanly with the "
        f"band the JD targets, independent of how the skills themselves look",
        f"both the {stated:.1f}-year claim and the {years:.1f}-year derived "
        f"figure land outside the role's target band -- a level mismatch "
        f"that's about scope, not capability",
        f"this profile is either ahead of or behind where the role is "
        f"pitched: {years:.1f} derived years sits outside the band, which "
        f"argues for a level-set conversation rather than a pass",
        f"seniority math puts this candidate at {years:.1f} years derived, "
        f"outside the experience band the role was scoped for -- a banding "
        f"question more than a competence one",
        f"{years:.1f} years (against a {stated:.1f}-year claim) doesn't "
        f"match the experience band this opening targets -- right person, "
        f"possibly wrong level",
        f"the experience-band mismatch here ({years:.1f} derived years "
        f"against the role's target) is structural, not a reflection of the "
        f"skills or career-fit signals elsewhere in this profile",
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
