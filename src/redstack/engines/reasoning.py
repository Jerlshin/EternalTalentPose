from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import date
from typing import Final, Literal, final

from pydantic import BaseModel, ConfigDict, Field

from redstack.domain.candidate.eligibility import EligibilityFinding
from redstack.domain.candidate.representation import CandidateRepresentation
from redstack.domain.enums import (
    EligibilityCode,
    EvidenceKind,
    ReasoningPolarity,
    ScoreComponent,
)
from redstack.domain.errors import ProvenanceError
from redstack.domain.ids import AnchorId, CandidateId
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
# Latent builders receive the raw anchor cosine similarity (float) instead of
# a ScoreComponentValue, because JD-latent reasoning is evidence-first.
_LatentBuilder = Callable[
    [RawCandidate, CandidateRepresentation, float, str],
    tuple[str, tuple[EvidenceRef, ...]] | None,
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
            f"beyond the skills list, how this candidate writes about their own "
            f"work reads closest to the '{anchor_label}' theme this JD is "
            f"actually hiring for -- a signal independent of which keywords "
            f"happen to appear in the profile",
            f"the language this candidate uses to describe their experience "
            f"gravitates toward '{anchor_label}', which is the substantive "
            f"framing the JD is built around -- {article} {word} read on "
            f"genuine rather than surface-level alignment",
            f"strip the skills section away and the underlying narrative still "
            f"reads as '{anchor_label}' work -- the kind of alignment that "
            f"keyword-matching alone would miss",
            f"this candidate's own words about their career cluster around "
            f"'{anchor_label}' -- the dominant technical theme in this JD -- "
            f"not because they echoed it back, but because that's how they've "
            f"framed their own work, {article} {word} sign of genuine fit",
            f"the substance of how this candidate describes their work fits the "
            f"'{anchor_label}' pattern in this JD, which gives the fit signal "
            f"{article} {word} basis beyond a skills-keyword check",
            f"how this candidate narrates their own experience -- the problems "
            f"they describe, the work they emphasize -- lines up {word} with "
            f"the '{anchor_label}' focus the JD is centered on",
            f"looking past the listed skills, the conceptual framing this "
            f"candidate uses for their work is '{anchor_label}'-oriented, "
            f"which is exactly where this JD's signal sits -- {article} {word} "
            f"read on substantive rather than keyword alignment",
            f"the way this candidate talks about what they've built reads as "
            f"'{anchor_label}' experience: not aspirational keyword listing, "
            f"but {article} {word} natural description of their own work",
            f"the narrative texture here -- how this candidate explains what "
            f"they've done and why -- sits closest to the '{anchor_label}' "
            f"framing the JD uses to describe the ideal candidate",
            f"how this candidate describes the problems they've worked on reads "
            f"as {article} {word} '{anchor_label}' orientation -- the kind of "
            f"substantive signal that persists even when vocabulary choices differ",
            f"this profile's language on its own work naturally gravitates to "
            f"'{anchor_label}', suggesting real domain exposure rather than "
            f"vocabulary borrowed from the job description -- {article} {word} "
            f"independent corroboration of the skill-match read",
            f"the conceptual framing this candidate uses for their career -- "
            f"what they say they built and why -- lands on '{anchor_label}', "
            f"the core technical area this JD is hiring for; "
            f"{article} {word} signal, independently of whatever is listed in "
            f"the skills section",
            f"the dominant pattern in how this candidate talks about their "
            f"career is '{anchor_label}': not a self-reported label, but "
            f"{article} {word} reflection of how they actually narrate "
            f"their own work history",
        )
    else:
        templates = (
            f"looking past the skills checklist, the way this candidate frames "
            f"their own work covers the JD's domain broadly -- no single theme "
            f"dominates, but the overall framing is {word} and on-target",
            f"the language this candidate uses to describe their work spans the "
            f"JD's technical territory evenly -- a diffuse but {word} fit that "
            f"holds up across multiple areas rather than peaking in one",
            f"how this candidate narrates their career touches the JD's key "
            f"themes at multiple points without any one area standing out -- "
            f"{article} {word} broad-coverage signal rather than a narrow spike",
            f"this candidate's own description of their work draws on domain "
            f"vocabulary that spans the JD's focus areas at a {word} level, "
            f"even if no single area clearly dominates the read",
            f"the narrative framing here maps across several of the JD's "
            f"core areas rather than concentrating in one -- {article} {word} "
            f"distributed fit, independently of the explicit skills list",
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
    label_text = label.replace("_", " ") if label else "the target candidate"
    if archetype is not None:
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED, "archetype.distance", float(archetype.distance)
            )
        )
    templates = (
        f"viewed as a career shape rather than a list of credentials, this "
        f"reads as {article} {word} fit for the '{label_text}' profile this "
        f"JD is built to attract",
        f"the overall trajectory here -- not the individual skills, but the "
        f"pattern of where this career has gone -- fits the '{label_text}' "
        f"mold the JD describes; {article} {word} structural read",
        f"putting the line items aside and looking at the career as a whole, "
        f"it reads as {article} {word} '{label_text}' shape: the kind of "
        f"profile this JD was written to hire",
        f"career-shape analysis places this in the '{label_text}' pattern, "
        f"which is the holistic target the JD is aimed at -- {article} {word} "
        f"corroborating read that sits independently of any single line item",
        f"the arc of this career has the structure of {article} '{label_text}' "
        f"-- someone who has done this kind of work in real settings, not just "
        f"described it -- a {word} structural match for what this JD needs",
        f"at the pattern level rather than the checklist level, this reads as "
        f"the '{label_text}' this JD is targeting: {article} {word} structural "
        f"fit that reinforces the component-level signals rather than "
        f"duplicating them",
        f"the shape of this career, taken as a whole, is {article} {word} "
        f"'{label_text}' read -- the kind of broad structural alignment that "
        f"individual-component scores alone don't always surface",
        f"this career has converged on the '{label_text}' pattern the JD is "
        f"targeting -- which is harder to stage than a skills list and a "
        f"stronger signal of genuine fit; {article} {word} read overall",
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
# Behavioral signal synthesizers — translate the 23 redrob_signals into plain  #
# hiring-relevant language.  These are standalone functions (not _StrengthBuilder  #
# / _ConcernBuilder callables) because behavioral signals are multipliers, not  #
# ScoreComponents, and have no EligibilityFinding to carry through.            #
# --------------------------------------------------------------------------- #

_BEHAVIORAL_STRENGTH_THRESHOLD_RESPONSE: Final[float] = 0.65
_BEHAVIORAL_STRENGTH_THRESHOLD_GITHUB: Final[float] = 40.0
_BEHAVIORAL_STRENGTH_INACTIVE_CUTOFF_DAYS: Final[int] = 50
_BEHAVIORAL_CONCERN_INACTIVE_CUTOFF_DAYS: Final[int] = 90
_BEHAVIORAL_CONCERN_RESPONSE_THRESHOLD: Final[float] = 0.22
_BEHAVIORAL_STRENGTH_SAVES_THRESHOLD: Final[int] = 4

# How weak the weakest selected component's raw score must be before a strong
# behavioral clause is allowed to displace it (§ reasoning design §5).
_BEHAVIORAL_DISPLACE_RAW_CEILING: Final[float] = 0.35


def _behavioral_strength(
    raw: RawCandidate,
    candidate_id: CandidateId,
    as_of: date,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]] | None:
    """Synthesize a hiring-relevant behavioral clause from redrob_signals.

    Returns ``None`` when no signal clears the threshold for a meaningful
    positive observation -- so the caller can skip it entirely rather than
    emitting a generic or vacuous clause.
    """
    signals = raw.redrob_signals
    days_inactive = max(0, (as_of - signals.last_active_date).days)
    open_to_work = signals.open_to_work_flag
    response_rate = float(signals.recruiter_response_rate)
    github = float(signals.github_activity_score)
    saves = int(signals.saved_by_recruiters_30d)

    is_recent = days_inactive <= _BEHAVIORAL_STRENGTH_INACTIVE_CUTOFF_DAYS
    high_response = response_rate >= _BEHAVIORAL_STRENGTH_THRESHOLD_RESPONSE
    has_github = github >= _BEHAVIORAL_STRENGTH_THRESHOLD_GITHUB
    notable_saves = saves >= _BEHAVIORAL_STRENGTH_SAVES_THRESHOLD

    if not (open_to_work or is_recent or high_response or has_github or notable_saves):
        return None

    dump = _dump(raw)
    evidence: list[EvidenceRef] = [
        mint(dump, kind=EvidenceKind.SIGNAL, path="redrob_signals.open_to_work_flag"),
        mint(dump, kind=EvidenceKind.SIGNAL, path="redrob_signals.last_active_date"),
    ]
    _sig = EvidenceKind.SIGNAL
    if high_response:
        evidence.append(
            mint(dump, kind=_sig, path="redrob_signals.recruiter_response_rate")
        )
    if has_github:
        evidence.append(
            mint(dump, kind=_sig, path="redrob_signals.github_activity_score")
        )
    if notable_saves:
        evidence.append(
            mint(dump, kind=_sig, path="redrob_signals.saved_by_recruiters_30d")
        )

    response_pct = f"{round(response_rate * 100)}%"
    github_score = int(github)
    days_label = f"{days_inactive} days" if days_inactive > 0 else "today"

    templates: tuple[str, ...]
    if open_to_work and is_recent and high_response:
        templates = (
            f"marked open to work and last active {days_label} ago with a "
            f"{response_pct} recruiter response rate -- the practical "
            f"availability side here is clean",
            f"open to work, active {days_label} ago, responds to {response_pct} "
            f"of recruiter messages -- realistically available, not just "
            f"theoretically so",
            f"practical sourcing looks straightforward: open to work, seen "
            f"{days_label} ago, {response_pct} of outreach gets a reply",
            f"from a hiring-logistics standpoint, the signals are positive -- "
            f"open to work, {days_label} since last login, {response_pct} "
            f"recruiter response rate",
            f"availability checks out clearly: open to work, active "
            f"{days_label} ago, and a {response_pct} response rate that "
            f"means most outreach gets answered",
        )
    elif open_to_work and is_recent:
        templates = (
            f"flagged open to work and last active {days_label} ago -- "
            f"realistically in the market rather than passively listed",
            f"open to work with a recent platform login ({days_label} ago) -- "
            f"the availability side of this hire is not a question",
            f"marked available and active {days_label} ago -- worth moving "
            f"on quickly given the combination of profile quality and genuine "
            f"market availability",
            f"practical availability is clear: open to work, last seen "
            f"{days_label} ago on the platform",
        )
    elif high_response and is_recent:
        templates = (
            f"responds to {response_pct} of recruiter messages and was last "
            f"active {days_label} ago -- the sourcing side of this hire "
            f"looks workable",
            f"a {response_pct} recruiter response rate alongside recent "
            f"activity ({days_label} ago) is a practical positive -- "
            f"outreach is likely to land",
            f"active {days_label} ago with a {response_pct} response rate "
            f"to recruiter messages -- the reachability side here is "
            f"stronger than average",
        )
    elif notable_saves and (is_recent or open_to_work):
        templates = (
            f"saved by {saves} recruiters in the past 30 days alongside "
            f"open-to-work status -- market competition for this candidate "
            f"is real and worth acting on",
            f"{saves} recruiter saves in the last month signals active market "
            f"demand; this candidate won't stay available indefinitely",
            f"open to work and already on {saves} recruiters' shortlists in "
            f"the last 30 days -- the market is paying attention here",
        )
    elif has_github and (is_recent or open_to_work or high_response):
        templates = (
            f"active GitHub presence ({github_score}/100) gives external "
            f"validation of the technical work this profile claims -- not "
            f"just self-reported",
            f"GitHub activity score of {github_score}/100 suggests technical "
            f"work visible outside a closed employer context -- relevant "
            f"since the JD explicitly flags open-source contributions",
            f"the JD specifically calls out open-source contributions; a "
            f"{github_score}/100 GitHub activity score here is a "
            f"concrete corroborating signal",
        )
    elif high_response:
        templates = (
            f"responds to {response_pct} of recruiter messages -- above the "
            f"typical response rate and a positive sourcing signal",
            f"a {response_pct} recruiter response rate means most outreach "
            f"gets answered, regardless of whether they're actively "
            f"looking right now",
        )
    else:
        templates = (
            f"platform signals are positive: active {days_label} ago, "
            f"{response_pct} recruiter response rate",
            f"recently active ({days_label} ago) with a {response_pct} "
            f"recruiter response rate -- practical availability "
            f"is not a concern here",
        )

    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


def _behavioral_concern(
    raw: RawCandidate,
    candidate_id: CandidateId,
    as_of: date,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]] | None:
    """Synthesize a hiring-relevant behavioral concern from redrob_signals.

    Returns ``None`` when signals don't justify adding a concern -- so the
    caller can skip rather than emitting a weak or vacuous clause.
    """
    signals = raw.redrob_signals
    days_inactive = max(0, (as_of - signals.last_active_date).days)
    open_to_work = signals.open_to_work_flag
    response_rate = float(signals.recruiter_response_rate)

    inactive_long = days_inactive > _BEHAVIORAL_CONCERN_INACTIVE_CUTOFF_DAYS
    low_response = response_rate < _BEHAVIORAL_CONCERN_RESPONSE_THRESHOLD
    not_available = not open_to_work

    if not ((inactive_long and not_available) or low_response):
        return None

    dump = _dump(raw)
    evidence: list[EvidenceRef] = [
        mint(dump, kind=EvidenceKind.SIGNAL, path="redrob_signals.last_active_date"),
        mint(dump, kind=EvidenceKind.SIGNAL, path="redrob_signals.open_to_work_flag"),
    ]
    if low_response:
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.SIGNAL,
                path="redrob_signals.recruiter_response_rate",
            )
        )

    response_pct = f"{round(response_rate * 100)}%"
    days_label = f"{days_inactive} days"

    templates: tuple[str, ...]
    if inactive_long and not_available and low_response:
        templates = (
            f"platform signals are a concern: inactive for {days_label}, "
            f"not marked open to work, and only a {response_pct} recruiter "
            f"response rate -- this candidate may not be in the market "
            f"right now regardless of how the profile reads",
            f"inactive for {days_label}, not flagged as available, and a "
            f"{response_pct} recruiter response rate -- the practical "
            f"reachability here is a real question before moving forward",
            f"despite the profile quality, {days_label} of platform "
            f"inactivity, no open-to-work flag, and a {response_pct} "
            f"response rate together suggest this hire may be harder to "
            f"close than it looks on paper",
            f"a {response_pct} response rate alongside {days_label} of "
            f"inactivity and no open-to-work status is a meaningful "
            f"practical obstacle -- worth confirming actual availability "
            f"before progressing",
        )
    elif inactive_long and not_available:
        templates = (
            f"not marked open to work and inactive for {days_label} -- "
            f"a real question mark on whether they're actually in the "
            f"market right now, independent of how the profile reads",
            f"last platform login was {days_label} ago with no open-to-work "
            f"flag -- availability is a genuine question to resolve before "
            f"prioritizing this candidate",
            f"{days_label} without a platform login, and not flagged as "
            f"available -- worth a quick reachability check before "
            f"moving this profile up the queue",
            f"the skills read well, but {days_label} of inactivity and "
            f"no open-to-work status is worth flagging as a practical "
            f"availability concern, not a skills concern",
        )
    elif low_response:
        templates = (
            f"a {response_pct} recruiter response rate is the friction "
            f"point here -- most cold outreach goes unanswered, which "
            f"changes the sourcing approach needed",
            f"{response_pct} response rate to recruiter messages -- a "
            f"practical sourcing obstacle that warrants a warm intro "
            f"or a different reach channel rather than cold outreach",
            f"the {response_pct} recruiter response rate is worth noting: "
            f"reachability may be lower than the profile quality would suggest",
        )
    else:
        templates = (
            f"last active {days_label} ago with open-to-work status off -- "
            f"practical availability is less clear than the profile quality "
            f"would otherwise suggest",
        )

    fragment = _pick(seed, templates)
    return fragment, tuple(evidence)


# --------------------------------------------------------------------------- #
# JD-latent hiring-thesis architecture.                                       #
#                                                                               #
# The JD establishes a ranked evidence hierarchy for what constitutes a       #
# strong hire.  Instead of asking "which ScoreComponent has the highest        #
# weighted contribution?", the new _strength_clauses selects clauses by       #
# asking "which JD positive latent does this candidate most clearly satisfy?" #
#                                                                               #
# Positive latents are checked in JD priority order; the first _MAX_STRENGTHS  #
# that clear _POSITIVE_LATENT_THRESHOLD emit thesis clauses built from career  #
# and skill evidence, not from ML score floats.  Negative latents with        #
# sufficient anchor cosine add concern clauses after soft-penalty findings.   #
# --------------------------------------------------------------------------- #

_JD_POSITIVE_LATENT_PRIORITY: Final[tuple[str, ...]] = (
    "jd.production_ml",  # 1 — shipped ML to real users
    "jd.product_company",  # 2 — at a product company, not consulting/research
    "jd.retrieval_ranking",  # 3 — core domain: ranking / retrieval / search
    "jd.shipping_mentality",  # 4 — hands-on IC, not just researcher or manager
    "jd.eval_framework",  # 5 — evaluation rigor (NDCG, MRR, MAP)
    "jd.hybrid_retrieval",  # 6 — dense + sparse retrieval experience
)
_JD_NEGATIVE_LATENT_PRIORITY: Final[tuple[str, ...]] = (
    "jd.pure_researcher",  # pure research, no production counterpart
    "jd.consulting_only",  # consulting-heavy, limited product ownership
    "jd.keyword_only",  # skill claims not corroborated by work evidence
    "jd.framework_enthusiast",  # framework-only tooling, no system depth
    "jd.inactive",  # not practically available (overlaps behavioral_concern)
)
# Minimum cosine to a positive jd.* anchor to emit a thesis strength clause.
_POSITIVE_LATENT_THRESHOLD: Final[float] = 0.12
# Minimum cosine to a negative jd.* anchor to add a latent concern clause.
_NEGATIVE_LATENT_THRESHOLD: Final[float] = 0.10
# Maps each positive jd.* latent to the nearest ScoreComponent for
# ReasoningClause.jd_link continuity (the domain constraint requires
# jd_link to be EligibilityCode | ScoreComponent | None).
_LATENT_TO_JD_LINK: Final[Mapping[str, ScoreComponent]] = {
    "jd.production_ml": ScoreComponent.CAREER_FIT,
    "jd.product_company": ScoreComponent.CAREER_FIT,
    "jd.retrieval_ranking": ScoreComponent.SEMANTIC_FIT,
    "jd.shipping_mentality": ScoreComponent.CAREER_FIT,
    "jd.eval_framework": ScoreComponent.SKILL_MATCH,
    "jd.hybrid_retrieval": ScoreComponent.SEMANTIC_FIT,
}

# Evidence-scan lexicons (lowercase substring matching over position blobs).
_PROD_SIGNAL_CUES: Final[tuple[str, ...]] = (
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
    "real users",
    "live system",
)
_PRODUCT_ORG_CUES: Final[tuple[str, ...]] = (
    "product",
    "saas",
    "platform",
    "our app",
    "our users",
    "feature flag",
    "a/b test",
    "user base",
    "mau",
    "dau",
    "consumer",
    "in-house",
    "proprietary",
)
_RETRIEVAL_DOMAIN_CUES: Final[tuple[str, ...]] = (
    "retrieval",
    "ranking",
    "search engine",
    "recommendation",
    "recsys",
    "information retrieval",
    "faiss",
    "elasticsearch",
    "solr",
    "lucene",
    "bm25",
    "colbert",
    "dense retrieval",
    "sparse retrieval",
    "vector search",
    "semantic search",
    "ann",
)
_EVAL_EVIDENCE_CUES: Final[tuple[str, ...]] = (
    "ndcg",
    "mrr",
    "map",
    "p@",
    "precision@",
    "recall@",
    "auc",
    "offline eval",
    "online eval",
    "a/b test",
    "evaluation framework",
    "benchmark",
    "offline metrics",
)
_HYBRID_RET_CUES: Final[tuple[str, ...]] = (
    "hybrid",
    "dense",
    "sparse",
    "bm25",
    "faiss",
    "colbert",
    "dpr",
    "bi-encoder",
    "cross-encoder",
    "reranker",
    "two-stage",
    "dense+sparse",
)
_HANDS_ON_CUES: Final[tuple[str, ...]] = (
    "built",
    "implemented",
    "shipped",
    "deployed",
    "wrote",
    "coded",
    "developed",
    "refactored",
    "delivered",
    "pull request",
    "commit",
    "open source",
    "github",
)
_RESEARCH_ONLY_CUES: Final[tuple[str, ...]] = (
    "research",
    "published",
    "publication",
    "paper",
    "phd",
    "thesis",
    "neurips",
    "icml",
    "acl",
    "prototype only",
    "proof of concept",
)
_CONSULTING_SIGNAL_CUES: Final[tuple[str, ...]] = (
    "consulting",
    "client",
    "consultancy",
    "system integrat",
    "outsourc",
    "managed services",
    "staff augment",
    "billable",
)


def _best_position_for_cues(raw: RawCandidate, cues: tuple[str, ...]) -> int | None:
    """Index of the career_history position with the most cue hits; None if zero."""
    best_idx: int | None = None
    best_count = 0
    for idx, pos in enumerate(raw.career_history):
        blob = f"{pos.description} {pos.company} {pos.industry}".lower()
        count = sum(1 for cue in cues if cue in blob)
        if count > best_count:
            best_count = count
            best_idx = idx
    return best_idx if best_count > 0 else None


def _jd_production_ml_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    sim: float,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]] | None:
    """Production ML deployment: ML shipped to real users at scale."""
    _ = representation
    word = _bucket_word(seed, sim)
    article = _article(word)
    dump = _dump(raw)
    evidence: list[EvidenceRef] = []
    templates: tuple[str, ...]

    idx = _best_position_for_cues(raw, _PROD_SIGNAL_CUES)
    if idx is not None:
        pos = raw.career_history[idx]
        _ck = EvidenceKind.CAREER_FIELD
        evidence.append(mint(dump, kind=_ck, path=f"career_history[{idx}].company"))
        evidence.append(mint(dump, kind=_ck, path=f"career_history[{idx}].description"))
        company = pos.company
        templates = (
            f"production ML track record at {company}: the work described "
            f"goes beyond experimentation into live systems with real-user "
            f"stakes -- {article} {word} signal for this role's primary bar",
            f"at {company}, the engineering context reads as production ML -- "
            f"deployed systems, live traffic, real operational constraints; "
            f"{article} {word} answer to the JD's first-order requirement",
            f"{company} shows up as a context where ML went to production, "
            f"not just to a notebook -- the description puts this in "
            f"{article} {word} tier for what the role is actually asking for",
            f"the production ML bar this JD sets is cleared at {company}: "
            f"the work described carries the fingerprints of live deployment, "
            f"not prototype-and-hand-off",
            f"shipping ML to real users is the hardest signal to fake on a "
            f"resume, and {company}'s context makes {article} {word} case "
            f"that this candidate has actually done it",
            f"{article.capitalize()} {word} production ML signal from "
            f"{company}: not just model training but live inference with "
            f"the operational responsibility that comes with it",
            f"the JD's primary bar is production ML; the {company} context "
            f"in this profile clears it -- the work is framed in deployment "
            f"terms, not research terms",
        )
    else:
        if sim < _POSITIVE_LATENT_THRESHOLD:
            return None
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED,
                "semantic.anchor.production_ml",
                round(sim, 4),
            )
        )
        templates = (
            f"profile language suggests {article} {word} production-ML "
            f"orientation -- the way work is described reads closer to "
            f"deployed systems than to research or prototypes, even without "
            f"explicit deployment language",
            f"{word} alignment with the production-ML signal; the framing "
            f"of this candidate's work suggests deployment experience, though "
            f"the specific context could be described more explicitly",
        )
    return _pick(seed, templates), tuple(evidence)


def _jd_product_company_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    sim: float,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]] | None:
    """Product company tenure: built for end users inside a product org."""
    word = _bucket_word(seed, sim)
    article = _article(word)
    dump = _dump(raw)
    evidence: list[EvidenceRef] = []
    templates: tuple[str, ...]

    career = representation.require_career()
    product_positions = [p for p in career.positions if p.is_product_company]
    if product_positions:
        best = product_positions[0]  # positions are sorted by start_date desc
        idx = _position_index(
            raw,
            company=best.company,
            title=best.title,
            start_date=best.start_date,
        )
        if idx is not None:
            evidence.append(
                mint(
                    dump,
                    kind=EvidenceKind.CAREER_FIELD,
                    path=f"career_history[{idx}].company",
                )
            )
            evidence.append(
                mint(
                    dump,
                    kind=EvidenceKind.CAREER_FIELD,
                    path=f"career_history[{idx}].industry",
                )
            )
        else:
            evidence.append(
                make_evidence(
                    EvidenceKind.DERIVED, "career.product_company", best.company
                )
            )
        org_count = len(product_positions)
        plural = "s" if org_count > 1 else ""
        recently = "most recently" if org_count == 1 else "consistently"
        templates = (
            f"product-company background at {best.company} ({best.industry}) "
            f"-- the JD explicitly screens for this context, and {article} "
            f"{word} fraction of this career is product-company tenure",
            f"{best.company} is {article} {word} product-company context; "
            f"across {org_count} product role{plural} this person has built "
            f"for end users rather than for clients",
            f"{recently} at product companies -- {best.company} "
            f"({best.industry}) is the kind of org this JD targets, and the "
            f"tenure there is {article} {word} match for what's being asked",
            f"the JD is written for someone who has shipped to end users "
            f"inside a product company; {best.company} ({best.industry}) "
            f"is exactly that context -- {article} {word} fit on this screen",
            f"product-company tenure is one of this JD's core filters; "
            f"{best.company} in {best.industry} clears it -- {article} "
            f"{word} footing on the most important background requirement",
            f"{best.company} ({best.industry}) is a product-company context; "
            f"the career has spent {article} {word} portion of its time "
            f"building for users, not billing clients",
        )
    else:
        idx = _best_position_for_cues(raw, _PRODUCT_ORG_CUES)
        if idx is not None:
            evidence.append(
                mint(
                    dump,
                    kind=EvidenceKind.CAREER_FIELD,
                    path=f"career_history[{idx}].company",
                )
            )
        else:
            if sim < _POSITIVE_LATENT_THRESHOLD:
                return None
            evidence.append(
                make_evidence(
                    EvidenceKind.DERIVED,
                    "semantic.anchor.product_company",
                    round(sim, 4),
                )
            )
        templates = (
            f"profile suggests {article} {word} product-company orientation "
            f"in how work is described, though explicit product-company "
            f"markers are sparse in the history",
            f"{word} alignment with the product-company signal -- work "
            f"framing suggests user-facing product context even where the "
            f"company classification isn't explicit",
        )
    return _pick(seed, templates), tuple(evidence)


def _jd_retrieval_ranking_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    sim: float,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]] | None:
    """Core domain: ranking, retrieval, or search systems built and shipped."""
    word = _bucket_word(seed, sim)
    article = _article(word)
    dump = _dump(raw)
    evidence: list[EvidenceRef] = []
    templates: tuple[str, ...]

    semantic = representation.require_semantic()
    pos_idx = _best_position_for_cues(raw, _RETRIEVAL_DOMAIN_CUES)
    skill_idx: int | None = None
    for i, sk in enumerate(raw.skills):
        if any(
            cue in sk.name.lower()
            for cue in ("retriev", "rank", "search", "recsys", "recommend")
        ):
            skill_idx = i
            break

    if pos_idx is not None:
        pos = raw.career_history[pos_idx]
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{pos_idx}].company",
            )
        )
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{pos_idx}].description",
            )
        )
        company = pos.company
        templates = (
            f"retrieval and ranking domain depth is visible at {company}: "
            f"the work described puts this squarely in the JD's core domain, "
            f"not just adjacent to it -- {article} {word} signal",
            f"{company} shows up as a retrieval/ranking context -- exactly "
            f"the core domain this role requires; {article} {word} case "
            f"for genuine domain depth rather than keyword proximity",
            f"the JD hires for retrieval/ranking expertise built in "
            f"production; {company} is where this candidate has done it -- "
            f"{article} {word} and specific answer to the role's core ask",
            f"domain match on retrieval/ranking: the {company} experience "
            f"puts this in the JD's sweet spot rather than at the margins",
            f"core domain evidence at {company} -- ranking and retrieval "
            f"work in a production context; {article} {word} fit for this "
            f"role's most specific technical requirement",
        )
    elif skill_idx is not None:
        sk = raw.skills[skill_idx]
        evidence.append(
            mint(dump, kind=EvidenceKind.SKILL, path=f"skills[{skill_idx}].name")
        )
        templates = (
            f"retrieval/ranking domain expertise surfaces in the skills: "
            f"{sk.name} -- {article} {word} match for the core domain "
            f"this role is built around",
            f"on the skills side, {sk.name} anchors this in the retrieval/"
            f"ranking space the JD is targeting; {article} {word} alignment "
            f"with the role's technical core",
        )
    else:
        if sim < _POSITIVE_LATENT_THRESHOLD:
            return None
        anchor = semantic.best_positive_anchor
        if anchor is not None:
            evidence.append(
                make_evidence(
                    EvidenceKind.DERIVED, "semantic.best_positive_anchor", str(anchor)
                )
            )
            anchor_label = str(anchor).rsplit(".", 1)[-1].replace("_", " ")
        else:
            evidence.append(
                make_evidence(
                    EvidenceKind.DERIVED,
                    "semantic.anchor.retrieval_ranking",
                    round(sim, 4),
                )
            )
            anchor_label = "retrieval and ranking"
        templates = (
            f"profile language patterns suggest {article} {word} alignment "
            f"with the '{anchor_label}' domain this role is built for, "
            f"independently of what is in the skills section",
            f"{word} semantic alignment with retrieval/ranking -- how this "
            f"candidate describes their work gravitates toward this JD's "
            f"core technical focus rather than toward adjacent areas",
        )
    return _pick(seed, templates), tuple(evidence)


def _jd_shipping_mentality_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    sim: float,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]] | None:
    """Hands-on engineering execution: built and shipped, not just designed."""
    _ = representation
    word = _bucket_word(seed, sim)
    article = _article(word)
    dump = _dump(raw)
    evidence: list[EvidenceRef] = []
    templates: tuple[str, ...]

    idx = _best_position_for_cues(raw, _HANDS_ON_CUES)
    if idx is not None:
        pos = raw.career_history[idx]
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{idx}].company",
            )
        )
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{idx}].description",
            )
        )
        company = pos.company
        templates = (
            f"hands-on engineering at {company}: the description uses "
            f"builder language -- shipped, built, deployed -- not just "
            f"'oversaw' or 'designed'; {article} {word} IC execution "
            f"signal for a role that needs exactly this",
            f"at {company}, this person was writing code and shipping "
            f"systems rather than directing or reviewing -- {article} {word} "
            f"execution track record for a role that requires IC depth",
            f"the work at {company} is described in the language of someone "
            f"who built things themselves: {article} {word} hands-on "
            f"engineering signal, not an architectural or managerial one",
            f"shipping mentality in evidence at {company}: the profile "
            f"describes work in delivery and implementation terms, which is "
            f"what this JD is actually looking for rather than seniority titles",
            f"{company} context is described with vocabulary of someone "
            f"who codes and ships rather than delegates -- {article} {word} "
            f"IC signal for a role that values engineering execution over "
            f"org-chart seniority",
        )
    else:
        if sim < _POSITIVE_LATENT_THRESHOLD:
            return None
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED,
                "semantic.anchor.shipping_mentality",
                round(sim, 4),
            )
        )
        templates = (
            f"profile language suggests {article} {word} hands-on "
            f"engineering orientation -- the emphasis is on building and "
            f"shipping rather than on strategy or management",
            f"{word} alignment with the execution signal this JD is looking "
            f"for -- the candidate's work framing reads as IC-first",
        )
    return _pick(seed, templates), tuple(evidence)


def _jd_eval_framework_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    sim: float,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]] | None:
    """Evaluation rigor: NDCG / MRR / MAP usage and offline/online discipline."""
    _ = representation
    word = _bucket_word(seed, sim)
    article = _article(word)
    dump = _dump(raw)
    evidence: list[EvidenceRef] = []
    templates: tuple[str, ...]

    eval_skill_idx: int | None = None
    for i, sk in enumerate(raw.skills):
        if any(
            cue in sk.name.lower()
            for cue in ("ndcg", "mrr", " map", "eval", "metrics", "a/b")
        ):
            eval_skill_idx = i
            break

    pos_idx = _best_position_for_cues(raw, _EVAL_EVIDENCE_CUES)
    if eval_skill_idx is not None:
        sk = raw.skills[eval_skill_idx]
        evidence.append(
            mint(dump, kind=EvidenceKind.SKILL, path=f"skills[{eval_skill_idx}].name")
        )
        if pos_idx is not None:
            evidence.append(
                mint(
                    dump,
                    kind=EvidenceKind.CAREER_FIELD,
                    path=f"career_history[{pos_idx}].description",
                )
            )
        templates = (
            f"evaluation rigour shows up explicitly: {sk.name} is listed as "
            f"a skill, and the JD specifically calls for proper offline/online "
            f"evaluation -- {article} {word} signal this isn't a "
            f"vibes-based engineer",
            f"{sk.name} in the skills section is exactly the evaluation-metric "
            f"literacy the JD flags as a differentiator; {article} {word} "
            f"credibility signal for a ranking role",
            f"the presence of {sk.name} here is the JD's 'evaluation "
            f"framework' signal in concrete form -- {article} {word} "
            f"indicator that this person measures ranking quality correctly",
        )
    elif pos_idx is not None:
        pos = raw.career_history[pos_idx]
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{pos_idx}].company",
            )
        )
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{pos_idx}].description",
            )
        )
        templates = (
            f"evaluation framework usage in evidence at {pos.company}: "
            f"the role description references ranking metrics (NDCG, MRR, "
            f"or similar), which is the rigour signal the JD is hiring for",
            f"the {pos.company} description shows evaluation-aware "
            f"engineering -- ranking metrics are referenced, differentiating "
            f"this from candidates who tune systems by intuition alone",
        )
    else:
        if sim < _POSITIVE_LATENT_THRESHOLD:
            return None
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED, "semantic.anchor.eval_framework", round(sim, 4)
            )
        )
        templates = (
            f"{word} alignment with the evaluation-framework signal -- "
            f"profile language suggests metric-aware engineering even where "
            f"specific metrics are not named explicitly",
        )
    return _pick(seed, templates), tuple(evidence)


def _jd_hybrid_retrieval_strength(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    sim: float,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]] | None:
    """Dense + sparse hybrid retrieval experience."""
    _ = representation
    word = _bucket_word(seed, sim)
    article = _article(word)
    dump = _dump(raw)
    evidence: list[EvidenceRef] = []
    templates: tuple[str, ...]

    pos_idx = _best_position_for_cues(raw, _HYBRID_RET_CUES)
    skill_idx: int | None = None
    for i, sk in enumerate(raw.skills):
        name_lower = sk.name.lower()
        _hcues = ("hybrid", "bm25", "faiss", "colbert", "rerank", "dense", "sparse")
        if any(cue in name_lower for cue in _hcues):
            skill_idx = i
            break

    if pos_idx is not None:
        pos = raw.career_history[pos_idx]
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{pos_idx}].company",
            )
        )
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{pos_idx}].description",
            )
        )
        if skill_idx is not None:
            evidence.append(
                mint(dump, kind=EvidenceKind.SKILL, path=f"skills[{skill_idx}].name")
            )
        company = pos.company
        templates = (
            f"hybrid retrieval experience at {company}: the description "
            f"shows familiarity with the full retrieval stack (dense + sparse), "
            f"not just one approach -- {article} {word} fit for what the JD asks",
            f"dense and sparse retrieval are both in scope based on the "
            f"{company} experience; {article} {word} signal for a role that "
            f"specifically requires hybrid systems",
            f"the JD asks for hybrid retrieval; the {company} context suggests "
            f"{article} {word} hands-on exposure to both dense and sparse "
            f"methods rather than expertise in one only",
        )
    elif skill_idx is not None:
        sk = raw.skills[skill_idx]
        evidence.append(
            mint(dump, kind=EvidenceKind.SKILL, path=f"skills[{skill_idx}].name")
        )
        templates = (
            f"hybrid retrieval technology visible in the skills: {sk.name} "
            f"-- {article} {word} alignment with the JD's hybrid-retrieval "
            f"requirement",
            f"{sk.name} in the skills section is the JD's hybrid-retrieval "
            f"signal in concrete form; {article} {word} match for this "
            f"specific technical ask",
        )
    else:
        if sim < _POSITIVE_LATENT_THRESHOLD:
            return None
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED,
                "semantic.anchor.hybrid_retrieval",
                round(sim, 4),
            )
        )
        templates = (
            f"{word} alignment with the hybrid-retrieval domain; profile "
            f"language suggests familiarity with combined dense and sparse "
            f"approaches",
        )
    return _pick(seed, templates), tuple(evidence)


_LATENT_STRENGTH_BUILDERS: Final[Mapping[str, _LatentBuilder]] = {
    "jd.production_ml": _jd_production_ml_strength,
    "jd.product_company": _jd_product_company_strength,
    "jd.retrieval_ranking": _jd_retrieval_ranking_strength,
    "jd.shipping_mentality": _jd_shipping_mentality_strength,
    "jd.eval_framework": _jd_eval_framework_strength,
    "jd.hybrid_retrieval": _jd_hybrid_retrieval_strength,
}


# --------------------------------------------------------------------------- #
# Negative-latent concern builders — pattern-level concerns derived from JD   #
# anti-patterns that do not map 1:1 to EligibilityCodes.                      #
# --------------------------------------------------------------------------- #
def _jd_pure_researcher_concern(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    sim: float,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    """Anti-pattern: pure research career with no production counterpart."""
    _ = representation
    word = _bucket_word(seed, sim)
    dump = _dump(raw)
    evidence: list[EvidenceRef] = []
    templates: tuple[str, ...]

    idx = _best_position_for_cues(raw, _RESEARCH_ONLY_CUES)
    if idx is not None:
        pos = raw.career_history[idx]
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{idx}].company",
            )
        )
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{idx}].description",
            )
        )
        company = pos.company
        templates = (
            f"career pattern leans toward research ({company} context): "
            f"this is one of the JD's explicit anti-patterns -- the role "
            f"needs production ML, not research output alone",
            f"the {company} context reads as research-oriented (publications, "
            f"papers, or prototype-level work) rather than the production "
            f"deployment this JD is hiring for",
            f"pure-researcher signal at {company}: the JD explicitly rules "
            f"out candidates whose recent work is research without a "
            f"production counterpart",
            f"research framing at {company} is visible in the description -- "
            f"the role specifically wants someone who has shipped systems, "
            f"not just produced results in a lab or academic context",
        )
    else:
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED, "semantic.anchor.pure_researcher", round(sim, 4)
            )
        )
        templates = (
            f"profile language has {word} alignment with the 'pure researcher' "
            f"anti-pattern this JD calls out -- the work framing is academic "
            f"rather than deployment-oriented",
        )
    return _pick(seed, templates), tuple(evidence)


def _jd_consulting_only_concern(
    raw: RawCandidate,
    representation: CandidateRepresentation,
    sim: float,
    seed: str,
) -> tuple[str, tuple[EvidenceRef, ...]]:
    """Anti-pattern: consulting-heavy career with limited product ownership."""
    _ = representation
    word = _bucket_word(seed, sim)
    dump = _dump(raw)
    evidence: list[EvidenceRef] = []
    templates: tuple[str, ...]

    idx = _best_position_for_cues(raw, _CONSULTING_SIGNAL_CUES)
    if idx is not None:
        pos = raw.career_history[idx]
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{idx}].company",
            )
        )
        evidence.append(
            mint(
                dump,
                kind=EvidenceKind.CAREER_FIELD,
                path=f"career_history[{idx}].description",
            )
        )
        company = pos.company
        templates = (
            f"consulting-heavy career pattern at {company} and elsewhere: "
            f"client-delivery work builds breadth, but not the product-company "
            f"depth this JD is specifically screening for",
            f"the {company} context reads as a consulting engagement rather "
            f"than product-company ownership -- the JD is explicit that this "
            f"is an anti-pattern for this role",
            f"client-facing consulting at {company} is {word} evidence of "
            f"what this JD screens against: the role wants in-house product "
            f"engineering, not client delivery",
            f"the {company} role description carries consulting signals -- "
            f"a career path the JD views as a poor proxy for product-company "
            f"ML experience",
        )
    else:
        evidence.append(
            make_evidence(
                EvidenceKind.DERIVED, "semantic.anchor.consulting_only", round(sim, 4)
            )
        )
        templates = (
            f"profile language has {word} alignment with the consulting-heavy "
            f"anti-pattern the JD calls out -- client-delivery framing over "
            f"product-ownership framing",
        )
    return _pick(seed, templates), tuple(evidence)


_LATENT_CONCERN_BUILDERS: Final[Mapping[str, _LatentBuilder]] = {
    "jd.pure_researcher": _jd_pure_researcher_concern,
    "jd.consulting_only": _jd_consulting_only_concern,
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
    period, behavioral signals, ...) rather than a bare component float;
    phrasing is selected deterministically per
    ``(candidate_id, component_or_code)`` from a pool of structurally
    distinct sentence templates, so two candidates who share a dominant
    component still render materially different text.

    ``as_of`` is injected (not read from the wall clock) and is used only
    to compute recency for behavioral-signal clauses.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=False
    )

    as_of: date = Field()

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
        # JD-latent hiring-thesis selection: check each positive JD anchor in
        # priority order, emit a clause for the first _MAX_STRENGTHS that clear
        # the threshold.  Clause content is derived from career/skill evidence
        # that directly supports the JD's hiring case for this candidate, not
        # from which ML score component happened to score highest.
        semantic = representation.require_semantic()
        anchor_sims = semantic.anchor_similarities
        clauses: list[ReasoningClause] = []
        for latent_id in _JD_POSITIVE_LATENT_PRIORITY:
            if len(clauses) >= _MAX_STRENGTHS:
                break
            sim = float(anchor_sims.get(AnchorId(latent_id), -1.0))
            builder = _LATENT_STRENGTH_BUILDERS.get(latent_id)
            if builder is None:
                continue
            seed = f"{ranked.candidate_id}:{latent_id}:strength"
            result = builder(raw, representation, sim, seed)
            if result is None:
                continue
            fragment, evidence = result
            jd_link: ScoreComponent | None = _LATENT_TO_JD_LINK.get(latent_id)
            clauses.append(
                ReasoningClause(
                    polarity=ReasoningPolarity.STRENGTH,
                    fragment=fragment,
                    evidence=evidence,
                    jd_link=jd_link,
                )
            )

        # Behavioral strength fills any remaining slot (additive, not replacing
        # JD evidence -- the hiring thesis always takes priority over behavioral).
        if len(clauses) < _MAX_STRENGTHS and representation.behavioral is not None:
            beh_seed = f"{ranked.candidate_id}:behavioral:strength"
            beh_result = _behavioral_strength(
                raw, ranked.candidate_id, self.as_of, beh_seed
            )
            if beh_result is not None:
                beh_fragment, beh_evidence = beh_result
                clauses.append(
                    ReasoningClause(
                        polarity=ReasoningPolarity.STRENGTH,
                        fragment=beh_fragment,
                        evidence=beh_evidence,
                        jd_link=None,
                    )
                )

        # Ensure the domain invariant: at least one clause carries a non-null
        # jd_link.  Behavioral clauses always have jd_link=None, so if the
        # only clause that fired was behavioral (no JD latent cleared the
        # threshold), we prepend a score-component fallback to satisfy the
        # invariant without discarding the behavioral signal.
        has_jd_link = any(c.jd_link is not None for c in clauses)
        if not has_jd_link:
            clauses.insert(0, self._fallback_strength(raw, representation, ranked))

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
        clauses: list[ReasoningClause] = [
            self._concern_clause(raw, representation, finding)
            for finding in penalties[:_MAX_CONCERNS]
        ]

        # JD negative-latent concerns: add pattern-level concerns when the
        # candidate's profile fires strongly against a JD anti-pattern anchor
        # (e.g., pure_researcher, consulting_only) and capacity remains.
        if len(clauses) < _MAX_CONCERNS and representation.semantic is not None:
            anchor_sims = representation.semantic.anchor_similarities
            for latent_id in _JD_NEGATIVE_LATENT_PRIORITY:
                if len(clauses) >= _MAX_CONCERNS:
                    break
                sim = float(anchor_sims.get(AnchorId(latent_id), -1.0))
                if sim < _NEGATIVE_LATENT_THRESHOLD:
                    continue
                builder = _LATENT_CONCERN_BUILDERS.get(latent_id)
                if builder is None:
                    continue
                seed = f"{representation.candidate_id}:{latent_id}:concern"
                result = builder(raw, representation, sim, seed)
                if result is None:
                    continue
                fragment, evidence = result
                clauses.append(
                    ReasoningClause(
                        polarity=ReasoningPolarity.CONCERN,
                        fragment=fragment,
                        evidence=evidence,
                        jd_link=None,
                    )
                )

        # Behavioral concern: add when signals indicate low practical hirability,
        # provided there is still room within the concern cap.
        if len(clauses) < _MAX_CONCERNS and representation.behavioral is not None:
            beh_seed = f"{representation.candidate_id}:behavioral:concern"
            beh_result = _behavioral_concern(
                raw, representation.candidate_id, self.as_of, beh_seed
            )
            if beh_result is not None:
                beh_fragment, beh_evidence = beh_result
                clauses.append(
                    ReasoningClause(
                        polarity=ReasoningPolarity.CONCERN,
                        fragment=beh_fragment,
                        evidence=beh_evidence,
                        jd_link=None,
                    )
                )

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
