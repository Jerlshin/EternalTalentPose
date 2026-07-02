from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Final, Literal, final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.enums import EligibilityCode, ReasoningPolarity, ScoreComponent
from redstack.domain.ids import CandidateId
from redstack.domain.provenance import EvidenceRef

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)

RankBand = Literal["top", "mid", "tail"]

#: Bands that must carry at least one STRENGTH clause (§J.2).
_STRENGTH_REQUIRED_BANDS: Final[frozenset[RankBand]] = frozenset({"top", "mid"})


def _stable_index(seed: str, n: int) -> int:
    """Domain-local twin of ``engines.reasoning._pick_index`` (§2 / §J.5).

    Re-implemented here rather than imported: ``domain`` may not depend on
    ``engines`` (Hexagonal Isolation / import-linter contract #2), so the
    SHA-256-remainder mechanism that is the sole source of textual variation
    is duplicated as a small local primitive instead of shared across the
    boundary.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


#: Rank-band tone qualifiers -- rank<=10 gets decisive, "this is a clear
#: yes" language; the rest gets objective, descriptive language. This is a
#: substantive claim about fit (varies with rank), not narrative throat-
#: clearing about how the paragraph is organized -- unlike the old framing
#: clauses ("Starting from...", "The arc of this career...") it replaces,
#: which announced a presentation choice rather than asserting a fact.
_TOP_BAND_QUALIFIERS: Final[tuple[str, ...]] = (
    "Exceptional fit",
    "Direct match",
    "Top-tier candidate",
    "Standout case",
    "Elite alignment",
    "A clear top pick",
)
_REST_BAND_QUALIFIERS: Final[tuple[str, ...]] = (
    "Solid fit",
    "Workable match",
    "Good alignment",
    "Reasonable case",
    "Passable fit",
    "A fair match",
)

#: Contrastive transition opening the (optional) second, concern sentence --
#: "exactly one contrastive transition" per the strict-sentence-count
#: contract. Picked deterministically (never randomness/wall-clock) so
#: concern sentences don't all share one bolted-on skeleton (§J.5).
_CONCERN_LEAD_INS: Final[tuple[str, ...]] = (
    "However,",
    "That said,",
    "Even so,",
    "Still,",
    "On the downside,",
    "Worth flagging:",
    "The one caveat:",
    "Set against that,",
    "The main friction point:",
    "Weighed against that,",
)

#: Terms that make a lead-in read as an echo when the concern fragment
#: already contains them ("The main friction point: the start-date math is
#: the friction point: ..."). A lead-in is dropped from the pool for a given
#: fragment when any of its echo terms appears in that fragment
#: (case-insensitive); the pick over the filtered pool stays deterministic,
#: and the transition is guaranteed never to repeat the clause's own wording.
_LEAD_IN_ECHO_TERMS: Final[Mapping[str, tuple[str, ...]]] = {
    "However,": ("however",),
    "That said,": ("that said",),
    "Even so,": ("even so",),
    "Still,": ("still",),
    "On the downside,": ("downside",),
    "Worth flagging:": ("flagging", "worth"),
    "The one caveat:": ("caveat",),
    "Set against that,": ("against",),
    "The main friction point:": ("friction",),
    "Weighed against that,": ("against", "weigh"),
}


@final
class ReasoningClause(BaseModel):
    """One evidence-backed reasoning fragment connected to a JD requirement."""

    model_config = _STRICT

    polarity: ReasoningPolarity
    fragment: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    jd_link: EligibilityCode | ScoreComponent | None


@final
class CandidateReasoning(BaseModel):
    """The rendered, deterministic explanation for one ranked candidate."""

    model_config = _STRICT

    candidate_id: CandidateId
    clauses: tuple[ReasoningClause, ...] = Field(min_length=1)
    rendered: str = Field(min_length=1)
    rank_band: RankBand
    #: Disambiguation salt (default 0). Two different candidates can
    #: legitimately land on the same single strength fragment and the same
    #: candidate_id-seeded qualifier (small pools + shared facts, e.g. the
    #: same employer); Stage-4 rejects any two identical top-100 renderings
    #: outright. ``ReasoningEngine.explain`` detects that batch-level
    #: collision (invisible to any single candidate's pure render) and
    #: re-assembles the later one with an incremented salt until its
    #: rendering is unique -- still fully deterministic, just no longer a
    #: function of ``(clauses, rank_band, candidate_id)`` alone.
    tie_break_salt: int = Field(default=0, ge=0)

    @staticmethod
    def _trim(fragment: str) -> str:
        """Strip a clause fragment's own terminal punctuation.

        Builders author each fragment as a free-standing mid-sentence clause
        (often itself ending in ``.``); stripping that before appending the
        sentence's own closing period is what keeps the final rendering from
        showing a stray ``"."`` mid-sentence or ``".."`` (§J.5).
        """
        return fragment.strip().rstrip(" .;,")

    @staticmethod
    def _capitalize(sentence: str) -> str:
        if not sentence:
            return sentence
        return sentence[0].upper() + sentence[1:]

    @staticmethod
    def render(
        clauses: tuple[ReasoningClause, ...],
        rank_band: RankBand,
        candidate_id: CandidateId,
        tie_break_salt: int = 0,
    ) -> str:
        """Deterministically render a clause set into a 1-2 sentence justification.

        Exactly one STRENGTH fragment leads, opened with a rank-band
        qualifier ("Exceptional fit" for rank<=10 / "Solid fit" for the
        rest) instead of a narrative framing clause -- callers upstream cap
        clause selection to a single strength (``engines.reasoning``'s
        ``_MAX_STRENGTHS=1``), so there is never a multi-clause chain to
        weave together here. At most one CONCERN fragment follows as its
        own sentence, joined by exactly one contrastive transition
        ("However,", ...) -- never a comma-chain of several concerns. No
        concern -> exactly one sentence; a concern present -> exactly two.

        The qualifier and transition are seeded by ``candidate_id`` rather
        than the fragment text: two different candidates who share a fact
        (e.g. the same employer) can legitimately land on the same
        evidence template, and seeding off the (then-identical) fragment
        text would make the qualifier collide too, compounding rather than
        breaking the tie -- exactly what produced byte-identical top-100
        rows before this fix (Stage-4 "identical reasoning" is a hard
        rejection). ``candidate_id`` is unique by construction, so this is
        still a pure, deterministic function of the clause set's owner --
        no randomness, no wall clock (§J.5) -- just not solely of its text.
        """
        if not clauses:
            return ""
        strengths = tuple(
            c for c in clauses if c.polarity is not ReasoningPolarity.CONCERN
        )
        concerns = tuple(
            CandidateReasoning._trim(c.fragment)
            for c in clauses
            if c.polarity is ReasoningPolarity.CONCERN
        )
        if not strengths:
            # Defensive only: upstream always attaches a strength clause for
            # a ranked candidate (see ReasoningEngine.reason_for's fallback).
            if not concerns:
                return ""
            return CandidateReasoning._capitalize(concerns[0]) + "."

        fragment = CandidateReasoning._trim(strengths[0].fragment)
        qualifier_pool = (
            _TOP_BAND_QUALIFIERS if rank_band == "top" else _REST_BAND_QUALIFIERS
        )
        qualifier = qualifier_pool[
            _stable_index(
                f"{candidate_id}|qualifier|{tie_break_salt}", len(qualifier_pool)
            )
        ]
        strength_sentence = f"{qualifier} -- {fragment}."

        if not concerns:
            return strength_sentence

        fragment_lower = concerns[0].lower()
        transition_pool = tuple(
            lead
            for lead in _CONCERN_LEAD_INS
            if not any(term in fragment_lower for term in _LEAD_IN_ECHO_TERMS[lead])
        )
        if not transition_pool:
            transition_pool = _CONCERN_LEAD_INS
        transition = transition_pool[
            _stable_index(
                f"{candidate_id}|transition|{tie_break_salt}", len(transition_pool)
            )
        ]
        concern_sentence = (
            CandidateReasoning._capitalize(f"{transition} {concerns[0]}") + "."
        )
        return f"{strength_sentence} {concern_sentence}"

    @classmethod
    def assemble(
        cls,
        *,
        candidate_id: CandidateId,
        clauses: tuple[ReasoningClause, ...],
        rank_band: RankBand,
        tie_break_salt: int = 0,
    ) -> CandidateReasoning:
        """Build a reasoning object with ``rendered`` derived from ``clauses``."""
        return cls(
            candidate_id=candidate_id,
            clauses=clauses,
            rendered=cls.render(clauses, rank_band, candidate_id, tie_break_salt),
            rank_band=rank_band,
            tie_break_salt=tie_break_salt,
        )

    @model_validator(mode="after")
    def _stage4_invariants(self) -> CandidateReasoning:
        # §J.5 — rendered is the deterministic render of the ordered clauses.
        if self.rendered != self.render(
            self.clauses, self.rank_band, self.candidate_id, self.tie_break_salt
        ):
            raise ValueError("rendered must equal the deterministic render of clauses")
        # §J.3 — at least one clause connects to a JD requirement.
        if not any(c.jd_link is not None for c in self.clauses):
            raise ValueError("at least one clause must carry a non-null jd_link")
        # §J.2 — top/mid bands require at least one STRENGTH clause.
        if self.rank_band in _STRENGTH_REQUIRED_BANDS and not any(
            c.polarity is ReasoningPolarity.STRENGTH for c in self.clauses
        ):
            raise ValueError(
                f"rank_band '{self.rank_band}' requires at least one STRENGTH clause"
            )
        return self


__all__: tuple[str, ...] = ("CandidateReasoning", "RankBand", "ReasoningClause")
