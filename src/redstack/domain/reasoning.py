from __future__ import annotations

import hashlib
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


#: Component groups that license a topic-led opening framing (§ below);
#: a component absent from all three (e.g. EDUCATION_FIT) simply renders
#: unframed -- framing is a bonus narrative device, never a requirement.
_CAREER_LED_COMPONENTS: Final[frozenset[ScoreComponent]] = frozenset(
    {ScoreComponent.CAREER_FIT, ScoreComponent.EXPERIENCE_FIT}
)
_COMPETENCY_LED_COMPONENTS: Final[frozenset[ScoreComponent]] = frozenset(
    {ScoreComponent.SKILL_MATCH, ScoreComponent.CREDIBILITY}
)
_SEMANTIC_LED_COMPONENTS: Final[frozenset[ScoreComponent]] = frozenset(
    {ScoreComponent.SEMANTIC_FIT, ScoreComponent.ARCHETYPE_FIT}
)

#: One of five seed-selected opening "topologies" -- which positive clause
#: leads, and under what framing -- so two candidates dominated by the same
#: component still open their paragraph differently (§J.5 structural variety).
_LEAD_CAREER: Final[int] = 0
_LEAD_COMPETENCY: Final[int] = 1
_LEAD_SEMANTIC: Final[int] = 2
_LEAD_CUMULATIVE: Final[int] = 3
_LEAD_DIRECT: Final[int] = 4
_LEAD_STYLE_COUNT: Final[int] = 5

_CAREER_FRAMINGS: Final[tuple[str, ...]] = (
    "looking at the career arc first,",
    "starting from where this career has actually gone,",
    "leading with trajectory rather than the skills list,",
    "the career history is the natural place to start here:",
    "viewed through career progression first,",
    "putting the career narrative up front,",
    "the arc of this career is worth framing before anything else:",
    "starting from career shape rather than any single line item,",
)
_COMPETENCY_FRAMINGS: Final[tuple[str, ...]] = (
    "leading with the concrete competencies on file,",
    "starting from hard skills rather than narrative arc,",
    "the corroborated competencies make the strongest opening case:",
    "putting the verified skill set first,",
    "competency evidence is the place to start:",
    "a skills-first read on this profile starts here:",
    "the most concrete starting point is what's actually been verified:",
    "leading with substance over story,",
)
_SEMANTIC_FRAMINGS: Final[tuple[str, ...]] = (
    "starting from how this profile's own language tracks the JD,",
    "leading with the semantic read rather than the keyword list,",
    "vector alignment is the most distinctive signal, so start there:",
    "the embedding-level read frames this best:",
    "before anything else, how this profile is actually written is telling:",
    "semantic fit is the most interesting starting point here:",
    "starting from meaning rather than vocabulary,",
    "the way this profile is phrased is worth leading with:",
)

#: Per-junction connectors gluing consecutive positive clauses into one
#: flowing sentence -- the direct replacement for the old fixed ``"; "``
#: join. Two pools (flowing / direct) give the unframed lead styles their
#: own register instead of reusing the framed styles' connective tissue.
_CONNECTORS_FLOWING: Final[tuple[str, ...]] = (
    "and just as tellingly,",
    "layered on top of that,",
    "compounding the case,",
    "rounding out the picture,",
    "adding to that base,",
    "and, beyond that,",
    "what reinforces this further is that",
    "just as relevant,",
    "and, building on that,",
    "to add another data point,",
    "and, tellingly,",
    "stacking another signal on top of that,",
    "further bolstering the case,",
    "and, rounding things out,",
    "which pairs with the fact that",
)
_CONNECTORS_DIRECT: Final[tuple[str, ...]] = (
    "plus,",
    "also,",
    "alongside that,",
    "and, equally,",
    "worth noting too,",
    "and, separately,",
    "add to that,",
    "and, just as relevant,",
    "not only that --",
    "and, no less important,",
    "in the same vein,",
    "and, on top of that,",
)
#: With 3 positive clauses, gluing all of them into a single comma chain
#: reads as a run-on; the third clause instead opens a short second
#: sentence introduced by one of these (capitalized) transitions.
_SECOND_SENTENCE_OPENERS: Final[tuple[str, ...]] = (
    "Also worth weighing:",
    "On top of that,",
    "It's also worth noting that",
    "Adding further support,",
    "There's a third data point here too:",
    "Rounding out the case,",
    "One more thing counts in this profile's favor:",
    "Further still,",
)
#: Above this many positive clauses, ``_compose_positive_case`` breaks into
#: two sentences (see ``_SECOND_SENTENCE_OPENERS``) instead of one long chain.
_MAX_POSITIVES_PER_SENTENCE: Final[int] = 2

#: Concern-sentence lead-ins; picked deterministically from the clause
#: fragments themselves (never randomness/wall-clock) so concern sentences
#: don't all share the same bolted-on "Concerns: " skeleton (§J.5).
_CONCERN_LEAD_INS: Final[tuple[str, ...]] = (
    "That said,",
    "On the downside,",
    "Worth flagging:",
    "The main caveat:",
    "Set against that,",
    "Less convincingly,",
    "On the other side of the ledger,",
    "The case isn't without friction:",
    "A counterpoint worth weighing:",
    "Pulling in the other direction,",
    "Not every signal points the same way here:",
    "The flip side:",
    "There's a real caveat, though:",
    "Tempering that read somewhat,",
    "It isn't all upside, however:",
    "Weighed against the above,",
    "Cutting the other way,",
    "To be even-handed about it,",
    "One thing worth flagging against the above:",
    "A point of friction worth naming:",
)
#: Joiners between *multiple* concern fragments within that one sentence.
_CONCERN_JOINERS: Final[tuple[str, ...]] = (
    "and",
    "and, separately,",
    "and, just as relevant,",
    "alongside that,",
    "and, adding to that,",
)
#: Mid-sentence contrastive connectors used only when a single concern is
#: folded directly onto the positive case rather than given its own
#: sentence (the ``_LEAD_DIRECT``-adjacent, most "cohesive paragraph" shape
#: the rewrite was asked for) -- each grammatically takes a full clause.
_CONCERN_FOLD_CONNECTORS: Final[tuple[str, ...]] = (
    "though",
    "even though",
    "even as",
    "while",
    "notwithstanding that",
    "even granting that",
    "all while",
    "though it bears mentioning that",
)
#: A folded concern is only grafted onto a *compact* positive case (<= 2
#: lead clauses); with 3, the combined sentence runs long enough that a
#: separate, clearly-delimited concern sentence reads better.
_MAX_POSITIVES_FOR_FOLD: Final[int] = 2


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

    @staticmethod
    def _trim(fragment: str) -> str:
        """Strip a clause fragment's own terminal punctuation.

        Builders author each fragment as a free-standing mid-sentence clause
        (often itself ending in ``.``); stripping that before weaving several
        clauses together via prose connectives is what keeps the final
        rendering from showing a stray ``"."`` mid-sentence or ``".."`` once
        the one closing period for the whole sentence is appended (§J.5).
        """
        return fragment.strip().rstrip(" .;,")

    @staticmethod
    def _capitalize(sentence: str) -> str:
        if not sentence:
            return sentence
        return sentence[0].upper() + sentence[1:]

    @staticmethod
    def _lead_style(seed_material: str) -> int:
        return _stable_index(seed_material + "|lead-style", _LEAD_STYLE_COUNT)

    @staticmethod
    def _find_lead(
        positives: tuple[ReasoningClause, ...], components: frozenset[ScoreComponent]
    ) -> int | None:
        """Index of the first positive clause whose ``jd_link`` is in ``components``.

        Used only to choose which fragment narratively *opens* the rendered
        paragraph; ``clauses`` itself (already ordered by weighted
        contribution upstream) is never reordered -- only this local,
        presentation-only fragment list is.
        """
        for index, clause in enumerate(positives):
            link = clause.jd_link
            if isinstance(link, ScoreComponent) and link in components:
                return index
        return None

    @staticmethod
    def _compose_positive_case(
        positives: tuple[ReasoningClause, ...], seed_material: str, lead_style: int
    ) -> str:
        """Weave the (<=3) positive fragments into one flowing, framed clause.

        Replaces the old fixed ``"; "`` join: ``lead_style`` picks whether a
        topic framing opens the paragraph (and which topic), and each
        clause-to-clause junction independently draws its own connector, so
        two candidates dominated by the same component still open and join
        their case differently (§J.5).
        """
        fragments = [CandidateReasoning._trim(c.fragment) for c in positives]
        lead_index: int | None
        framing_pool: tuple[str, ...]
        if lead_style == _LEAD_CAREER:
            lead_index = CandidateReasoning._find_lead(
                positives, _CAREER_LED_COMPONENTS
            )
            framing_pool = _CAREER_FRAMINGS
        elif lead_style == _LEAD_COMPETENCY:
            lead_index = CandidateReasoning._find_lead(
                positives, _COMPETENCY_LED_COMPONENTS
            )
            framing_pool = _COMPETENCY_FRAMINGS
        elif lead_style == _LEAD_SEMANTIC:
            lead_index = CandidateReasoning._find_lead(
                positives, _SEMANTIC_LED_COMPONENTS
            )
            framing_pool = _SEMANTIC_FRAMINGS
        else:
            # _LEAD_CUMULATIVE / _LEAD_DIRECT: no topic framing, direct opening.
            lead_index, framing_pool = None, ()

        order = list(range(len(fragments)))
        if lead_index is not None and lead_index != 0:
            order.insert(0, order.pop(lead_index))
        ordered = [fragments[i] for i in order]

        if lead_index is not None:
            framing = framing_pool[
                _stable_index(seed_material + "|framing", len(framing_pool))
            ]
            opening = f"{framing} {ordered[0]}"
        else:
            opening = ordered[0]

        connectors = (
            _CONNECTORS_DIRECT if lead_style == _LEAD_DIRECT else _CONNECTORS_FLOWING
        )

        def _joined(fragments_to_join: list[str], *, offset: int) -> str:
            parts = [fragments_to_join[0]]
            for position, fragment in enumerate(fragments_to_join[1:]):
                junction_seed = f"{seed_material}|junction:{offset + position}"
                connector = connectors[_stable_index(junction_seed, len(connectors))]
                parts.append(f"{connector} {fragment}")
            return ", ".join(parts)

        if len(ordered) <= _MAX_POSITIVES_PER_SENTENCE:
            return _joined([opening, *ordered[1:]], offset=0)

        # 3 positive clauses: a single comma chain over all three reads as a
        # run-on, so the last clause opens its own short second sentence.
        first_sentence = _joined([opening, *ordered[1:-1]], offset=0)
        opener_pool_size = len(_SECOND_SENTENCE_OPENERS)
        opener = _SECOND_SENTENCE_OPENERS[
            _stable_index(seed_material + "|second-sentence", opener_pool_size)
        ]
        second_sentence = f"{opener} {ordered[-1]}"
        return f"{first_sentence}. {second_sentence}"

    @staticmethod
    def _join_concerns(concerns: list[str], concern_seed: str) -> str:
        if len(concerns) == 1:
            return concerns[0]
        parts = [concerns[0]]
        for position, fragment in enumerate(concerns[1:]):
            joiner_seed = f"{concern_seed}|concern-junction:{position}"
            joiner = _CONCERN_JOINERS[_stable_index(joiner_seed, len(_CONCERN_JOINERS))]
            parts.append(f"{joiner} {fragment}")
        return ", ".join(parts)

    @staticmethod
    def _render_concern_sentence(concerns: list[str]) -> str:
        if not concerns:
            return ""
        concern_seed = "||".join(concerns)
        lead_in = _CONCERN_LEAD_INS[_stable_index(concern_seed, len(_CONCERN_LEAD_INS))]
        body = CandidateReasoning._join_concerns(concerns, concern_seed)
        return CandidateReasoning._capitalize(f"{lead_in} {body}") + "."

    @staticmethod
    def render(clauses: tuple[ReasoningClause, ...]) -> str:
        """Deterministically render an ordered clause set into a cohesive paragraph.

        Positive (STRENGTH/CONTEXT) fragments are woven into one sentence via
        a seed-selected opening framing plus per-junction connectors -- never
        a fixed ``"; "`` join, so the structure itself (not just the wording)
        varies candidate to candidate. A single concern is sometimes grafted
        onto that same sentence with a contrastive connector instead of
        getting its own; otherwise concerns form a second sentence introduced
        by a lead-in drawn from a wide, varied pool. Every selection is a pure
        SHA-256-derived function of the clause fragments themselves -- no
        randomness, no wall clock, no name/index templating (§J.5).
        """
        if not clauses:
            return ""
        positives = tuple(
            c for c in clauses if c.polarity is not ReasoningPolarity.CONCERN
        )
        concerns = [
            CandidateReasoning._trim(c.fragment)
            for c in clauses
            if c.polarity is ReasoningPolarity.CONCERN
        ]
        if not positives:
            return CandidateReasoning._render_concern_sentence(concerns)

        seed_material = "||".join(f"{c.polarity.value}:{c.fragment}" for c in clauses)
        lead_style = CandidateReasoning._lead_style(seed_material)
        body = CandidateReasoning._compose_positive_case(
            positives, seed_material, lead_style
        )

        if not concerns:
            return CandidateReasoning._capitalize(body) + "."

        should_fold = len(concerns) == 1 and len(positives) <= _MAX_POSITIVES_FOR_FOLD
        if should_fold and _stable_index(seed_material + "|fold-coin", 2) == 1:
            fold_pool_size = len(_CONCERN_FOLD_CONNECTORS)
            connector = _CONCERN_FOLD_CONNECTORS[
                _stable_index(seed_material + "|fold-connector", fold_pool_size)
            ]
            combined = f"{body}, {connector} {concerns[0]}"
            return CandidateReasoning._capitalize(combined) + "."

        strength_sentence = CandidateReasoning._capitalize(body) + "."
        concern_sentence = CandidateReasoning._render_concern_sentence(concerns)
        return f"{strength_sentence} {concern_sentence}"

    @classmethod
    def assemble(
        cls,
        *,
        candidate_id: CandidateId,
        clauses: tuple[ReasoningClause, ...],
        rank_band: RankBand,
    ) -> CandidateReasoning:
        """Build a reasoning object with ``rendered`` derived from ``clauses``."""
        return cls(
            candidate_id=candidate_id,
            clauses=clauses,
            rendered=cls.render(clauses),
            rank_band=rank_band,
        )

    @model_validator(mode="after")
    def _stage4_invariants(self) -> CandidateReasoning:
        # §J.5 — rendered is the deterministic render of the ordered clauses.
        if self.rendered != self.render(self.clauses):
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
