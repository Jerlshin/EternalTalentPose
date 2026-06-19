

from __future__ import annotations

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
_CONCERN_PREFIX: Final[str] = "Concerns: "


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
    def render(clauses: tuple[ReasoningClause, ...]) -> str:
        """Deterministically render an ordered clause set into ≤2 sentences.

        Positive (STRENGTH/CONTEXT) fragments fold into one sentence; CONCERN
        fragments fold into a second. Purely a function of clause order and
        content — never randomness or templated name insertion (§J.5).
        """
        positives = [
            c.fragment
            for c in clauses
            if c.polarity is not ReasoningPolarity.CONCERN
        ]
        concerns = [
            c.fragment for c in clauses if c.polarity is ReasoningPolarity.CONCERN
        ]
        sentences: list[str] = []
        if positives:
            sentences.append("; ".join(positives) + ".")
        if concerns:
            sentences.append(_CONCERN_PREFIX + "; ".join(concerns) + ".")
        return " ".join(sentences)

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
            raise ValueError(
                "rendered must equal the deterministic render of clauses"
            )
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