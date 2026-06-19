

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Final, final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redstack.domain.errors import RankingInvariantError
from redstack.domain.ids import CANDIDATE_ID_PATTERN, CandidateId, Score
from redstack.domain.reasoning import CandidateReasoning
from redstack.domain.scoring import ScoredCandidate

_STRICT = ConfigDict(
    frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
)

_DEFAULT_SIZE: Final[int] = 100
_ID_RE: Final[re.Pattern[str]] = re.compile(CANDIDATE_ID_PATTERN)


@final
class RankedCandidate(BaseModel):
    """One placed candidate: rank, score, the scored record, and reasoning."""

    model_config = _STRICT

    candidate_id: CandidateId = Field(pattern=CANDIDATE_ID_PATTERN)
    rank: int = Field(ge=1)
    score: Score = Field(allow_inf_nan=False)
    scored: ScoredCandidate
    reasoning: CandidateReasoning | None

    @model_validator(mode="after")
    def _consistency(self) -> RankedCandidate:
        if self.candidate_id != self.scored.candidate_id:
            raise ValueError(
                "RankedCandidate.candidate_id must equal scored.candidate_id"
            )
        if self.score != self.scored.final_score:
            raise ValueError("RankedCandidate.score must equal scored.final_score")
        if (
            self.reasoning is not None
            and self.reasoning.candidate_id != self.candidate_id
        ):
            raise ValueError("reasoning.candidate_id must equal candidate_id")
        return self

    def with_reasoning(self, reasoning: CandidateReasoning) -> RankedCandidate:
        """Return a copy with reasoning attached (R7)."""
        if reasoning.candidate_id != self.candidate_id:
            raise RankingInvariantError(
                "reasoning.candidate_id must equal candidate_id"
            )
        return self.model_copy(update={"reasoning": reasoning})


@final
class Ranking(BaseModel):
    """The final ordered ranking; all six validator rules hold by construction."""

    model_config = _STRICT

    ordered: tuple[RankedCandidate, ...]
    size: int = Field(default=_DEFAULT_SIZE, ge=1)
    honeypot_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validator_rules(self) -> Ranking:
        ordered = self.ordered
        # Rule 1 — exact size.
        if len(ordered) != self.size:
            raise RankingInvariantError(
                f"ranking must contain exactly {self.size} rows, got {len(ordered)}"
            )
        # Rule 2 — ranks are exactly 1..size, each once, in order.
        for position, rc in enumerate(ordered):
            if rc.rank != position + 1:
                raise RankingInvariantError(
                    "ranks must be the dense sequence 1..size in order"
                )
        # Rule 3 — candidate ids unique and well-formed.
        seen: set[str] = set()
        for rc in ordered:
            if _ID_RE.fullmatch(rc.candidate_id) is None:
                raise RankingInvariantError(f"malformed candidate_id: {rc.candidate_id}")
            if rc.candidate_id in seen:
                raise RankingInvariantError(f"duplicate candidate_id: {rc.candidate_id}")
            seen.add(rc.candidate_id)
        # Rules 4-6 — non-increasing score, tie-break by id, full sort order.
        for prev, curr in zip(ordered, ordered[1:]):
            if curr.score > prev.score:
                raise RankingInvariantError("scores must be non-increasing by rank")
            if curr.score == prev.score and not (prev.candidate_id < curr.candidate_id):
                raise RankingInvariantError(
                    "tie-break violated: equal scores must order by ascending id"
                )
        if self.honeypot_count > self.size:
            raise RankingInvariantError("honeypot_count cannot exceed size")
        return self

    @classmethod
    def from_scored(
        cls,
        scored: Iterable[ScoredCandidate],
        *,
        size: int = _DEFAULT_SIZE,
        honeypot_count: int = 0,
    ) -> Ranking:
        """Sort by ``(-final_score, candidate_id)``, take top ``size``, rank them.

        The deterministic ordering law lives here so every producer shares it.
        """
        pool = sorted(scored, key=lambda sc: (-sc.final_score, sc.candidate_id))
        if len(pool) < size:
            raise RankingInvariantError(
                f"need at least {size} scored candidates, got {len(pool)}"
            )
        top = pool[:size]
        ordered = tuple(
            RankedCandidate(
                candidate_id=sc.candidate_id,
                rank=position + 1,
                score=sc.final_score,
                scored=sc,
                reasoning=None,
            )
            for position, sc in enumerate(top)
        )
        return cls(ordered=ordered, size=size, honeypot_count=honeypot_count)

    def with_reasoning(
        self, mapping: Mapping[CandidateId, CandidateReasoning]
    ) -> Ranking:
        """Attach reasoning by candidate id where present; re-assert invariants."""
        ordered = tuple(
            rc.with_reasoning(mapping[rc.candidate_id])
            if rc.candidate_id in mapping
            else rc
            for rc in self.ordered
        )
        return self.model_copy(update={"ordered": ordered})


__all__: tuple[str, ...] = ("Ranking", "RankedCandidate")