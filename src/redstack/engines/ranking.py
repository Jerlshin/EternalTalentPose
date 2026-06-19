

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pydantic import BaseModel, ConfigDict

from redstack.domain.errors import RankingInvariantError
from redstack.domain.ids import CandidateId, Score
from redstack.domain.ranking import RankedCandidate, Ranking
from redstack.domain.scoring import ScoredCandidate

_DEFAULT_SIZE: int = 100


@final
class RankingEngine(BaseModel):
    """Stateless, pure ranking engine."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    size: int = _DEFAULT_SIZE

    # ------------------------------------------------------------------ public
    def rank(self, scored: Sequence[ScoredCandidate]) -> Ranking:
        """Sort, partition floored to the tail, take top ``size``, build ``Ranking``."""
        if len(scored) < self.size:
            raise RankingInvariantError(
                f"need >= {self.size} scored candidates, got {len(scored)}"
            )

        ordered_scored = sorted(scored, key=self._sort_key)
        top = ordered_scored[: self.size]

        ranked: list[RankedCandidate] = [
            RankedCandidate(
                candidate_id=candidate.candidate_id,
                rank=index + 1,
                score=Score(float(candidate.final_score)),
                scored=candidate,
                reasoning=None,
            )
            for index, candidate in enumerate(top)
        ]
        honeypot_count = sum(
            1 for c in top if not c.breakdown.integrity_gate.passed
        )
        return Ranking(
            ordered=tuple(ranked),
            size=self.size,
            honeypot_count=honeypot_count,
        )

    # --------------------------------------------------------------- internals
    @staticmethod
    def _is_floored(candidate: ScoredCandidate) -> bool:
        gates = candidate.breakdown
        return not (gates.integrity_gate.passed and gates.eligibility_gate.passed)

    def _sort_key(self, candidate: ScoredCandidate) -> tuple[int, float, CandidateId]:
        # (non-floored first, score descending, candidate_id ascending).
        floored_rank = 1 if self._is_floored(candidate) else 0
        return (floored_rank, -float(candidate.final_score), candidate.candidate_id)


__all__: tuple[str, ...] = ("RankingEngine",)