"""``RankingEngine`` — deterministic sort + top-K extraction (``engines/ranking.py``).

Owner layer: engines. Allowed imports: ``domain``. Forbidden: ``adapters``,
``pipelines``, ``observability``, sibling engines, clock, online RNG, ports.

Logical→physical (Repo §9): the ``CandidateRankingEngine``. A single, total,
deterministic order produces a spec-valid ``Ranking``:

- floored candidates (failed integrity OR eligibility gate) partition to the
  filler **tail**; non-floored candidates fill ranks 1..size first (Online R6),
  keeping the top-100 honeypot rate ≈ 0 by construction, not by special-casing;
- within each partition, sort by ``(−final_score, candidate_id)`` — score
  descending, then ``candidate_id`` **alphanumeric ascending** as the rigid
  tie-break that protects the submission from non-determinism;
- assign ranks ``1..size`` and hand the slice to the ``Ranking`` factory, which
  re-asserts the six validator invariants (count / unique-rank / id-pattern /
  monotonicity / tie-break / sortedness) and raises ``RankingInvariantError`` on
  any violation.

Pure, single-threaded stable sort; identical inputs ⇒ identical ``Ranking``.
"""

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