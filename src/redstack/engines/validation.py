"""``ValidationEngine`` — submission + Stage-4 checks as code (``engines/validation.py``).

Owner layer: engines. Allowed imports: ``domain``. Forbidden: ``adapters``,
``pipelines``, ``observability``, sibling engines, ports, clock, online RNG.

Logical→physical (Repo §9): the ``ValidationEngine`` (the code half of
``SubmissionGenerationEngine``). It mirrors the organizer's
``validate_submission.py`` structural rules and adds the six Stage-4 reasoning
checks, as defence-in-depth at R9 — the ``Ranking`` factory already makes most
structural violations unrepresentable, but a submission is never written unless
this report is valid.

Report contract (Domain §K): ``is_valid == (no finding with severity == HARD)``;
``checks_run`` records every evaluated ``ValidationCode``. Structural failures and
the hard reasoning failures (empty / identical / templated / hallucinated) are
``HARD``; rank-band/reasoning inconsistency is ``SOFT``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Final, final

from pydantic import BaseModel, ConfigDict

from redstack.domain.enums import ReasoningPolarity, Severity, ValidationCode
from redstack.domain.ranking import RankedCandidate, Ranking
from redstack.domain.validation import ValidationFinding, ValidationReport

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^CAND_[0-9]{7}$")
# Normalization for the templated-reasoning skeleton: collapse ids and numbers.
_NUM_TOKEN: Final[re.Pattern[str]] = re.compile(r"\d+(?:\.\d+)?")
_ID_TOKEN: Final[re.Pattern[str]] = re.compile(r"CAND_[0-9]{7}")
_TEMPLATED_FRACTION: Final[float] = 0.70

_ALL_CODES: Final[frozenset[ValidationCode]] = frozenset(ValidationCode)


@final
class ValidationEngine(BaseModel):
    """Stateless, pure validator over a (reasoned) ``Ranking``."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    expected_size: int = 100

    # ------------------------------------------------------------------ public
    def validate(self, ranking: Ranking) -> ValidationReport:
        """Run all structural + Stage-4 checks; assemble the ``ValidationReport``."""
        findings: list[ValidationFinding] = []
        ordered = ranking.ordered

        findings.extend(self._row_count(ordered))
        findings.extend(self._ranks(ordered))
        findings.extend(self._ids(ordered))
        findings.extend(self._scores(ordered))
        findings.extend(self._reasoning(ordered))

        is_valid = not any(f.severity is Severity.HARD for f in findings)
        return ValidationReport(
            findings=tuple(findings),
            is_valid=is_valid,
            checks_run=_ALL_CODES,
        )

    # ----------------------------------------------------------- structural
    def _row_count(
        self, ordered: tuple[RankedCandidate, ...]
    ) -> tuple[ValidationFinding, ...]:
        if len(ordered) == self.expected_size:
            return ()
        return (
            ValidationFinding(
                code=ValidationCode.WRONG_ROW_COUNT,
                severity=Severity.HARD,
                message=f"expected {self.expected_size} rows, got {len(ordered)}",
                location=None,
            ),
        )

    def _ranks(
        self, ordered: tuple[RankedCandidate, ...]
    ) -> tuple[ValidationFinding, ...]:
        out: list[ValidationFinding] = []
        ranks = [c.rank for c in ordered]
        for candidate in ordered:
            if candidate.rank < 1 or candidate.rank > self.expected_size:
                out.append(
                    ValidationFinding(
                        code=ValidationCode.RANK_OUT_OF_RANGE,
                        severity=Severity.HARD,
                        message=f"rank {candidate.rank} out of [1, {self.expected_size}]",
                        location=candidate.candidate_id,
                    )
                )
        counts = Counter(ranks)
        for rank, freq in sorted(counts.items()):
            if freq > 1:
                out.append(
                    ValidationFinding(
                        code=ValidationCode.DUPLICATE_RANK,
                        severity=Severity.HARD,
                        message=f"rank {rank} appears {freq} times",
                        location=None,
                    )
                )
        expected = set(range(1, self.expected_size + 1))
        missing = expected - set(ranks)
        if missing:
            out.append(
                ValidationFinding(
                    code=ValidationCode.MISSING_RANK,
                    severity=Severity.HARD,
                    message=f"missing ranks: {sorted(missing)[:10]}",
                    location=None,
                )
            )
        return tuple(out)

    def _ids(
        self, ordered: tuple[RankedCandidate, ...]
    ) -> tuple[ValidationFinding, ...]:
        out: list[ValidationFinding] = []
        seen: Counter[str] = Counter(c.candidate_id for c in ordered)
        for candidate in ordered:
            if _ID_PATTERN.match(candidate.candidate_id) is None:
                out.append(
                    ValidationFinding(
                        code=ValidationCode.BAD_ID_FORMAT,
                        severity=Severity.HARD,
                        message=f"id {candidate.candidate_id!r} fails ^CAND_[0-9]{{7}}$",
                        location=candidate.candidate_id,
                    )
                )
        for cid, freq in sorted(seen.items()):
            if freq > 1:
                out.append(
                    ValidationFinding(
                        code=ValidationCode.DUPLICATE_ID,
                        severity=Severity.HARD,
                        message=f"id {cid} appears {freq} times",
                        location=cid,
                    )
                )
        return tuple(out)

    def _scores(
        self, ordered: tuple[RankedCandidate, ...]
    ) -> tuple[ValidationFinding, ...]:
        out: list[ValidationFinding] = []
        for candidate in ordered:
            value = candidate.score
            if not isinstance(value, float) or not math.isfinite(value):
                out.append(
                    ValidationFinding(
                        code=ValidationCode.SCORE_NOT_FLOAT,
                        severity=Severity.HARD,
                        message=f"score {value!r} is not a finite float",
                        location=candidate.candidate_id,
                    )
                )
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.score > earlier.score:
                out.append(
                    ValidationFinding(
                        code=ValidationCode.SCORE_INCREASING,
                        severity=Severity.HARD,
                        message=(
                            f"score rises at rank {later.rank} "
                            f"({earlier.score} -> {later.score})"
                        ),
                        location=later.candidate_id,
                    )
                )
            elif (
                later.score == earlier.score
                and earlier.candidate_id >= later.candidate_id
            ):
                out.append(
                    ValidationFinding(
                        code=ValidationCode.TIEBREAK_VIOLATION,
                        severity=Severity.HARD,
                        message=(
                            f"tie at rank {later.rank}: "
                            f"{earlier.candidate_id} !< {later.candidate_id}"
                        ),
                        location=later.candidate_id,
                    )
                )
        return tuple(out)

    # ------------------------------------------------------------- Stage-4
    def _reasoning(
        self, ordered: tuple[RankedCandidate, ...]
    ) -> tuple[ValidationFinding, ...]:
        out: list[ValidationFinding] = []
        rendered: list[str] = []
        skeletons: list[str] = []

        for candidate in ordered:
            reasoning = candidate.reasoning
            if reasoning is None or not reasoning.rendered.strip():
                out.append(
                    ValidationFinding(
                        code=ValidationCode.EMPTY_REASONING,
                        severity=Severity.HARD,
                        message="reasoning missing or blank",
                        location=candidate.candidate_id,
                    )
                )
                continue

            if any(len(clause.evidence) == 0 for clause in reasoning.clauses):
                out.append(
                    ValidationFinding(
                        code=ValidationCode.HALLUCINATION,
                        severity=Severity.HARD,
                        message="a reasoning clause carries no evidence",
                        location=candidate.candidate_id,
                    )
                )

            out.extend(self._rank_band_consistency(candidate))

            rendered.append(reasoning.rendered)
            skeletons.append(self._skeleton(reasoning.rendered))

        out.extend(self._identical(ordered, rendered))
        out.extend(self._templated(skeletons))
        return tuple(out)

    def _rank_band_consistency(
        self, candidate: RankedCandidate
    ) -> tuple[ValidationFinding, ...]:
        reasoning = candidate.reasoning
        if reasoning is None:
            return ()
        expected = self._expected_band(candidate.rank)
        out: list[ValidationFinding] = []
        if reasoning.rank_band != expected:
            out.append(
                ValidationFinding(
                    code=ValidationCode.RANK_REASONING_MISMATCH,
                    severity=Severity.SOFT,
                    message=(
                        f"rank {candidate.rank} band {reasoning.rank_band!r} "
                        f"!= expected {expected!r}"
                    ),
                    location=candidate.candidate_id,
                )
            )
        has_strength = any(
            c.polarity is ReasoningPolarity.STRENGTH for c in reasoning.clauses
        )
        if expected in ("top", "mid") and not has_strength:
            out.append(
                ValidationFinding(
                    code=ValidationCode.RANK_REASONING_MISMATCH,
                    severity=Severity.SOFT,
                    message=f"{expected}-band candidate has no strength clause",
                    location=candidate.candidate_id,
                )
            )
        return tuple(out)

    def _expected_band(self, rank: int) -> str:
        top_cut = max(1, round(self.expected_size * 0.10))
        mid_cut = max(top_cut + 1, round(self.expected_size * 0.50))
        if rank <= top_cut:
            return "top"
        if rank <= mid_cut:
            return "mid"
        return "tail"

    @staticmethod
    def _identical(
        ordered: tuple[RankedCandidate, ...], rendered: list[str]
    ) -> tuple[ValidationFinding, ...]:
        if len(rendered) < 2:
            return ()
        counts = Counter(rendered)
        out: list[ValidationFinding] = []
        for text, freq in sorted(counts.items()):
            if freq > 1:
                out.append(
                    ValidationFinding(
                        code=ValidationCode.IDENTICAL_REASONING,
                        severity=Severity.HARD,
                        message=f"{freq} candidates share identical reasoning text",
                        location=None,
                    )
                )
        return tuple(out)

    @staticmethod
    def _templated(skeletons: list[str]) -> tuple[ValidationFinding, ...]:
        if len(skeletons) < 2:
            return ()
        counts = Counter(skeletons)
        _, top_freq = counts.most_common(1)[0]
        fraction = top_freq / len(skeletons)
        if fraction >= _TEMPLATED_FRACTION:
            return (
                ValidationFinding(
                    code=ValidationCode.TEMPLATED_REASONING,
                    severity=Severity.HARD,
                    message=(
                        f"{top_freq}/{len(skeletons)} renderings collapse to one "
                        f"templated skeleton ({fraction:.0%})"
                    ),
                    location=None,
                ),
            )
        return ()

    @staticmethod
    def _skeleton(text: str) -> str:
        """Normalize ids and numbers out so genuine variation survives but a fixed
        template (varying only by number/id) collapses to a single skeleton."""
        without_ids = _ID_TOKEN.sub("<id>", text)
        without_nums = _NUM_TOKEN.sub("<n>", without_ids)
        return " ".join(without_nums.lower().split())


__all__: tuple[str, ...] = ("ValidationEngine",)