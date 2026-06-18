"""``CsvSubmissionSinkAdapter`` — implements ``SubmissionSinkPort`` (Adapters §6).

Owner layer: adapters (infrastructure — impure IO).
Allowed imports: stdlib ``csv``/``io``/``hashlib``/``os``/``tempfile``/``pathlib``;
``domain.ranking``; ``ports``.
Forbidden: ``engines``, ``pipelines``, any ML/network runtime, business logic.

Serializes a fully-constructed, already-invariant-checked ``Ranking`` to a
validator-compliant, byte-stable CSV and returns a ``SubmissionReceipt``. The
adapter does not rank, re-order, or build reasoning — it writes the content of
the ``Ranking`` it is given.

Output contract: header ``candidate_id,rank,score,reasoning``; exactly
``ranking.size`` data rows in rank order; ``score`` at a fixed decimal
precision; RFC-4180 minimal quoting; UTF-8 (no BOM); ``\\n`` line endings; no
trailing whitespace. Before writing, the sink re-asserts non-increasing-by-rank
and id-ascending tie-break *at the emitted precision* (rounding may create ties,
permitted only if id order within ties holds) and raises
``SubmissionContractError`` rather than emit a rejectable file. The write is
atomic (temp file in the target directory → fsync → ``os.replace``), so a crash
never leaves a partial submission.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Final

from redstack.domain.ranking import Ranking
from redstack.ports._types import SubmissionReceipt
from redstack.ports.submission_sink import (
    SubmissionContractError,
    SubmissionWriteError,
)

#: The frozen, validator-mandated column order.
_HEADER: Final[tuple[str, str, str, str]] = (
    "candidate_id",
    "rank",
    "score",
    "reasoning",
)


class CsvSubmissionSinkAdapter:
    """Single-use, byte-stable CSV writer for a finished ``Ranking``.

    Constructed by the pipeline composition root with the resolved output path
    and the fixed score precision agreed with the scoring contract. One
    :meth:`write` per run; no persistent handle.
    """

    __slots__ = ("_output_path", "_score_decimals")

    def __init__(self, output_path: Path, *, score_decimals: int = 6) -> None:
        """Bind the sink to its output target and score precision.

        Args:
            output_path: Resolved ``<participant_id>.csv`` target path (the
                filename/extension are a pipeline concern; the sink writes here).
            score_decimals: Fixed number of decimal places for the ``score``
                column. Must be non-negative; pins byte-stable float formatting.
        """
        if score_decimals < 0:
            raise SubmissionContractError(
                f"score_decimals must be non-negative, got {score_decimals}"
            )
        self._output_path: Final[Path] = output_path
        self._score_decimals: Final[int] = score_decimals

    @property
    def output_path(self) -> Path:
        """The bound output path (audit-only)."""
        return self._output_path

    @property
    def score_decimals(self) -> int:
        """The fixed score precision (audit-only)."""
        return self._score_decimals

    def _format_score(self, score: float) -> str:
        """Format a score to the fixed decimal precision (byte-stable)."""
        return f"{float(score):.{self._score_decimals}f}"

    def _build_rows(self, ranking: Ranking) -> list[tuple[str, str, str, str]]:
        """Project the ranking into ordered, formatted CSV row tuples.

        Emits rows in the ranking's canonical ``ordered`` sequence (already
        sorted by ``(-score, candidate_id)`` and rank-assigned at construction).
        Missing reasoning serializes as an empty field; reasoning *content*
        quality is a ``ValidationEngine`` concern, not the sink's.
        """
        rows: list[tuple[str, str, str, str]] = []
        for ranked in ranking.ordered:
            reasoning = "" if ranked.reasoning is None else ranked.reasoning.rendered
            rows.append(
                (
                    str(ranked.candidate_id),
                    str(ranked.rank),
                    self._format_score(ranked.score),
                    reasoning,
                )
            )
        return self._stabilize_emitted_ties(rows)

    @staticmethod
    def _stabilize_emitted_ties(
        rows: list[tuple[str, str, str, str]]
    ) -> list[tuple[str, str, str, str]]:
        """Re-sort runs of equal *emitted* score by ``candidate_id`` ascending.

        Rounding to ``score_decimals`` can collapse two distinct full-precision
        scores that the ``RankingEngine`` already ordered correctly (by true
        score, then ``candidate_id``) into the same displayed value. The
        spec's tie-break rule is defined on the displayed score, so any such
        run must independently satisfy id-ascending order; rank numbers are
        purely positional and are renumbered to match. Every row within a run
        carries the identical emitted score, so non-increasing-by-rank still
        holds trivially.
        """
        n = len(rows)
        start = 0
        out = list(rows)
        while start < n:
            end = start
            while end + 1 < n and out[end + 1][2] == out[start][2]:
                end += 1
            if end > start:
                group = sorted(out[start : end + 1], key=lambda row: row[0])
                for offset, (cid, _rank, score, reasoning) in enumerate(group):
                    out[start + offset] = (cid, str(start + offset + 1), score, reasoning)
            start = end + 1
        return out

    def _reassert_invariants(
        self, ranking: Ranking, rows: list[tuple[str, str, str, str]]
    ) -> None:
        """Re-check validator invariants at the *emitted* precision; fail-fast.

        Catches rounding that would break monotonicity in a way the id tie-break
        cannot satisfy, before any byte hits disk.

        Raises:
            SubmissionContractError: row count, rank sequence, non-increasing
                score, or id-ascending tie-break would be violated as emitted.
        """
        if len(rows) != ranking.size:
            raise SubmissionContractError(
                f"row count {len(rows)} != ranking size {ranking.size}"
            )

        prev_score: float | None = None
        prev_cid: str | None = None
        for position, (cid, rank_str, score_str, _reasoning) in enumerate(rows):
            expected_rank = position + 1
            if rank_str != str(expected_rank):
                raise SubmissionContractError(
                    f"rank at position {position} is {rank_str!r}, "
                    f"expected {expected_rank}"
                )

            score = float(score_str)
            if prev_score is not None:
                if score > prev_score:
                    raise SubmissionContractError(
                        f"score increases at rank {expected_rank}: "
                        f"{prev_score!r} -> {score!r}"
                    )
                if score == prev_score and prev_cid is not None and cid <= prev_cid:
                    raise SubmissionContractError(
                        f"tie-break violation at rank {expected_rank}: "
                        f"{prev_cid!r} not < {cid!r} at equal emitted score"
                    )
            prev_score = score
            prev_cid = cid

    @staticmethod
    def _serialize(rows: list[tuple[str, str, str, str]]) -> bytes:
        """Serialize header + rows to RFC-4180 / UTF-8 (no BOM) / ``\\n`` bytes.

        ``QUOTE_MINIMAL`` quotes only fields containing a delimiter, quote, or
        newline (``\\r``/``\\n``), doubling embedded quotes per RFC-4180. The
        ``\\n`` line terminator and UTF-8 encoding (no BOM) make the output
        byte-stable.
        """
        buffer = io.StringIO(newline="")
        writer = csv.writer(
            buffer,
            delimiter=",",
            quotechar='"',
            doublequote=True,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerow(_HEADER)
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")

    def _atomic_write(self, data: bytes) -> None:
        """Write ``data`` via temp file in the target directory + ``os.replace``.

        Raises:
            SubmissionWriteError: any IO failure; the temp file is removed so no
                partial/rejectable submission is left behind.
        """
        parent = self._output_path.parent
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=parent, prefix=".tmp_submission_", suffix=".csv"
            )
        except OSError as exc:
            raise SubmissionWriteError(
                f"cannot create temp file in {parent!s}: {exc}"
            ) from exc

        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._output_path)
        except OSError as exc:
            with suppress(OSError):
                tmp_path.unlink()
            raise SubmissionWriteError(
                f"cannot write submission to {self._output_path!s}: {exc}"
            ) from exc

    def write(self, ranking: Ranking) -> SubmissionReceipt:
        """Serialize ``ranking`` to the output CSV and return a receipt.

        Order of operations: project rows → re-assert invariants at emitted
        precision (abort before any IO on violation) → serialize → hash →
        atomic write. The receipt's ``output_sha256`` is computed over the exact
        bytes written.

        Raises:
            SubmissionContractError: a serialized invariant would break (no file
                is written).
            SubmissionWriteError: an IO error during the atomic write.
        """
        rows = self._build_rows(ranking)
        self._reassert_invariants(ranking, rows)

        data = self._serialize(rows)
        output_sha256 = hashlib.sha256(data).hexdigest()

        self._atomic_write(data)

        return SubmissionReceipt(
            row_count=len(rows),
            bytes_written=len(data),
            output_sha256=output_sha256,
        )


if TYPE_CHECKING:
    from redstack.ports.submission_sink import SubmissionSinkPort

    # Compile-time structural conformance to the frozen port surface.
    _PORT_CONFORMANCE: type[SubmissionSinkPort] = CsvSubmissionSinkAdapter


__all__: tuple[str, ...] = ("CsvSubmissionSinkAdapter",)