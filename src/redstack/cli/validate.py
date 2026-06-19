

from __future__ import annotations

import csv
import re
from pathlib import Path

import typer

from redstack.domain.ids import CANDIDATE_ID_PATTERN

__all__: tuple[str, ...] = ("validate",)

_HEADER: tuple[str, ...] = ("candidate_id", "rank", "score", "reasoning")
_ID_RE = re.compile(CANDIDATE_ID_PATTERN)


def _check_rows(rows: list[dict[str, str]], expected_size: int) -> list[str]:
    """Return every structural violation found in ``rows`` (empty if valid)."""
    errors: list[str] = []
    if len(rows) != expected_size:
        errors.append(f"expected {expected_size} rows, got {len(rows)}")

    seen_ids: set[str] = set()
    prev_score: float | None = None
    prev_id: str | None = None
    for position, row in enumerate(rows, start=1):
        candidate_id = row["candidate_id"]
        try:
            rank = int(row["rank"])
        except ValueError:
            errors.append(f"row {position}: rank {row['rank']!r} is not an integer")
            continue
        try:
            score = float(row["score"])
        except ValueError:
            errors.append(f"row {position}: score {row['score']!r} is not a float")
            continue

        if rank != position:
            errors.append(f"row {position}: rank {rank} != row position {position}")
        if _ID_RE.fullmatch(candidate_id) is None:
            errors.append(f"row {position}: malformed candidate_id {candidate_id!r}")
        if candidate_id in seen_ids:
            errors.append(f"row {position}: duplicate candidate_id {candidate_id!r}")
        seen_ids.add(candidate_id)
        if not row["reasoning"].strip():
            errors.append(f"row {position} ({candidate_id}): reasoning is blank")

        if prev_score is not None and prev_id is not None:
            if score > prev_score:
                errors.append(
                    f"row {position}: score {score} > previous row's score {prev_score}"
                )
            elif score == prev_score and not (prev_id < candidate_id):
                errors.append(
                    f"row {position}: tie-break violated — equal scores must order "
                    f"by ascending candidate_id ({prev_id!r} then {candidate_id!r})"
                )
        prev_score, prev_id = score, candidate_id

    return errors


def validate(
    submission: Path = typer.Option(
        ..., "--submission", help="Path to the finished submission.csv."
    ),
    expected_size: int = typer.Option(
        100, "--expected-size", help="Required row count (the top-K cut)."
    ),
) -> None:
    """Validate ``submission`` against the structural submission rules."""
    try:
        with submission.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != _HEADER:
                typer.secho(
                    f"header mismatch: expected {_HEADER}, got {reader.fieldnames}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)
            rows = list(reader)
    except OSError as exc:
        typer.secho(f"cannot read {submission}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    errors = _check_rows(rows, expected_size)
    if errors:
        for error in errors:
            typer.secho(error, fg=typer.colors.RED, err=True)
        typer.secho(f"INVALID: {len(errors)} violation(s)", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"VALID: {len(rows)} rows", fg=typer.colors.GREEN)
