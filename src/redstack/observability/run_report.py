"""RunReport model builder: reproducible + audit + timings + budget blocks.

Owner layer: observability.
Allowed imports: domain.

Builds an object conforming structurally to the ports-owned ``RunReport``
Protocol (``ports._types.RunReport``). ``ports`` and ``observability`` are
mutually independent (Repository Layout §8c) — neither imports the other.
``RunReportSnapshot`` and its three nested blocks therefore do not inherit
from ``ports._types.RunReport`` and its sibling Protocols; they are plain
frozen dataclasses whose fields exactly mirror those Protocols' properties by
name and type. mypy verifies that correspondence *structurally*, at whatever
call site (the pipelines composition root) passes a built snapshot to
something typed as the ports ``RunReport`` Protocol — e.g.
``RunReportSinkPort.write``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import final

from redstack.domain.errors import InvariantViolation

__all__: tuple[str, ...] = (
    "AuditSnapshot",
    "BudgetSnapshot",
    "ReproducibleSnapshot",
    "RunReportBuilder",
    "RunReportBuilderError",
    "RunReportSnapshot",
)


class RunReportBuilderError(InvariantViolation):
    """:meth:`RunReportBuilder.build` was called before every block was bound."""


@dataclass(frozen=True, slots=True)
class ReproducibleSnapshot:
    """The deterministic region — the only region compared by determinism tests.

    Mirrors ``ports._types.ReproducibleBlock`` field-for-field.
    """

    code_version: str
    config_hash: str
    manifest_hash: str
    artifact_hashes: Mapping[str, str]
    input_file_sha256: str
    candidate_count: int
    output_sha256: str
    honeypot_count_top100: int
    honeypot_rate: float
    eligibility_summary: Mapping[str, int]
    score_distribution_digest: str


@dataclass(frozen=True, slots=True)
class AuditSnapshot:
    """Wall-clock / identity region; excluded from any reproducibility hash.

    Mirrors ``ports._types.AuditBlock`` field-for-field.
    """

    run_id: str
    started_at: str
    ended_at: str
    host_label: str


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Budget accounting for the run, frozen at report-build time.

    Mirrors ``ports._types.BudgetBlock`` field-for-field. Built from a
    ``timing.BudgetGuard``'s final readings rather than embedding the guard
    itself, so the report stays an immutable value once built.
    """

    limit_seconds: float
    used_seconds: float
    within_budget: bool
    peak_rss_mb: float


@dataclass(frozen=True, slots=True)
class RunReportSnapshot:
    """The complete, immutable run report; structurally a ``RunReport`` (Ports §13)."""

    reproducible: ReproducibleSnapshot
    audit: AuditSnapshot
    timings: Mapping[str, float]
    budget: BudgetSnapshot


@final
class RunReportBuilder:
    """Mutable accumulator for the four ``RunReport`` blocks (Ports §13).

    Each ``with_*`` call binds one block and returns ``self`` for chaining;
    :meth:`build` raises :class:`RunReportBuilderError` if any block is still
    unbound, and otherwise freezes the accumulated state into a
    :class:`RunReportSnapshot`.
    """

    __slots__ = ("_audit", "_budget", "_reproducible", "_timings")

    def __init__(self) -> None:
        self._reproducible: ReproducibleSnapshot | None = None
        self._audit: AuditSnapshot | None = None
        self._timings: Mapping[str, float] | None = None
        self._budget: BudgetSnapshot | None = None

    def with_reproducible(self, reproducible: ReproducibleSnapshot) -> RunReportBuilder:
        """Bind the deterministic region and return ``self`` for chaining."""
        self._reproducible = reproducible
        return self

    def with_audit(self, audit: AuditSnapshot) -> RunReportBuilder:
        """Bind the wall-clock/identity region and return ``self`` for chaining."""
        self._audit = audit
        return self

    def with_timings(self, timings: Mapping[str, float]) -> RunReportBuilder:
        """Bind the per-stage wall-ms timings and return ``self`` for chaining."""
        self._timings = dict(timings)
        return self

    def with_budget(self, budget: BudgetSnapshot) -> RunReportBuilder:
        """Bind the budget accounting region and return ``self`` for chaining."""
        self._budget = budget
        return self

    def build(self) -> RunReportSnapshot:
        """Freeze the accumulated blocks into a complete ``RunReportSnapshot``.

        Raises:
            RunReportBuilderError: any of the four blocks was never bound.
        """
        if self._reproducible is None:
            raise RunReportBuilderError(
                "RunReportBuilder.build(): 'reproducible' block not bound"
            )
        if self._audit is None:
            raise RunReportBuilderError(
                "RunReportBuilder.build(): 'audit' block not bound"
            )
        if self._timings is None:
            raise RunReportBuilderError(
                "RunReportBuilder.build(): 'timings' block not bound"
            )
        if self._budget is None:
            raise RunReportBuilderError(
                "RunReportBuilder.build(): 'budget' block not bound"
            )
        return RunReportSnapshot(
            reproducible=self._reproducible,
            audit=self._audit,
            timings=self._timings,
            budget=self._budget,
        )
