"""The R0…R9 online orchestrator — the literal Stage-3 reproduce spine.

Builds the immutable ``OnlineRunContext`` (binds ``CandidateSource``,
``SemanticVectorStore``, ``EmbeddingModel(onnx)``, ``SubmissionSink``,
``RunReportSink``, ``OnlineEntropy``; loads + verifies every artifact), then runs
the stages strictly sequentially R0→R9, threading the growing state
copy-on-write. Ports are touched only at R0/R1/R3/R8/R9; R2/R4/R5/R6/R7 are pure
engine work. Fail-fast on any integrity/coherence failure; the context is never
returned partially bound (R0 builds it whole or raises).

This module is the composition root: it imports the online stages and the bound
ports, never adapters directly (those are injected). The online containment
contract (no ``sentence_transformers``/``sklearn``/networking imports) holds —
nothing here or in ``stages.py`` pulls a budget-busting dependency.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import final

from redstack.domain.errors import ArtifactContractError
from redstack.domain.ids import CandidateId
from redstack.ports.artifact_store import ArtifactStorePort
from redstack.ports.candidate_source import CandidateSourcePort
from redstack.ports.embedding import EmbeddingModelPort
from redstack.ports.online import (
    OnlineEntropyPort,
    RunReportSinkPort,
    SemanticVectorStorePort,
    SubmissionSinkPort,
)
from redstack.pipelines.online import stages

__all__: tuple[str, ...] = (
    "OnlineRunConfig",
    "OnlineRunContext",
    "OnlinePipeline",
    "OnlinePipelineResult",
)


@final
@dataclass(frozen=True, slots=True)
class OnlineRunConfig:
    """Resolved online runtime config (IO/runtime only — never weights).

    Sourced from ``runtime/online.yaml`` by the composition root.
    """

    participant_id: str
    code_version: str
    config_hash: str
    seed: int
    floor_sentinel: float = 0.0
    ranking_size: int = 100
    score_decimals: int = 6
    abort_on_malformed: bool = True
    budget_limit_seconds: float = 300.0


@final
@dataclass(frozen=True, slots=True)
class OnlineRunContext:
    """The immutable R0 output: resolved config + bound ports + verified artifacts.

    Assembled whole by R0 (``OnlinePipeline._r0``) or not at all — there is no
    partially-bound context. ``artifacts`` holds the loaded artifact objects
    (parsed JSON/YAML, npy matrices) keyed by registry key; ``manifest`` is the
    verified manifest mapping; ``as_of``/``seed`` are the determinism anchors.
    """

    config: OnlineRunConfig
    candidate_source: CandidateSourcePort
    vector_store: SemanticVectorStorePort
    embedding_model: EmbeddingModelPort
    submission_sink: SubmissionSinkPort
    run_report_sink: RunReportSinkPort
    entropy: OnlineEntropyPort
    artifact_store: ArtifactStorePort
    artifacts: Mapping[str, object]
    manifest: Mapping[str, object]
    as_of: date
    seed: int
    input_file_sha256: str

    @property
    def manifest_hash(self) -> str:
        """The verified manifest self-hash (recorded into the run report)."""
        value = self.manifest.get("manifest_sha256")
        return value if isinstance(value, str) else ""

    @property
    def layout_version(self) -> str:
        value = self.manifest.get("layout_version")
        return value if isinstance(value, str) else ""


@final
@dataclass(frozen=True, slots=True)
class OnlinePipelineResult:
    """The terminal result of a full R0→R9 run."""

    submission_path: str
    output_sha256: str
    row_count: int
    report_path: str
    honeypot_rate_top100: float
    within_budget: bool
    stage_timings_ms: Mapping[str, float]


@final
@dataclass(slots=True)
class _StageClock:
    """Per-stage wall-time accumulator (audit only; not in the repro block)."""

    timings_ms: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, started: float) -> None:
        self.timings_ms[stage] = round((time.perf_counter() - started) * 1000.0, 3)


class OnlinePipeline:
    """The R0…R9 composition root; runs the stages strictly sequentially."""

    def __init__(
        self,
        *,
        config: OnlineRunConfig,
        candidate_source: CandidateSourcePort,
        vector_store: SemanticVectorStorePort,
        embedding_model: EmbeddingModelPort,
        submission_sink: SubmissionSinkPort,
        run_report_sink: RunReportSinkPort,
        entropy: OnlineEntropyPort,
        artifact_store: ArtifactStorePort,
    ) -> None:
        self._config = config
        self._candidate_source = candidate_source
        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._submission_sink = submission_sink
        self._run_report_sink = run_report_sink
        self._entropy = entropy
        self._artifact_store = artifact_store

    def run(self, *, input_file_sha256: str) -> OnlinePipelineResult:
        """Execute R0→R9 and return the terminal result.

        Strictly sequential; the representation set flows forward copy-on-write.
        Any integrity/coherence failure raises (fail-fast); outputs are written
        atomically only at R8/R9, so a crash before R8 leaves no partial file.
        """
        clock = _StageClock()

        # R0 — load + verify all artifacts; bind ports; build the context whole.
        started = time.perf_counter()
        ctx = self._r0(input_file_sha256)
        clock.record("R0", started)

        # R1 — stream + validate candidates (lazy; materialized for the bulk path).
        started = time.perf_counter()
        ingested = stages.r1_ingest(ctx)
        clock.record("R1", started)
        if not ingested:
            raise ArtifactContractError("R1 produced zero candidates (empty input)")

        # R2 — bulk structural feature extraction into the (N, D) CQV.
        started = time.perf_counter()
        featured = stages.r2_features(ctx, ingested)
        clock.record("R2", started)

        # R3 — semantic hydration by lookup (+ onnx fallback for misses).
        started = time.perf_counter()
        situated = stages.r3_semantic(ctx, featured)
        clock.record("R3", started)

        # R4 — integrity + eligibility gates → floor mask.
        started = time.perf_counter()
        gated = stages.r4_gates(ctx, situated)
        clock.record("R4", started)

        # R5 — scoring (locked weights, gates, bounded multipliers).
        started = time.perf_counter()
        scored = stages.r5_score(ctx, gated)
        clock.record("R5", started)

        # R6 — ranking (raw scores, floor-partitioned, top-100, invariants).
        started = time.perf_counter()
        ranking = stages.r6_rank(ctx, scored)
        clock.record("R6", started)

        # R7 — reasoning for the top-100 (evidence-bound, no reorder).
        started = time.perf_counter()
        reasoned = stages.r7_reason(ctx, ranking, gated)
        clock.record("R7", started)

        # R8 — validate + write the submission CSV atomically.
        started = time.perf_counter()
        receipt = stages.r8_submit(ctx, reasoned)
        clock.record("R8", started)

        # R9 — run report (reproducible + audit + timings + budget).
        started = time.perf_counter()
        report_outcome = stages.r9_report(
            ctx,
            ranking=reasoned,
            scored=scored,
            gated=gated,
            receipt=receipt,
            stage_timings_ms=dict(clock.timings_ms),
        )
        clock.record("R9", started)

        used_seconds = sum(clock.timings_ms.values()) / 1000.0
        within_budget = used_seconds <= self._config.budget_limit_seconds
        return OnlinePipelineResult(
            submission_path=report_outcome.submission_path,
            output_sha256=receipt.output_sha256,
            row_count=receipt.row_count,
            report_path=report_outcome.report_path,
            honeypot_rate_top100=report_outcome.honeypot_rate_top100,
            within_budget=within_budget,
            stage_timings_ms=dict(clock.timings_ms),
        )

    # ------------------------------------------------------------------ #
    # R0 — the only stage that builds the context (ports + artifacts)    #
    # ------------------------------------------------------------------ #
    def _r0(self, input_file_sha256: str) -> OnlineRunContext:
        """Load + verify artifacts, then assemble the immutable context.

        Delegates the manifest verification + artifact loading + cross-artifact
        coherence to ``stages.r0_load``; binds every port; never returns a
        partially-bound context (a failure inside ``r0_load`` raises before the
        context is constructed).
        """
        loaded = stages.r0_load(
            artifact_store=self._artifact_store,
            embedding_model=self._embedding_model,
            vector_store=self._vector_store,
            entropy=self._entropy,
            config=self._config,
        )
        return OnlineRunContext(
            config=self._config,
            candidate_source=self._candidate_source,
            vector_store=self._vector_store,
            embedding_model=self._embedding_model,
            submission_sink=self._submission_sink,
            run_report_sink=self._run_report_sink,
            entropy=self._entropy,
            artifact_store=self._artifact_store,
            artifacts=loaded.artifacts,
            manifest=loaded.manifest,
            as_of=self._entropy.as_of(),
            seed=self._entropy.seed,
            input_file_sha256=input_file_sha256,
        )


def _resolve_candidate_ids(records: Sequence[object]) -> tuple[CandidateId, ...]:
    """Defensive helper: extract candidate ids from ingested records (audit use)."""
    out: list[CandidateId] = []
    for rec in records:
        cid = getattr(rec, "candidate_id", None)
        if isinstance(cid, str):
            out.append(CandidateId(cid))
    return tuple(out)