
from __future__ import annotations

import gc
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import final

from redstack.config.determinism import DeterminismPolicy, pin_determinism
from redstack.domain.errors import ArtifactContractError
from redstack.observability.timing import sample_peak_rss_mb
from redstack.pipelines.online import stages
from redstack.ports.artifact_store import ArtifactStorePort
from redstack.ports.candidate_source import CandidateSourcePort
from redstack.ports.embedding import EmbeddingModelPort
from redstack.ports.online import (
    OnlineEntropyPort,
    RunReportSinkPort,
    SemanticVectorStorePort,
    SubmissionSinkPort,
)

__all__: tuple[str, ...] = (
    "OnlinePipeline",
    "OnlinePipelineResult",
    "OnlineRunConfig",
    "OnlineRunContext",
)


@final
@dataclass(frozen=True, slots=True)
class OnlineRunConfig:
    """Resolved online runtime config (IO/runtime only — never weights).

    Sourced from ``configs/runtime/online.yaml`` + ``base.yaml`` + the active
    profile by the composition root. Holds the participant id (output filename
    stem), the determinism seed + thread pins, the ingestion abort policy, the
    FLOOR sentinel, the score-presentation decimals, and the wall-clock budget
    ceiling. Behaviour-defining weights/thresholds are consumed online from
    ``artifacts/``, never from here.
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
    blas_threads: int = 1
    onnx_intra_op_threads: int = 1
    onnx_inter_op_threads: int = 1

    def determinism_policy(self) -> DeterminismPolicy:
        """The determinism policy this config pins at startup (one owner)."""
        return DeterminismPolicy(
            seed=self.seed,
            blas_threads=self.blas_threads,
            intra_op_threads=self.onnx_intra_op_threads,
            inter_op_threads=self.onnx_inter_op_threads,
        )


@final
@dataclass(frozen=True, slots=True)
class OnlineRunContext:
    """The immutable R0 output: resolved config + bound ports + verified artifacts.

    Assembled whole by R0 (``OnlinePipeline._r0``) or not at all — there is no
    partially-bound context. ``artifacts`` holds the loaded artifact objects
    (parsed JSON/YAML, npy matrices) keyed by registry key; ``manifest`` is the
    verified manifest mapping; ``as_of``/``seed`` are the determinism anchors. The
    stages read this immutable carrier and return new representation state — they
    never mutate the context.
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
        """The layout_version the manifest pins (asserted coherent at R0)."""
        value = self.manifest.get("layout_version")
        return value if isinstance(value, str) else ""


@final
@dataclass(frozen=True, slots=True)
class OnlinePipelineResult:
    """The terminal result of a full R0→R9 run (the orchestrator's return)."""

    submission_path: str
    output_sha256: str
    row_count: int
    report_path: str
    honeypot_rate_top100: float
    within_budget: bool
    peak_rss_mb: float
    stage_timings_ms: Mapping[str, float]


@final
@dataclass(slots=True)
class _StageClock:
    """Per-stage wall-time accumulator (audit only; not in the repro block)."""

    timings_ms: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, started: float) -> None:
        """Record ``stage``'s elapsed wall-ms since ``started`` (monotonic clock)."""
        self.timings_ms[stage] = round((time.perf_counter() - started) * 1000.0, 3)

    def total_seconds(self) -> float:
        """Total wall-seconds across all recorded stages."""
        return sum(self.timings_ms.values()) / 1000.0


def _peak_rss_mb() -> float:
    """Peak resident set size in MiB, on whichever platform this runs on.

    Pure of any wall clock; reads the OS-reported high-water mark so R9's budget
    block can record ``peak_rss_mb`` without sampling. Delegates to the shared
    cross-platform reader (POSIX ``resource`` or Windows ``ctypes``/``psapi``) so
    this never assumes a Linux-only sandbox (CLAUDE.md §1: cross-platform
    execution stability).
    """
    return round(sample_peak_rss_mb(), 3)


class OnlinePipeline:
    """The R0…R9 composition root; runs the stages strictly sequentially.

    Constructed with every bound port (the composition root injects concrete
    adapters); ``run`` pins determinism, executes R0→R9 fail-fast, and returns the
    terminal result. The instance is stateless across stages — all run state lives
    on the immutable ``OnlineRunContext`` and the forward-threaded stage outputs.
    """

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

        Pins determinism *first* (before any numeric work / onnx init), then runs
        the stages strictly sequentially with the representation set flowing
        forward copy-on-write. Any integrity/coherence failure raises (fail-fast);
        outputs are written atomically only at R8/R9, so a crash before R8 leaves
        no partial submission file.

        Args:
            input_file_sha256: the sha256 of ``candidates.jsonl`` (the provenance
                anchor recorded in the R9 reproducible block; computed by the
                composition root, not re-read here — no wall clock / extra IO).

        Returns:
            The terminal :class:`OnlinePipelineResult` (submission + report paths,
            output hash, honeypot rate, budget verdict, peak RSS, stage timings).
        """
        # Determinism is owned in config/determinism.py and asserted here, at the
        # very top, before any numeric reduction or onnx session creation — so the
        # output is thread-count-invariant (Architecture §2 / Online Part 1).
        pin_determinism(self._config.determinism_policy())

        clock = _StageClock()

        # The R0..R9 representation set briefly holds tens of millions of
        # small, acyclic, immutable objects (one FeatureCell + EvidenceRef
        # per feature per candidate, at full candidate-pool scale). None of
        # them ever form a reference cycle, so the generational collector's
        # cycle scans buy nothing here but cost real wall-time once the live
        # object count gets large — refcounting alone still reclaims every
        # one of them the instant a stage's superseded generation is
        # dropped (see the `del` calls below). Disabled only for this run's
        # duration and restored in `finally` so it never leaks into any other
        # work sharing this process (e.g. a test runner invoking several
        # pipelines). Pure timing/CPU effect — does not change output values,
        # so determinism is unaffected.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            # R0 — load + verify all artifacts; bind ports; build the context whole.
            started = time.perf_counter()
            ctx = self._r0(input_file_sha256)
            clock.record("R0", started)

            # R1 — stream + validate candidates (lazy at the port; bulk path here).
            started = time.perf_counter()
            ingested = stages.r1_ingest(ctx)
            clock.record("R1", started)
            if not ingested:
                raise ArtifactContractError("R1 produced zero candidates (empty input)")

            # R2 — bulk structural feature extraction into the (N, D) CQV.
            started = time.perf_counter()
            featured = stages.r2_features(ctx, ingested)
            clock.record("R2", started)
            # `ingested` is fully superseded by `featured` (same candidates,
            # same IngestedCandidate objects, plus the new feature slices) —
            # drop the reference now rather than letting the orchestrator
            # hold every stage's full 100k-candidate generation alive
            # simultaneously through to R9 (the ≤16 GB online RAM ceiling;
            # CLAUDE.md §1).
            del ingested

            # R3 — semantic hydration by lookup (+ onnx fallback for misses).
            started = time.perf_counter()
            situated = stages.r3_semantic(ctx, featured)
            clock.record("R3", started)
            del featured

            # R4 — integrity + eligibility gates → floor mask (verdicts are data).
            started = time.perf_counter()
            gated = stages.r4_gates(ctx, situated)
            clock.record("R4", started)
            del situated

            # R5 — scoring (locked weights, gates, bounded multipliers; no calibration).
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

            # R8 — validate (structural + Stage-4) then write the CSV atomically.
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
        finally:
            # gc.disable() never promotes anything out of generation 0 (that
            # only happens during a collection), so every object allocated
            # since the call above — tens of millions, at full pool scale —
            # is still sitting in gen0. Re-enabling naively leaves the *next*
            # allocation-triggered collection to cycle-scan that entire
            # backlog in one pass, even though it's claimed acyclic (so the
            # scan is pure waste): a multi-minute stall in practice, after
            # the run's actual output has already been written. Freezing
            # moves the live set into the permanent generation first, so the
            # cyclic collector skips it entirely; refcounting (unaffected by
            # freeze) still reclaims every object normally as soon as it's
            # dropped, so this is free as long as the acyclic claim holds.
            gc.freeze()
            if gc_was_enabled:
                gc.enable()

        used_seconds = clock.total_seconds()
        within_budget = used_seconds <= self._config.budget_limit_seconds
        return OnlinePipelineResult(
            submission_path=report_outcome.submission_path,
            output_sha256=receipt.output_sha256,
            row_count=receipt.row_count,
            report_path=report_outcome.report_path,
            honeypot_rate_top100=report_outcome.honeypot_rate_top100,
            within_budget=within_budget,
            peak_rss_mb=_peak_rss_mb(),
            stage_timings_ms=dict(clock.timings_ms),
        )

    # ------------------------------------------------------------------ #
    # R0 — the only stage that builds the context (ports + artifacts)    #
    # ------------------------------------------------------------------ #
    def _r0(self, input_file_sha256: str) -> OnlineRunContext:
        """Load + verify artifacts, then assemble the immutable context.

        Delegates manifest verification + artifact loading + cross-artifact
        coherence to ``stages.r0_load``; binds every port; never returns a
        partially-bound context (a failure inside ``r0_load`` raises before the
        ``OnlineRunContext`` is constructed). ``as_of``/``seed`` come from the
        injected ``OnlineEntropy`` (no wall clock; RNG forbidden).
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
