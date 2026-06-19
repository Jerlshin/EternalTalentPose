

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from redstack.pipelines.offline.context import OfflinePipelineContext
from redstack.pipelines.offline.graph import (
    OFFLINE_EXECUTION_GRAPH,
    OfflineExecutionGraph,
)
from redstack.pipelines.offline.registry import (
    OFFLINE_ARTIFACT_REGISTRY,
    OfflineArtifactRegistry,
)
from redstack.pipelines.offline.runner import (
    OfflinePipelineRunner,
    StageCallable,
    StageReceipt,
    StageStatus,
)

__all__: tuple[str, ...] = (
    "OfflinePipelineReport",
    "OfflinePipeline",
)


@dataclass(frozen=True, slots=True)
class OfflinePipelineReport:
    """The build audit (Offline Pipeline Part 1, ``OfflinePipelineReport``).

    Mirrors the port-owned ``RunReport`` shape with an explicit ``reproducible``
    vs ``audit`` split (Ports §13). Per-stage timings + metrics, seeds, hashes,
    and the produced-artifact set are surfaced here; the ``reproducible`` block
    is deterministic across rebuilds, the ``audit`` block (wall timings) is not.

    Attributes:
        reproducible: deterministic provenance — ``code_version``, ``config_hash``,
            ``seed``, ``layout_version``, and per-stage produced-artifact hashes.
        audit: per-stage wall-ms and this-run status (excluded from repro hashes).
        stage_metrics: per-stage deterministic metrics (census stats, NDCG, …).
        executed: stage ids that ran this build (vs resumed-from-checkpoint).
        skipped: stage ids served from a fresh checkpoint.
        failed: non-critical stage ids flagged under ``continue_on_error``.
    """

    reproducible: Mapping[str, object]
    audit: Mapping[str, object]
    stage_metrics: Mapping[str, Mapping[str, object]]
    executed: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[str, ...]

    @classmethod
    def from_receipts(
        cls,
        ctx: OfflinePipelineContext,
        receipts: Mapping[str, StageReceipt],
    ) -> OfflinePipelineReport:
        """Assemble a report from a completed run's receipts."""
        artifact_hashes: dict[str, str] = {}
        for receipt in receipts.values():
            for key, digest in receipt.artifact_hashes.items():
                artifact_hashes[key] = digest

        executed = tuple(
            sorted(
                sid for sid, r in receipts.items()
                if r.status is StageStatus.EXECUTED
            )
        )
        skipped = tuple(
            sorted(
                sid for sid, r in receipts.items()
                if r.status is StageStatus.SKIPPED
            )
        )
        failed = tuple(
            sorted(
                sid for sid, r in receipts.items()
                if r.status is StageStatus.FAILED
            )
        )
        reproducible: dict[str, object] = {
            "code_version": ctx.code_version,
            "config_hash": ctx.config_hash,
            "seed": ctx.seed,
            "layout_version": ctx.layout_version,
            "as_of": ctx.as_of.isoformat(),
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "stage_versions": {
                sid: receipts[sid].stage_version for sid in sorted(receipts)
            },
        }
        audit: dict[str, object] = {
            "stage_wall_ms": {
                sid: receipts[sid].wall_ms for sid in sorted(receipts)
            },
            "stage_status": {
                sid: receipts[sid].status.value for sid in sorted(receipts)
            },
        }
        stage_metrics = {
            sid: dict(receipts[sid].metrics) for sid in sorted(receipts)
        }
        return cls(
            reproducible=reproducible,
            audit=audit,
            stage_metrics=stage_metrics,
            executed=executed,
            skipped=skipped,
            failed=failed,
        )


@dataclass(frozen=True, slots=True)
class OfflinePipeline:
    """Pure orchestration over injected stage callables (Part 1).

    Construct with the injected ``stages`` mapping (the composition root wires
    O0–O18 callables, each binding its needed ports from the context) and,
    optionally, a custom graph/registry/runner for testing. Defaults to the
    frozen Part 11 graph and Part 10 registry.

    Attributes:
        stages: ``stage_id -> StageCallable`` for every node in the graph.
        graph: the dependency DAG (defaults to the frozen topology).
        registry: the artifact catalog (defaults to the frozen catalog).
        runner: the resume/checkpoint executor (defaults to a fresh runner over
            the same graph + registry).
    """

    stages: Mapping[str, StageCallable]
    graph: OfflineExecutionGraph = OFFLINE_EXECUTION_GRAPH
    registry: OfflineArtifactRegistry = OFFLINE_ARTIFACT_REGISTRY
    runner: OfflinePipelineRunner | None = None

    def __post_init__(self) -> None:
        # Default the runner to one bound to this pipeline's graph + registry.
        if self.runner is None:
            object.__setattr__(
                self,
                "runner",
                OfflinePipelineRunner(graph=self.graph, registry=self.registry),
            )
        self._assert_stage_coverage()

    @property
    def _runner(self) -> OfflinePipelineRunner:
        """Return the bound runner, guaranteed non-``None`` after construction."""
        assert self.runner is not None  # noqa: S101 — invariant set in __post_init__
        return self.runner

    def _assert_stage_coverage(self) -> None:
        """Every graph node must have a registered callable; raise otherwise.

        A missing callable is a composition error (a stage was declared in the
        DAG but never wired). Caught here so ``plan()``/``run()`` never discover
        it mid-execution.
        """
        graph_ids = set(self.graph.stage_ids())
        wired_ids = set(self.stages)
        missing = sorted(graph_ids - wired_ids)
        if missing:
            msg = f"offline pipeline missing stage callable(s): {', '.join(missing)}"
            raise ValueError(msg)
        extra = sorted(wired_ids - graph_ids)
        if extra:
            msg = f"offline pipeline has stage callable(s) not in graph: {', '.join(extra)}"
            raise ValueError(msg)
        for stage_id, stage in self.stages.items():
            if stage.stage_id != stage_id:
                msg = (
                    f"stage callable keyed {stage_id!r} reports id "
                    f"{stage.stage_id!r}"
                )
                raise ValueError(msg)

    def plan(self) -> tuple[str, ...]:
        """Return the deterministic topological execution order (Part 1).

        Cycle detection already happened at graph construction; this exposes the
        stable plan the runner will walk.
        """
        return self._runner.plan()

    def run(
        self,
        ctx: OfflinePipelineContext,
        *,
        force: tuple[str, ...] | None = None,
    ) -> dict[str, StageReceipt]:
        """Execute stale stages in dependency order via the resume runner.

        Args:
            ctx: the immutable build context (ports + roots + provenance).
            force: optional stage ids to recompute regardless of freshness; their
                downstream closure is forced too.
        """
        return self._runner.run(ctx, self.stages, force=force)

    def finalize(
        self,
        ctx: OfflinePipelineContext,
        receipts: Mapping[str, StageReceipt],
    ) -> OfflinePipelineReport:
        """Package-validate the produced set and build the build report.

        Asserts the required-online artifact set is complete against the registry
        (a partial set is fatal — Ports §8). The manifest's self-hash + per-key
        sha256 + cross-artifact coherence are produced by the O17/O18 stages this
        orchestrator already ran; ``finalize`` confirms completeness and emits the
        :class:`OfflinePipelineReport`.

        Raises:
            ValueError: a required-online artifact is missing from the build
                (surfaced from the registry's completeness check).
        """
        produced: set[str] = set()
        for receipt in receipts.values():
            if receipt.status is StageStatus.FAILED:
                continue
            produced.update(receipt.artifact_hashes)
        self.registry.assert_required_present(frozenset(produced))
        return OfflinePipelineReport.from_receipts(ctx, receipts)

    def execute(
        self,
        ctx: OfflinePipelineContext,
        *,
        force: tuple[str, ...] | None = None,
    ) -> OfflinePipelineReport:
        """Run the full ``plan() → run() → finalize()`` lifecycle end-to-end."""
        self.plan()
        receipts = self.run(ctx, force=force)
        return self.finalize(ctx, receipts)