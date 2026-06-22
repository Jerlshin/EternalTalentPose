

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from redstack.domain.errors import ArtifactContractError, DomainError
from redstack.pipelines.offline.graph import OfflineExecutionGraph
from redstack.pipelines.offline.registry import (
    OfflineArtifactRegistry,
    ValidationOutcome,
)

if TYPE_CHECKING:
    from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "StageStatus",
    "ArtifactPayload",
    "StageResult",
    "StageCallable",
    "StageReceipt",
    "OfflinePipelineRunner",
    "RunnerError",
)


class RunnerError(DomainError):
    """A runner-level orchestration failure (checkpoint corruption, IO)."""


class StageStatus(StrEnum):
    """The terminal disposition of a stage in a single ``run()`` invocation."""

    EXECUTED = "executed"
    SKIPPED = "skipped"
    QUARANTINED = "quarantined"
    FAILED = "failed"


# An artifact payload is the parsed, registry-validatable view of one produced
# artifact: for JSON/YAML the decoded mapping; for npy/parquet the header/schema
# metadata the stage reports. The runner never re-reads bytes to validate —
# stages hand back exactly what the registry validator inspects, plus the raw
# bytes' sha256 the stage already computed when it wrote the artifact.
@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    """One artifact produced by a stage, ready for registry validation + hashing.

    Attributes:
        key: The registry key this artifact satisfies.
        sha256: The hex sha256 the stage computed over the written bytes (the
            value that lands in ``MANIFEST.json``). The runner trusts the stage's
            hash for the receipt; O17 re-streams it during packaging.
        bytes_written: Size on disk, recorded for the manifest/report.
        validation_view: The structural payload the registry validator inspects
            (decoded mapping or extracted npy/parquet metadata).
    """

    key: str
    sha256: str
    bytes_written: int
    validation_view: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StageResult:
    """What a stage callable returns: its produced artifacts + metrics.

    Attributes:
        artifacts: The artifacts the stage produced this run, each registry-keyed.
        metrics: Deterministic stage metrics for the build report (census stats,
            honeypot counts, NDCG, cluster quality, …). Must be JSON-serializable
            and reproducible; excludes wall-clock.
    """

    artifacts: tuple[ArtifactPayload, ...]
    metrics: Mapping[str, object] = field(default_factory=dict)


class StageCallable(Protocol):
    """The injected stage contract the runner orchestrates (Part 1).

    A stage is ``f(ctx, upstream) -> StageResult``: pure given the context's
    ports + injected entropy/clock. ``upstream`` exposes the receipts of the
    stage's dependencies so a stage can read its inputs' identities (hashes)
    without the runner knowing stage internals. ``stage_version`` is the stage's
    own semver, part of its staleness key — bump it to force recompute on a
    behavior change.
    """

    @property
    def stage_id(self) -> str: ...

    @property
    def stage_version(self) -> str: ...

    def __call__(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class StageReceipt:
   
    stage_id: str
    stage_version: str
    staleness_key: str
    artifact_hashes: Mapping[str, str]
    artifact_bytes: Mapping[str, int]
    metrics: Mapping[str, object] = field(default_factory=dict)
    status: StageStatus = StageStatus.EXECUTED
    wall_ms: float = 0.0

    def reproducible_view(self) -> dict[str, object]:
        """Return the deterministic subset for checkpoint serialization."""
        return {
            "stage_id": self.stage_id,
            "stage_version": self.stage_version,
            "staleness_key": self.staleness_key,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "artifact_bytes": dict(sorted(self.artifact_bytes.items())),
            "metrics": _canonical_jsonable(self.metrics),
        }


def _canonical_jsonable(value: object) -> object:
    """Round-trip ``value`` through canonical JSON to normalize for hashing."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _sha256_hex(text: str) -> str:
    """Return the hex sha256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


MonotonicClock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class OfflinePipelineRunner:
    """Executes an offline plan with resume, checkpointing, and quarantine.

    Stateless across invocations except for the on-disk checkpoint store under
    ``ctx.checkpoints_root``; two runs with identical inputs produce identical
    receipts' reproducible views (the audit ``wall_ms`` differs and is ignored).

    Attributes:
        graph: The dependency DAG providing topo order + downstream closure.
        registry: The artifact catalog; every produced artifact is validated
            against it before its receipt is written.
        continue_on_error: If ``True``, a *non-critical* stage failure is
            flagged and the build continues; a critical failure always aborts.
        clock: Injected monotonic clock for stage timing (audit only).
            Defaults to ``time.perf_counter``, the highest-resolution monotonic
            clock available, consistent with :class:`~redstack.observability
            .timing.StageTimer` and the online pipeline's per-stage timing.
    """

    graph: OfflineExecutionGraph
    registry: OfflineArtifactRegistry
    continue_on_error: bool = False
    clock: MonotonicClock = time.perf_counter

    # ------------------------------------------------------------------ #
    # Staleness                                                          #
    # ------------------------------------------------------------------ #
    def _staleness_key(
        self,
        stage: StageCallable,
        upstream: Mapping[str, StageReceipt],
        config_hash: str,
    ) -> str:
        """Compute ``hash(input_hashes + stage_version + config_slice)`` (Part 11).

        Input hashes are the produced-artifact sha256s of *this stage's direct
        dependencies*, taken in deterministic id order, so the key is stable and
        order-independent. The ``config_slice`` is the context's canonical
        ``config_hash`` — a config change invalidates dependent stages.
        """
        parts: list[str] = [stage.stage_id, stage.stage_version, config_hash]
        for dep_id in sorted(self.graph.dependencies(stage.stage_id)):
            receipt = upstream.get(dep_id)
            if receipt is None:
                # Dependency produced no receipt (e.g. skipped non-critical
                # branch): contribute a stable sentinel so the key still differs
                # from a present-dependency key.
                parts.append(f"{dep_id}=∅")
                continue
            for key, digest in sorted(receipt.artifact_hashes.items()):
                parts.append(f"{dep_id}:{key}={digest}")
        return _sha256_hex("\u0000".join(parts))

    # ------------------------------------------------------------------ #
    # Checkpoint persistence                                             #
    # ------------------------------------------------------------------ #
    def _checkpoint_path(self, ctx: OfflinePipelineContext, stage_id: str) -> Path:
        return ctx.checkpoints_root / f"{stage_id}.receipt.json"

    def _load_checkpoint(
        self, ctx: OfflinePipelineContext, stage_id: str
    ) -> StageReceipt | None:
        """Load a persisted receipt, or ``None`` if absent/corrupt.

        Checkpoint corruption is treated as "recompute" (Part 11: "checkpoint
        corruption → recompute"), not a hard failure: a malformed sidecar simply
        yields ``None`` so the stage re-runs.
        """
        path = self._checkpoint_path(ctx, stage_id)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return StageReceipt(
                stage_id=str(raw["stage_id"]),
                stage_version=str(raw["stage_version"]),
                staleness_key=str(raw["staleness_key"]),
                artifact_hashes=dict(raw["artifact_hashes"]),
                artifact_bytes=dict(raw["artifact_bytes"]),
                metrics=dict(raw.get("metrics", {})),
                status=StageStatus.SKIPPED,
                wall_ms=0.0,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _persist_checkpoint(
        self, ctx: OfflinePipelineContext, receipt: StageReceipt
    ) -> None:
        """Atomically persist a receipt's reproducible view as a sidecar."""
        ctx.checkpoints_root.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(ctx, receipt.stage_id)
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(
            receipt.reproducible_view(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            msg = f"failed to persist checkpoint for {receipt.stage_id!r}: {exc}"
            raise RunnerError(msg) from exc

    # ------------------------------------------------------------------ #
    # Validation + quarantine                                            #
    # ------------------------------------------------------------------ #
    def _validate_outputs(self, result: StageResult) -> None:
        """Validate every produced artifact against the registry; raise on breach.

        A validator rejection is an invariant breach (Offline-testing §8:
        "validated against the registry before manifesting"), so it raises
        ``ArtifactContractError`` naming the offending key + reason — the hard
        fail path. The happy path returns ``None``.
        """
        for payload in result.artifacts:
            outcome: ValidationOutcome = self.registry.validate(
                payload.key, payload.validation_view
            )
            if not outcome.ok:
                msg = (
                    f"artifact {payload.key!r} failed registry validation: "
                    f"{outcome.reason}"
                )
                raise ArtifactContractError(msg)

    def _quarantine(
        self, ctx: OfflinePipelineContext, stage_id: str
    ) -> None:
        """Move a failed stage's partial outputs out of the manifested tree.

        Best-effort: any artifact whose relative path is owned by ``stage_id`` is
        moved under ``quarantine_root/<stage_id>/`` so it can never be hashed
        into ``MANIFEST.json`` (Part 11: "partial/failed outputs are
        quarantined"). Missing files are ignored — a stage may have failed before
        writing anything.
        """
        owned = [
            spec
            for spec in self.registry.specs
            if stage_id in spec.owner_stages
        ]
        if not owned:
            return
        dest_root = ctx.quarantine_root / stage_id
        dest_root.mkdir(parents=True, exist_ok=True)
        for spec in owned:
            src = ctx.artifacts_root / spec.relative_path
            if not src.exists():
                continue
            dest = dest_root / Path(spec.relative_path).name
            try:
                shutil.move(str(src), str(dest))
            except OSError:
                # Quarantine is defence-in-depth; never let it mask the original
                # stage error. A failed move leaves the partial file, which O17's
                # required-key/coherence pass will still reject.
                continue

    # ------------------------------------------------------------------ #
    # Plan + run                                                         #
    # ------------------------------------------------------------------ #
    def plan(self, only: Sequence[str] | None = None) -> tuple[str, ...]:
        """Return the deterministic execution order, optionally restricted.

        With ``only`` given, the plan is the topo-ordered downstream closure of
        those stages (their transitive dependents), so a forced re-run of an
        upstream stage drags its closure (Part 11). Without it, the full topo
        order is returned. Cycle detection already happened at graph construction.
        """
        full = self.graph.topo_order()
        if only is None:
            return full
        closure = self.graph.downstream_closure(only)
        return tuple(sid for sid in full if sid in closure)

    def run(
        self,
        ctx: OfflinePipelineContext,
        stages: Mapping[str, StageCallable],
        *,
        force: Sequence[str] | None = None,
    ) -> dict[str, StageReceipt]:
        """Execute the plan, skipping up-to-date stages, and return receipts.

        Args:
            ctx: The immutable build context (ports + roots + provenance).
            stages: ``stage_id -> callable`` for every stage in the plan. A plan
                stage absent from this mapping is a programming error.
            force: Stage ids to recompute regardless of checkpoint freshness;
                their downstream closure is forced stale too.

        Returns:
            ``stage_id -> StageReceipt`` for every planned stage (executed,
            skipped, or — under ``continue_on_error`` — flagged failed).

        Raises:
            RunnerError: a planned stage has no registered callable, or
                checkpoint IO failed.
            ArtifactContractError: a produced artifact failed registry validation.
            Exception: re-raised from a *critical* stage after quarantine.
        """
        order = self.plan()
        forced = self.graph.downstream_closure(force) if force else frozenset()
        receipts: dict[str, StageReceipt] = {}
        reran: set[str] = set()

        for stage_id in order:
            stage = stages.get(stage_id)
            if stage is None:
                msg = f"no stage callable registered for planned stage {stage_id!r}"
                raise RunnerError(msg)

            upstream = {
                dep: receipts[dep]
                for dep in self.graph.dependencies(stage_id)
                if dep in receipts
            }
            staleness_key = self._staleness_key(stage, upstream, ctx.config_hash)
            checkpoint = self._load_checkpoint(ctx, stage_id)

            upstream_reran = any(
                dep in reran for dep in self.graph.dependencies(stage_id)
            )
            is_forced = stage_id in forced
            is_fresh = (
                checkpoint is not None
                and checkpoint.staleness_key == staleness_key
                and checkpoint.stage_version == stage.stage_version
                and not upstream_reran
                and not is_forced
            )

            if is_fresh and checkpoint is not None:
                receipts[stage_id] = checkpoint  # status == SKIPPED
                continue

            receipts[stage_id] = self._execute_stage(
                ctx, stage, upstream, staleness_key
            )
            if receipts[stage_id].status is StageStatus.EXECUTED:
                reran.add(stage_id)

        return receipts

    def _execute_stage(
        self,
        ctx: OfflinePipelineContext,
        stage: StageCallable,
        upstream: Mapping[str, StageReceipt],
        staleness_key: str,
    ) -> StageReceipt:
        """Run one stage, validate + checkpoint on success, quarantine on failure.

        Encapsulates the per-stage failure policy: validator breach and critical
        stage exceptions propagate (after quarantine); a non-critical failure
        under ``continue_on_error`` is captured as a ``FAILED`` receipt and the
        build continues.
        """
        started = self.clock()
        try:
            result = stage(ctx, upstream)
            self._validate_outputs(result)
        except ArtifactContractError:
            self._quarantine(ctx, stage.stage_id)
            raise
        except Exception:
            self._quarantine(ctx, stage.stage_id)
            node = self.graph.node(stage.stage_id)
            if node.critical or not self.continue_on_error:
                raise
            return StageReceipt(
                stage_id=stage.stage_id,
                stage_version=stage.stage_version,
                staleness_key=staleness_key,
                artifact_hashes={},
                artifact_bytes={},
                metrics={},
                status=StageStatus.FAILED,
                wall_ms=(self.clock() - started) * 1000.0,
            )

        receipt = StageReceipt(
            stage_id=stage.stage_id,
            stage_version=stage.stage_version,
            staleness_key=staleness_key,
            artifact_hashes={a.key: a.sha256 for a in result.artifacts},
            artifact_bytes={a.key: a.bytes_written for a in result.artifacts},
            metrics=result.metrics,
            status=StageStatus.EXECUTED,
            wall_ms=(self.clock() - started) * 1000.0,
        )
        self._persist_checkpoint(ctx, receipt)
        return receipt