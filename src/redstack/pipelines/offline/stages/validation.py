

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from redstack.config.schema import MalformedRecordPolicy
from redstack.domain.errors import SchemaError
from redstack.features.parsing import validate as parse
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.ports._types import SourceMalformed, SourceOk

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "ValidationStage",
    "validation_stage",
)

#: Maximum rejections retained verbatim in the report (the rest are counted only).
_REJECT_LOG_CAP: Final[int] = 1000

#: Fraction of rejected records above which the build aborts (registry bound).
_MAX_REJECT_RATE: Final[float] = 0.01


class ValidationStage(OfflineStage):
    """O2 — stream the pool, validate to ``RawCandidate``, emit a reject report."""

    stage_id = "O2"
    stage_version = "1.0"

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        policy = self._resolve_policy(ctx)
        accepted = 0
        rejected = 0
        reject_log: list[dict[str, object]] = []

        for record in ctx.candidate_source.stream():
            if isinstance(record, SourceMalformed):
                rejected += 1
                self._log_reject(reject_log, record.line_no, f"malformed: {record.error}")
                if policy is MalformedRecordPolicy.ABORT:
                    msg = f"malformed record at line {record.line_no}: {record.error}"
                    raise SchemaError(msg)
                continue
            if not isinstance(record, SourceOk):
                continue
            outcome = self._validate_one(record.raw)
            if outcome is None:
                accepted += 1
                continue
            rejected += 1
            self._log_reject(reject_log, record.line_no, outcome)
            if policy is MalformedRecordPolicy.ABORT:
                msg = f"schema rejection at line {record.line_no}: {outcome}"
                raise SchemaError(msg)

        total = accepted + rejected
        reject_rate = (rejected / total) if total else 0.0
        if reject_rate > _MAX_REJECT_RATE:
            msg = (
                f"reject rate {reject_rate:.4f} exceeds bound {_MAX_REJECT_RATE:.4f} "
                f"({rejected}/{total})"
            )
            raise SchemaError(msg)

        report: dict[str, object] = {
            "accepted": accepted,
            "rejected": rejected,
            "total": total,
            "reject_rate": reject_rate,
            "reject_log_truncated": rejected > _REJECT_LOG_CAP,
            "reject_log": reject_log,
        }
        artifact = self.emit_json(ctx, "validation_report", report)
        metrics: dict[str, object] = {
            "accepted": accepted,
            "rejected": rejected,
            "reject_rate": reject_rate,
        }
        return StageResult(artifacts=(artifact,), metrics=metrics)

    @staticmethod
    def _resolve_policy(ctx: OfflinePipelineContext) -> MalformedRecordPolicy:
        """Resolve the malformed-record policy.

        The online block owns the explicit policy knob; offline builds default to
        ``ABORT`` (exactly 100K well-formed rows expected — Ports §11). If an
        online block is present in this config it is honored, else ABORT.
        """
        online = ctx.config.online
        if online is not None:
            return online.malformed_record_policy
        return MalformedRecordPolicy.ABORT

    @staticmethod
    def _validate_one(raw: Mapping[str, object]) -> str | None:
        """Parse one record; return ``None`` on accept or a reason string on reject.

        Only type/shape breaches reject (``SchemaError``); semantic contradictions
        pass through untouched for downstream integrity discovery.
        """
        try:
            parse(raw)
        except SchemaError as exc:
            return str(exc).splitlines()[0] if str(exc) else "schema violation"
        return None

    @staticmethod
    def _log_reject(log: list[dict[str, object]], line_no: int, reason: str) -> None:
        """Append a rejection to the capped log (kept bounded for O(1) memory)."""
        if len(log) < _REJECT_LOG_CAP:
            log.append({"line_no": line_no, "reason": reason})


def validation_stage() -> ValidationStage:
    """Factory: construct the O2 validation stage bound to the frozen registry."""
    return ValidationStage()