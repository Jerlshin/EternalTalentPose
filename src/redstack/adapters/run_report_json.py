"""``JsonRunReportSinkAdapter`` — implements ``RunReportSinkPort`` (Adapters §7).

Owner layer: adapters (infrastructure — impure IO).
Allowed imports: stdlib ``json``/``io``/``hashlib``/``os``/``tempfile``/``pathlib``;
``ports``.
Forbidden: ``engines``, ``pipelines``, business logic, metric computation.

Persists a ``RunReport`` (the port-owned structural Protocol) as deterministic
JSON: sorted keys, fixed float formatting, the ``reproducible`` and ``audit``
regions explicitly separated. The ``reproducible`` block is byte-stable across
runs with identical inputs; the ``audit`` block (wall-clock, ``run_id``) is
serialized but excluded from the determinism contract. A
``report_schema_version`` travels with the report. The write is atomic (temp
file in the target directory → fsync → ``os.replace``); the receipt carries the
sha256 over the exact bytes written.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Final

from redstack.ports._types import ReportReceipt, RunReport
from redstack.ports.run_report_sink import ReportWriteError


class JsonRunReportSinkAdapter:
    """Single-use, deterministic JSON writer for a ``RunReport``.

    Constructed by the pipeline composition root with the output path; one
    :meth:`write` per run; no persistent handle.
    """

    __slots__ = ("_output_path", "_report_schema_version", "_float_decimals")

    def __init__(
        self,
        output_path: Path,
        *,
        report_schema_version: str = "1.0",
        float_decimals: int = 6,
    ) -> None:
        """Bind the sink to its output target and serialization precision.

        Args:
            output_path: Resolved ``run_report.json`` target path.
            report_schema_version: Version stamped into the report so consumers
                can check compatibility.
            float_decimals: Fixed rounding precision for all serialized floats;
                pins byte-stable output. Must be non-negative.
        """
        if float_decimals < 0:
            raise ReportWriteError(
                f"float_decimals must be non-negative, got {float_decimals}"
            )
        self._output_path: Final[Path] = output_path
        self._report_schema_version: Final[str] = report_schema_version
        self._float_decimals: Final[int] = float_decimals

    @property
    def output_path(self) -> Path:
        """The bound output path (audit-only)."""
        return self._output_path

    def _fixed(self, value: float) -> float:
        """Round a float to the fixed precision for deterministic output."""
        return round(float(value), self._float_decimals)

    def _build(self, report: RunReport) -> dict[str, object]:
        """Project the structural ``RunReport`` into a JSON-native dict.

        Accessing the Protocol's properties surfaces any structurally missing
        field as a programming error (``AttributeError``), per the contract.
        """
        repro = report.reproducible
        audit = report.audit
        budget = report.budget

        reproducible: dict[str, object] = {
            "code_version": repro.code_version,
            "config_hash": repro.config_hash,
            "manifest_hash": repro.manifest_hash,
            "artifact_hashes": dict(repro.artifact_hashes),
            "input_file_sha256": repro.input_file_sha256,
            "candidate_count": repro.candidate_count,
            "output_sha256": repro.output_sha256,
            "honeypot_count_top100": repro.honeypot_count_top100,
            "honeypot_rate": self._fixed(repro.honeypot_rate),
            "eligibility_summary": dict(repro.eligibility_summary),
            "score_distribution_digest": repro.score_distribution_digest,
        }
        audit_block: dict[str, object] = {
            "run_id": audit.run_id,
            "started_at": audit.started_at,
            "ended_at": audit.ended_at,
            "host_label": audit.host_label,
        }
        timings: dict[str, float] = {
            stage: self._fixed(ms) for stage, ms in report.timings.items()
        }
        budget_block: dict[str, object] = {
            "limit_seconds": self._fixed(budget.limit_seconds),
            "used_seconds": self._fixed(budget.used_seconds),
            "within_budget": budget.within_budget,
            "peak_rss_mb": self._fixed(budget.peak_rss_mb),
        }

        return {
            "report_schema_version": self._report_schema_version,
            "reproducible": reproducible,
            "audit": audit_block,
            "timings": timings,
            "budget": budget_block,
        }

    def _serialize(self, report: RunReport) -> bytes:
        """Serialize the report deterministically: sorted keys, compact, UTF-8."""
        payload = self._build(report)
        text = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return text.encode("utf-8")

    def _atomic_write(self, data: bytes) -> None:
        """Write ``data`` via temp file in the target directory + ``os.replace``.

        Raises:
            ReportWriteError: any IO failure; the temp file is removed so no
                partial report is left behind.
        """
        parent = self._output_path.parent
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=parent, prefix=".tmp_run_report_", suffix=".json"
            )
        except OSError as exc:
            raise ReportWriteError(
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
            raise ReportWriteError(
                f"cannot write run report to {self._output_path!s}: {exc}"
            ) from exc

    def write(self, report: RunReport) -> ReportReceipt:
        """Serialize ``report`` to JSON atomically and return a receipt.

        Raises:
            ReportWriteError: an IO error during the atomic write.
        """
        data = self._serialize(report)
        report_sha256 = hashlib.sha256(data).hexdigest()
        self._atomic_write(data)
        return ReportReceipt(bytes_written=len(data), report_sha256=report_sha256)


if TYPE_CHECKING:
    from redstack.ports.run_report_sink import RunReportSinkPort

    # Compile-time structural conformance to the frozen port surface.
    _PORT_CONFORMANCE: type[RunReportSinkPort] = JsonRunReportSinkAdapter


__all__: tuple[str, ...] = ("JsonRunReportSinkAdapter",)