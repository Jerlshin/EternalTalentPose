
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, NewType, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from redstack.domain.ids import CandidateId

# --------------------------------------------------------------------------- #
# Vector / mapping aliases.
# --------------------------------------------------------------------------- #
#: A single L2-normalized embedding row, shape ``(dim,)``, read-only.
type FloatVector = npt.NDArray[np.float32]
#: A stack of embedding rows, shape ``(n, dim)``; row order == input order.
type FloatMatrix = npt.NDArray[np.float32]
#: An undecoded candidate record (a JSON object), values left as ``object``.
type RawMapping = Mapping[str, object]

#: A manifest key, e.g. ``"embeddings/candidate_vectors"``.
ArtifactKey = NewType("ArtifactKey", str)

#: The on-disk encoding an artifact was packaged in.
type ArtifactKind = Literal["npy", "parquet", "json", "onnx", "yaml"]


# --------------------------------------------------------------------------- #
# Artifact-store DTOs.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ArtifactLocator:
    """A hash-verified handle for mmap consumers.

    ``opaque_handle`` is path-like but intentionally opaque to ``ports``: the
    adapter that produced it and the consumer that mmaps it agree on its
    concrete form out-of-band, so ``ports`` never assumes a filesystem.
    """

    key: ArtifactKey
    verified: bool
    opaque_handle: object


@dataclass(frozen=True, slots=True)
class EmbeddingManifest:
    """The embedding contract recorded in the manifest (``model_id`` + ``dim``)."""

    model_id: str
    dim: int


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """One artifact's manifest record.

    ``size_bytes`` mirrors the manifest's ``bytes`` field (renamed to avoid
    shadowing the ``bytes`` builtin).
    """

    key: ArtifactKey
    path: str
    sha256: str
    size_bytes: int
    schema_version: str
    kind: ArtifactKind


@dataclass(frozen=True, slots=True)
class Manifest:
    """The verified artifact registry (``MANIFEST.json``), §8.

    ``created_at`` is audit-only and excluded from the manifest self-hash.
    """

    manifest_schema_version: str
    builder_version: str
    created_at: str
    embedding: EmbeddingManifest
    layout_version: str
    artifacts: Mapping[ArtifactKey, ArtifactEntry]
    manifest_sha256: str


# --------------------------------------------------------------------------- #
# Candidate-source records (tagged union; the port reports, the pipeline decides).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SourceOk:
    """A decoded source line (the JSON object, not yet schema-validated)."""

    raw: RawMapping
    line_no: int
    source_index: int


@dataclass(frozen=True, slots=True)
class SourceMalformed:
    """A line that failed UTF-8/JSON structural decode; reported as data."""

    line_no: int
    error: str


#: A single streamed record: either decoded (``SourceOk``) or ``SourceMalformed``.
type SourceRecord = SourceOk | SourceMalformed


# --------------------------------------------------------------------------- #
# Vector-store bulk result.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BulkVectorResult:
    """Order-preserving bulk gather: ``found`` aligns row-for-row with ``vectors``.

    ``vectors`` is excluded from equality/hash (ndarray truthiness is ambiguous);
    callers compare ``found``/``missing`` and inspect ``vectors`` explicitly.
    """

    vectors: FloatMatrix = field(compare=False)
    found: tuple[CandidateId, ...]
    missing: tuple[CandidateId, ...]


# --------------------------------------------------------------------------- #
# Sink receipts.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    """Outcome of writing the submission CSV; feeds the Run Report."""

    row_count: int
    bytes_written: int
    output_sha256: str


@dataclass(frozen=True, slots=True)
class ReportReceipt:
    """Outcome of writing the run report JSON."""

    bytes_written: int
    report_sha256: str


# --------------------------------------------------------------------------- #
# Run report (port-owned structural Protocol, §13).
# --------------------------------------------------------------------------- #
@runtime_checkable
class ReproducibleBlock(Protocol):
    """The deterministic region — the only region compared by determinism tests."""

    @property
    def code_version(self) -> str: ...
    @property
    def config_hash(self) -> str: ...
    @property
    def manifest_hash(self) -> str: ...
    @property
    def artifact_hashes(self) -> Mapping[str, str]: ...
    @property
    def input_file_sha256(self) -> str: ...
    @property
    def candidate_count(self) -> int: ...
    @property
    def output_sha256(self) -> str: ...
    @property
    def honeypot_count_top100(self) -> int: ...
    @property
    def honeypot_rate(self) -> float: ...
    @property
    def eligibility_summary(self) -> Mapping[str, int]: ...
    @property
    def score_distribution_digest(self) -> str: ...


@runtime_checkable
class AuditBlock(Protocol):
    """Wall-clock / identity region; excluded from any reproducibility hash."""

    @property
    def run_id(self) -> str: ...
    @property
    def started_at(self) -> str: ...
    @property
    def ended_at(self) -> str: ...
    @property
    def host_label(self) -> str: ...


@runtime_checkable
class BudgetBlock(Protocol):
    """Budget accounting for the run."""

    @property
    def limit_seconds(self) -> float: ...
    @property
    def used_seconds(self) -> float: ...
    @property
    def within_budget(self) -> bool: ...
    @property
    def peak_rss_mb(self) -> float: ...


@runtime_checkable
class RunReport(Protocol):
    """Structural contract observability builds; persisted by ``RunReportSinkPort``."""

    @property
    def reproducible(self) -> ReproducibleBlock: ...
    @property
    def audit(self) -> AuditBlock: ...
    @property
    def timings(self) -> Mapping[str, float]: ...
    @property
    def budget(self) -> BudgetBlock: ...


__all__: tuple[str, ...] = (
    "ArtifactEntry",
    "ArtifactKey",
    "ArtifactKind",
    "ArtifactLocator",
    "AuditBlock",
    "BudgetBlock",
    "BulkVectorResult",
    "EmbeddingManifest",
    "FloatMatrix",
    "FloatVector",
    "Manifest",
    "RawMapping",
    "ReportReceipt",
    "ReproducibleBlock",
    "RunReport",
    "SourceMalformed",
    "SourceOk",
    "SourceRecord",
    "SubmissionReceipt",
)
