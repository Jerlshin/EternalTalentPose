"""Behavioral port fakes (Testing Strategy §6/§16).

Owner: tests/fixtures (shared, versioned ground truth).
Allowed imports: stdlib, numpy, ``redstack.domain``, ``redstack.ports``.

Per §6 ("Port Contract Testing"), every fake here is a *behavioral* in-memory
implementation of its port — not a ``unittest.mock`` — so the shared contract
suite in ``tests/contract/`` can run it through the exact same assertions the
real adapter faces. A fake intentionally re-implements adapter-equivalent
logic from scratch (rather than delegating to the real adapter) so that a
divergence between the two is something the contract suite can actually
catch.

One fake per port (§6 table):

================================  ===========================
Port                              Fake
================================  ===========================
``CandidateSourcePort``           ``ListCandidateSource``
``ArtifactStorePort``             ``InMemoryArtifactStore``
``EmbeddingModelPort``            ``StubEmbeddingModel``
``SemanticVectorStorePort``       ``InMemoryVectorStore``
``SubmissionSinkPort``            ``CapturingSubmissionSink``
``RunReportSinkPort``             ``CapturingRunReportSink``
``DeterministicEntropyPort``      ``FixedEntropy``
================================  ===========================
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from redstack.domain.errors import ArtifactContractError
from redstack.domain.ids import CandidateId
from redstack.ports._types import (
    ArtifactEntry,
    ArtifactKey,
    ArtifactKind,
    ArtifactLocator,
    BulkVectorResult,
    EmbeddingManifest,
    FloatMatrix,
    FloatVector,
    Manifest,
    ReportReceipt,
    RunReport,
    SourceMalformed,
    SourceOk,
    SourceRecord,
    SubmissionReceipt,
)
from redstack.ports.artifact_store import ManifestError
from redstack.ports.candidate_source import CandidateSourceError
from redstack.ports.embedding import EmbeddingError
from redstack.ports.rng import EntropyDisabledError
from redstack.ports.run_report_sink import ReportWriteError
from redstack.ports.semantic_index import VectorStoreError
from redstack.ports.submission_sink import SubmissionContractError

if TYPE_CHECKING:
    from redstack.domain.ranking import Ranking
    from redstack.ports.artifact_store import ArtifactStorePort
    from redstack.ports.candidate_source import CandidateSourcePort
    from redstack.ports.embedding import EmbeddingModelPort
    from redstack.ports.rng import DeterministicEntropyPort
    from redstack.ports.run_report_sink import RunReportSinkPort
    from redstack.ports.semantic_index import SemanticVectorStorePort
    from redstack.ports.submission_sink import SubmissionSinkPort

#: The frozen, validator-mandated submission column order (mirrors the adapter).
_HEADER: Final[tuple[str, str, str, str]] = (
    "candidate_id",
    "rank",
    "score",
    "reasoning",
)
#: Width (bytes) of the sha256 prefix folded into a numpy seed (64-bit).
_SUBSEED_BYTES: Final[int] = 8


# --------------------------------------------------------------------------- #
# CandidateSourcePort — ListCandidateSource.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BadLine:
    """Marks one fixture line as malformed (``SourceMalformed``), not raised."""

    error: str


class ListCandidateSource:
    """In-memory, order-preserving ``CandidateSourcePort`` fake.

    ``lines`` represents physical lines in file order: a ``Mapping`` decodes to
    ``SourceOk``; a ``BadLine`` decodes to ``SourceMalformed``; ``None``
    represents a blank/whitespace-only line and is skipped, consuming a
    ``line_no`` but no ``source_index`` — matching
    ``JsonlCandidateSourceAdapter`` exactly.
    """

    __slots__ = ("_count", "_lines", "_open_error")

    def __init__(
        self,
        lines: Sequence[Mapping[str, object] | BadLine | None],
        *,
        open_error: Exception | None = None,
        count: int | None = None,
    ) -> None:
        self._lines: Final[tuple[Mapping[str, object] | BadLine | None, ...]] = tuple(
            lines
        )
        self._open_error: Final[Exception | None] = open_error
        self._count: Final[int | None] = count

    def stream(self) -> Iterator[SourceRecord]:
        """Yield records lazily in file order; a fresh independent pass each call.

        Raises:
            CandidateSourceError: when constructed with ``open_error``.
        """
        if self._open_error is not None:
            raise CandidateSourceError(str(self._open_error)) from self._open_error

        source_index = 0
        for line_no, item in enumerate(self._lines, start=1):
            if item is None:
                continue
            if isinstance(item, BadLine):
                yield SourceMalformed(line_no=line_no, error=item.error)
                continue
            yield SourceOk(raw=dict(item), line_no=line_no, source_index=source_index)
            source_index += 1

    def count(self) -> int | None:
        """Return the configured count, or ``None`` if not supplied."""
        return self._count


# --------------------------------------------------------------------------- #
# ArtifactStorePort — InMemoryArtifactStore.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _StoredArtifact:
    """One artifact's verified record, canonical bytes, and decoded value."""

    entry: ArtifactEntry
    content: bytes
    value: object


def _canonical_bytes(value: object) -> bytes:
    """Canonicalize a fixture value to the bytes the store hashes and verifies."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return np.asarray(value, dtype=np.float32).tobytes()


class InMemoryArtifactStore:
    """Hash-verified, in-memory ``ArtifactStorePort`` fake.

    Built via :meth:`build` from plain Python values (``bytes``/``str``/
    ``Mapping``/array-like); each load re-verifies the stored content's
    sha256 against its recorded manifest entry, exactly mirroring
    ``FilesystemArtifactStoreAdapter``'s verify-on-load discipline.
    :meth:`tamper` and :meth:`corrupt_manifest` are test-only hooks for
    exercising the corruption paths (§7) without a real filesystem.
    """

    __slots__ = ("_manifest", "_store")

    def __init__(
        self,
        manifest: Manifest,
        store: Mapping[ArtifactKey, _StoredArtifact],
    ) -> None:
        self._manifest: Manifest = manifest
        self._store: dict[ArtifactKey, _StoredArtifact] = dict(store)

    @staticmethod
    def _self_hash(
        entries: Mapping[ArtifactKey, ArtifactEntry],
        *,
        manifest_schema_version: str,
        builder_version: str,
        layout_version: str,
        embedding_model_id: str,
        embedding_dim: int,
    ) -> str:
        """Recompute the manifest self-hash from its current entries/metadata."""
        canonical = json.dumps(
            {
                "manifest_schema_version": manifest_schema_version,
                "builder_version": builder_version,
                "layout_version": layout_version,
                "embedding": {"model_id": embedding_model_id, "dim": embedding_dim},
                "artifacts": {
                    str(key): {
                        "path": entry.path,
                        "sha256": entry.sha256,
                        "size_bytes": entry.size_bytes,
                        "schema_version": entry.schema_version,
                        "kind": entry.kind,
                    }
                    for key, entry in sorted(entries.items())
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def build(
        cls,
        artifacts: Mapping[str, tuple[ArtifactKind, object]],
        *,
        manifest_schema_version: str = "1.0",
        builder_version: str = "test",
        layout_version: str = "1",
        embedding_model_id: str = "stub-embedder",
        embedding_dim: int = 8,
    ) -> InMemoryArtifactStore:
        """Build a self-consistent store + manifest from ``{key: (kind, value)}``."""
        entries: dict[ArtifactKey, ArtifactEntry] = {}
        store: dict[ArtifactKey, _StoredArtifact] = {}
        for raw_key, (kind, value) in artifacts.items():
            key = ArtifactKey(raw_key)
            content = _canonical_bytes(value)
            sha256 = hashlib.sha256(content).hexdigest()
            entry = ArtifactEntry(
                key=key,
                path=f"memory://{raw_key}",
                sha256=sha256,
                size_bytes=len(content),
                schema_version=manifest_schema_version,
                kind=kind,
            )
            entries[key] = entry
            store[key] = _StoredArtifact(entry=entry, content=content, value=value)

        manifest_sha256 = cls._self_hash(
            entries,
            manifest_schema_version=manifest_schema_version,
            builder_version=builder_version,
            layout_version=layout_version,
            embedding_model_id=embedding_model_id,
            embedding_dim=embedding_dim,
        )
        manifest = Manifest(
            manifest_schema_version=manifest_schema_version,
            builder_version=builder_version,
            created_at="1970-01-01T00:00:00Z",
            embedding=EmbeddingManifest(model_id=embedding_model_id, dim=embedding_dim),
            layout_version=layout_version,
            artifacts=entries,
            manifest_sha256=manifest_sha256,
        )
        return cls(manifest, store)

    def manifest(self) -> Manifest:
        """Return the manifest after recomputing and checking its self-hash.

        Raises:
            ManifestError: the self-hash no longer matches (see
                :meth:`corrupt_manifest`).
        """
        m = self._manifest
        expected = self._self_hash(
            m.artifacts,
            manifest_schema_version=m.manifest_schema_version,
            builder_version=m.builder_version,
            layout_version=m.layout_version,
            embedding_model_id=m.embedding.model_id,
            embedding_dim=m.embedding.dim,
        )
        if expected != m.manifest_sha256:
            raise ManifestError("manifest self-hash mismatch")
        return m

    def verify_all(self) -> None:
        """Eagerly verify the manifest self-hash and every artifact's sha256."""
        self.manifest()
        for key in self._store:
            self._verified(key)

    def _verified(self, key: ArtifactKey) -> _StoredArtifact:
        """Look up ``key`` and verify its content's sha256 before returning it."""
        try:
            artifact = self._store[key]
        except KeyError:
            raise ArtifactContractError(f"missing artifact key: {key!s}") from None
        actual = hashlib.sha256(artifact.content).hexdigest()
        if actual != artifact.entry.sha256:
            raise ArtifactContractError(
                f"sha256 mismatch for {key!s}: "
                f"expected {artifact.entry.sha256}, got {actual}"
            )
        return artifact

    def load_bytes(self, key: ArtifactKey) -> bytes:
        return self._verified(key).content

    def load_text(self, key: ArtifactKey) -> str:
        artifact = self._verified(key)
        if isinstance(artifact.value, str):
            return artifact.value
        return artifact.content.decode("utf-8")

    def load_json(self, key: ArtifactKey) -> Mapping[str, object]:
        artifact = self._verified(key)
        if not isinstance(artifact.value, Mapping):
            raise ArtifactContractError(f"artifact {key!s} is not a JSON object")
        return artifact.value

    def load_npy(self, key: ArtifactKey) -> npt.NDArray[np.float32]:
        artifact = self._verified(key)
        try:
            array = np.asarray(artifact.value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ArtifactContractError(
                f"artifact {key!s} is not array-like: {exc}"
            ) from exc
        array = array.copy()
        array.setflags(write=False)
        return array

    def locate(self, key: ArtifactKey) -> ArtifactLocator:
        artifact = self._verified(key)
        return ArtifactLocator(key=key, verified=True, opaque_handle=artifact.content)

    def tamper(self, key: str) -> None:
        """Test-only: flip a byte of ``key``'s stored content, simulating corruption."""
        artifact_key = ArtifactKey(key)
        artifact = self._store[artifact_key]
        mutated = bytearray(artifact.content) or bytearray(b"\x00")
        mutated[0] ^= 0xFF
        self._store[artifact_key] = replace(artifact, content=bytes(mutated))

    def corrupt_manifest(self) -> None:
        """Test-only: invalidate the manifest self-hash."""
        self._manifest = replace(self._manifest, manifest_sha256="0" * 64)


# --------------------------------------------------------------------------- #
# EmbeddingModelPort — StubEmbeddingModel.
# --------------------------------------------------------------------------- #
class StubEmbeddingModel:
    """Deterministic, hash-derived ``EmbeddingModelPort`` fake (§6).

    Each text's vector is derived from a sha256 of the text seeding a PCG64
    generator, then L2-normalized — reproducible, never a model download,
    never an all-zero vector (the contract's "never mask failure" rule).
    """

    __slots__ = ("_dim", "_fail_on", "_model_id", "encoded_texts")

    def __init__(
        self,
        *,
        dim: int = 8,
        model_id: str = "stub-embedder-v1",
        fail_on: Iterable[str] = (),
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim: Final[int] = dim
        self._model_id: Final[str] = model_id
        self._fail_on: Final[frozenset[str]] = frozenset(fail_on)
        #: every text ever handed to :meth:`encode`, in call order -- lets a
        #: test assert the (expensive) encode path was never reached for a
        #: given candidate (e.g. one skipped via a pre-embedding hard gate).
        self.encoded_texts: list[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def encode(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> FloatMatrix:
        """Encode ``texts`` to unit-norm rows, independent of ``batch_size``.

        Raises:
            EmbeddingError: ``text`` is one of the configured ``fail_on`` texts.
        """
        self.encoded_texts.extend(texts)
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows = [self._encode_one(text) for text in texts]
        return np.stack(rows).astype(np.float32)

    def _encode_one(self, text: str) -> npt.NDArray[np.float32]:
        if text in self._fail_on:
            raise EmbeddingError(f"stub embedding failure for text: {text!r}")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:_SUBSEED_BYTES], "big")
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector = np.ones(self._dim, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
        return (vector / norm).astype(np.float32)


# --------------------------------------------------------------------------- #
# SemanticVectorStorePort — InMemoryVectorStore.
# --------------------------------------------------------------------------- #
class InMemoryVectorStore:
    """Read-only, dict-backed ``SemanticVectorStorePort`` fake.

    Missing ids are reported as ``None``/``missing`` (never raised); structural
    corruption (:meth:`corrupt`) makes :meth:`view_all` raise
    ``VectorStoreError``, mirroring the real mmap'd store's failure mode.
    """

    __slots__ = ("_corrupted", "_dim", "_vectors")

    def __init__(
        self,
        dim: int,
        vectors: Mapping[CandidateId, npt.ArrayLike] | None = None,
        *,
        corrupted: bool = False,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim: Final[int] = dim
        self._corrupted = corrupted
        self._vectors: dict[CandidateId, npt.NDArray[np.float32]] = {}
        for cid, vec in (vectors or {}).items():
            self._add(cid, vec)

    def _add(self, cid: CandidateId, vec: npt.ArrayLike) -> None:
        array = np.asarray(vec, dtype=np.float32)
        if array.shape != (self._dim,):
            raise VectorStoreError(
                f"vector for {cid!s} has shape {array.shape}, expected ({self._dim},)"
            )
        array = array.copy()
        array.setflags(write=False)
        self._vectors[cid] = array

    @property
    def dim(self) -> int:
        return self._dim

    def contains(self, cid: CandidateId) -> bool:
        return cid in self._vectors

    def get(self, cid: CandidateId) -> FloatVector | None:
        return self._vectors.get(cid)

    def get_many(self, cids: Sequence[CandidateId]) -> BulkVectorResult:
        found: list[CandidateId] = []
        missing: list[CandidateId] = []
        rows: list[npt.NDArray[np.float32]] = []
        for cid in cids:
            vector = self._vectors.get(cid)
            if vector is None:
                missing.append(cid)
            else:
                found.append(cid)
                rows.append(vector)
        matrix = (
            np.stack(rows).astype(np.float32)
            if rows
            else np.zeros((0, self._dim), dtype=np.float32)
        )
        matrix.setflags(write=False)
        return BulkVectorResult(
            vectors=matrix, found=tuple(found), missing=tuple(missing)
        )

    def view_all(self) -> tuple[FloatMatrix, Sequence[CandidateId]]:
        """Return the full matrix and its aligned id order.

        Raises:
            VectorStoreError: the store was marked corrupt via :meth:`corrupt`.
        """
        if self._corrupted:
            raise VectorStoreError("vector store is corrupt")
        id_order = tuple(self._vectors.keys())
        matrix = (
            np.stack([self._vectors[cid] for cid in id_order]).astype(np.float32)
            if id_order
            else np.zeros((0, self._dim), dtype=np.float32)
        )
        matrix.setflags(write=False)
        return matrix, id_order

    def corrupt(self) -> None:
        """Test-only: mark the store corrupt so :meth:`view_all` raises."""
        self._corrupted = True


# --------------------------------------------------------------------------- #
# SubmissionSinkPort — CapturingSubmissionSink.
# --------------------------------------------------------------------------- #
class CapturingSubmissionSink:
    """In-memory ``SubmissionSinkPort`` fake — captures bytes instead of a file.

    Mirrors ``CsvSubmissionSinkAdapter`` row-for-row: same header, same fixed
    score precision, same RFC-4180 minimal quoting, the same
    re-assert-invariants-at-emitted-precision defence before "writing".
    """

    __slots__ = ("_score_decimals", "writes")

    def __init__(self, *, score_decimals: int = 6) -> None:
        if score_decimals < 0:
            raise SubmissionContractError(
                f"score_decimals must be non-negative, got {score_decimals}"
            )
        self._score_decimals: Final[int] = score_decimals
        self.writes: list[bytes] = []

    @property
    def score_decimals(self) -> int:
        return self._score_decimals

    @property
    def last_csv_text(self) -> str | None:
        """The most recently captured CSV, decoded — or ``None`` if unwritten."""
        return self.writes[-1].decode("utf-8") if self.writes else None

    def _format_score(self, score: float) -> str:
        return f"{float(score):.{self._score_decimals}f}"

    def _build_rows(self, ranking: Ranking) -> list[tuple[str, str, str, str]]:
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
        return rows

    def _reassert_invariants(
        self, ranking: Ranking, rows: list[tuple[str, str, str, str]]
    ) -> None:
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

    def write(self, ranking: Ranking) -> SubmissionReceipt:
        """Capture ``ranking`` as bytes (no real IO) and return its receipt.

        Raises:
            SubmissionContractError: a serialized invariant would be violated
                at the emitted precision.
        """
        rows = self._build_rows(ranking)
        self._reassert_invariants(ranking, rows)

        data = self._serialize(rows)
        output_sha256 = hashlib.sha256(data).hexdigest()
        self.writes.append(data)

        return SubmissionReceipt(
            row_count=len(rows), bytes_written=len(data), output_sha256=output_sha256
        )


# --------------------------------------------------------------------------- #
# RunReportSinkPort — CapturingRunReportSink.
# --------------------------------------------------------------------------- #
class CapturingRunReportSink:
    """In-memory ``RunReportSinkPort`` fake — captures the JSON payload.

    Mirrors ``JsonRunReportSinkAdapter``: sorted keys, fixed float precision,
    the ``reproducible``/``audit``/``timings``/``budget`` regions kept distinct.
    """

    __slots__ = ("_float_decimals", "_report_schema_version", "writes")

    def __init__(
        self, *, report_schema_version: str = "1.0", float_decimals: int = 6
    ) -> None:
        if float_decimals < 0:
            raise ReportWriteError(
                f"float_decimals must be non-negative, got {float_decimals}"
            )
        self._report_schema_version: Final[str] = report_schema_version
        self._float_decimals: Final[int] = float_decimals
        self.writes: list[dict[str, object]] = []

    @property
    def last_report(self) -> Mapping[str, object] | None:
        """The most recently captured report payload, or ``None`` if unwritten."""
        return self.writes[-1] if self.writes else None

    def _fixed(self, value: float) -> float:
        return round(float(value), self._float_decimals)

    def _build(self, report: RunReport) -> dict[str, object]:
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

    def write(self, report: RunReport) -> ReportReceipt:
        """Capture ``report`` as a deterministic JSON-native payload."""
        payload = self._build(report)
        text = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        data = text.encode("utf-8")
        report_sha256 = hashlib.sha256(data).hexdigest()
        self.writes.append(payload)
        return ReportReceipt(bytes_written=len(data), report_sha256=report_sha256)


# --------------------------------------------------------------------------- #
# DeterministicEntropyPort — FixedEntropy.
# --------------------------------------------------------------------------- #
def _derive_subseed(seed: int, label: str) -> int:
    """Deterministically fold ``(seed, label)`` into a stable 64-bit sub-seed."""
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:_SUBSEED_BYTES], "big")


class FixedEntropy:
    """Seeded ``DeterministicEntropyPort`` fake; ``online()`` is RNG-disabled.

    Independent re-implementation of the ``(seed, label)`` -> sha256 -> sub-seed mix
    the real ``OfflineEntropy``/``OnlineEntropy`` adapters use, so the shared contract
    suite can catch any divergence between fake and adapter.
    """

    __slots__ = ("_as_of", "_rng_disabled", "_seed")

    def __init__(self, seed: int, as_of: date, *, rng_disabled: bool = False) -> None:
        self._seed: Final[int] = seed
        self._as_of: Final[date] = as_of
        self._rng_disabled: Final[bool] = rng_disabled

    @classmethod
    def online(cls, seed: int, as_of: date) -> FixedEntropy:
        """Build the RNG-disabled variant (``derive``/``numpy_generator`` raise)."""
        return cls(seed, as_of, rng_disabled=True)

    @property
    def seed(self) -> int:
        return self._seed

    def as_of(self) -> date:
        return self._as_of

    def derive(self, label: str) -> int:
        """Return a stable sub-seed for ``label``.

        Raises:
            EntropyDisabledError: this is the RNG-disabled (online) variant.
        """
        if self._rng_disabled:
            raise EntropyDisabledError(
                f"entropy is disabled; derive({label!r}) is not permitted"
            )
        return _derive_subseed(self._seed, label)

    def numpy_generator(self, label: str) -> np.random.Generator:
        """Return a seeded PCG64 ``Generator`` for ``label``.

        Raises:
            EntropyDisabledError: this is the RNG-disabled (online) variant.
        """
        if self._rng_disabled:
            raise EntropyDisabledError(
                f"entropy is disabled; numpy_generator({label!r}) is not permitted"
            )
        return np.random.default_rng(self.derive(label))


if TYPE_CHECKING:
    # Compile-time structural conformance to each frozen port surface.
    _CANDIDATE_SOURCE_CONFORMANCE: type[CandidateSourcePort] = ListCandidateSource
    _ARTIFACT_STORE_CONFORMANCE: type[ArtifactStorePort] = InMemoryArtifactStore
    _EMBEDDING_CONFORMANCE: type[EmbeddingModelPort] = StubEmbeddingModel
    _VECTOR_STORE_CONFORMANCE: type[SemanticVectorStorePort] = InMemoryVectorStore
    _SUBMISSION_SINK_CONFORMANCE: type[SubmissionSinkPort] = CapturingSubmissionSink
    _RUN_REPORT_SINK_CONFORMANCE: type[RunReportSinkPort] = CapturingRunReportSink
    _ENTROPY_CONFORMANCE: type[DeterministicEntropyPort] = FixedEntropy


__all__: tuple[str, ...] = (
    "BadLine",
    "CapturingRunReportSink",
    "CapturingSubmissionSink",
    "FixedEntropy",
    "InMemoryArtifactStore",
    "InMemoryVectorStore",
    "ListCandidateSource",
    "StubEmbeddingModel",
)
