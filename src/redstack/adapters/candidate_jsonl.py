"""``JsonlCandidateSourceAdapter`` — implements ``CandidateSourcePort`` (Adapters §1).

Owner layer: adapters (infrastructure — impure IO).
Allowed imports: stdlib ``io``/``gzip``/``json``/``pathlib``/``typing``; ``ports``.
Forbidden: ``engines``, ``pipelines``, any ML/network runtime.

Streams ``.jsonl`` / ``.jsonl.gz`` with constant per-record memory, preserving
file order exactly. The adapter performs UTF-8 (strict) + JSON structural decode
only; schema validation (``candidate_schema.json`` → ``RawCandidate``) lives
downstream in ``features.parsing``. A per-line decode failure is *data*
(``SourceMalformed``); an open/decompress/low-level IO failure *raises*
``CandidateSourceError``.

Determinism contract: identical file ⇒ identical record sequence, including
identical ``SourceMalformed`` placement; ``source_index`` is the enumeration
order over emitted records (blank/whitespace-only lines are skipped and consume
no index); ``line_no`` is the physical 1-based line number.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Final

from redstack.ports._types import RawMapping, SourceMalformed, SourceOk, SourceRecord
from redstack.ports.candidate_source import CandidateSourceError

#: gzip magic prefix; used for suffix-independent auto-detection of compression.
_GZIP_MAGIC: Final[bytes] = b"\x1f\x8b"


class JsonlCandidateSourceAdapter:
    """Lazy, single-pass, order-preserving JSONL/JSONL.GZ candidate source.

    Constructed by the pipeline composition root with a resolved path; never
    self-constructs. Each :meth:`stream` call opens a fresh handle and returns
    an independent one-pass iterator; the handle closes on iterator exhaustion
    or generator close. Not re-entrant; not shared across threads.
    """

    __slots__ = ("_path", "_gzipped")

    def __init__(self, path: Path, *, gzipped: bool | None = None) -> None:
        """Bind the adapter to a resolved input path.

        Args:
            path: Filesystem path to the ``.jsonl`` / ``.jsonl.gz`` source.
            gzipped: Force gzip handling (``True``) or plain (``False``); when
                ``None`` (default) compression is auto-detected from the
                ``.gz`` suffix and confirmed by the file's magic bytes.
        """
        self._path: Final[Path] = path
        self._gzipped: Final[bool] = (
            self._detect_gzip(path) if gzipped is None else gzipped
        )

    @property
    def path(self) -> Path:
        """The bound source path (audit-only)."""
        return self._path

    @property
    def gzipped(self) -> bool:
        """Whether the source is read through gzip decompression (audit-only)."""
        return self._gzipped

    @staticmethod
    def _detect_gzip(path: Path) -> bool:
        """Auto-detect gzip by ``.gz`` suffix, confirmed by the magic prefix.

        Suffix is authoritative for naming; the magic-byte read is a cheap guard
        against a mislabeled file. A missing/unreadable file is left for
        :meth:`stream` to surface as ``CandidateSourceError`` (detection is
        best-effort and never raises).
        """
        if path.suffix != ".gz":
            return False
        try:
            with open(path, "rb") as probe:
                return probe.read(2) == _GZIP_MAGIC
        except OSError:
            # Defer the IO failure to stream(); assume gzip per the suffix.
            return True

    def _open(self) -> gzip.GzipFile | io.BufferedReader:
        """Open a fresh binary handle, transparently gzip-wrapping when needed.

        Raises:
            CandidateSourceError: the source cannot be opened or decompressed.
        """
        try:
            if self._gzipped:
                return gzip.open(self._path, "rb")
            return open(self._path, "rb")
        except (OSError, EOFError) as exc:
            raise CandidateSourceError(
                f"cannot open source {self._path!s}: {exc}"
            ) from exc

    def stream(self) -> Iterator[SourceRecord]:
        """Yield records lazily in file order, one at a time.

        ``SourceOk`` carries the undecoded JSON object plus its ``source_index``
        (enumeration order over emitted records) and physical ``line_no``;
        ``SourceMalformed`` carries ``line_no`` and an error string. Calling
        ``stream`` again starts a fresh, independent pass.

        Raises:
            CandidateSourceError: the source cannot be opened/decompressed, or a
                low-level IO/decompression error occurs mid-stream.
        """
        handle = self._open()
        try:
            line_no = 0
            source_index = 0
            while True:
                try:
                    raw_line = handle.readline()
                except (OSError, EOFError) as exc:
                    raise CandidateSourceError(
                        f"IO error reading {self._path!s} at line {line_no + 1}: {exc}"
                    ) from exc
                if not raw_line:
                    break
                line_no += 1

                if not raw_line.strip():
                    # Blank / whitespace-only line: not a record (validator row
                    # semantics). Consumes a physical line, no source_index.
                    continue

                record = self._decode_line(raw_line, line_no, source_index)
                source_index += 1
                yield record
        finally:
            handle.close()

    @staticmethod
    def _decode_line(
        raw_line: bytes, line_no: int, source_index: int
    ) -> SourceRecord:
        """Decode one non-blank physical line into a ``SourceRecord``.

        UTF-8 strict, then ``json.loads``, then a top-level-object structural
        check. Any failure is reported as ``SourceMalformed`` (data, never an
        exception); the schema boundary is downstream.
        """
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            return SourceMalformed(line_no=line_no, error=f"utf-8 decode: {exc}")

        try:
            decoded: object = json.loads(text)
        except json.JSONDecodeError as exc:
            return SourceMalformed(line_no=line_no, error=f"json decode: {exc.msg}")

        if not isinstance(decoded, dict):
            return SourceMalformed(
                line_no=line_no,
                error=f"top-level JSON value is {type(decoded).__name__}, not object",
            )

        raw: RawMapping = {str(key): value for key, value in decoded.items()}
        return SourceOk(raw=raw, line_no=line_no, source_index=source_index)

    def count(self) -> int | None:
        """Return ``None``: the record count is unknown without a full pass.

        Counting would require streaming the entire source, which would violate
        the single-pass / O(1) contract, so the honest cheap answer is ``None``.
        """
        return None


if TYPE_CHECKING:
    from redstack.ports.candidate_source import CandidateSourcePort

    # Compile-time structural conformance to the frozen port surface.
    _PORT_CONFORMANCE: type[CandidateSourcePort] = JsonlCandidateSourceAdapter


__all__: tuple[str, ...] = ("JsonlCandidateSourceAdapter",)