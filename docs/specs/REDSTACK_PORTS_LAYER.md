# REDSTACK v1.1 — Ports Layer Specification

**Scope:** `src/redstack/ports/` only. Repository architecture and Domain Layer are frozen; this document does not alter them. No implementation code — only the port contracts, detailed enough that adapter implementation can begin immediately.

**Ports Layer purity contract (inherited, restated):** `ports/` may import **only** stdlib `typing`, `domain/`, and `numpy` typing (for vector shapes). It may **not** import `adapters/`, `engines/`, `pipelines/`, `config/loader`, `observability/`, or any ML/IO runtime. Ports are `typing.Protocol`s plus small frozen input/output DTOs they own. They are the hexagon's edges: engines depend on these abstractions; adapters implement them; only `pipelines` (the composition root) instantiates an adapter and binds it to a port.

**Verdicts-vs-failures discipline (inherited):** a *missing candidate* or a *malformed input line* is **data** (returned as a value), not an exception. **Integrity/contract violations** (hash mismatch, wrong dimensionality, schema-version incompatibility, broken submission invariant) are **failures** and raise — because with no live leaderboard, silent degradation is undetectable and therefore unacceptable.

---

## §0. Shared port types (`ports/_types.py`)

Declared once, imported by the port modules. These are the only non-`domain` types the ports introduce.

| Alias / DTO | Definition | Notes |
|---|---|---|
| `FloatVector` | `npt.NDArray[np.float32]` shape `(dim,)` | read-only, L2-normalized |
| `FloatMatrix` | `npt.NDArray[np.float32]` shape `(n, dim)` | row order == input order |
| `RawMapping` | `Mapping[str, object]` | undecoded candidate record (JSON object) |
| `ArtifactKey` | `NewType('ArtifactKey', str)` | manifest key, e.g. `"embeddings/candidate_vectors"` |
| `ArtifactLocator` | frozen DTO: `key, verified: bool, opaque_handle` | hash-verified handle for mmap consumers; opaque (path-like) |
| `SourceRecord` | tagged union: `Ok(raw: RawMapping, line_no: int, source_index: int)` \| `Malformed(line_no: int, error: str)` | port reports; pipeline decides policy |
| `BulkVectorResult` | frozen DTO: `vectors: FloatMatrix, found: tuple[CandidateId,...], missing: tuple[CandidateId,...]` | order-preserving for `found` |
| `SubmissionReceipt` | frozen DTO: `row_count: int, bytes_written: int, output_sha256: str` | feeds Run Report |
| `ReportReceipt` | frozen DTO: `bytes_written: int, report_sha256: str` | — |
| `Manifest` | frozen DTO (see §8) | the verified artifact registry |
| `RunReport` | **port-owned structural Protocol** (see §13) | observability builds a conforming object |

Domain types referenced: `CandidateId, AnchorId, ArchetypeId, Score` (`domain.ids`), `RawCandidate` (`domain.source`), `Ranking` (`domain.ranking`), and the `DomainError` hierarchy (`domain.errors`), extended here with port-level errors (§ per port).

---

## §1. EmbeddingModelPort (`ports/embedding.py`)

- **Purpose:** turn already-composed text documents into normalized dense vectors. The single text→vector seam, used offline to *build* candidate/anchor vectors and online as a *fallback* for candidates absent from the vector store.
- **Responsibilities:** deterministic encoding; L2 normalization; batching transparently; exposing `dim` and a `model_id` for provenance.
- **Non-responsibilities:** does **not** store vectors (that's `SemanticVectorStorePort`); does **not** compute similarity; does **not** compose/template the input text (caller assembles the document); does **not** choose pooling (fixed by the model); does **not** access the network.
- **Method signatures:**
  ```
  class EmbeddingModelPort(Protocol):
      @property
      def dim(self) -> int: ...
      @property
      def model_id(self) -> str: ...
      def encode(self, texts: Sequence[str], *, batch_size: int | None = None) -> FloatMatrix: ...
  ```
- **Inputs:** a `Sequence[str]` of pre-composed documents; optional adapter-hint `batch_size` (does not affect results).
- **Outputs:** `FloatMatrix` `(len(texts), dim)`, `float32`, each row L2-normalized, **row order == input order**, array read-only.
- **Exceptions:** `EmbeddingError` (encode runtime failure). Empty string is valid input (encodes to the model's representation; never raises). Never returns zeros to mask failure.
- **Performance:** offline unbounded; online fallback is invoked only on store misses, so per-call latency must be low but aggregate volume is tiny in full reproduction. Must batch to keep CPU throughput high.
- **Thread-safety:** one session per instance; `encode` is safe to call sequentially; instances are **not** shared mutably across threads. onnxruntime session thread/op counts are pinned for determinism.
- **Determinism:** within a runtime, identical input ⇒ identical output bytes (fixed weights, pooling, normalization, `float32`, pinned threads, no sampling). **Cross-runtime** (sentence-transformers offline vs onnxruntime online) bitwise equality is **not** guaranteed; the contract guarantees same vector space and cosine agreement within ε (see §9).
- **Validation:** contract tests assert output shape, `float32` dtype, and per-row unit norm within ε.
- **Error-handling strategy:** raise `EmbeddingError`; the pipeline decides whether a fallback miss is fatal (in practice misses are near-zero in full reproduction).
- **Expected implementations:** `adapters/st_embedder.py` (offline, sentence-transformers — **offline-only, import-guarded**); `adapters/onnx_embedder.py` (online, onnxruntime, `CPUExecutionProvider`); `tests` `StubEmbeddingModel` (deterministic hash-derived vectors).
- **Dependency rules:** port → `domain.ids`, `_types`, numpy typing. Adapters → the ML runtime + `ArtifactStorePort` (onnx adapter loads `model/encoder.onnx`).

---

## §2. SemanticVectorStorePort (`ports/semantic_index.py`)

- **Purpose:** O(1) retrieval of precomputed candidate vectors by `CandidateId`, plus a read-only view of the full matrix for the columnar scoring path. This port is *why* R3 is "Semantic Lookup," not "encode."
- **Responsibilities:** id→vector lookup; bulk gather; full-matrix view aligned to a stable id order; report misses as data.
- **Non-responsibilities:** does **not** encode (that's `EmbeddingModelPort`); does **not** compute similarity or assign archetypes (`SemanticEngine` does, in numpy); does **not** verify artifact integrity (delegated to `ArtifactStorePort` at open time).
- **Method signatures:**
  ```
  class SemanticVectorStorePort(Protocol):
      @property
      def dim(self) -> int: ...
      def contains(self, cid: CandidateId) -> bool: ...
      def get(self, cid: CandidateId) -> FloatVector | None: ...
      def get_many(self, cids: Sequence[CandidateId]) -> BulkVectorResult: ...
      def view_all(self) -> tuple[FloatMatrix, Sequence[CandidateId]]: ...
  ```
- **Inputs:** `CandidateId`(s). `view_all` takes none.
- **Outputs:** read-only `float32` vectors/matrix, normalized (inherits the embedding norm guarantee). `get` → `(dim,)` or `None`; `get_many` → `BulkVectorResult` (order-preserving `found`, plus `missing`); `view_all` → `(matrix, id_order)` where `id_order[i]` labels row `i`.
- **Exceptions:** `VectorStoreError` (corrupt/unreadable store, dim mismatch). **A missing candidate is `None`/`missing`, never an exception.**
- **Performance:** `get` ~µs via an in-memory id→row-index map; `get_many` is a vectorized gather O(K); `view_all` returns a memory-mapped matrix (no copy). The id index (100K ids) loads fully (a few MB); vector bytes (≈100K×`dim`×4 ≈ 150 MB) stay mmap'd, resident only on touched pages. Comfortably within the 5-min budget.
- **Thread-safety:** read-only after open; concurrent reads safe (mmap). No online writes.
- **Determinism:** id→row mapping fixed by the artifact; `view_all` id order is the stored order; returned arrays read-only.
- **Validation:** at open, assert `dim` matches the embedding contract and the manifest; assert id uniqueness; assert `schema_version`.
- **Error-handling strategy:** structural corruption raises and aborts R0; misses flow to the encode fallback.
- **Expected implementations:** `adapters/vector_store_parquet.py` (pyarrow/`.npy` mmap + id index); `tests` `InMemoryVectorStore`.
- **Dependency rules:** port → `domain.ids`, `_types`, numpy. Adapter → pyarrow/numpy + `ArtifactStorePort.locate(...)`.

---

## §3. ArtifactStorePort (`ports/artifact_store.py`)

- **Purpose:** load **and integrity-verify** build artifacts by manifest key. The contract backbone between the offline build and the online run.
- **Responsibilities:** parse + self-verify the manifest; verify each artifact's sha256 (streaming, during load); expose typed loaders and a verified locator for mmap consumers; enforce schema-version compatibility (§8).
- **Non-responsibilities:** does **not** build/write artifacts (offline packaging does); does **not** interpret artifact semantics (returns bytes/arrays/locators); does **not** mmap itself (provides a verified locator the consumer mmaps).
- **Method signatures:**
  ```
  class ArtifactStorePort(Protocol):
      def manifest(self) -> Manifest: ...
      def verify_all(self) -> None: ...
      def load_bytes(self, key: ArtifactKey) -> bytes: ...
      def load_text(self, key: ArtifactKey) -> str: ...
      def load_json(self, key: ArtifactKey) -> Mapping[str, object]: ...
      def load_npy(self, key: ArtifactKey) -> npt.NDArray: ...
      def locate(self, key: ArtifactKey) -> ArtifactLocator: ...
  ```
- **Inputs:** manifest keys.
- **Outputs:** verified bytes / text / json / ndarray / `ArtifactLocator`. Every loader verifies sha256 against the manifest before returning.
- **Exceptions:** `ManifestError` (manifest missing/unparseable/self-hash fail); `ArtifactContractError` (missing key, sha256 mismatch, schema-version incompatible, cross-artifact incoherence — see §8). All are fatal at R0.
- **Performance:** verification uses **streaming sha256 while reading**, so integrity costs ~nothing on top of the load that would happen anyway. Hashing the large artifacts (vectors, onnx model) is counted in the R0 budget; design assumes total R0 < ~30 s.
- **Thread-safety:** read-only; concurrent loads safe; verification idempotent and result-cached per key.
- **Determinism:** same artifact bytes ⇒ same hashes; key→artifact mapping fixed by the manifest.
- **Validation:** manifest schema; per-key sha256; schema-version compatibility; cross-artifact coherence (§8).
- **Error-handling strategy:** **fail-fast, no degraded mode.** Any integrity/contract failure aborts the run with a precise reason (which key, expected vs actual).
- **Expected implementations:** `adapters/artifact_store_fs.py` (filesystem + `MANIFEST.json`); `tests` `InMemoryArtifactStore`.
- **Dependency rules:** port → `domain.errors`, `_types`. Adapter → filesystem, `hashlib`, `json`, numpy.

---

## §4. CandidateSourcePort (`ports/candidate_source.py`)

- **Purpose:** stream raw candidate records from the input (`.jsonl` / `.jsonl.gz`) with constant memory, preserving file order.
- **Responsibilities:** open/decompress transparently; decode each line as a JSON object; yield `SourceRecord`s lazily with a `source_index` and `line_no`; report per-line decode failures as data.
- **Non-responsibilities:** does **not** validate against `candidate_schema.json` (that's `features.parsing` → `RawCandidate`); does **not** decide skip-vs-abort policy on malformed lines (the pipeline does); does **not** reorder; does **not** type records into the domain.
- **Method signatures:**
  ```
  class CandidateSourcePort(Protocol):
      def stream(self) -> Iterator[SourceRecord]: ...
      def count(self) -> int | None: ...   # may be None if unknown without a pass
  ```
- **Inputs:** none per call (path/compression configured on the adapter).
- **Outputs:** a lazy `Iterator[SourceRecord]` in file order; `Ok` carries the undecoded `RawMapping` + `source_index` (enumeration order) + `line_no`; `Malformed` carries `line_no` + error.
- **Exceptions:** `CandidateSourceError` (cannot open/decompress, IO error). **Per-line bad JSON is a `Malformed` record, not an exception.**
- **Performance:** O(1) per-record memory (generator); single pass; gzip streamed; reads the 100K pool (≈52 MB gz / 465 MB raw) well within budget. Never materializes the whole file.
- **Thread-safety:** single-consumer iterator; not re-entrant; call `stream()` again for a fresh pass (creates a new iterator).
- **Determinism:** file order preserved; `source_index` == enumeration order; identical file ⇒ identical sequence.
- **Validation:** UTF-8 + JSON structural decode only (the **schema-validation boundary** — see §11).
- **Error-handling strategy:** IO/decompression errors raise; malformed lines reported as data; the pipeline applies config policy (default for full runs: abort on any malformed, since exactly 100K well-formed rows are expected).
- **Expected implementations:** `adapters/candidate_jsonl.py` (gzip-aware streaming reader); `tests` `ListCandidateSource` (yields fixtures).
- **Dependency rules:** port → `_types`. Adapter → `gzip`, `json`, `io`.

---

## §5. SubmissionSinkPort (`ports/submission_sink.py`)

- **Purpose:** write a fully-constructed `Ranking` to a validator-compliant CSV, byte-stably.
- **Responsibilities:** serialize the header and 100 rows exactly per spec; RFC-4180 quote the `reasoning`; format `score` deterministically; write atomically; return a receipt with the output sha256.
- **Non-responsibilities:** does **not** rank or re-order; does **not** build reasoning; does **not** own the filename/extension (a CLI/pipeline concern) — it writes content to the target it was given.
- **Method signatures:**
  ```
  class SubmissionSinkPort(Protocol):
      def write(self, ranking: Ranking) -> SubmissionReceipt: ...
  ```
- **Inputs:** a `Ranking` (already invariant-checked at domain construction — exactly 100, unique ranks/ids, non-increasing scores, id-ascending tie-break).
- **Outputs:** a UTF-8 (no BOM) CSV: header `candidate_id,rank,score,reasoning`; 100 data rows; `\n` line endings; deterministic `score` precision; RFC-4180 quoting. Returns `SubmissionReceipt`.
- **Exceptions:** `SubmissionWriteError` (IO); `SubmissionContractError` (defence-in-depth: a serialized invariant would be broken — should be impossible given a valid `Ranking`).
- **Performance:** trivial (100 rows).
- **Thread-safety:** single writer.
- **Determinism:** byte-for-byte reproducible — fixed column order, fixed float format, fixed quoting, fixed newline, no trailing whitespace. The output sha256 is therefore stable for identical rankings (this hash lands in the Run Report).
- **Validation:** produces output that passes `validate_submission.py`. After formatting, the sink re-checks non-increasing-by-rank and the id-ascending tie-break **at the emitted precision** (rounding may create ties, which the validator permits, provided id order within ties holds); on violation it raises `SubmissionContractError` rather than emit a rejectable file.
- **Error-handling strategy:** write to a temp file then atomic rename; never leave a partial submission.
- **Expected implementations:** `adapters/submission_csv.py`; `tests` `CapturingSubmissionSink` (in-memory, asserts byte equality).
- **Dependency rules:** port → `domain.ranking`, `_types`. Adapter → `csv`, `io`, `hashlib`.

---

## §6. RunReportSinkPort (`ports/run_report_sink.py`)

- **Purpose:** persist the run's audit + reproducibility report as deterministic JSON.
- **Responsibilities:** serialize a `RunReport` (§13) with stable key order and fixed float formatting; return a receipt.
- **Non-responsibilities:** does **not** compute the metrics (observability/pipeline do); does **not** gate the run.
- **Method signatures:**
  ```
  class RunReportSinkPort(Protocol):
      def write(self, report: RunReport) -> ReportReceipt: ...
  ```
- **Inputs:** a `RunReport` (the port-owned structural contract, §13).
- **Outputs:** a JSON file; `ReportReceipt` with the report sha256.
- **Exceptions:** `ReportWriteError`.
- **Performance:** trivial.
- **Thread-safety:** single writer.
- **Determinism:** the report's `reproducible` block (§13) serializes deterministically; the `audit` block (wall-clock, run_id) is explicitly excluded from any reproducibility hash/equality check.
- **Validation:** the report must expose the required structural fields (§13); missing fields are a programming error.
- **Error-handling strategy:** raise on IO failure; atomic write.
- **Expected implementations:** `adapters/run_report_json.py`; `tests` `CapturingRunReportSink`.
- **Dependency rules:** port → `_types` (owns `RunReport`). Adapter → `json`, `io`, `hashlib`. **`observability` builds an object conforming to the port's `RunReport` Protocol — `ports/` never imports `observability`.**

---

## §7. DeterministicEntropyPort (`ports/rng.py`)

- **Purpose:** the single seam for *all* pseudo-randomness and the injected reference date, so domain/engines never touch a global RNG or the wall clock. Online ranking has **no** legitimate randomness (ties are id-based); offline stages (KMeans init, weight search, sampling) need seeded, reproducible streams.
- **Responsibilities:** expose the run `seed`, seeded numpy `Generator`s, deterministic labeled substreams, and the injected `as_of` date.
- **Non-responsibilities:** never used for tie-breaking or any decision that affects output ordering; not a clock for logic.
- **Method signatures:**
  ```
  class DeterministicEntropyPort(Protocol):
      @property
      def seed(self) -> int: ...
      def as_of(self) -> date: ...
      def derive(self, label: str) -> int: ...                 # stable sub-seed from (seed, label)
      def numpy_generator(self, label: str) -> np.random.Generator: ...
  ```
- **Inputs:** a `label` to name an independent substream.
- **Outputs:** deterministic `int` sub-seeds and seeded `Generator`s; the fixed `as_of` date from config.
- **Exceptions:** `EntropyDisabledError` — raised by the **online** variant if `numpy_generator`/`derive` is called (online must be RNG-free; only `as_of` is permitted online).
- **Performance:** negligible.
- **Thread-safety:** `Generator`s are **not** thread-safe; callers obtain a per-thread substream via a distinct `label` rather than sharing one generator.
- **Determinism:** same `seed` ⇒ identical generators and substreams; `as_of` fixed from config; substreams independent and reproducible across runs.
- **Validation:** `seed` is an int from config; `as_of` a valid date.
- **Error-handling strategy:** misuse on the online path raises `EntropyDisabledError`, enforcing the RNG-free guarantee.
- **Expected implementations:** `adapters/entropy.py` with `OfflineEntropy` (full RNG) and `OnlineEntropy` (RNG disabled; `as_of` only); `tests` `FixedEntropy`.
- **Dependency rules:** port → `domain` (date), `_types`, numpy. Adapter → numpy only.

---

## §8. Artifact Contract System

**Manifest architecture (`MANIFEST.json`):**
```
{
  "manifest_schema_version": "1.x",
  "builder_version": "<git-describe>",
  "created_at": "<audit only, excluded from self-hash>",
  "embedding": { "model_id": "...", "dim": 384 },
  "layout_version": "cqv-1.x",
  "artifacts": {
     "<key>": { "path": "...", "sha256": "...", "bytes": <int>,
                "schema_version": "...", "kind": "npy|parquet|json|onnx|yaml" },
     ...
  },
  "manifest_sha256": "<self-hash over the canonical serialization excluding this field>"
}
```

**Hash verification strategy:** verify `manifest_sha256` first (self-integrity), then per-artifact sha256 **streamed during load** so verification is essentially free. `verify_all()` (optional eager pass at R0) forces fail-fast before any work; per-key loaders verify lazily and cache the result.

**Schema-version strategy:** every artifact carries a `schema_version`; the manifest carries a `layout_version` for the CQV feature layout. Consumers in code declare the versions they support. Rule: `major` must match exactly; `minor` is backward-compatible (consumer ≥ artifact minor).

**Compatibility rules (asserted at R0):**
1. `manifest_schema_version` supported by the loader.
2. Every **required** key present.
3. Each artifact `schema_version` compatible with the code's expected version.
4. **Cross-artifact coherence:** `weights.layout_version == manifest.layout_version == FeatureLayout` constant; `embedding.dim == vector store dim == online encoder dim`; `embedding.model_id` consistent between `candidate_vectors` and the onnx fallback; the JD spec's anchor set ⊆ `anchor_vectors` keys; centroid `dim == embedding.dim`.

**Failure modes (all → `ArtifactContractError`/`ManifestError`, all fatal at R0):** missing key; sha256 mismatch (corruption/tamper); incompatible `schema_version`/`layout_version`; cross-artifact incoherence; manifest self-hash failure; partial artifact set. Each error names the offending key and the expected-vs-actual.

---

## §9. Embedding Contract

- **Dimensionality:** a single fixed `dim` (declared in the manifest's `embedding.dim` and on `EmbeddingModelPort.dim`); offline build, vector store, and online fallback must all agree (enforced by §8 rule 4).
- **Normalization guarantee:** every output row is L2-normalized to unit length within ε, so cosine similarity reduces to a dot product downstream.
- **Batching guarantee:** output order == input order irrespective of internal batch size; `batch_size` is an adapter hint that never changes results.
- **Fallback behavior:** online, the onnx adapter encodes **only** candidates missing from the vector store. JD anchors are precomputed offline and never re-encoded online. Fallback vectors live in the same space as the offline vectors (same base model exported to ONNX).
- **Deterministic encoding requirements:** fixed weights, pooling, normalization, `float32`, pinned thread/op counts, no sampling. Within a runtime: bitwise-deterministic. **Cross-runtime (st ↔ onnx): cosine agreement within ε (target > 0.999), not bitwise equality** — documented and acceptable because lookup is the primary path and fallback is rare.

---

## §10. Vector Store Contract

- **Lookup behavior:** `get(cid)` is O(1) via an in-memory id→row-index map, returning a read-only normalized `(dim,)` vector or `None`.
- **Missing-candidate behavior:** `None` / `missing` list — **normal**, triggers the encode fallback. In full reproduction the 100K pool is fully precomputed, so misses ≈ 0; the ≤100 sandbox sample may contain ids outside the precomputed parquet, which the fallback covers.
- **Bulk retrieval:** `get_many` returns order-preserving `found` vectors plus `missing` ids; `view_all` returns the full mmap'd matrix and its aligned id order for the columnar scoring path.
- **Memory-mapping strategy:** vectors stored contiguously (`float32`) and mmap'd read-only; resident memory ≈ touched pages. The id index loads fully (~MBs).
- **Performance expectations:** `get` ~µs; `get_many` O(K) gather; `view_all` zero-copy. This is the architectural reason R3 fits the 5-min budget.

---

## §11. Candidate Source Contract

- **Streaming guarantees:** lazy, one `SourceRecord` at a time, file order preserved, `source_index` == enumeration order, single pass per `stream()`.
- **Memory guarantees:** O(1) per record; the 465 MB raw / 52 MB gz file is never fully materialized; gzip streamed.
- **Malformed-row handling:** a line that is not a valid JSON object yields `Malformed(line_no, error)`. The **port reports; the pipeline decides** (skip-and-log vs abort) per config. Default full-run policy: abort (exactly 100K well-formed rows expected).
- **Schema-validation boundary:** the port performs **UTF-8 + JSON structural decode only**. Validation against `candidate_schema.json` (required fields, enums, ranges) happens in `features.parsing` → `RawCandidate` and surfaces as `SchemaError`. Thus "valid JSON but missing `redrob_signals`" is **not** a source error; "not valid JSON" **is**.

---

## §12. Submission Contract

- **Validator compliance:** output must pass `validate_submission.py` — exact header; exactly 100 rows; ranks 1–100 each once; ids unique and `^CAND_[0-9]{7}$`; score non-increasing by rank; ties broken by id ascending. Because `Ranking` enforces these at construction, the sink's job is **faithful serialization** that doesn't break them.
- **UTF-8 requirements:** UTF-8, no BOM; `reasoning` text UTF-8; RFC-4180 quoting for any `reasoning` containing comma, quote, or newline.
- **Tie-breaking guarantees:** the id-ascending order within equal scores is taken verbatim from the `Ranking`; the sink never re-orders.
- **Output guarantees:** fixed column order; deterministic `score` precision; `\n` newlines; no trailing whitespace; atomic write; a `SubmissionReceipt` carrying the output sha256. After formatting, the sink re-asserts monotonicity + tie-break at the emitted precision and raises `SubmissionContractError` rather than write a rejectable file.

---

## §13. Run Report Contract (`RunReport` Protocol, port-owned)

The port owns a **structural** `RunReport` Protocol so `observability` can build a conforming object without `ports/` importing `observability`. Two explicitly separated regions:

- **`reproducible` (deterministic; the only region compared by determinism tests):**
  `code_version, config_hash, manifest_hash, artifact_hashes: Mapping[str,str], input_file_sha256, candidate_count, output_sha256, honeypot_count_top100, honeypot_rate, eligibility_summary, score_distribution_digest`.
- **`audit` (excluded from any reproducibility hash):** `run_id, started_at, ended_at, host_label`.
- **`timings`:** per-stage wall-ms for R0–R9 (or O1–O10 for offline build reports), plus budget headroom.
- **`budget`:** `limit_seconds, used_seconds, within_budget: bool, peak_rss_mb`.

**Audit requirements:** the report ties a submission to the exact artifact set (`manifest_hash` + per-artifact `artifact_hashes`), config (`config_hash`), code (`code_version`), input (`input_file_sha256`), and output (`output_sha256`). **Reproducibility requirements:** identical inputs ⇒ identical `reproducible` block. **Manifest tracking:** `manifest_hash` and `artifact_hashes` are sourced from `ArtifactStorePort.manifest()`. **Timing reporting:** per-stage timings + `within_budget` flag from the budget guard; the honeypot rate is recorded for the Stage-3 ≤10% gate.

---

## §14. Port Ownership Matrix

| Port | Owning concern | Consumed by (engine / stage) | Read/Write |
|---|---|---|---|
| `EmbeddingModelPort` | text→vector | `SemanticEngine` (R3 fallback); offline O5/O6 | read-compute |
| `SemanticVectorStorePort` | precomputed vectors | `SemanticEngine` (R3) | read |
| `ArtifactStorePort` | artifact load + integrity | pipeline R0; offline O9 verify | read |
| `CandidateSourcePort` | raw input streaming | pipeline R1; offline O1/O2/O5 | read |
| `SubmissionSinkPort` | final CSV | pipeline R8 | write |
| `RunReportSinkPort` | audit/repro report | pipeline R9 (online), offline build report | write |
| `DeterministicEntropyPort` | seeds + `as_of` | engines via injected `as_of`; offline O7/O8 RNG | compute |

---

## §15. Dependency Flow Matrix

| From → To | `domain` | `ports` | `engines` | `adapters` | `pipelines` |
|---|---|---|---|---|---|
| `domain` | — | ✗ | ✗ | ✗ | ✗ |
| `ports` | ✓ (+numpy) | — | ✗ | ✗ | ✗ |
| `engines` | ✓ | ✓ | (no cross-engine) | ✗ | ✗ |
| `adapters` | ✓ | ✓ (implements) | ✗ | — | ✗ |
| `pipelines` | ✓ | ✓ | ✓ | ✓ (instantiates) | — |

Rule: engines receive ports by injection; **only `pipelines` constructs adapters and binds them to ports** (composition root). Import-linter enforces every ✗.

---

## §16. Contract Testing Strategy

- **Shared contract suites.** Each port has one abstract, parametrized behavioral suite run against **every** implementation — the real adapter *and* the fake. This is what prevents fakes from drifting from reality.
  - `EmbeddingModelPort`: output shape == `(n, dim)`; `float32`; per-row unit norm ± ε; order preserved; determinism across two calls.
  - `SemanticVectorStorePort`: `get` round-trips a known id; `get_many` preserves order and partitions found/missing; `view_all` id order aligns with rows; missing id ⇒ `None`/`missing`, never raises.
  - `ArtifactStorePort`: tampered byte ⇒ `ArtifactContractError`; missing key ⇒ error; incompatible `schema_version` ⇒ error; happy path returns expected bytes.
  - `CandidateSourcePort`: order preservation; `source_index` monotonic; malformed line ⇒ `Malformed` (not exception); gzip == plain parity.
  - `SubmissionSinkPort`: emitted CSV passes `validate_submission.py`; byte-identical for identical `Ranking`; reasoning with commas/quotes/newlines round-trips.
  - `RunReportSinkPort`: `reproducible` block byte-stable across runs with identical inputs; `audit` block ignored by the determinism assertion.
  - `DeterministicEntropyPort`: same seed ⇒ identical streams; distinct labels ⇒ independent streams; online variant raises on `numpy_generator`.
- **Property tests:** embedding norm; store get/put identity; source order; submission validator round-trip.
- **Golden tests:** fixed inputs → fixed output bytes (submission, run-report `reproducible`).

---

## §17. Mocking Strategy

- **Prefer fakes (in-memory real implementations) over mocks.** Engines are pure and need no mocks; mocking happens **only** at the port boundary.
- Canonical fakes (in `tests/fixtures`, all passing the §16 contract suites): `InMemoryArtifactStore`, `InMemoryVectorStore`, `StubEmbeddingModel` (deterministic hash-derived unit vectors of configurable `dim`), `ListCandidateSource`, `CapturingSubmissionSink`, `CapturingRunReportSink`, `FixedEntropy`.
- `StubEmbeddingModel` derives each vector deterministically from a hash of the input text, then normalizes — reproducible, norm-1, no model download, fast.
- No `unittest.mock` inside engine tests; behavioral fakes only, so tests exercise real contract behavior rather than asserting call sequences.

---

## §18. Offline vs Online Usage Matrix

| Port | Offline (O-stages) | Online (R-stages) | Adapter offline / online |
|---|---|---|---|
| `EmbeddingModelPort` | O5 (candidate vectors), O6 (anchors) | R3 (fallback only) | `st_embedder` / `onnx_embedder` |
| `SemanticVectorStorePort` | — (built, not consumed) | R3 | — / `vector_store_parquet` |
| `ArtifactStorePort` | O9 (verify dry-run) | R0 | `artifact_store_fs` (both) |
| `CandidateSourcePort` | O1, O2, O5 | R1 | `candidate_jsonl` (both) |
| `SubmissionSinkPort` | O9 (golden dry-run, optional) | R8 | `submission_csv` (both) |
| `RunReportSinkPort` | O-build report (optional) | R9 | `run_report_json` (both) |
| `DeterministicEntropyPort` | O7 (KMeans), O8 (weight search) | `as_of` only (RNG disabled) | `OfflineEntropy` / `OnlineEntropy` |

---

## §19. Adapter Mapping Matrix

| Port | Adapter file | Underlying tech | Used in |
|---|---|---|---|
| `EmbeddingModelPort` | `adapters/st_embedder.py` | sentence-transformers | offline only (import-guarded) |
| `EmbeddingModelPort` | `adapters/onnx_embedder.py` | onnxruntime (CPU EP) | online fallback |
| `SemanticVectorStorePort` | `adapters/vector_store_parquet.py` | pyarrow / numpy mmap | online |
| `ArtifactStorePort` | `adapters/artifact_store_fs.py` | fs, hashlib, json | offline + online |
| `CandidateSourcePort` | `adapters/candidate_jsonl.py` | gzip, json, io | offline + online |
| `SubmissionSinkPort` | `adapters/submission_csv.py` | csv, io, hashlib | offline + online |
| `RunReportSinkPort` | `adapters/run_report_json.py` | json, io, hashlib | offline + online |
| `DeterministicEntropyPort` | `adapters/entropy.py` | numpy | offline (RNG) / online (`as_of`) |

---

## §20. Sequence diagrams (port usage)

**Offline build — O1 … O10** (composition root: `pipelines/offline/pipeline.py`)
```
O1 Census            CandidateSourcePort.stream() ─▶ tally schema/coverage
O2 Feature Extract   CandidateSourcePort.stream() ─▶ features.* ─▶ feature tables (fs write, internal)
O3 Integrity Calib   (read features) ─▶ integrity_thresholds.json (fs)
O4 Lexicon Discovery (read features + configs) ─▶ lexicon.compiled.json (fs)
O5 Embedding Gen     CandidateSourcePort.stream() ─▶ compose docs
                     EmbeddingModelPort(st).encode(batch) ─▶ candidate_vectors.parquet (fs)
                     + export encoder.onnx
O6 Anchor Authoring  (configs/anchors) ─▶ EmbeddingModelPort(st).encode() ─▶ anchor_vectors.npy (fs)
O7 Archetype Disc.   (read candidate_vectors) + DeterministicEntropyPort.numpy_generator("kmeans")
                     ─▶ KMeans ─▶ centroids.npy (fs)
O8 Weight Search     (read features + data/golden) + DeterministicEntropyPort.derive("search")
                     ─▶ scoring_weights.locked.yaml (fs)
O9 Validation Battery ArtifactStorePort.verify_all()  (dry-run integrity over freshly built set)
                     [optional] SubmissionSinkPort.write(golden_ranking) ─▶ offline quality check
O10 Packaging        compute sha256 per artifact ─▶ write MANIFEST.json (+ manifest_sha256)
                     [optional] RunReportSinkPort.write(build_report)
```

**Online rank — R0 … R9** (composition root: `pipelines/online/pipeline.py`)
```
R0 Artifact Loading  ArtifactStorePort.manifest(); verify_all()
                     load weights/lexicon/centroids/anchors (load_json/load_npy)
                     SemanticVectorStorePort opened over ArtifactStorePort.locate("candidate_vectors")
                     EmbeddingModelPort(onnx) initialized from locate("encoder")
                     DeterministicEntropyPort(online) created with as_of from config
R1 Candidate Parsing CandidateSourcePort.stream() ─▶ features.parsing ─▶ RawCandidate (constant memory)
R2 Feature Extract   features.* (pure) ─▶ career/credibility/behavioral/logistics slices
R3 Semantic Lookup   SemanticVectorStorePort.view_all()/get_many(ids) ─▶ vectors
                     misses ─▶ EmbeddingModelPort(onnx).encode(miss_docs)
                     anchors (from R0) + centroids ─▶ SemanticEngine (numpy) ─▶ semantic/archetype slices
R4 Gates             IntegrityEngine, EligibilityEngine (pure) ─▶ integrity/eligibility slices
                     (survivors retain RawCandidate for later reasoning)
R5 Scoring           CQVAssembler + ScoringEngine (pure, weights from R0) ─▶ ScoredCandidate[]
R6 Ranking           RankingEngine.rank(...) (pure) ─▶ Ranking (invariants enforced)
R7 Reasoning         ReasoningEngine (pure) over top-K retained RawCandidate ─▶ Ranking.with_reasoning(...)
R8 Submission        SubmissionSinkPort.write(ranking) ─▶ submission.csv + SubmissionReceipt
R9 Run Report        RunReportSinkPort.write(report) using ArtifactStorePort.manifest() hashes,
                     stage timings, SubmissionReceipt.output_sha256, honeypot_rate
```

Note R3→R7: ports appear only at R0/R1/R3/R8/R9; R2/R4/R5/R6/R7 are pure engine work with **no port calls**, which is exactly what keeps the hot path testable and the online budget predictable.

---

## Build order for adapter implementation

1. `_types.py` + per-port `Protocol`s + port-level error classes (no external deps).
2. Test fakes for all seven ports + the §16 shared contract suites (so every adapter has a target to satisfy).
3. `artifact_store_fs` (everything else depends on artifacts) → `candidate_jsonl` → `vector_store_parquet` → `onnx_embedder` / `st_embedder` → `submission_csv` → `run_report_json` → `entropy`.
4. Each adapter is merged only when it passes its shared contract suite identically to its fake.
