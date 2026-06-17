# REDSTACK_ADAPTERS_LAYER.md

REDSTACK v1.1 — Adapters (Infrastructure) Layer Specification

**Scope:** `src/redstack/adapters/` only. Architecture, Domain, Ports, Feature, Engine, Offline, and Online layers are frozen; this document changes none of them. It specifies the concrete implementations of the seven frozen ports. No code, no pseudocode — architecture-level, implementation-ready.

**What adapters are.** Infrastructure only. Adapters are the *only* layer permitted to touch the filesystem, read/write CSV/JSONL/parquet/numpy, load ONNX or sentence-transformers, and access external runtimes. Each adapter implements exactly one port contract and nothing more.

**What adapters are not.** Adapters contain **no** business logic — no scoring, eligibility, reasoning, feature extraction, ranking, or integrity logic. Those live in `engines/` and `features/`. An adapter that finds itself making a domain decision is mis-scoped.

**Import boundary (enforced by import-linter, CI-blocking).** Adapters may import `domain/`, `ports/`, `config.schema`, and necessary infrastructure libraries. Adapters may **never** import `engines/` or `pipelines/`. Adapters are instantiated only by the pipeline composition roots (`pipelines/offline`, `pipelines/online`) and bound to their ports there; an adapter never self-constructs another adapter except through an injected port.

**Frozen-surface note (ArtifactStorePort).** The port surface is exactly `manifest() · verify_all() · load_bytes(key) · load_text(key) · load_json(key) · load_npy(key) · locate(key) → ArtifactLocator`. "Load YAML / parquet / ONNX" are realized through this surface, not by new methods: **YAML** = `load_text` + `config.schema` parse; **parquet** and **ONNX** = `locate()` → a hash-verified `ArtifactLocator` the consumer (vector store / embedder) mmaps or opens. The port is unchanged.

---

## §0 Adapter → Port → Runtime Mapping

| # | Adapter | Implements (port) | Runtime / library | Offline | Online |
|---|---|---|---|---|---|
| 1 | `JsonlCandidateSourceAdapter` | `CandidateSourcePort` | `io`, `gzip`, `json` (stdlib) | ✓ (O0/O1/O2/O13a/O14) | ✓ (R1) |
| 2 | `FilesystemArtifactStoreAdapter` | `ArtifactStorePort` | `pathlib`, `hashlib`, `json`, numpy, `config.schema` | ✓ (O17/O18) | ✓ (R0) |
| 3 | `OnnxEmbeddingModelAdapter` | `EmbeddingModelPort` | `onnxruntime` (CPU EP), numpy | — | ✓ (R3 fallback) |
| 4 | `SentenceTransformerEmbeddingAdapter` | `EmbeddingModelPort` | `sentence-transformers`, `torch` | ✓ (O13) | ✗ (import-guarded) |
| 5 | `ParquetSemanticVectorStoreAdapter` | `SemanticVectorStorePort` | `pyarrow`/numpy mmap | — | ✓ (R3) |
| 6 | `CsvSubmissionSinkAdapter` | `SubmissionSinkPort` | `csv`, `io`, `hashlib` | ✓ (O9 dry-run) | ✓ (R8) |
| 7 | `JsonRunReportSinkAdapter` | `RunReportSinkPort` | `json`, `io`, `hashlib` | ✓ (O18 report) | ✓ (R9) |
| 8 | `DeterministicEntropyAdapter` | `DeterministicEntropyPort` | numpy | ✓ (Offline variant, RNG) | ✓ (Online variant, `as_of` only) |

Adapters depend only on the port DTOs and domain types they exchange (`SourceRecord`, `ArtifactLocator`, `Manifest`, `FloatMatrix/Vector`, `BulkVectorResult`, `Ranking`, `RunReport`, `SubmissionReceipt`, `ReportReceipt`, `CandidateId`).

---

## §1 JsonlCandidateSourceAdapter — implements `CandidateSourcePort`

- **Purpose.** Stream raw candidate records from `.jsonl` / `.jsonl.gz` with constant memory and deterministic ordering, reporting per-line decode outcomes as data.
- **Inputs.** A configured path + compression flag (held on the adapter, not passed per call); no per-call inputs to `stream()`.
- **Outputs.** `Iterator[SourceRecord]` — `Ok(raw: Mapping[str,object], line_no, source_index)` or `Malformed(line_no, error)`; `count() → int | None`.
- **Dependencies.** stdlib `io`, `gzip`, `json`; `ports._types`. No domain typing performed here (schema validation is `features.parsing`, downstream).
- **Lifecycle.** Constructed by the pipeline with a resolved path; `stream()` opens a fresh handle each call (one-pass generator); the handle closes on iterator exhaustion or context exit. Re-reading requires a new `stream()`.
- **Internal workflow.** Open (transparently gzip-wrap if `.gz`); read line-by-line in binary, decode UTF-8 **strict**; attempt `json.loads`; on success emit `Ok` with the running `source_index` (enumeration order) and `line_no`; on `JSONDecodeError`/`UnicodeDecodeError` emit `Malformed`. Blank/whitespace-only lines are skipped (not counted as records), matching the validator's row semantics.
- **Failure modes.** `CandidateSourceError` on open/IO/decompression failure (raised). Per-line bad JSON or bad UTF-8 → `Malformed` record (not an exception). The adapter never decides skip-vs-abort — the pipeline applies policy.
- **Determinism guarantees.** File order preserved exactly; `source_index` == enumeration order; identical file ⇒ identical sequence (incl. identical `Malformed` placement). gzip decode is deterministic.
- **Thread safety.** Single-consumer iterator; not re-entrant; not shared across threads. Concurrency is achieved by the pipeline consuming sequentially and fanning out *after* ingestion.
- **Performance characteristics.** ~2,500–3,500 records/s (decode-bound); 100K in ~30–40 s; constant per-record cost. gzip adds modest CPU, saves IO.
- **Memory characteristics.** O(1): one line buffer at a time; never materializes the file (≈52 MB gz / 465 MB raw stays on disk).
- **Auditability.** Emits `rows_read, malformed_count, bytes_read, gzip: bool`; every `Malformed` carries `line_no` + reason for the run report.
- **Testability.** Contract suite (Ports §16) + gzip-vs-plain parity on identical content; malformed-line fixture yields a `Malformed` at the right `line_no`; order/`source_index` monotonicity property; empty-line skipping.

---

## §2 FilesystemArtifactStoreAdapter — implements `ArtifactStorePort`

- **Purpose.** Load and integrity-verify build artifacts by manifest key; provide verified locators for mmap/session consumers. The fail-fast integrity gate between offline and online.
- **Inputs.** An artifact-root path + `MANIFEST.json`; manifest keys per call.
- **Outputs.** `Manifest`; `load_bytes/text/json/npy(key)`; `locate(key) → ArtifactLocator`. (YAML via `load_text` + `config.schema`; parquet/ONNX via `locate()`.)
- **Dependencies.** `pathlib`, `hashlib`, `json`, numpy (for `load_npy`), `config.schema` (for typed config parsing), `domain.errors`, `ports._types`. **safe-deserialization libs only.**
- **Lifecycle.** Constructed with the root; reads + self-verifies the manifest at init; per-key loads verify lazily and cache the verified result; `verify_all()` forces an eager pass at R0/O17. Read-only; no disposal beyond closing transient handles.
- **Internal workflow.** (1) Read manifest; recompute `manifest_sha256` over its canonical serialization (excluding the field) and compare. (2) On each load, resolve `key → path` strictly within the root; **stream the file while computing sha256**, compare to the manifest entry, then materialize (`load_npy` with `allow_pickle=False`; `load_json`/`load_text` UTF-8; `locate` returns an `ArtifactLocator{key, verified=True, path}` only after the hash passes). (3) Coherence checks on demand (Ports §8): `layout_version` agreement, `embedding.dim`/`model_id` consistency, anchor set ⊆ `jd_concepts`, schema-version compatibility.
- **Failure modes.** `ManifestError` (missing/unparseable/self-hash fail); `ArtifactContractError` (missing key, sha256 mismatch, incompatible schema/layout version, cross-artifact incoherence, path-escape attempt). All fatal — **no degraded mode**.
- **Determinism guarantees.** Same artifact bytes ⇒ same hashes; key→artifact mapping fixed by the manifest; verification idempotent and cached.
- **Thread safety.** Read-only after init; concurrent loads safe; the verification cache is write-once per key (safe under concurrent first-touch via idempotent recompute).
- **Performance characteristics.** Streaming sha256 makes verification ~free atop the load. Hashing the ~150 MB vector file + onnx dominates R0 (≤ ~20 s); small artifacts negligible. `locate()` does not read large files into RAM.
- **Memory characteristics.** Small artifacts loaded fully (a few MB total); large artifacts (parquet/onnx) **not** loaded — `locate()` hands a path the consumer mmaps/opens. Hash buffer is a fixed chunk.
- **Auditability.** Emits `manifest_hash`, per-key `artifact_hashes`, `verified_keys`, and any failure key+reason — the provenance backbone of the run report.
- **Testability.** Contract suite + tampered-byte detection (flip one byte ⇒ `ArtifactContractError`); missing-key, incompatible-version, path-traversal (`..`/absolute key) rejection; happy-path byte fidelity; manifest self-hash failure.

---

## §3 OnnxEmbeddingModelAdapter — implements `EmbeddingModelPort`

- **Purpose.** The **online fallback** encoder: turn composed documents into normalized dense vectors via ONNX Runtime, CPU-only, deterministic, no network. Invoked only for `CandidateId`s missing from the vector store.
- **Inputs.** A verified `ArtifactLocator` for `model/encoder.onnx` (from the artifact store); `encode(texts, batch_size)`.
- **Outputs.** `FloatMatrix (n, dim)`, float32, L2-normalized rows, order-preserving; `dim` and `model_id` properties.
- **Dependencies.** `onnxruntime`, numpy, the tokenizer assets (hash-pinned in the artifact set), `ports._types`.
- **Lifecycle.** Constructed at R0 from the located onnx file; creates one `InferenceSession` with pinned options; serves `encode` for the run; releases the session at run end.
- **Internal workflow.** Session options: `providers=['CPUExecutionProvider']`, `intra_op_num_threads` and `inter_op_num_threads` pinned (from determinism config), `execution_mode=SEQUENTIAL`, fixed graph-optimization level. Per `encode`: tokenize (pinned tokenizer), run in batches of `batch_size`, apply the fixed pooling, L2-normalize, assemble the output preserving input order. **No network** (no model download; the model is the local hash-verified artifact).
- **Failure modes.** `EmbeddingError` on session/run failure or tokenizer error; `ArtifactContractError` if `dim`/`model_id` disagree with the manifest. Never returns zeros to mask failure; a per-candidate encode failure is surfaced so R3 can mark that candidate's semantics `UNKNOWN`.
- **Determinism guarantees.** Within this runtime: bitwise-deterministic (pinned threads, sequential exec, fixed pooling/normalization, float32, no sampling). **Cross-runtime** vs the offline sentence-transformers vectors: cosine within ε (target > 0.999), not bitwise — the documented and accepted bound (Ports §9).
- **Thread safety.** One session per instance; `encode` called sequentially; the instance is not shared mutably across threads. Thread/op counts are fixed regardless of host cores (determinism + predictable latency).
- **Performance characteristics.** Fallback only — invoked for ≈ 0 candidates in full reproduction, a handful in the sandbox sample; batched encode keeps per-call latency low. Online budget impact ≈ 0.
- **Memory characteristics.** Session ≈ 100–200 MB; batch buffers bounded by `batch_size`.
- **Auditability.** Emits `encode_calls, fallback_docs, model_id, dim, batch_size, encode_ms`.
- **Testability.** Contract suite (norm/shape/order/determinism) + an ε-parity check vs a reference sentence-transformers vector on a fixture; verifies no network is attempted (offline assertion).

**Online fallback path.** Used by `CandidateRetrievalEngine` at R3 only when `SemanticVectorStorePort` reports a miss. **Model compatibility.** The onnx graph is the exported twin of the offline base model (same `model_id`/`dim`); compatibility is asserted at R0. **Runtime budget.** ≤ a few seconds aggregate; never the critical path.

---

## §4 SentenceTransformerEmbeddingAdapter — implements `EmbeddingModelPort` (OFFLINE ONLY)

- **Purpose.** The offline embedding generator (O13): produce candidate, anchor, career-history, and concept vectors, and export the ONNX twin for the online fallback.
- **Inputs.** Composed documents (candidate/anchor/concept texts) from the offline stages; a pinned model id + revision.
- **Outputs.** `FloatMatrix (n, dim)`, float32, L2-normalized, order-preserving; the exported `model/encoder.onnx`; `dim`/`model_id`.
- **Dependencies.** `sentence-transformers`, `torch`, numpy, `optimum`/`torch.onnx` for export. **Offline-only.**
- **Lifecycle.** Constructed only inside `pipelines/offline`; loads the pinned model in `eval()` mode under `torch.no_grad`; encodes; exports onnx; released after O13.
- **Internal workflow.** Set offline env (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`); load the pinned-revision model; batched encode with the fixed pooling; L2-normalize; write `candidate_vectors.parquet` / `anchor_vectors.npy` keyed/ordered deterministically; export onnx at a fixed opset and **verify st↔onnx parity** (cosine ≥ 0.999 on a sample) before the artifact is accepted.
- **Failure modes.** `EmbeddingError` on load/encode/export; **`RuntimeError` on import if an "online" environment marker is set** (import guard); parity-check failure aborts O13.
- **Determinism guarantees.** Within a host: deterministic (eval mode, no dropout, pinned `torch` threads, fixed pooling/normalization). **Cross-host:** ε-stable, not bitwise — consistent with the Offline Pipeline's O18 treatment of embedding artifacts as ε-stable outputs (or build-once-and-freeze, per the team's ratified choice).
- **Thread safety.** Single-threaded controlled batches; offline, no concurrency requirement.
- **Performance characteristics.** Minutes for 100K × dim — the compute-dominant offline stage; unbounded by online budget.
- **Memory characteristics.** Model + batch resident; outputs streamed to disk.
- **Auditability.** Emits `encoded_docs, model_id, dim, opset, parity_cosine, encode_ms`.
- **Testability.** Run only in offline test profiles; **import-guard test** (importing under an online marker raises); export-parity test; norm/shape/order contract.

**Offline-only guarantees.** Import-guarded + offline env vars + never referenced by `pipelines/online` (import-linter forbids `pipelines.online → adapters.st_embedder` and `→ sentence_transformers`). **Export-to-onnx workflow.** Pinned opset, parity-verified, hash-registered in the manifest. **Reproducibility constraints.** Pinned model revision + pinned thread counts; ε-stable across hosts.

---

## §5 ParquetSemanticVectorStoreAdapter — implements `SemanticVectorStorePort`

- **Purpose.** O(1) retrieval of precomputed candidate vectors by `CandidateId`, plus a zero-copy full-matrix view for the columnar scoring path. The reason R3 is "lookup," not "encode."
- **Inputs.** A verified `ArtifactLocator` for `candidate_vectors.parquet` (+ its id index); `CandidateId`(s) per call.
- **Outputs.** `dim`; `contains(cid)`; `get(cid) → FloatVector | None`; `get_many(cids) → BulkVectorResult(vectors, found, missing)`; `view_all() → (FloatMatrix, id_order)`.
- **Dependencies.** `pyarrow`/numpy (mmap), `ports._types`, `domain.ids`.
- **Lifecycle.** Constructed at R0 from the located parquet; loads the id→row index fully (small), memory-maps the vector block read-only; serves lookups for the run; unmaps at run end.
- **Internal workflow.** Load the `CandidateId → row_index` map (a parquet column / sidecar) into an in-memory dict. `get` resolves the row and returns a read-only `(dim,)` slice. `get_many` gathers found rows in input order and partitions misses. `view_all` returns the mmap'd `(N, dim)` matrix + the aligned `id_order` (stored order) without copying. Vectors are already L2-normalized (offline guarantee); the adapter does not re-normalize.
- **Failure modes.** `VectorStoreError` on corrupt/unreadable store or `dim` mismatch vs manifest (raised). **A missing `CandidateId` is `None`/`missing` — never an exception** (it triggers the R3 onnx fallback).
- **Determinism guarantees.** id→row mapping fixed by the artifact; `view_all` id order == stored order; returned arrays read-only; deterministic gather order.
- **Thread safety.** Read-only after open; concurrent reads safe (mmap + immutable index). No online writes.
- **Performance characteristics.** `get` ~µs (dict + slice); `get_many` O(K) vectorized gather; `view_all` zero-copy. R3 stays within ≤ 15 s with large headroom.
- **Memory characteristics.** Vector block mmap'd read-only (≈ 100K×dim×4 ≈ 150 MB, resident only on touched pages); id index a few MB fully resident; no full-matrix copy.
- **Auditability.** Emits `lookups, hits, misses, dim, rows`.
- **Testability.** Contract suite (round-trip a known id; `get_many` order + found/missing partition; `view_all` id-row alignment; miss ⇒ `None`); corruption ⇒ `VectorStoreError`; dim-mismatch detection.

**Index structure.** In-memory `CandidateId → row_index` dict over a contiguous mmap'd float32 matrix. **Lookup complexity.** `get` O(1); `get_many` O(K); `view_all` O(1) (view). **Dimension validation.** Asserted equal to `embedding.dim` from the manifest at open.

---

## §6 CsvSubmissionSinkAdapter — implements `SubmissionSinkPort`

- **Purpose.** Serialize a finished `Ranking` to a validator-compliant, byte-stable CSV and return a receipt.
- **Inputs.** A `Ranking` (already invariant-checked at domain construction); a configured output path (`<participant_id>.csv`).
- **Outputs.** `submission.csv` on disk; `SubmissionReceipt(row_count, bytes_written, output_sha256)`.
- **Dependencies.** stdlib `csv`, `io`, `hashlib`; `domain.ranking`, `ports._types`, `domain.errors`.
- **Lifecycle.** Constructed with the output path; `write(ranking)` performs a single atomic write (temp file in the same directory → `fsync` → atomic rename); no persistent handle.
- **Internal workflow.** Emit header `candidate_id,rank,score,reasoning`; emit 100 rows in rank order; format `score` at the fixed configured precision; RFC-4180 quote any `reasoning` containing comma/quote/newline; UTF-8 (no BOM); `\n` line endings; no trailing whitespace. **Before finalizing**, re-assert non-increasing-by-rank + id-ascending tie-break at the emitted precision; compute the output sha256.
- **Failure modes.** `SubmissionContractError` if a serialized invariant would break (e.g., rounding violates monotonicity in a way the tie-break can't satisfy) — abort, no file. `SubmissionWriteError` on IO. Never leaves a partial file (temp + rename).
- **Determinism guarantees.** Byte-for-byte reproducible: fixed column order, float format, quoting, newline, encoding; identical `Ranking` ⇒ identical bytes ⇒ identical `output_sha256`.
- **Thread safety.** Single writer; one `write` per run.
- **Performance characteristics.** Trivial (100 rows); ≤ 1 s including hash.
- **Memory characteristics.** Negligible (buffered 100 rows).
- **Auditability.** Returns `output_sha256, row_count, bytes_written` for the run report.
- **Testability.** Contract suite (emitted CSV passes the organizer `validate_submission.py`; byte equality for identical `Ranking`; reasoning with commas/quotes/newlines round-trips); precision/monotonicity-preservation test; atomic-write (no partial file on simulated failure).

**Formatting guarantees.** Exact header/order; RFC-4180; UTF-8 no BOM; `\n`. **Precision guarantees.** Fixed decimal precision agreed with the scoring contract; equal rounded scores are permitted (validator allows ties) provided id-ascending order holds. **Output sha256.** Computed over the final bytes, surfaced in the receipt.

---

## §7 JsonRunReportSinkAdapter — implements `RunReportSinkPort`

- **Purpose.** Persist the `RunReport` (audit + reproducibility) as deterministic JSON.
- **Inputs.** A `RunReport` (the port-owned structural contract; `observability` builds it).
- **Outputs.** `run_report.json` on disk; `ReportReceipt(bytes_written, report_sha256)`.
- **Dependencies.** stdlib `json`, `io`, `hashlib`; `ports._types`.
- **Lifecycle.** Constructed with the output path; `write(report)` does an atomic temp+rename; no persistent handle.
- **Internal workflow.** Serialize with **sorted keys**, fixed float formatting, enums by value; partition into `reproducible` and `audit` blocks; compute the report sha256 over the serialized bytes; write atomically. The `reproducible` block excludes wall-clock/`run_id`.
- **Failure modes.** `ReportWriteError` on IO. A `RunReport` missing required structural fields is a programming error (raised).
- **Determinism guarantees.** The `reproducible` block is byte-stable across runs with identical inputs; the `audit` block is excluded from any determinism comparison/hash.
- **Thread safety.** Single writer.
- **Performance characteristics.** Trivial; ≤ 1 s.
- **Memory characteristics.** Negligible.
- **Auditability.** Returns `report_sha256, bytes_written`; the report itself *is* the audit artifact.
- **Testability.** Contract suite (`reproducible` byte-stable across two identical runs; `audit` ignored by the determinism assertion); schema-completeness test.

**Stable JSON ordering.** Sorted keys + fixed float format + enums-by-value. **Hashing behavior.** sha256 over final bytes; the `reproducible` block is the comparison unit. **Report versioning.** A `report_schema_version` field travels with the report; consumers check compatibility.

---

## §8 DeterministicEntropyAdapter — implements `DeterministicEntropyPort`

- **Purpose.** The single seam for the injected `as_of` date and (offline only) seeded randomness, so domain/engines never touch a global RNG or the wall clock.
- **Inputs.** `seed: int` and `as_of: date` from config; a `label` per RNG request.
- **Outputs.** `seed`; `as_of()`; `derive(label) → int`; `numpy_generator(label) → np.random.Generator`.
- **Dependencies.** numpy; `domain` (date); `ports._types`.
- **Lifecycle.** Two variants constructed by the respective composition roots: `OfflineEntropy` (full RNG) and `OnlineEntropy` (RNG disabled). Immutable after construction.
- **Internal workflow.** `derive(label)` mixes `(seed, label)` into a stable sub-seed (deterministic hash → int); `numpy_generator(label)` builds a `Generator` from that sub-seed (independent substreams per label). `as_of()` returns the fixed configured date. **Online variant:** `derive`/`numpy_generator` raise `EntropyDisabledError`; only `as_of()` is served.
- **Failure modes.** `EntropyDisabledError` on any RNG access via the online variant; config error if `seed`/`as_of` invalid.
- **Determinism guarantees.** Same `seed` ⇒ identical generators and substreams; distinct labels ⇒ independent reproducible streams; `as_of` fixed.
- **Thread safety.** `Generator`s are not thread-safe; callers obtain a **per-thread** substream via a distinct `label` rather than sharing one generator.
- **Performance characteristics.** Negligible.
- **Memory characteristics.** Negligible.
- **Auditability.** Emits `seed, as_of, rng_enabled: bool`.
- **Testability.** Contract suite (same seed ⇒ identical streams; distinct labels independent; **online variant raises on `numpy_generator`/`derive`**).

**Offline entropy.** Seeded, labeled, reproducible substreams for KMeans (O7) and weight search (O9). **Online restrictions.** RNG disabled — ranking ties are id-based, not random; only `as_of` is exposed. **Reproducibility guarantees.** Deterministic sub-seed derivation; identical results across runs.

---

## §9 Adapter Dependency Graph

```
                 domain/  ports/  config.schema   (the only allowed project deps)
                    ▲        ▲          ▲
   ┌────────────────┼────────┼──────────┼───────────────────────────┐
   │ JsonlCandidateSource   ArtifactStore   CsvSubmissionSink         │
   │ JsonRunReportSink      DeterministicEntropy                      │
   │                                                                  │
   │ OnnxEmbeddingModel ──────┐  (locate "encoder.onnx")              │
   │ ParquetSemanticVectorStore ┐ (locate "candidate_vectors.parquet")│
   │                            └▶ ArtifactStore (via injected port)  │
   │ SentenceTransformerEmbedding  (OFFLINE; writes vectors+onnx)     │
   └──────────────────────────────────────────────────────────────────┘
                 ✗ never import engines/ or pipelines/
```

- All adapters depend on `domain/`, `ports/`, `config.schema` + their infra libraries.
- `OnnxEmbeddingModelAdapter` and `ParquetSemanticVectorStoreAdapter` consume artifacts **through the injected `ArtifactStorePort`** (`locate()`), not by importing the artifact-store adapter directly — preserving the port seam even between adapters.
- No adapter imports another adapter's class; cross-adapter use is always via a port.

---

## §10 Adapter Runtime Lifecycle

1. **Construction (composition root).** `pipelines/online` (or `pipelines/offline`) instantiates each adapter from resolved config + the artifact root, binding it to its port. Adapters never self-construct.
2. **Initialization.** Open handles/sessions/mmaps; verify what must be verified (artifact-store manifest self-hash, vector-store dim, onnx model id/dim).
3. **Steady state.** Serve port calls; read-only adapters tolerate concurrent reads; writer adapters are single-use.
4. **Disposal.** Close file handles, unmap vectors, release the onnx/torch session. Writer adapters finalize via atomic rename. Disposal is explicit (context-managed) so no handle leaks across a run.
5. **Re-entry.** A fresh run re-constructs adapters from scratch (deterministic); no adapter carries state between runs.

---

## §11 Offline Adapter Usage Matrix

| Stage | Adapters used |
|---|---|
| O0 Census | Jsonl source |
| O1 Normalization / O2 Validation | Jsonl source |
| O3 Honeypot / O4 Lexicon | (none — pure stages over O2 output) |
| O5/O6 Vocab/Concepts | SentenceTransformer (embed terms/concepts) |
| O7 Archetypes | DeterministicEntropy (offline RNG) |
| O8 Labeling | (offline tool; no port adapters) |
| O9 Weight Calib | DeterministicEntropy; CsvSubmissionSink (golden dry-run) |
| O13 Embeddings | SentenceTransformer (vectors + onnx export) |
| O14 Representation | Jsonl source (re-read) |
| O17 Packaging | ArtifactStore (verify) |
| O18 Repro Validation | ArtifactStore (verify_all); Jsonl source; Csv/JsonReport (dry-run); Onnx (parity) |

## §12 Online Adapter Usage Matrix

| Stage | Adapters used |
|---|---|
| R0 Artifact Loading | ArtifactStore (manifest + verify + locate); DeterministicEntropy (Online, `as_of`) |
| R1 Ingestion | Jsonl source |
| R2 Features | (none — pure) |
| R3 Semantic Hydration | ParquetSemanticVectorStore; OnnxEmbeddingModel (fallback only) |
| R4 Gates / R5 Scoring / R6 Ranking / R7 Reasoning | (none — pure engines) |
| R8 Submission | CsvSubmissionSink |
| R9 Run Report | JsonRunReportSink; ArtifactStore (manifest hashes) |

Ports/adapters appear only at R0/R1/R3/R8/R9 — the pure middle (R2/R4/R5/R6/R7) keeps the hot path testable and the budget predictable.

---

## §13 Failure Recovery Strategy

| Class | Adapter | Behavior |
|---|---|---|
| Artifact integrity (hash/version/coherence) | ArtifactStore | **Fail-fast, abort** — no degraded mode. |
| Manifest self-hash failure | ArtifactStore | `ManifestError`, abort. |
| Path escape / unsafe key | ArtifactStore | Reject (`..`/absolute/symlink-out) → `ArtifactContractError`. |
| IO / decompression | Jsonl source | `CandidateSourceError`, abort; deterministic full re-run recovers. |
| Malformed line | Jsonl source | `Malformed` record (data); pipeline policy decides. |
| Vector store corruption / dim mismatch | ParquetVectorStore | `VectorStoreError`, abort. |
| Missing candidate vector | ParquetVectorStore | `None`/`missing` (data) → onnx fallback. Never fatal. |
| Encode failure | Onnx | `EmbeddingError`; R3 marks candidate semantics `UNKNOWN`, run continues. |
| Online RNG misuse | DeterministicEntropy (Online) | `EntropyDisabledError` (programming error). |
| Submission would be invalid | CsvSubmissionSink | `SubmissionContractError`, abort, **no file** (atomic). |
| Report write failure | JsonRunReportSink | `ReportWriteError`, abort. |

**No network anywhere.** All recovery is local. Writers use temp+rename so a crash never yields a partial/rejectable artifact. Because the online run is short and deterministic, the canonical recovery is a full re-run.

---

## §14 Performance Budget

| Adapter | Online contribution | Bound |
|---|---|---|
| ArtifactStore | R0 streaming-hash of large artifacts | ≤ 20 s |
| Jsonl source | R1 stream + decode 100K | ≤ 35 s |
| ParquetVectorStore | R3 mmap gather + view | ≤ a few seconds |
| Onnx | R3 fallback (≈ 0 docs in full run) | ≈ 0 |
| CsvSubmissionSink | R8 write + hash | ≤ 1 s |
| JsonRunReportSink | R9 serialize | ≤ 1 s |
| DeterministicEntropy | `as_of` lookup | negligible |

Adapters consume a minority of the ~130 s online budget; the pure engines (R2/R5) dominate. **How it's held:** streaming hashing (verification ≈ free), mmap (no vector copy), lookup-not-encode, pinned threads (predictable latency), atomic small writes. Offline adapters (SentenceTransformer) are unbudgeted by design.

## §15 Memory Budget

| Adapter | Online peak (additional) |
|---|---|
| ArtifactStore | small artifacts (≤ few MB) + fixed hash buffer; large artifacts not loaded |
| Jsonl source | O(1) (one-line buffer) |
| ParquetVectorStore | mmap (~150 MB resident on touched pages) + id index (~few MB) |
| Onnx | session ~100–200 MB (loaded at R0) |
| CsvSubmissionSink / JsonRunReportSink / Entropy | negligible |

Total adapter contribution sits comfortably within the ≤ 4 GB online target; the `(N,D)` feature matrix (an engine/feature concern) is the larger consumer.

## §16 Determinism Guarantees

- **No wall clock** in any adapter logic; `DeterministicEntropy.as_of()` is the only time source, and it is injected.
- **No online RNG** (`OnlineEntropy` raises); offline RNG is seeded + labeled.
- **Byte-stable outputs:** CSV and the run-report `reproducible` block are reproducible; `output_sha256`/`report_sha256` stable for identical inputs.
- **Thread-invariant:** onnx threads pinned; vector ops are read-only gathers; reductions belong to engines (fixed order). Adapter outputs do not vary with host core count.
- **ε-bounded cross-runtime embeddings:** onnx ↔ sentence-transformers cosine ≥ 0.999, not bitwise — the single documented non-bitwise boundary.

## §17 Testing Strategy

- **Shared contract suites (Ports §16)** run identically against each real adapter **and** its in-memory fake, so the fake cannot drift from reality.
- **Adapter-specific:** Jsonl gzip-vs-plain parity + malformed placement; ArtifactStore tampered-byte + path-traversal + version-incompatibility; Onnx norm/shape/order + ε-parity + no-network; SentenceTransformer import-guard + export-parity (offline profile only); ParquetVectorStore round-trip + miss + dim-mismatch; CsvSink organizer-validator pass + byte equality + atomicity; JsonReportSink reproducible-block stability; Entropy online-raises + seeded-reproducibility.
- **Fakes over mocks:** behavioral fakes (`InMemoryArtifactStore`, `InMemoryVectorStore`, `StubEmbeddingModel`, `ListCandidateSource`, `CapturingSubmissionSink`, `CapturingRunReportSink`, `FixedEntropy`) — engines/pipelines test against fakes; adapters test against the contract suite + real IO in temp dirs.

## §18 Security Model

- **No network at online time.** Onnx adapter loads only the local hash-verified model; sentence-transformers is offline-only (import-guarded, `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`). No telemetry egress.
- **Artifact integrity before use.** Every artifact is sha256-verified against the manifest (which is self-hashed) before it is parsed, mmap'd, or turned into an onnx session — tamper/corruption is caught pre-use.
- **Safe deserialization only.** `numpy.load(allow_pickle=False)`; YAML via safe parsing through `config.schema`; JSON via stdlib; **no `pickle`, no arbitrary code execution** from any artifact.
- **Path containment.** Artifact keys resolve strictly within the artifact root; `..`, absolute keys, and symlinks escaping the root are rejected. Output writes go to a temp file **in the same directory** as the target (no cross-device/symlink races) then atomic rename.
- **Input hygiene.** Input lines decoded UTF-8 strict; malformed lines isolated as data; oversized lines/files flagged.
- **ONNX/model trust.** The model is treated as a hash-pinned artifact, not a downloadable; session creation happens only after hash verification.
- **Determinism as integrity.** No clock/RNG online means outputs can't be perturbed by environment; reproducibility hashes make any deviation detectable.

## §19 Import Boundary Rules

1. Adapters import only `domain/`, `ports/`, `config.schema`, and infrastructure libraries.
2. Adapters **never** import `engines/` or `pipelines/` (import-linter forbidden contract, CI-blocking).
3. Cross-adapter dependency is **always via a port** (`OnnxEmbeddingModelAdapter`/`ParquetSemanticVectorStoreAdapter` reach artifacts through the injected `ArtifactStorePort`, not the concrete store adapter).
4. `pipelines.online.*` is forbidden from importing `adapters.st_embedder`, `sentence_transformers`, `sklearn`, or any networking module (online containment).
5. Only the pipeline composition roots construct adapters; nothing else instantiates them.

## §20 Adapter Implementation Readiness Checklist

- [ ] Each adapter implements exactly its port's frozen surface — no extra public methods, no business logic.
- [ ] `JsonlCandidateSourceAdapter`: gzip-transparent, UTF-8 strict, `source_index` enumeration, `Ok`/`Malformed` tagging, blank-line skip, O(1) memory.
- [ ] `FilesystemArtifactStoreAdapter`: manifest self-hash, streaming per-key sha256, `locate()` for parquet/onnx, YAML via `load_text`+`config.schema`, path containment, coherence checks, fail-fast.
- [ ] `OnnxEmbeddingModelAdapter`: CPU EP, pinned threads, sequential exec, fixed pooling+L2 norm, order-preserving, no network, `dim`/`model_id` asserted.
- [ ] `SentenceTransformerEmbeddingAdapter`: import-guarded offline, pinned revision+threads, vector + onnx export with ε-parity verification.
- [ ] `ParquetSemanticVectorStoreAdapter`: mmap read-only, in-memory id index, `get`/`get_many`/`view_all`, miss ⇒ `None`/`missing`, dim validation.
- [ ] `CsvSubmissionSinkAdapter`: header/order/precision fixed, RFC-4180, UTF-8 no BOM, `\n`, atomic temp+rename, post-format invariant re-assert, `output_sha256`.
- [ ] `JsonRunReportSinkAdapter`: sorted keys, fixed floats, enums-by-value, `reproducible`/`audit` split, `report_sha256`, atomic write.
- [ ] `DeterministicEntropyAdapter`: Offline (seeded, labeled substreams) + Online (RNG-disabled, `as_of`-only, raises on RNG).
- [ ] All adapters pass the shared port contract suite **and** their adapter-specific tests, identically to their fakes.
- [ ] import-linter contracts green (no `engines`/`pipelines` import; online containment).
- [ ] Security checks: safe deserialization, path containment, offline guarantees, hash-before-use.
- [ ] Determinism: thread-invariant outputs; byte-stable CSV + run-report `reproducible` block across two runs.

---

**Implementation order.** `FilesystemArtifactStoreAdapter` first (everything else verifies through it) → `JsonlCandidateSourceAdapter` → `ParquetSemanticVectorStoreAdapter` → `OnnxEmbeddingModelAdapter` → `CsvSubmissionSinkAdapter` → `JsonRunReportSinkAdapter` → `DeterministicEntropyAdapter` → `SentenceTransformerEmbeddingAdapter` (offline profile, last). Each adapter is merged only when it passes its shared contract suite identically to its fake and its adapter-specific IO tests in a temp directory.
