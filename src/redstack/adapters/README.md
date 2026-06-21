# `adapters/` — Infrastructure

**This is the only package in the repository permitted to touch the filesystem, read or write CSV/JSONL/Parquet/NumPy, or load ONNX Runtime or sentence-transformers.** Each adapter implements exactly one port and contains no business logic — an adapter that finds itself making a ranking, scoring, or eligibility decision is mis-scoped. Adapters may import `domain/`, `ports/`, `config.schema`, and infrastructure libraries; they may never import `engines/` or `pipelines/`. Adapters are instantiated only inside a pipeline composition root (`pipelines/online/pipeline.py`, `pipelines/offline/pipeline.py`) and bound to a port there. See [`/ARCHITECTURE.md` §4](../../../ARCHITECTURE.md#4-layer-reference) and [`docs/specs/REDSTACK_ADAPTERS_LAYER.md`](../../../docs/specs/REDSTACK_ADAPTERS_LAYER.md).

## File inventory

| File | Class | Implements (port) | Library | Offline | Online |
|---|---|---|---|---|---|
| [`candidate_jsonl.py`](candidate_jsonl.py) | `JsonlCandidateSourceAdapter` | `CandidateSourcePort` | stdlib `io`/`gzip`/`json` | yes | yes (R1) |
| [`artifact_store_fs.py`](artifact_store_fs.py) | `FilesystemArtifactStoreAdapter` | `ArtifactStorePort` | `pathlib`/`hashlib`/`json`/`numpy` | yes (packaging, verification) | yes (R0) |
| [`onnx_embedder.py`](onnx_embedder.py) | `OnnxEmbeddingModelAdapter` | `EmbeddingModelPort` | `onnxruntime` (CPU execution provider) | no | yes (R3 fallback only) |
| [`st_embedder.py`](st_embedder.py) | `SentenceTransformerEmbeddingAdapter` | `EmbeddingModelPort` | `sentence-transformers`, `torch` | yes — **offline only, import-guarded** | no |
| [`vector_store_parquet.py`](vector_store_parquet.py) | `ParquetSemanticVectorStoreAdapter` | `SemanticVectorStorePort` | `pyarrow` / numpy memory-map | no | yes (R3) |
| [`submission_csv.py`](submission_csv.py) | `CsvSubmissionSinkAdapter` | `SubmissionSinkPort` | stdlib `csv`/`io`/`hashlib` | optional (dry-run validation) | yes (R8) |
| [`run_report_json.py`](run_report_json.py) | `JsonRunReportSinkAdapter` | `RunReportSinkPort` | stdlib `json`/`io`/`hashlib` | yes (build report) | yes (R9) |
| [`entropy.py`](entropy.py) | `OfflineEntropy`, `OnlineEntropy` | `DeterministicEntropyPort` (`OnlineEntropy` also satisfies the narrower `OnlineEntropyPort`) | `numpy` | yes (full RNG) | yes (`as_of` only — raises `EntropyDisabledError` on any RNG call) |

## Why `st_embedder.py` is import-guarded

`adapters/st_embedder.py` is the only adapter never permitted to run online — it raises on import if an "online" environment marker is set, as defense-in-depth on top of the import-linter contract that already forbids `pipelines.online.*` from importing it, `sentence_transformers`, or `sklearn` at all. Two independent layers have to fail simultaneously for a heavyweight training runtime to reach the online ranking path.

## Key implementation guarantees

- **`artifact_store_fs.py`** streams a SHA-256 hash of every artifact *while reading it*, so integrity verification costs essentially nothing beyond the read that would happen anyway. It rejects path traversal and uses only safe deserialization (`numpy.load(allow_pickle=False)`, no `pickle`).
- **`vector_store_parquet.py`** memory-maps the candidate vector block read-only; a missing candidate id returns `None`/an empty match, never an exception — that's the signal that triggers the rare ONNX fallback encode.
- **`onnx_embedder.py`** pins thread/op counts and uses a sequential execution mode for bitwise determinism within a run; cross-runtime agreement with the offline `sentence-transformers` vectors is guaranteed only within a cosine-similarity epsilon, not bitwise — the one documented non-bitwise boundary in the system.
- **`submission_csv.py`** writes atomically (temp file + rename) and re-asserts the ranking's monotonicity and tie-break invariants at the emitted decimal precision before finalizing — it will abort and leave no file at all rather than write a submission that could fail external validation.
- **`entropy.py`**'s `OnlineEntropy` variant exists specifically to make "the online run has no source of randomness" enforceable: calling its RNG methods raises rather than silently returning something.

## Testing

Every adapter is tested against the same shared port contract suite as its in-memory fake (`tests/contract/`), plus adapter-specific tests covering corruption handling, cross-runtime parity, and failure injection using real IO in a temporary directory. See [`tests/README.md`](../../../tests/README.md).
