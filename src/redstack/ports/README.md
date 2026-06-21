# `ports/` — The Hexagon Boundary

`ports/` declares, as `typing.Protocol` interfaces, every contract the core (`domain`, `features`, `engines`) needs from the outside world. A `Protocol` has zero runtime cost and zero inheritance coupling — an adapter satisfies a port simply by having the right methods, not by subclassing anything from this package. This package may import only `domain/` and stdlib `typing`/numpy typing; it may never import `adapters/`, `engines/`, `pipelines/`, `config.loader`, or `observability/`. See [`/ARCHITECTURE.md` §4](../../../ARCHITECTURE.md#4-layer-reference) and [`docs/specs/REDSTACK_PORTS_LAYER.md`](../../../docs/specs/REDSTACK_PORTS_LAYER.md).

## The verdicts-vs-failures discipline

A **missing candidate** or a **malformed input line** is data — returned as a value (`None`, a tagged `Malformed` record), never an exception. An **integrity or contract violation** — a hash mismatch, a wrong vector dimension, a broken submission invariant — always raises, because there is no way to detect a silently-degraded ranking after the fact, so silent degradation is treated as strictly worse than a loud failure.

## File inventory

| File | Protocol(s) / contents | Real adapter | Used by |
|---|---|---|---|
| [`_types.py`](_types.py) | Shared DTOs with no behavior: `ArtifactLocator`, `EmbeddingManifest`, `ArtifactEntry`, `Manifest`, `SourceOk`/`SourceMalformed`, `BulkVectorResult`, `SubmissionReceipt`, `ReportReceipt`, and the structural `RunReport`/`ReproducibleBlock`/`AuditBlock`/`BudgetBlock` protocols. | — (imported by every other port) | all ports |
| [`embedding.py`](embedding.py) | `EmbeddingModelPort` (`dim`, `model_id`, `encode`), `OnnxExportCapable`, `EmbeddingError` | `adapters/onnx_embedder.py` (online), `adapters/st_embedder.py` (offline) | `SemanticEngine` (R3 fallback); offline embedding generation |
| [`semantic_index.py`](semantic_index.py) | `SemanticVectorStorePort` (`dim`, `contains`, `get`, `get_many`, `view_all`), `VectorStoreError` | `adapters/vector_store_parquet.py` | `SemanticEngine` (R3) |
| [`artifact_store.py`](artifact_store.py) | `ArtifactStorePort` (`manifest`, `verify_all`, `load_bytes/text/json/npy`, `locate`), `ManifestError` | `adapters/artifact_store_fs.py` | pipeline R0; offline packaging/verification |
| [`candidate_source.py`](candidate_source.py) | `CandidateSourcePort` (`stream`, `count`), `CandidateSourceError` | `adapters/candidate_jsonl.py` | pipeline R1; multiple offline stages |
| [`submission_sink.py`](submission_sink.py) | `SubmissionSinkPort` (`write`), `SubmissionWriteError`, `SubmissionContractError` | `adapters/submission_csv.py` | pipeline R8 |
| [`run_report_sink.py`](run_report_sink.py) | `RunReportSinkPort` (`write`), `ReportWriteError` | `adapters/run_report_json.py` | pipeline R9; offline build report |
| [`rng.py`](rng.py) | `DeterministicEntropyPort` (`seed`, `as_of`, `derive`, `numpy_generator`), `EntropyDisabledError` | `adapters/entropy.py` (`OfflineEntropy`/`OnlineEntropy`) | engines via injected `as_of`; offline clustering/calibration RNG |
| [`online.py`](online.py) | `OnlineEntropyPort` — the narrower `seed` + `as_of`-only surface the online run is restricted to, plus re-exports of `RunReportSinkPort`/`SemanticVectorStorePort`/`SubmissionSinkPort` for convenient online-side importing. | `adapters/entropy.py` (`OnlineEntropy`) | `pipelines/online/*` |

## Why `EntropyDisabledError` exists

Online ranking has no legitimate source of randomness — every tie is resolved by ascending candidate ID. `DeterministicEntropyPort.derive()`/`numpy_generator()` exist for the *offline* build (seeded k-means initialization, the gold-label weight search), and the online entropy adapter raises `EntropyDisabledError` if either is ever called from the online path — a structural guarantee, not a code-review rule, that the online run cannot introduce nondeterminism through its one RNG seam.

## Testing

Every port has exactly one shared, parametrized contract suite (`tests/contract/`) that runs identically against the real adapter and an in-memory fake. This is what prevents a fake from drifting from what the real adapter actually does — if the fake passes a check the adapter fails, the suite is red. See [`tests/README.md`](../../../tests/README.md).
