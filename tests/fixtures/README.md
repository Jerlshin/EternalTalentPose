# `tests/fixtures/` — Shared Fakes and Candidate Fixtures

The shared, versioned ground truth every other test category builds on. Fixtures here are either **behavioral fakes** (in-memory implementations of a port, passing the same contract suite as the real adapter) or **frozen candidate records** with an expected-output table. Fixtures are owned centrally so a single change to one propagates consistently to every category that depends on it.

## File inventory

| File | Contents |
|---|---|
| [`fake_ports.py`](fake_ports.py) | Every canonical port fake: `ListCandidateSource` (`CandidateSourcePort`, including a `BadLine` helper for malformed-input fixtures), `InMemoryArtifactStore` (`ArtifactStorePort`), `StubEmbeddingModel` (`EmbeddingModelPort` — derives each vector deterministically from a hash of the input text, then normalizes, so it's reproducible with no model download), `InMemoryVectorStore` (`SemanticVectorStorePort`), `CapturingSubmissionSink` (`SubmissionSinkPort`), `CapturingRunReportSink` (`RunReportSinkPort`), and `FixedEntropy` (`DeterministicEntropyPort`). |

## Why fakes, not mocks

Every fake here is a small, real, in-memory implementation of its port — not a record of expected call sequences. This means a test using `InMemoryVectorStore` is exercising the same `get`/`get_many`/`view_all` behavior a real `ParquetSemanticVectorStoreAdapter` call site would see, just backed by a Python dict instead of a memory-mapped file. Every fake here is required to pass the exact same shared contract suite (in [`../contract/`](../contract/README.md)) that the real adapter passes — that's what guarantees a fake can't silently drift from reality.

## Adding a candidate fixture

Per [`docs/specs/REDSTACK_TESTING_STRATEGY.md` §16](../../docs/specs/REDSTACK_TESTING_STRATEGY.md), the intended fixture families beyond the port fakes are: representative valid candidates, one exemplar per honeypot/integrity-flag class, a keyword-stuffer, a consulting-only career, an ideal job-description match, one exemplar per eligibility code (hard and soft), malformed/schema-invalid records, and edge-case behavioral-signal profiles (all-sentinel values, stale activity, boundary notice periods). A fixture change that alters an expected output requires re-blessing the dependent golden snapshots in [`../golden/`](../golden/README.md) in the same commit.
