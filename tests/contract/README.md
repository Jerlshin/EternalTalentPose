# `tests/contract/` — Shared Port Conformance Suites

One abstract, parametrized behavioral suite **per port**, run against both the real adapter (in [`src/redstack/adapters/`](../../src/redstack/adapters/README.md)) and its in-memory fake (in [`../fixtures/`](../fixtures/README.md)). This is the mechanism that prevents a fake from drifting from what the real adapter actually does in production — if the fake passes a check the adapter fails, or vice versa, the suite is red.

## What belongs here

Per [`docs/specs/REDSTACK_TESTING_STRATEGY.md` §6](../../docs/specs/REDSTACK_TESTING_STRATEGY.md), one suite per port in [`src/redstack/ports/README.md`](../../src/redstack/ports/README.md):

| Port | Compliance criteria the suite must assert |
|---|---|
| `CandidateSourcePort` | File order preserved; a malformed line yields a tagged record, never an exception; gzip and plain input produce identical output. |
| `ArtifactStorePort` | A tampered byte raises `ArtifactContractError`; a missing key raises; the happy path returns exact byte fidelity. |
| `EmbeddingModelPort` | Output shape `(n, dim)`, `float32`, unit-norm rows within epsilon; two calls on the same input produce identical output. |
| `SemanticVectorStorePort` | A known id round-trips exactly; a missing id returns `None`/an empty match, never raises. |
| `SubmissionSinkPort` | The emitted CSV passes the external structural validator; byte-identical output for an identical `Ranking`. |
| `RunReportSinkPort` | The reproducible block is byte-stable across two runs with identical inputs; the audit block is excluded from that comparison. |
| `DeterministicEntropyPort` | The same seed yields identical streams; the online variant raises `EntropyDisabledError` on any RNG call. |

This directory currently holds only its package marker (`__init__.py`) — the suites above are the conformance contract to implement as each adapter is exercised in CI. See [`tests/fixtures/README.md`](../fixtures/README.md) for the fakes each suite parametrizes against.
