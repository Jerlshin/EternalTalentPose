# `pipelines/online/` — The Online Ranking Run

Runs every time `redstack rank` is invoked, against an already-built `artifacts/` set, inside a fixed time and memory budget, with no network access. This package — and everything it transitively imports — is forbidden by an `import-linter` contract from importing `sentence_transformers`, `sklearn`, `adapters.st_embedder`, or any networking module: a build break, not a runtime check. See [`/ARCHITECTURE.md` §5.2](../../../../ARCHITECTURE.md#52-online-pipeline--r0-through-r9) and [`docs/specs/REDSTACK_ONLINE_PIPELINE.md`](../../../../docs/specs/REDSTACK_ONLINE_PIPELINE.md) for the exhaustive specification.

## File inventory

| File | Role |
|---|---|
| [`pipeline.py`](pipeline.py) | The pure R0–R9 orchestrator — the literal ranking spine. Sequences the stage functions in `stages.py` strictly in order, threading the growing `CandidateRepresentation` population, and records per-stage wall time and peak RSS via `resource`/`time` for the run report's budget block. |
| [`compose.py`](compose.py) | The composition root: `run_online_rank()`. Instantiates every online-safe adapter (`FilesystemArtifactStoreAdapter`, `JsonlCandidateSourceAdapter`, `OnnxEmbeddingModelAdapter`, `ParquetSemanticVectorStoreAdapter`, `CsvSubmissionSinkAdapter`, `JsonRunReportSinkAdapter`, `OnlineEntropy`), builds the ONNX session options via `config.determinism`, and assembles the immutable run context. This is the function `redstack rank` calls. |
| [`stages.py`](stages.py) | One pure callable per stage, `r0_load` through `r9_report`. Ports are touched only inside `r0`, `r1`, `r3`, `r8`, and `r9` — `r2`, `r4`, `r5`, `r6`, `r7` make no port calls at all, which is what keeps the middle of the run's timing predictable and the stages independently unit-testable with plain domain fixtures. |

## Stage summary

| Stage | Function | Touches a port? | What it does |
|---|---|---|---|
| R0 | `r0_load` | yes | Load and hash-verify every artifact; bind adapters; build the immutable run context. Aborts on any integrity failure. |
| R1 | `r1_ingest` | yes | Stream the candidate file into validated, typed records in constant memory. |
| R2 | (feature extraction) | no | Compute every structural and behavioral feature into the columnar quality-vector matrix. |
| R3 | (semantic hydration) | yes | Look up each candidate's precomputed vector (O(1) gather); fall back to on-the-fly encoding only for a store miss. |
| R4 | (gates & eligibility) | no | Apply the integrity and eligibility verdicts; build the score floor mask. |
| R5 | (scoring) | no | Apply the locked weights, the floor mask, and the bounded multipliers. |
| R6 | (ranking) | no | Deterministic sort, top-100 cut, ascending-id tie-break; the `Ranking` factory enforces all six structural invariants. |
| R7 | (reasoning) | no | Generate evidence-grounded reasoning for the top 100 only. |
| R8 | `r8_submit` (or equivalent) | yes | Validate and write `submission.csv`, atomically. |
| R9 | `r9_report` | yes | Write `run_report.json`: artifact hashes, timings, honeypot rate, eligibility summary, budget compliance. |

## Why the sequencing matters

The strictly-sequential order is what makes the run's worst-case timing additive and auditable: each stage's wall time is recorded independently, and `redstack rank` exits non-zero if the total falls outside budget — there is no parallel-stage interleaving that could hide a regression in one stage behind a fast one elsewhere. See [`/ARCHITECTURE.md` §10](../../../../ARCHITECTURE.md#10-determinism-and-performance) for the determinism guarantees this ordering and the rest of the online package provide.
