# Design Specifications

This directory holds the frozen, line-level design specifications for every layer of RedStack. They are the engineering source of truth: [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) and every package-level `README.md` in this repository are distilled, dual-audience summaries of what is specified exhaustively here. Where a summary and a spec disagree, **the spec is authoritative.**

These documents are frozen design artifacts, not living guides — they are not edited as part of routine development. A change to one represents a deliberate architecture revision, not a documentation update.

## Reading order

| Document | Scope |
|---|---|
| [`REDSTACK_ARCHITECTURE.md`](REDSTACK_ARCHITECTURE.md) | The master architecture: repository tree, module-by-module breakdown, dependency direction rules, configuration/artifact/testing architecture, CLI architecture, performance considerations. The primary source for [`/ARCHITECTURE.md`](../../ARCHITECTURE.md). |
| [`REDSTACK_REPOSITORY_LAYOUT.md`](REDSTACK_REPOSITORY_LAYOUT.md) | The repository-and-file-layout translation of the architecture: every directory and file, owner, dependencies, import restrictions, and the eight import-linter contracts, file by file. |
| [`REDSTACK_DOMAIN_LAYER.md`](REDSTACK_DOMAIN_LAYER.md) | Every value object under `src/redstack/domain/`: NewTypes, enums, the ten `CandidateRepresentation` slices, scoring/ranking/reasoning/validation models, immutability and copy-on-write strategy, memory and determinism considerations. |
| [`REDSTACK_PORTS_LAYER.md`](REDSTACK_PORTS_LAYER.md) | The seven hexagon-boundary `Protocol`s under `src/redstack/ports/`: method signatures, contracts, failure modes, the artifact/embedding/vector-store/submission/run-report contracts, and the shared contract-testing strategy. |
| [`REDSTACK_FEATURE_LAYER.md`](REDSTACK_FEATURE_LAYER.md) | All 30 feature groups, the job-description latent families, the career-intelligence and behavioral-composite features, the honeypot detectors, the feature store design, and the explainability chain. |
| [`REDSTACK_ENGINE_LAYER.md`](REDSTACK_ENGINE_LAYER.md) | The fourteen logical engine services (mapped onto the eleven physical modules under `src/redstack/engines/`): inputs, outputs, internal workflow, failure modes, performance budget, and testability for each. |
| [`REDSTACK_ADAPTERS_LAYER.md`](REDSTACK_ADAPTERS_LAYER.md) | The eight concrete adapters under `src/redstack/adapters/`: runtime/library, lifecycle, internal workflow, security model, and performance/memory budget for each. |
| [`REDSTACK_OFFLINE_PIPELINE.md`](REDSTACK_OFFLINE_PIPELINE.md) | The offline build, stage by stage (O0–O18): purpose, inputs, outputs, algorithm, complexity, and the artifact registry and execution graph. |
| [`REDSTACK_ONLINE_PIPELINE.md`](REDSTACK_ONLINE_PIPELINE.md) | The online ranking run, stage by stage (R0–R9): runtime/memory budget per stage, validation checkpoints, failure modes, and the one-page final specification. |
| [`REDSTACK_TESTING_STRATEGY.md`](REDSTACK_TESTING_STRATEGY.md) | The full testing architecture: the test pyramid, per-layer test obligations, the honeypot and reasoning validation suites, determinism and performance testing, coverage strategy, and the reproducibility certification gate. |

## How these relate to the rest of the repository

- [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) is the reader-facing synthesis of all ten documents above.
- Every `README.md` under `src/redstack/*` is the package-level synthesis of the one or two specs that govern that package (e.g. `src/redstack/domain/README.md` summarizes `REDSTACK_DOMAIN_LAYER.md`).
- [`tests/README.md`](../../tests/README.md) summarizes `REDSTACK_TESTING_STRATEGY.md`.
