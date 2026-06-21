# `pipelines/offline/` — The Offline Build

Runs once per artifact rebuild, with an unbounded compute budget. Reads `data/` and `configs/`; writes the entire `artifacts/` tree. See [`/ARCHITECTURE.md` §5.1](../../../../ARCHITECTURE.md#51-offline-pipeline--o0-through-o18) for the full O0–O18 stage table and [`docs/specs/REDSTACK_OFFLINE_PIPELINE.md`](../../../../docs/specs/REDSTACK_OFFLINE_PIPELINE.md) for the exhaustive specification.

## File inventory

| File | Role |
|---|---|
| [`pipeline.py`](pipeline.py) | `OfflinePipeline` — the pure orchestrator: declares the O0–O18 stage set and its dependency graph, delegates `plan()` (topologically order and identify stale stages) / `run()` (execute) / `finalize()` (package) to the runner and registry below. Owns no IO of its own. |
| [`compose.py`](compose.py) | The composition root: `run_offline_build()` and `run_offline_build_with_locked_heuristics()`. Instantiates every adapter (`JsonlCandidateSourceAdapter`, `OfflineEntropy`, `SentenceTransformerEmbeddingAdapter`, ...), binds them to ports, builds the `OfflinePipelineContext`, and drives the run. This is the function the `redstack build` CLI verb calls. |
| [`context.py`](context.py) | `OfflinePipelineContext` — the resolved, immutable build environment: config, bound ports, seed, `as_of`, output roots, the `FeatureRegistry`/`FeatureLayout` constant. |
| [`graph.py`](graph.py) | `OfflineExecutionGraph`, `OFFLINE_EXECUTION_GRAPH`, `StageNode` — the fixed O0–O18 dependency DAG and its deterministic topological order (ties broken by stage id). |
| [`registry.py`](registry.py) | `OfflineArtifactRegistry`, `OFFLINE_ARTIFACT_REGISTRY`, `ArtifactSpec`, `ArtifactKind`, `ValidationOutcome` — the typed catalog of every artifact this build is expected to produce: key, schema, owner stage, version, and validator. Every produced artifact is checked against this registry before being manifested. |
| [`runner.py`](runner.py) | `OfflinePipelineRunner` — executes the plan with checkpoint-based resume (skip a stage whose inputs/version/config haven't changed since its last successful run), per-stage timing, and failure quarantine (`StageReceipt`s). |
| [`build_artifact_store.py`](build_artifact_store.py) | Builds the final, hash-verified artifact-store representation (the manifest construction logic) from the registry's validated outputs — the bridge between "the runner finished" and "`artifacts/MANIFEST.json` exists and is self-consistent." |
| [`stages/`](stages/README.md) | One module per offline stage, O0 through O18. See [`stages/README.md`](stages/README.md). |

## Resume and incremental rebuilds

The runner keys each stage's staleness on `(input artifact hashes, stage code version, relevant config slice)`. A clean rebuild and an incremental rebuild of an unchanged stage set produce byte-identical artifacts for every deterministic stage; only the embedding-generation stage (which depends on a third-party model's floating-point behavior across hosts) is held to an epsilon-similarity bound instead of bit-exact equality.

## How a build fails

A stage raises → its partial output is quarantined (never registered in the manifest) → every downstream stage is marked stale → the build aborts (or continues past non-critical stages, if configured to). There is no path by which a corrupted or partially-built artifact set is silently manifested as complete.
