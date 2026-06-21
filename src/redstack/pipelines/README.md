# `pipelines/` — Orchestration and the Composition Roots

This is the application layer, and the **only** package permitted to import both `engines/` and `adapters/` at once — every concrete adapter in the system is instantiated and bound to a port somewhere under this package, never anywhere else. See [`/ARCHITECTURE.md` §5](../../../ARCHITECTURE.md#5-two-pipelines-one-codebase) for the full offline/online split and [`docs/specs/REDSTACK_OFFLINE_PIPELINE.md`](../../../docs/specs/REDSTACK_OFFLINE_PIPELINE.md) / [`docs/specs/REDSTACK_ONLINE_PIPELINE.md`](../../../docs/specs/REDSTACK_ONLINE_PIPELINE.md) for the full stage-by-stage specification.

## File inventory

| Path | Contents |
|---|---|
| [`context.py`](context.py) | `RunContext` and `canonical_config_hash()` — the small, immutable carrier shared by both the offline and online composition roots: resolved config plus the config hash recorded in every run report. |
| [`offline/`](offline/README.md) | The offline build: stages O0–O18, the execution graph, the artifact registry, the resumable runner, and the composition root. |
| [`online/`](online/README.md) | The online ranking run: stages R0–R9 and the composition root. |

## The two physically separate pipelines

| | [`offline/`](offline/README.md) | [`online/`](online/README.md) |
|---|---|---|
| Entry point | `redstack build` | `redstack rank` |
| Runtime budget | unbounded | ≤5 minutes (internal target ≤150s) |
| May import | anything, including `sentence-transformers`, `scikit-learn` | `engines`, `domain`, `ports`, `config`, `observability`, and only the online-safe adapters — `import-linter` forbids it from importing `sentence_transformers`, `sklearn`, `adapters.st_embedder`, or any networking module, transitively |
| Writes | `artifacts/` | `submission.csv`, `run_report.json` |

This split is enforced mechanically, not by convention: the **online containment** import-linter contract is one of the eight boundary contracts in `pyproject.toml`, and a violation fails the build the same way a syntax error would.

## Composition-root pattern

Both `offline/compose.py` and `online/compose.py` are the only modules in the entire codebase that construct a concrete adapter (`FilesystemArtifactStoreAdapter`, `OnnxEmbeddingModelAdapter`, `ParquetSemanticVectorStoreAdapter`, etc.) and wire it to the port an engine expects. `offline/pipeline.py` and `online/pipeline.py` are pure orchestrators over the already-wired context — they sequence stages, they never instantiate infrastructure themselves.
