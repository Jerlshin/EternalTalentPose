# `src/redstack/` — Package Overview

Every importable line of RedStack lives under this package. The subdirectories below are the hexagonal layers described in [`/ARCHITECTURE.md` §4](../../ARCHITECTURE.md#4-layer-reference); dependencies flow inward only, and the boundary between them is enforced by eight `import-linter` contracts in `pyproject.toml` — a violated import is a CI-blocking build break, not a style note.

```text
cli ──▶ pipelines ──▶ engines ──▶ features ──▶ domain
                       │            │            ▲
                       ▼            └───────────▶│
                     ports ─────────────────────▶ domain
adapters ──▶ ports ──▶ domain          config ──▶ domain
pipelines ──▶ adapters   (composition root ONLY)
observability ──▶ domain
```

## Subpackages

| Package | Responsibility | README |
|---|---|---|
| `domain/` | Pure data models and invariants — the candidate aggregate, scoring, ranking, and reasoning value objects. Zero IO, zero ML, zero clock. | [`domain/README.md`](domain/README.md) |
| `ports/` | The seven `typing.Protocol` interfaces that are the hexagon's boundary — what the core needs from the outside world, with no concrete dependency. | [`ports/README.md`](ports/README.md) |
| `features/` | Pure, deterministic feature extraction — the 30 feature groups that turn a raw candidate record into structured, evidence-backed signal. | [`features/README.md`](features/README.md) |
| `engines/` | The 11 domain services that apply business judgment: integrity, eligibility, semantic fit, scoring, ranking, reasoning. | [`engines/README.md`](engines/README.md) |
| `config/` | Typed configuration schema, the deterministic YAML loader, and the determinism policy (seeds, thread pinning). | [`config/README.md`](config/README.md) |
| `adapters/` | Concrete infrastructure implementations of the ports — the only layer permitted to touch ONNX Runtime, Parquet, or the filesystem. | [`adapters/README.md`](adapters/README.md) |
| `pipelines/` | Orchestration and the composition roots: the offline build (O0–O18) and the online ranking run (R0–R9). | [`pipelines/README.md`](pipelines/README.md) |
| `observability/` | Structured logging, per-stage timing with a hard budget guard, and the run-report model. | [`observability/README.md`](observability/README.md) |
| `cli/` | The `redstack` command-line entrypoints — the thinnest layer, no business logic. | [`cli/README.md`](cli/README.md) |

## Layer import rules at a glance

| Layer | May import | May never import |
|---|---|---|
| `domain` | stdlib, `pydantic`, `numpy` | everything else in this package |
| `ports` | `domain` | `features`, `engines`, `adapters`, `pipelines`, `config`, `observability`, `cli` |
| `features` | `domain`, `config.schema` | `ports`, `engines`, `adapters`, `pipelines`, `observability`, `cli`, `config.loader`, any ML/network module |
| `engines` | `domain`, `ports`, `features`, `config.schema` | `adapters`, `pipelines`, `observability` IO, `config.loader`, any ML/network module, **each other** |
| `config.schema` | `domain`, `pydantic` | — |
| `config.loader` | `config.schema`, `pyyaml`, stdlib | only reachable from `pipelines`/`cli` |
| `adapters` | `domain`, `ports`, `config.schema`, infrastructure libraries | `engines`, `pipelines` |
| `pipelines` | all of the above (and is the only package that instantiates `adapters`) | — |
| `observability` | `domain` | `ports` (and `ports` never imports `observability`) |
| `cli` | `pipelines`, `config`, `observability` | direct business logic |

The single most important rule for the system's compute budget: **`pipelines.online` and everything it transitively imports is forbidden from importing `sentence_transformers`, `sklearn`, `adapters.st_embedder`, or any networking module.** This is what makes "the online ranking run cannot pull in a training runtime" a structural fact rather than a hope.
