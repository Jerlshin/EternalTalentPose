# RedStack

**A deterministic, evidence-grounded semantic talent-intelligence and candidate-ranking platform.**

RedStack takes a job description and a pool of candidate profiles and produces a ranked, explained shortlist. It is built to resist keyword gaming, gate out internally-impossible ("honeypot") profiles before they're ever scored, and produce a reasoning sentence for every ranked candidate that is mechanically tied to a real fact on their profile — never a templated or hallucinated claim.

The system runs in two physically separate modes: an unbounded **offline build** that pre-computes embeddings, clusters, and calibrated scoring weights into a hash-pinned artifact set, and a **CPU-only, network-isolated online ranking run** that applies those artifacts to a live candidate pool within a fixed time and memory budget. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design — it is the place to start.

> **Architecture is frozen.** The system design, layer boundaries, and stage contracts described in [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/specs/](docs/specs/README.md) are not to be redesigned in place; changes are additive within the existing structure.

## Why it's structured this way

- **Keyword-stuffing resistance.** Competency is computed as an evidence aggregate (endorsement × time-on-skill × corroboration in actual role descriptions × semantic match), never a keyword flag.
- **Integrity before relevance.** A dedicated integrity engine detects impossible profiles and floors them out of the ranking before scoring runs, not as a post-hoc filter.
- **Hexagonal isolation.** Business logic (`domain`, `features`, `engines`) depends only on abstract interfaces (`ports`); infrastructure (`adapters`) implements them. The online ranking package is import-linter-forbidden from ever pulling in a training runtime or a network library — enforced as a CI-blocking build break, not a convention.
- **Determinism by construction.** No wall clock, no online randomness, fixed-precision arithmetic, pinned thread counts. Identical inputs always produce a byte-identical output.

## Repository layout

```text
redstack/
├── README.md                 you are here
├── ARCHITECTURE.md           full system architecture — start here
├── pyproject.toml            package metadata, ruff/mypy/pytest/import-linter config
├── Makefile                  task aliases over uv + the redstack CLI
├── submission_metadata.yaml  release/provenance metadata
│
├── configs/                  human-authored, declarative behavior (weights, anchors, gates, lexicon...)
├── data/                     raw inputs — gitignored, see docs/runbook.md to populate
├── artifacts/                offline build output — gitignored, hash-pinned, rebuilt via `redstack build`
│
├── src/redstack/             all importable code
│   ├── domain/                pure data models + invariants
│   ├── ports/                 Protocol interfaces (the hexagon boundary)
│   ├── features/              pure feature extraction (30 feature groups)
│   ├── engines/                the 11 domain services that apply business judgment
│   ├── config/                 typed config schema, loader, determinism policy
│   ├── adapters/               infrastructure implementations of the ports
│   ├── pipelines/              orchestration: offline build (O0-O18), online ranking (R0-R9)
│   ├── observability/          logging, stage timing, the run-report model
│   └── cli/                    the `redstack` command-line entrypoints
│
├── tests/                     unit, property, contract, golden, integration, determinism suites
├── scripts/                   reproduce wrapper, sample-pool builder, submission analytics
└── docs/                      architecture companion, ADRs, runbook, frozen design specs, reference assets
```

Every directory above has its own `README.md` describing its specific contents and linking back to the relevant section of [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

```bash
uv sync --frozen --group core --group dev   # hot path + tooling
uv pip install -e .                          # src/ layout: install before tests run
```

For the offline build's heavier dependencies (`sentence-transformers`, `scikit-learn`), add the `offline` group:

```bash
uv sync --frozen --group core --group offline --group dev
```

Populate `data/raw/candidates.jsonl` (and, optionally, `data/golden/golden_labels.csv` for weight calibration) before building — see [docs/runbook.md](docs/runbook.md).

## Build, rank, validate

```bash
make build      # offline:  O0..O18  -> artifacts/ + MANIFEST.json
make rank       # online:   R0..R9   -> artifacts/submission.csv + run_report.json
make validate   # validate a finished submission.csv against the format rules
```

Equivalently, via the CLI directly:

```bash
uv run redstack build --config configs --no-golden-labels   # locked-heuristic weights, no calibration search
uv run redstack rank --input data/raw/candidates.jsonl --output artifacts/submission.csv
uv run redstack validate --submission artifacts/submission.csv
```

Run `redstack build` without `--no-golden-labels` to run the real O8–O10 gold-label calibration search against `data/golden/golden_labels.csv`. `scripts/reproduce.sh` wraps the canonical rank command for a clean, repeatable invocation.

## Quality gates

```bash
uv run ruff check src/ tests/              # lint
uv run ruff format src/ tests/ --check     # format check
uv run mypy --strict src/ tests/           # static typing
uv run lint-imports                        # the eight architecture boundary contracts
uv run pytest                              # full test suite
```

See [tests/README.md](tests/README.md) for the test architecture and [docs/specs/README.md](docs/specs/README.md) for the full testing strategy this suite implements.

## Where to go next

| Goal | Read |
|---|---|
| Understand the full system design | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Populate `data/` and run a first build | [docs/runbook.md](docs/runbook.md) |
| Understand one specific package | the `README.md` inside that directory |
| See the frozen, line-level design specifications | [docs/specs/README.md](docs/specs/README.md) |
