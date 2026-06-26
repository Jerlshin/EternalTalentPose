# RedStack

**A deterministic, evidence-grounded semantic talent-intelligence and candidate-ranking platform.**

RedStack takes a job description and a pool of candidate profiles and produces a ranked, explained shortlist. It resists keyword gaming, gates out internally-impossible ("honeypot") profiles before scoring, and produces a reasoning sentence for every ranked candidate mechanically tied to a real fact on their profile — never a templated or hallucinated claim.

---

## Evaluation — Judge Stage 3

The single command required by the evaluation spec (§10.3):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` is a root-level proxy that pins deterministic thread counts and delegates to `python -m redstack.cli.app rank`. Equivalent make target:

```bash
make rank
```

---

## Architecture Summary

RedStack is structured as a **Hexagonal (Ports & Adapters)** domain service with a strict layered import graph:

```
cli  ──▶  pipelines  ──▶  engines  ──▶  features  ──▶  domain
                  ↘                ↗
               adapters ──▶ ports
```

**Domain layer** (`src/redstack/domain/`) holds pure, immutable value objects and aggregate roots (`CandidateRepresentation`, `Ranking`). No infrastructure dependency is allowed here — enforced by `import-linter` as a build break.

**Ports layer** (`src/redstack/ports/`) defines `Protocol` interfaces (the hexagon boundary). Engines depend on these; adapters implement them.

**Engines** (`src/redstack/engines/`) — eleven stateless domain services that apply business judgment. Each engine is `f(CandidateRepresentation) → CandidateRepresentation'`; they never call each other (the pipeline is the only orchestrator):

| Engine | Responsibility |
|---|---|
| `integrity` | Detects internally-impossible profiles (timeline contradictions, expert-zero-usage claims, overlapping roles) → honeypot verdict |
| `eligibility` | Applies JD hard disqualifiers (research-only career, no production code) and soft penalties |
| `semantic` | Dense vector lookup + cosine similarity against JD anchors; archetype assignment |
| `lexicon` | Compiled-lexicon symbolic matching — corroborates claimed skills against actual role descriptions |
| `cqv` | Career quality and velocity signals |
| `behavioral` | Platform engagement signals → bounded multiplier |
| `logistics` | Location, notice period, salary-band sanity → bounded multiplier |
| `scoring` | Applies locked weights; gates via integrity/eligibility floor; emits fully decomposed `ScoreBreakdown` |
| `ranking` | Deterministic stable sort, top-100 cut, tie-break by ascending candidate ID |
| `reasoning` | Builds evidence-grounded reasoning text (every clause is bound to a resolvable profile fact) |
| `validation` | Submission-format and reasoning-quality checks before the CSV is written |

**Adapters** (`src/redstack/adapters/`) implement ports using concrete infrastructure (ONNX encoder, file I/O). Only the pipeline's composition root may import adapters — enforced as a build break.

**Pipelines** (`src/redstack/pipelines/`) orchestrate two physically separated execution modes:

- **Offline build** (stages O0–O18): runs once, pre-computes embeddings, clusters, calibrated weights → hash-verified artifact set in `artifacts/`.
- **Online ranking** (stages R0–R9): CPU-only, network-isolated, O(1) streaming memory, applies artifacts to live candidates, completes in < 1 minute.

---

## Installation

> Requires Python 3.12. [`uv`](https://docs.astral.sh/uv/) is the recommended toolchain.

**Universal onboarding — run this once before anything else:**

```bash
uv sync --frozen --all-groups
```

`--all-groups` installs every dependency group (`core`, `offline`, `dev`) from
the committed lockfile in one step. `--frozen` guarantees byte-identical
resolution on every machine and fails fast if the lockfile is out of sync.

**Hot-path only (no heavy ML deps):**

```bash
uv sync --frozen --group core
```

**Fallback (pip, no `uv`):**

```bash
pip install -r requirements.txt
pip install -e .
```

Populate `data/raw/candidates.jsonl` before running — see [docs/runbook.md](docs/runbook.md).

---

## Pre-computation Layer

The offline build packages parity matrices, ONNX encoder, locked scoring weights, and the archetype centroids into `artifacts/`. This must run once before ranking:

```bash
uv run build
# or
python scripts/run.py build
# or
make build
```

Approximate wall-clock time: **~11 minutes** on an 8-core CPU with 16 GB RAM. The online ranking run requires only the frozen `artifacts/` — it never retrain or re-embed.

---

## Build, Rank, Validate

All lifecycle commands are cross-platform, routed through `scripts/run.py`:

| Command | What it does |
|---|---|
| `python scripts/run.py build` / `make build` | Offline O0–O18 → `artifacts/` + `MANIFEST.json` |
| `python scripts/run.py rank` / `make rank` | Online R0–R9 → `artifacts/submission.csv` + `run_report.json` |
| `python scripts/run.py validate` / `make validate` | Validates `artifacts/submission.csv` against spec rules |
| `python scripts/run.py clean` / `make clean` | Purges `.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `__pycache__`, `dist`, `build` |

---

## Quality Gates

```bash
uv run ruff check src/ tests/          # lint
uv run ruff format src/ tests/ --check # format check
uv run mypy --strict src/ tests/       # strict static typing
uv run lint-imports                    # eight architecture boundary contracts
uv run pytest                          # full test suite (unit, property, contract, golden, integration, determinism)
```

---

## Repository Layout

```text
redstack/
├── rank.py                   root-level judge entrypoint proxy (§10.3)
├── README.md                 this file
├── ARCHITECTURE.md           full system design — start here
├── pyproject.toml            package metadata, ruff/mypy/pytest/import-linter/uv.scripts config
├── Makefile                  task aliases (routes through scripts/run.py)
├── requirements.txt          pip fallback for environments without uv
├── submission_metadata.yaml  team identity, compute declaration, reproduce command
│
├── configs/                  human-authored declarative behavior (weights, anchors, gates, lexicon…)
├── data/                     raw inputs — gitignored; see docs/runbook.md to populate
├── artifacts/                offline build output — gitignored, hash-pinned, rebuilt via `uv run build`
│
├── scripts/
│   ├── run.py                cross-platform pure-Python task runner
│   └── reproduce.sh          POSIX reproduce wrapper (legacy alias)
│
├── src/redstack/
│   ├── domain/               pure data models + invariants
│   ├── ports/                Protocol interfaces (the hexagon boundary)
│   ├── features/             pure feature extraction (30 feature groups)
│   ├── engines/              the 11 domain services (business judgment)
│   ├── config/               typed config schema, loader, determinism policy
│   ├── adapters/             infrastructure implementations of the ports
│   ├── pipelines/            orchestration: offline O0–O18, online R0–R9
│   ├── observability/        logging, stage timing, run-report model
│   └── cli/                  the `redstack` CLI entrypoints
│
├── tests/                    unit, property, contract, golden, integration, determinism suites
└── docs/                     architecture companion, ADRs, runbook, reference assets
```

---

## Where to go next

| Goal | Read |
|---|---|
| Full system design | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Populate `data/` and run a first build | [docs/runbook.md](docs/runbook.md) |
| Understand one specific package | the `README.md` inside that directory |
