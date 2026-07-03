---
title: EternalTalentSpace
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.46.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# EternalTalentPose

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

## Compute Compliance — Submission Spec §3

Every constraint below applies to the **ranking step only** (`rank.py` / `redstack rank`, stages R0–R9). Pre-computation (`redstack build`, stages O0–O18) is offline and unconstrained by this table — see [§10.3 of the spec](docs/guide/submission_spec.docx).

| Constraint | Limit | RedStack's guarantee | Measured (100k pool, 10-core / 16 GB host) |
|---|---|---|---|
| Total runtime | ≤ 5 minutes wall-clock | `redstack rank` enforces `budget_limit_seconds=300.0` (`configs/base.yaml: budget.max_wall_seconds`) against true wall-clock — from before adapter construction through submission write — and exits non-zero if exceeded. | **~256s real** (`/usr/bin/time`), `run_report.json.budget.used_seconds ≈ 253s`. See the timing note below. |
| Memory | ≤ 16 GB RAM | O(1) streaming memory over `candidates.jsonl` — no full-pool materialization. `run_report.json` records `peak_rss_mb` from the OS high-water mark. | ~3 GB peak RSS |
| Compute | CPU only — no GPU | `redstack.pipelines.online.*` and `redstack.engines.*` are forbidden, by `import-linter` contract, from importing any GPU-capable runtime; the online path loads a CPU-only ONNX encoder only as a fallback. | n/a (statically enforced) |
| Network | Off — no hosted LLM/API calls | `import-linter` contract "7. Online pipeline containment" forbids `socket`, `requests`, `httpx`, `sentence-transformers`, and `scikit-learn` anywhere under `pipelines/online/` or `engines/`, transitively. Verify with `uv run lint-imports`. | n/a (statically enforced) |
| Disk | ≤ 5 GB intermediate state | The online run reads the frozen `artifacts/` tree (produced once, offline) and writes only `submission.csv` + `run_report.json` — no intermediate spill files. | n/a |



Reproduce and verify locally before submitting:

```bash
uv run lint-imports          # confirms the online-containment / CPU-only / no-network boundary
uv run redstack build        # one-time offline pre-computation (unconstrained)
uv run redstack rank         # the constrained step — writes run_report.json with peak_rss_mb / within_budget
uv run redstack validate     # re-checks submission.csv against the spec's format rules
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
- **Online ranking** (stages R0–R9): CPU-only, network-isolated, O(1) streaming memory, applies artifacts to live candidates. Measured ~4m16s wall-clock over the full 100k-candidate pool (well within the spec's 5-minute ceiling) — see [Compute Compliance](#compute-compliance--submission-spec-3) above.

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
uv run redstack build
# or
python scripts/run.py build
# or
make build
```

Approximate wall-clock time: **~12 minutes** on a 10-core CPU with 16 GB RAM (measured: 12:11 for the full 100k-candidate pool with `--no-golden-labels`). Unconstrained by the spec's 5-minute ranking budget — this step runs once, offline. The online ranking run requires only the frozen `artifacts/` — it never retrains or re-embeds.

---
## Build, Rank, Validate

RedStack exposes the same lifecycle operations through multiple interfaces:

- **`redstack` CLI** — the canonical packaged interface.
- **`scripts/run.py`** — a cross-platform Python task runner.
- **`Makefile`** — developer convenience aliases that delegate to the Python task runner.

All interfaces execute the same deterministic pipeline and produce identical outputs.

| Operation | CLI | Python Runner | Make | Output |
|---|---|---|---|---|
| **Offline Build** | `uv run redstack build` | `python scripts/run.py build` | `make build` | Executes stages **O0–O18**, producing the immutable artifact store in `artifacts/` and `MANIFEST.json`. |
| **Online Ranking** | `uv run redstack rank` | `python scripts/run.py rank` | `make rank` | Executes stages **R0–R9**, producing `artifacts/submission.csv` and `run_report.json`. |
| **Validation** | `uv run redstack validate` | `python scripts/run.py validate` | `make validate` | Validates the generated submission and evidence-based reasoning against the competition specification. |
| **Clean** | — | `python scripts/run.py clean` | `make clean` | Removes generated caches (`.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `__pycache__`, `dist`, `build`) without modifying project data, configuration, or artifacts. |

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
│   ├── features/             pure feature extraction (32 feature groups)
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


**
