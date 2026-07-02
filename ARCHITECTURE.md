# RedStack Architecture

RedStack is a **semantic talent-intelligence and candidate-ranking platform**. Given a job description and a pool of candidate profiles, it produces a ranked shortlist with per-candidate, evidence-grounded reasoning — deterministically, reproducibly, and within a fixed CPU/RAM/time budget.

This document is the single map of the system: how data flows from raw candidate records to a ranked, explained shortlist, how the codebase is organized to make that flow safe to change, and what every layer is and is not allowed to do. Every subdirectory in the repository has its own `README.md` that goes one level deeper than this document and links back to the relevant section here.

> **Audience.** §1–§2 are written for anyone evaluating what the system does and why it's structured this way. §3 onward assume familiarity with the codebase and are the reference engineers use day to day.

---

## 1. What RedStack Does

Given:
- a **job description**, expressed as a set of weighted positive/negative semantic anchors and a small set of explicit eligibility rules, and
- a **pool of candidate profiles** (career history, education, skills, certifications, and platform engagement signals),

RedStack produces:
- a **ranked top-100 shortlist** (`candidate_id, rank, score, reasoning`), and
- a **run report** documenting exactly how that ranking was produced — which artifacts, which configuration, which code version, and per-stage timing/memory — so any placement is auditable after the fact.

Three properties make this more than a scored search index:

1. **Resistance to gaming.** A candidate who lists every fashionable skill keyword with no corroborating evidence (no endorsements, no time-on-skill, no mention in actual role descriptions, no semantic match between claimed skills and described work) is *not* rewarded for the keywords. Every competency feature is an **evidence aggregate** — trust × career corroboration × semantic similarity — not a keyword flag, so keyword-stuffing produces a fused competency score near zero by construction.
2. **Integrity gating.** Profiles with internally impossible facts (a 10-year tenure at a 2-year-old company, "expert" proficiency claimed with zero months of use, overlapping full-time roles) are detected by a dedicated integrity engine and floored out of the ranking *before* scoring — never silently degraded, never penalized just enough to slip into the tail.
3. **Grounded explanations.** Every sentence of generated reasoning is mechanically tied to a real field on the candidate's profile. A reasoning clause that cannot cite at least one resolvable fact cannot be constructed — this is enforced by the domain model's type system, not by a style guideline.

---

## 2. Design Principles

| Principle | What it means here |
|---|---|
| **Hexagonal architecture (ports & adapters)** | The core business logic (`domain`, `features`, `engines`) depends only on abstract `Protocol` interfaces (`ports`). Concrete infrastructure (`adapters`) implements those interfaces. Only the orchestration layer (`pipelines`) ever wires a concrete adapter to a port. An engine *cannot* accidentally import a heavyweight ML runtime, because it cannot import `adapters` at all — the boundary is enforced by `import-linter` contracts that fail the build, not by convention. |
| **Domain-driven design** | `CandidateRepresentation` is an aggregate root, assembled left-to-right as a candidate moves through the pipeline. Every slice (career profile, integrity report, score breakdown, ...) is an immutable value object attached via copy-on-write. Business verdicts ("this candidate is disqualified") are data objects that flow through the system, not exceptions. |
| **Pipes and filters** | Both the offline build and the online ranking run are ordered sequences of pure stages over one threaded object. Every stage is `f(x) -> x'` — referentially transparent and independently testable. |
| **Strict determinism** | No wall clock in any business logic (a recency calculation takes an injected reference date, never `datetime.now()`); no online randomness (tie-breaks resolve by ascending candidate ID); fixed-precision arithmetic with a fixed reduction order; pinned thread counts. Identical inputs produce byte-identical output, every time, regardless of host or thread count. |
| **Offline / online separation** | Everything computationally expensive (embedding generation, clustering, weight calibration) runs once, offline, and is compiled into a hash-verified artifact set. The online ranking run only *applies* those artifacts — it never trains, clusters, or recomputes them. This is what lets a 100,000-candidate ranking run complete in CPU-only, network-isolated conditions, well inside the submission spec's 5-minute ceiling (measured ~4m16s wall-clock; see README §Compute Compliance). |
| **Fail fast, never degrade silently** | Because a ranking run is not interactively monitored, any integrity violation (a corrupted artifact, a malformed input file, a broken invariant) aborts the run loudly rather than producing a quietly-wrong result. |

---

## 3. Repository Layout

```text
redstack/
├── README.md                      setup + the single build/rank/validate commands
├── ARCHITECTURE.md                 this document
├── pyproject.toml                  package metadata + ruff/mypy/pytest/import-linter config
├── Makefile                        task aliases over uv + the redstack CLI
├── submission_metadata.yaml        release/provenance metadata (team, compute declaration, reproduce command)
│
├── configs/                        human-authored, declarative behavior — see configs/README.md
│   ├── base.yaml, runtime/, weights/, lexicon/, anchors/, gates/, integrity/, profiles/
│
├── data/                           raw inputs (gitignored) — see data/README.md
│   ├── raw/candidates.jsonl
│   └── golden/golden_labels.csv
│
├── artifacts/                      offline build output (gitignored, hash-pinned) — see artifacts/README.md
│   ├── MANIFEST.json, model/, embeddings/, lexicon/, archetypes/, calibration/, weights/, ...
│
├── src/redstack/                   all importable code — see src/redstack/README.md
│   ├── domain/                     pure data + invariants
│   ├── ports/                      Protocol interfaces (the hexagon boundary)
│   ├── features/                   pure feature extraction
│   ├── engines/                    the 11 domain services (business judgment)
│   ├── config/                     typed config schema + loader + determinism policy
│   ├── adapters/                   infrastructure implementations of the ports
│   ├── pipelines/                  orchestration + composition roots (offline O0-O18, online R0-R9)
│   ├── observability/              logging, timing/budget guard, run-report model
│   └── cli/                        the `redstack` command-line entrypoints
│
├── tests/                          unit, property, contract, golden, integration, determinism
├── scripts/                        operational glue (reproduce wrapper, sample builder, analytics)
└── docs/                           architecture companion, ADRs, runbook, frozen layer specs, reference assets
```

---

## 4. Layer Reference

Dependencies flow inward only — outer layers depend on inner layers, never the reverse:

```text
cli ──▶ pipelines ──▶ engines ──▶ features ──▶ domain
                       │            │            ▲
                       ▼            └───────────▶│
                     ports ─────────────────────▶ domain
adapters ──▶ ports ──▶ domain          config ──▶ domain
pipelines ──▶ adapters   (composition root ONLY)
observability ──▶ domain
```

| Layer | Responsibility | May import | Detail |
|---|---|---|---|
| `domain` | Express valid candidate/ranking states and their invariants. Pure data, zero IO. | stdlib, `pydantic`, `numpy` | [src/redstack/domain/README.md](src/redstack/domain/README.md) |
| `ports` | Declare the contracts the core needs from the outside world, as `typing.Protocol`s. | `domain` | [src/redstack/ports/README.md](src/redstack/ports/README.md) |
| `features` | Derive structured features deterministically from raw candidate data. | `domain`, `config.schema` | [src/redstack/features/README.md](src/redstack/features/README.md) |
| `engines` | Apply business judgment: integrity, eligibility, semantic fit, scoring, ranking, reasoning. | `domain`, `ports`, `features`, `config.schema` | [src/redstack/engines/README.md](src/redstack/engines/README.md) |
| `config` | Load, validate, and freeze configuration; own the determinism policy. | `domain`, `pydantic`, `pyyaml` | [src/redstack/config/README.md](src/redstack/config/README.md) |
| `adapters` | Implement ports against real infrastructure (ONNX Runtime, Parquet, filesystem). | everything technical + `ports` + `domain` | [src/redstack/adapters/README.md](src/redstack/adapters/README.md) |
| `pipelines` | Sequence stages; wire adapters to ports. The only composition root. | all of the above | [src/redstack/pipelines/README.md](src/redstack/pipelines/README.md) |
| `observability` | Logging, stage timing, the run-report model. | `domain` | [src/redstack/observability/README.md](src/redstack/observability/README.md) |
| `cli` | Translate command-line arguments into a pipeline invocation. | `pipelines`, `config`, `observability` | [src/redstack/cli/README.md](src/redstack/cli/README.md) |

These boundaries are not stylistic — they are encoded as eight `import-linter` contracts in `pyproject.toml`, enforced in CI and pre-commit. A violated import arrow is a build break, not a code-review comment. The most important of the eight: **online containment** — the online ranking pipeline (`pipelines/online/*`) is forbidden, transitively, from importing `sentence-transformers`, `scikit-learn`, the offline embedding adapter, or any networking module. The online package is physically incapable of pulling in a budget-busting dependency.

---

## 5. Two Pipelines, One Codebase

RedStack runs in exactly two modes, kept physically separate by the import boundary above:

| | **Offline build** (`pipelines/offline`, stages O0–O18) | **Online ranking** (`pipelines/online`, stages R0–R9) |
|---|---|---|
| Purpose | Pre-compute everything expensive once, into a hash-pinned artifact set | Apply the locked artifacts to rank a candidate pool |
| Runtime budget | Unbounded (author-time) | Internal target ≤150s, hard ceiling ≤5min |
| Memory budget | Unbounded | Internal target ≤4GB, hard ceiling ≤16GB |
| Compute | CPU/GPU, may use heavy ML runtimes (sentence-transformers, scikit-learn) | CPU-only; ONNX Runtime only, and only as a rare fallback |
| Network | Permitted at author time | **Forbidden** — enforced by import-linter |
| Reads | `data/`, `configs/` | `artifacts/`, `configs/runtime/online.yaml`, the candidate file |
| Writes | `artifacts/` | `submission.csv`, `run_report.json` |
| Entry point | `redstack build` | `redstack rank` |

### 5.1 Offline pipeline — O0 through O18

Each offline stage is a pure, ordered transform that reads upstream artifacts/configs and writes one or more downstream artifacts. The runner resumes from checkpoints, so a partial rebuild only recomputes what's stale.

| Stage | Purpose | Key output artifact(s) |
|---|---|---|
| **O0** Dataset Census | Profile the candidate pool before anything else: coverage, distributions, outliers — every later threshold is set relative to the observed data, not hard-coded. | `dataset_profile.json` |
| **O1** Normalization | Canonicalize text, dates, skill tokens, and company names. | `canonical_maps.json` |
| **O2** Validation | Validate the normalized pool against the candidate schema; structural rejects only — semantic contradictions are *preserved* for the integrity stage to catch. | `validation_report.json` |
| **O3** Honeypot Discovery | Discover and calibrate the impossible-profile ("honeypot") detection thresholds against the census. | `integrity_rules.json`, `honeypot_catalog.json`, `calibration/integrity_thresholds.json` |
| **O4** Lexicon Discovery | Mine domain terminology from role descriptions so competency matching depends on meaning, not exact keyword choice. | `lexicon/lexicon.compiled.json`, `concepts.json`, `term_graph.json`, `phrase_graph.json` |
| **O5** Vocabulary Expansion | Expand the lexicon with embedding-nearest synonyms a keyword-stuffer wouldn't anticipate. | `concepts.json` (expanded) |
| **O6** JD Concept Extraction | Author the job description's positive/negative semantic anchors and the eligibility rule set. | `jd_concepts.json`, `gates/eligibility_rules.yaml` |
| **O7** Archetype Discovery | Cluster the candidate pool into archetypes (e.g. retrieval specialist, consultant, keyword-stuffer, startup generalist) via seeded k-means. | `archetypes.json`, `archetypes/centroids.npy` |
| **O8** Labeling Workspace | Human-in-the-loop gold-label collection used to calibrate scoring weights. | `gold_labels.json`, `calibration_split.json` |
| **O9** Weight Calibration | Calibrate `ScoringWeights` against the gold labels (cross-validated, tier-prior regularized). | `weights/scoring_weights.locked.yaml`, `calibration_report.json` |
| **O10** Feature Importance | Quantify per-feature contribution, used later to select which evidence the reasoning engine cites. | `feature_importance.json` |
| **O11** Behavioral Calibration | Fit the bounded behavioral-multiplier curves. | `behavioral_weights.json` |
| **O12** Risk Calibration | Set the honeypot composite threshold and confidence-shrink parameters. | merged into `calibration/integrity_thresholds.json`, `risk_weights.json` |
| **O13** Embedding Generation | The compute-dominant stage: encode every candidate, anchor, and (optionally) career-history document; export the ONNX twin used online. | `embeddings/candidate_vectors.parquet`, `embeddings/anchor_vectors.npy`, `model/encoder.onnx`, `embedding_manifest.json` |
| **O14** Feature Snapshot | Run every feature extractor over the full pool into the canonical `(N, D)` feature matrix — the calibration substrate *and* the online correctness oracle. | `feature_snapshot.parquet`, `feature_manifest.json` |
| **O15** Ranking Calibration | Fit an order-preserving score-presentation curve; confirm tie-break behavior. | `ranking_calibration.json` |
| **O16** Reasoning Templates | Build the evidence-slot reasoning templates from gold reference reasonings — no model call happens online. | `reasoning_templates.json` |
| **O17** Packaging | Hash every artifact and write the manifest — the single contract the online run verifies against. | `MANIFEST.json` |
| **O18** Reproducibility Validation | Prove the build is reproducible and online-consumable: reload and verify, recompute features online for a sample and diff against the snapshot, dry-run rank the golden set. | `reproducibility_report.json` |

Full detail: [src/redstack/pipelines/offline/README.md](src/redstack/pipelines/offline/README.md) and [src/redstack/pipelines/offline/stages/README.md](src/redstack/pipelines/offline/stages/README.md).

### 5.2 Online pipeline — R0 through R9

The online run is strictly sequential. Ports (the only places that touch infrastructure) appear *only* at R0, R1, R3, R8, and R9 — R2/R4/R5/R6/R7 are pure, in-memory engine work, which is exactly what keeps the run's timing predictable.

| Stage | Purpose | Touches infrastructure? |
|---|---|---|
| **R0** Artifact Loading | Load `MANIFEST.json`, verify every artifact's hash and cross-artifact coherence, bind every port to its adapter. Aborts the run on any integrity failure — there is no degraded mode. | Yes |
| **R1** Candidate Ingestion | Stream the input file into validated, typed candidate records, in constant memory. | Yes |
| **R2** Feature Extraction | Compute every structural and behavioral feature into the columnar `(N, D)` quality-vector matrix. | No |
| **R3** Semantic Hydration | Look up each candidate's precomputed embedding (O(1) gather from a memory-mapped store); compute anchor similarity and archetype assignment. Encoding is a rare *fallback*, only for an id absent from the store. | Yes (lookup; fallback encode) |
| **R4** Gates & Eligibility | Apply the integrity (honeypot) and eligibility (job-description hard/soft rule) verdicts; build the score floor mask. | No |
| **R5** Scoring | Apply the locked weights to the feature vector, gate by R4's floor mask, apply bounded behavioral/logistics multipliers. No weight is calibrated at this point — only applied. | No |
| **R6** Ranking | Deterministic stable sort, top-100 cut, tie-break by ascending candidate ID — the six structural invariants of a valid ranking are enforced at construction, not checked after the fact. | No |
| **R7** Reasoning | Generate evidence-grounded reasoning for the top 100 only. | No |
| **R8** Submission | Validate and write the final CSV, atomically. | Yes |
| **R9** Run Report | Write the audit/reproducibility report: artifact hashes, timings, honeypot rate, eligibility summary. | Yes |

Full detail: [src/redstack/pipelines/online/README.md](src/redstack/pipelines/online/README.md).

---

## 6. The Engines

`src/redstack/engines/` holds eleven physical modules realizing fourteen logical services. Engines are stateless callables: they take a `CandidateRepresentation` (plus typed config or an injected port) and return an enriched copy, or a verdict. **Engines never call each other** — the engine graph is a forest, and the pipeline is the only thing that threads the growing representation between them.

| Engine module | Logical service(s) | What it decides |
|---|---|---|
| **`integrity.py`** — `IntegrityEngine` | Consistency + Risk | Detects internally-impossible profiles (timeline contradictions, "expert" skill claimed with zero months of use, overlapping employment) and aggregates them into a honeypot verdict. A candidate is floored when it carries two or more hard impossibilities, or a calibrated composite risk score above threshold. This runs *before* scoring, never as a post-hoc filter. |
| **`eligibility.py`** — `EligibilityEngine` | Fit (eligibility half) | Applies the job description's explicit hard disqualifiers (e.g. research-only career, no production code in 18 months, consulting-only career) and soft penalties (e.g. notice period over 30 days) as declarative predicates. A hard block floors the candidate; a soft penalty only down-weights. |
| **`semantic.py`** — `SemanticEngine` | Embedding + Retrieval | The only engine that depends on an injected port. Looks up each candidate's precomputed dense vector, computes cosine similarity against the job description's positive/negative semantic anchors, and assigns the nearest archetype. Falls back to on-the-fly encoding only when a candidate's vector is missing from the precomputed store. |
| **`lexicon.py`** — `LexiconEngine` | (part of Feature) | Compiled-lexicon symbolic matching — the lexical half of the anti-keyword-stuffing defense, corroborating (or failing to corroborate) claimed skills against actual role descriptions. |
| **`cqv.py`** — `CQVAssembler` | Feature (assembly) | Folds every populated profile slice into the fixed-length, versioned candidate quality vector that scoring consumes. |
| **`behavioral.py`** — `BehavioralEngine` | Behavior | Converts platform engagement signals (availability, responsiveness, engagement, reliability) into a bounded multiplier — never a base-relevance source. |
| **`logistics.py`** — `LogisticsEngine` | Fit (logistics half) | Location, notice period, relocation willingness, and salary-band sanity into a bounded multiplier. |
| **`scoring.py`** — `ScoringEngine` | Scoring | Computes `base_relevance` as the weighted sum of score components (using the *locked* weights produced offline), applies the integrity/eligibility floor, applies the behavioral and logistics multipliers, and emits a fully decomposed, evidence-attached `ScoreBreakdown`. No calibration happens here — only application of an already-calibrated formula. |
| **`ranking.py`** — `RankingEngine` | Ranking | Deterministic sort, top-100 cut, tie-break — and the domain factory it calls enforces all six structural validity rules at construction time. |
| **`reasoning.py`** — `ReasoningEngine` | Reasoning | Builds evidence-grounded, non-templated reasoning text for the top 100 only. A reasoning clause cannot be constructed without at least one resolvable fact reference — this is a type-level guarantee, not a runtime check. |
| **`validation.py`** — `ValidationEngine` | Submission Generation (validation half) | Mirrors the external submission-format validator and the reasoning-quality checks, as a defense-in-depth pass before the file is written. |

See [src/redstack/engines/README.md](src/redstack/engines/README.md) for the full per-engine contract (inputs, outputs, failure modes, performance budget).

---

## 7. The Domain Model

The aggregate root is `CandidateRepresentation`, assembled left to right as a candidate moves through the online pipeline:

```text
CandidateRepresentation
├── identity      (R1)  Identity                ← features.parsing
├── career        (R2)  CareerProfile           ← features.career
├── credibility   (R2)  CredibilityProfile      ← features.skills + LexiconEngine
├── behavioral    (R2)  BehavioralProfile       ← BehavioralEngine
├── logistics     (R2)  LogisticsProfile        ← LogisticsEngine
├── semantic      (R3)  SemanticProfile         ← SemanticEngine
├── archetype     (R3)  ArchetypeAssignment     ← SemanticEngine
├── integrity     (R4)  IntegrityReport         ← IntegrityEngine
├── eligibility   (R4)  EligibilityReport       ← EligibilityEngine
└── quality       (R5)  CandidateQualityVector  ← CQVAssembler
```

Design rules that hold across every value object in `domain/`:

- **Illegal states are unrepresentable.** A `Ranking` cannot be constructed with a duplicate rank — the constructor rejects it. A similarity score and a final ranking score are distinct types at the type-checker level, so one cannot be passed where the other is expected.
- **Everything is frozen.** Mutation happens only through copy-on-write `with_*()` builders that return a new aggregate, advancing a monotonic build-stage marker. This makes every pipeline stage a pure, independently-testable function.
- **Provenance is carried, never discarded.** Every profile slice keeps a handle back to the exact source fields it was derived from, so the reasoning engine can cite real facts and a `ProvenanceError` is raised immediately if a citation doesn't resolve.
- **The job description is a frozen value object**, parsed once and shared read-only across the entire candidate pool.

Full detail: [src/redstack/domain/README.md](src/redstack/domain/README.md).

---

## 8. Persistence and the Artifact Contract

The offline build and the online run communicate through exactly one channel: a **hash-pinned artifact set**, registered in `artifacts/MANIFEST.json`. Artifacts are treated like a compiled binary — built once, content-addressed, never hand-edited.

| Storage form | Used for | Example |
|---|---|---|
| **JSON** | Small structured metadata, calibrated thresholds, rule sets | `integrity_thresholds.json`, `feature_manifest.json` |
| **YAML** | Locked, validated configuration produced by calibration | `weights/scoring_weights.locked.yaml` |
| **Parquet** | Large columnar data consumed by memory-mapped lookup | `embeddings/candidate_vectors.parquet`, `feature_snapshot.parquet` |
| **NumPy `.npy`** | Small dense arrays | `embeddings/anchor_vectors.npy`, `archetypes/centroids.npy` |
| **ONNX** | The exported embedding model used by the online fallback encoder | `model/encoder.onnx` |

At load time (`ArtifactStorePort`, stage R0), every artifact's SHA-256 is verified against the manifest, the manifest itself is self-hash-verified, and cross-artifact coherence is asserted (embedding dimensions agree across the vector store, the anchor set, the centroids, and the ONNX model; the feature-layout version agrees between the feature manifest and the locked weights). **Any failure aborts the run.** There is no degraded mode, because a silently-wrong ranking with no way to detect it after the fact is worse than a hard failure.

> **Why R3 is "lookup," not "encode."** All ~100K candidate vectors are precomputed offline by `sentence-transformers` and stored once. Online, retrieving a candidate's vector is an O(1) memory-mapped row gather, not a model forward pass. This single design decision is what makes the online ranking budget achievable on CPU.

Full detail: [artifacts/README.md](artifacts/README.md).

---

## 9. Configuration Architecture

Configuration is resolved through a fixed three-layer deep merge, validated against a typed schema (unknown keys are rejected, so a typo fails loudly instead of silently disabling a rule):

```text
base.yaml  →  runtime/{online|offline}.yaml  →  profiles/{ci|local}.yaml
```

The critical distinction: **`configs/` is intent, `artifacts/` is derived fact.** Anchors, eligibility rules, the lexicon seed, integrity rule shapes, and scoring weights are *authored* in `configs/` but *compiled and calibrated* into `artifacts/` by the offline build. The online run never reads `configs/weights`, `configs/lexicon`, `configs/anchors`, `configs/gates`, or `configs/integrity` directly — it reads their frozen artifact counterparts. This prevents a configuration edit from silently changing a ranking outcome without a rebuild.

Full detail: [configs/README.md](configs/README.md).

---

## 10. Determinism and Performance

| Guarantee | Mechanism |
|---|---|
| No wall-clock dependence | Every recency/tenure calculation takes an injected reference date (`as_of`); business logic never calls the system clock. |
| No online randomness | All ties resolve by ascending candidate ID; the online entropy port raises if anything attempts to draw a random number. |
| Fixed-precision, fixed-order arithmetic | `float32` throughout; component sums reduced in a fixed, versioned order; BLAS/OpenMP thread counts pinned, so output is thread-count-invariant. |
| Byte-identical reproducibility | Identical inputs produce a byte-identical submission file and an identical run-report reproducibility block, verified by running the pipeline twice (same process, fresh process, and 1-thread vs. N-thread). |
| Fixed compute budget | Heavy work (embedding, clustering, calibration) is exclusively offline. Online, R3 is a lookup and R5 applies an already-locked formula — the two stages that would otherwise dominate runtime are reduced to vectorized array operations. |

Full detail: [docs/specs/README.md](docs/specs/README.md) (testing strategy and performance budget per stage).

---

## 11. Quality Gates

Every change is checked by, in order: `ruff` (lint + format), `mypy --strict`, the eight `import-linter` boundary contracts, then the test pyramid — unit, property (Hypothesis-driven invariants), contract (one shared suite per port, run against every real adapter and its in-memory fake), golden (byte-exact snapshots), integration (the full pipeline against a sample pool), and determinism (repeat-run and thread-count-invariance checks). See [tests/README.md](tests/README.md) and [docs/specs/README.md](docs/specs/README.md) for the full testing architecture.

---

## 12. Where to Go Next

| If you want to... | Read |
|---|---|
| Set up and run the system | [README.md](README.md) |
| Understand one specific package | the `README.md` inside that package's directory |
| See the full, line-level frozen design specifications this architecture is distilled from | [docs/specs/README.md](docs/specs/README.md) |
| Understand the test strategy | [tests/README.md](tests/README.md) |
| Rebuild or inspect the offline artifact set | [artifacts/README.md](artifacts/README.md) |
