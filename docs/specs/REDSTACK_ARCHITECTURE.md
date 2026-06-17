# REDSTACK v1.1 — Repository & Implementation Architecture

**Redrob Evidence-Driven Symbolic + Semantic Ranker**
Status: architecture frozen. This document is implementation planning only. No component is added, removed, or redesigned.

Stack baseline: Python 3.12 · `uv` · `ruff` · `mypy --strict` · `pytest` · `pydantic v2` · `numpy` · `pandas` · `pyarrow` · `onnxruntime` · `scikit-learn` · `sentence-transformers` (offline artifact generation only) · YAML config · CPU-only · deterministic · reproducible.

Governing constraints from `submission_spec.docx` that shape every structural decision:

- **Online ranking step ≤ 5 min wall-clock, ≤ 16 GB RAM, CPU-only, no network.** → heavy work is physically impossible inside the online package; it is *architecturally* impossible because the online package cannot import any module that opens a socket or loads a training runtime.
- **Stage-3 reproduction:** a single command rebuilds `submission.csv` from `candidates.jsonl` inside a clean sandbox. → one CLI verb, one composition root, all dependencies declared, all artifacts hash-pinned.
- **Validator is law** (`validate_submission.py`): exactly 100 data rows, ranks 1–100 each once, `candidate_id` `^CAND_[0-9]{7}$` unique, score non-increasing by rank, ties broken by `candidate_id` ascending. → the Ranking and Validation engines encode these as invariants, not as best-effort.
- **Honeypot rate > 10 % in top 100 = disqualification.** → the Integrity Engine is a first-class gate upstream of scoring, not a post-hoc filter.
- **Behavioral signals modulate fit**, they do not define it. → Behavioral/Logistics engines produce *multipliers*, never base relevance.

---

## 1. High-level repository tree

```
redstack/
├── pyproject.toml                 # single source of build + tool config (ruff, mypy, pytest)
├── uv.lock                        # fully pinned, committed
├── .python-version                # 3.12
├── Makefile                       # thin task aliases over uv/cli
├── README.md                      # setup + the ONE reproduce command (spec 10.3)
├── ARCHITECTURE.md                # this document
├── submission_metadata.yaml       # mirrors portal metadata (spec 10.3)
├── .gitignore                     # ignores data/ and artifacts/
├── .pre-commit-config.yaml        # ruff + mypy + import-linter gate
│
├── configs/                       # declarative, versioned, human-authored knobs
│   ├── base.yaml
│   ├── runtime/{online.yaml, offline.yaml}
│   ├── weights/scoring_weights.yaml         # candidate weights; locked copy lands in artifacts/
│   ├── lexicon/lexicon.seed.yaml            # human seed consumed by O4
│   ├── anchors/jd_anchors.yaml              # O6 authored positive/negative anchors
│   ├── gates/eligibility_rules.yaml         # R4 disqualifiers derived from the JD
│   ├── integrity/honeypot_rules.yaml        # O3-calibrated thresholds
│   └── profiles/{ci.yaml, local.yaml}
│
├── artifacts/                     # GITIGNORED. Produced by offline, consumed by online. Immutable per build.
│   ├── MANIFEST.json              # artifact registry: paths, sha256, builder version, schema version
│   ├── model/encoder.onnx
│   ├── embeddings/{candidate_vectors.parquet, anchor_vectors.npy}
│   ├── lexicon/lexicon.compiled.json
│   ├── archetypes/centroids.npy
│   ├── calibration/integrity_thresholds.json
│   └── weights/scoring_weights.locked.yaml
│
├── data/                          # GITIGNORED. Raw inputs only.
│   ├── raw/candidates.jsonl
│   └── golden/golden_labels.csv             # O8 hand-labelled relevance for weight search
│
├── src/redstack/
│   ├── __init__.py                # version constant only; no side effects
│   ├── domain/                    # PURE. Data + invariants. No IO, no ML, no pandas, no numpy-at-the-edges.
│   ├── ports/                     # Protocols. The hexagon boundary.
│   ├── features/                  # Pure structured feature extraction.
│   ├── engines/                   # The 11 Core Systems (domain services).
│   ├── config/                    # Typed config schema + YAML composition + determinism policy.
│   ├── adapters/                  # INFRASTRUCTURE. Impure. Implements ports. Only place imports onnx/st/pyarrow.
│   ├── pipelines/                 # APPLICATION. Orchestration + composition roots (offline O*, online R*).
│   ├── observability/             # Logging, timing/budget guard, run-report model.
│   └── cli/                       # User-facing entrypoints. Thinnest layer.
│
├── tests/
│   ├── unit/  property/  golden/  integration/  determinism/  fixtures/  conftest.py
│
├── scripts/
│   ├── reproduce.sh               # the literal Stage-3 command wrapper
│   └── make_sandbox_sample.py     # builds a ≤100-candidate sample for the demo sandbox (spec 10.5)
│
└── docs/
    ├── architecture.md  adr/  runbook.md
```

`src/` layout (not flat) is mandatory: it forces the package to be installed (`uv pip install -e .`) before tests run, which guarantees the Stage-3 sandbox exercises the *installed* package, not a working-directory accident.

---

## 2. Module-by-module breakdown

### `src/redstack/domain/`
Pure data and business invariants. Pydantic v2 models with `frozen=True`, `extra="forbid"`, validators that reject impossible states at construction.

```
domain/
├── ids.py            # NewType wrappers: CandidateId, ArchetypeId, AnchorId
├── enums.py          # CompanySize, InstitutionTier, WorkMode, RelevanceTier(0..4), ScoreComponent
├── errors.py         # DomainError hierarchy (IntegrityViolation, IneligibleCandidate, ...)
├── jd.py             # JobDescriptionSpec: the frozen, parsed requirements/disqualifiers/anchors-as-intent
├── candidate/
│   ├── representation.py  # CandidateRepresentation — the aggregate root
│   ├── identity.py        # CandidateId + anonymized name + raw provenance handle
│   ├── integrity.py       # IntegrityReport: flags[], honeypot_score, is_honeypot
│   ├── eligibility.py     # EligibilityReport: passed_gates, hard_blocks[], soft_penalties[]
│   ├── career.py          # CareerProfile: tenures, recency, product_vs_services, title trajectory
│   ├── semantic.py        # SemanticProfile: anchor similarities, best-anchor, dense-vector handle
│   ├── credibility.py     # CredibilityProfile: skill-trust (endorsement×duration), assessment coherence
│   ├── behavioral.py      # BehavioralProfile: availability, responsiveness, engagement (normalized 0..1)
│   ├── logistics.py       # LogisticsProfile: location fit, notice, relocation, salary-band sanity
│   ├── archetype.py       # ArchetypeAssignment: id, distance, membership confidence
│   └── quality.py         # CandidateQualityVector (CQV): the fixed-length numeric feature vector
├── scoring.py        # ScoreBreakdown (per-component), ScoredCandidate
├── ranking.py        # RankedCandidate, Ranking (ordered, invariant-checked collection)
└── reasoning.py      # ReasoningEvidence, ReasoningSentence, CandidateReasoning
```

The `CandidateRepresentation` is an **aggregate built progressively**: each online stage attaches one sub-object. Sub-objects are individually frozen; the representation is assembled via copy-on-write `with_*()` methods so partial states are never mutated in place (critical for determinism and for testing each stage in isolation).

### `src/redstack/ports/`
The hexagon's edges, as `typing.Protocol` classes (structural, zero runtime cost, no inheritance coupling).

```
ports/
├── embedding.py        # EmbeddingModelPort
├── semantic_index.py   # SemanticVectorStorePort (id-lookup + optional nearest-neighbour)
├── artifact_store.py   # ArtifactStorePort (load/verify by manifest key)
├── candidate_source.py # CandidateSourcePort (stream raw candidate dicts)
├── submission_sink.py  # SubmissionSinkPort (write validated CSV)
├── run_report_sink.py  # RunReportSinkPort
└── rng.py              # DeterministicEntropyPort (seeded RNG seam; no module reads the OS clock for logic)
```

### `src/redstack/features/`
Pure deterministic transforms turning raw candidate dicts into typed structured features. No business *judgement* lives here — only derivation. (Judgement lives in engines.)

```
features/
├── parsing.py    # raw dict -> typed Candidate model (pydantic), tolerant of schema drift, never silently coerces
├── career.py     # duration math, recency, services-vs-product classification, title-vs-history consistency
├── skills.py     # skill-name normalization, proficiency encoding, endorsement×duration trust feature
├── education.py  # tier mapping, degree/field encoding, start/end sanity
└── signals.py    # normalize redrob_signals into bounded 0..1 features; handle sentinel -1 / empty dicts
```

### `src/redstack/engines/`
The 11 Core Systems as **domain services**. Stateless callables that take a `CandidateRepresentation` (+ typed config) and return an enriched one, or a verdict. Most are 100 % pure; only `SemanticEngine` depends on a port.

```
engines/
├── integrity.py    # IntegrityEngine    — honeypot rules (O3 thresholds): exp>company-age, expert@0-months, etc.
├── eligibility.py  # EligibilityEngine  — JD hard disqualifiers + soft penalties (R4 gates)
├── lexicon.py      # LexiconEngine      — compiled-lexicon symbolic matching; anti keyword-stuffing
├── semantic.py     # SemanticEngine     — anchor similarity & archetype proximity (consumes ports only)
├── cqv.py          # CQVAssembler       — folds all sub-profiles into the CandidateQualityVector
├── behavioral.py   # BehavioralEngine   — availability/responsiveness/engagement -> bounded multiplier
├── logistics.py    # LogisticsEngine    — location/notice/relocation/salary -> bounded multiplier
├── scoring.py      # ScoringEngine      — weighted CQV combination × behavioral × logistics, gated by integrity/eligibility
├── ranking.py      # RankingEngine      — stable sort, top-100 cut, deterministic tie-break, invariant enforcement
├── reasoning.py    # ReasoningEngine    — evidence-grounded, template-free, no-LLM, no-network sentence assembly
└── validation.py   # ValidationEngine   — the spec's rules as code; mirrors validate_submission.py + reasoning checks
```

### `src/redstack/config/`
```
config/
├── schema.py        # pydantic models for every YAML file; the typed view engines/pipelines receive
├── loader.py        # deterministic deep-merge (base <- runtime <- profile), then validate-or-die
└── determinism.py   # seeds, onnxruntime thread/provider pinning, numpy/BLAS thread caps
```

### `src/redstack/adapters/`
The only packages permitted to import `onnxruntime`, `sentence_transformers`, `pyarrow`, or touch the filesystem.

```
adapters/
├── onnx_embedder.py        # EmbeddingModelPort via onnxruntime (online fallback for unseen ids)
├── st_embedder.py          # sentence-transformers — OFFLINE ONLY; guarded against online import
├── vector_store_parquet.py # candidate vectors lookup by CandidateId (mmap'd, O(1) gather)
├── artifact_store_fs.py     # manifest-verified artifact loading (sha256 check on every load)
├── candidate_jsonl.py       # streaming JSONL/JSONL.GZ reader (constant memory)
├── submission_csv.py        # UTF-8 CSV writer matching the validator byte-for-byte
└── run_report_json.py
```

### `src/redstack/pipelines/`
Application layer. The **composition roots** live here: this is the *only* place where adapters are instantiated and bound to ports.

```
pipelines/
├── context.py          # RunContext: resolved config + bound ports + seeds + loaded manifest
├── online/
│   ├── pipeline.py     # R0..R9 orchestrator (the reproduce command's spine)
│   └── stages.py       # one pure callable per R-stage
└── offline/
    ├── pipeline.py     # O1..O10 orchestrator
    └── stages/         # census, feature_extraction, integrity_calib, lexicon_discovery,
                        # embedding_gen, anchor_authoring, archetype_discovery,
                        # weight_search, validation_battery, packaging
```

### `src/redstack/observability/` and `src/redstack/cli/`
Cross-cutting (logging, stage timing with a hard budget guard, the `RunReport` model for R9) and the thinnest possible Typer entrypoint that parses args, builds a `RunContext`, and calls a pipeline.

---

## 3. Purpose of every directory

| Directory | Purpose | Mutability | Committed? |
|---|---|---|---|
| `configs/` | Human-authored, declarative behaviour. The "what", never the "how". | edited by humans | yes |
| `artifacts/` | Machine-produced offline outputs the online run depends on. | write-once per build | **no** (rebuilt) |
| `data/` | Raw inputs (candidate pool, golden labels). | read-only at runtime | **no** |
| `src/redstack/` | All importable code. | code | yes |
| `tests/` | Verification, including determinism and golden snapshots. | code | yes |
| `scripts/` | Operational glue (reproduce wrapper, sandbox-sample builder). | scripts | yes |
| `docs/` | Architecture, ADRs, runbook. | prose | yes |

The hard line: **`configs/` is intent, `artifacts/` is derived fact, `data/` is raw fact.** No code is permitted to write into `configs/` or `data/`; only the offline pipeline writes `artifacts/`.

---

## 4. Purpose of every package

| Package | Single responsibility | May import |
|---|---|---|
| `domain` | Express valid candidate/ranking states and invariants. | stdlib, `pydantic`, `numpy` (vectors only) |
| `ports` | Declare contracts the domain needs from the outside world. | `domain`, stdlib `typing` |
| `features` | Derive structured features deterministically from raw input. | `domain` |
| `engines` | Apply business judgement (integrity, fit, scoring, ranking, reasoning). | `domain`, `ports`, `features`, `config.schema` |
| `config` | Load + validate + freeze configuration; own the determinism policy. | `domain`, `pydantic`, `pyyaml` |
| `adapters` | Implement ports against real infrastructure/ML runtimes. | everything technical (`onnxruntime`, `pyarrow`, …) + `ports` + `domain` |
| `pipelines` | Sequence stages; wire adapters → ports (composition root). | all of the above |
| `observability` | Logging, timing/budget, run report. | `domain` |
| `cli` | Translate argv into a pipeline invocation. | `pipelines`, `config`, `observability` |

**Rule of thumb:** the further left in this table, the purer and more stable. `domain` should change least often; `adapters`/`cli` most often.

---

## 5. Public interfaces between modules

Interfaces are `Protocol`s in `ports/` plus the public method surface of each engine. Contracts (signatures only, no bodies):

```python
# ports/embedding.py
class EmbeddingModelPort(Protocol):
    dim: int
    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]: ...   # (n, dim), L2-normalized

# ports/semantic_index.py
class SemanticVectorStorePort(Protocol):
    def get(self, cid: CandidateId) -> npt.NDArray[np.float32] | None: ...
    def get_many(self, cids: Sequence[CandidateId]) -> tuple[npt.NDArray, list[CandidateId]]: ...

# ports/candidate_source.py
class CandidateSourcePort(Protocol):
    def stream(self) -> Iterator[Mapping[str, object]]: ...   # constant-memory

# ports/artifact_store.py
class ArtifactStorePort(Protocol):
    def load_bytes(self, key: str) -> bytes: ...              # verifies sha256 vs MANIFEST
    def load_npy(self, key: str) -> npt.NDArray: ...
```

Engine surfaces (the inter-module API the pipeline calls):

```python
class IntegrityEngine(Protocol):
    def assess(self, c: CareerProfile, raw: ParsedCandidate) -> IntegrityReport: ...
class EligibilityEngine(Protocol):
    def evaluate(self, rep: CandidateRepresentation, jd: JobDescriptionSpec) -> EligibilityReport: ...
class SemanticEngine(Protocol):
    def profile(self, cid: CandidateId, raw: ParsedCandidate) -> SemanticProfile: ...   # lookup-first, encode-fallback
class CQVAssembler(Protocol):
    def assemble(self, rep: CandidateRepresentation) -> CandidateQualityVector: ...
class ScoringEngine(Protocol):
    def score(self, rep: CandidateRepresentation, w: ScoringWeights) -> ScoredCandidate: ...
class RankingEngine(Protocol):
    def rank(self, scored: Sequence[ScoredCandidate]) -> Ranking: ...   # invariant: spec-valid top-100
class ReasoningEngine(Protocol):
    def explain(self, ranked: RankedCandidate, rep: CandidateRepresentation) -> CandidateReasoning: ...
class ValidationEngine(Protocol):
    def validate(self, ranking: Ranking) -> list[ValidationFinding]: ...
```

The pipeline depends only on these protocols. Engines never call each other directly — the pipeline sequences them and threads the growing `CandidateRepresentation` between them. This keeps the engine graph a **forest, not a web**.

---

## 6. Dependency direction rules

Inward-only (hexagonal). Allowed import arrows:

```
cli ─▶ pipelines ─▶ engines ─▶ features ─▶ domain
                      │           │           ▲
                      ▼           └──────────▶│
                    ports ──────────────────▶ domain
adapters ─▶ ports ─▶ domain          config ─▶ domain
pipelines ─▶ adapters   (composition root ONLY)
observability ─▶ domain
```

Non-negotiable:
- `domain` imports **nothing** from the project. It is the stable core.
- `engines` may import `ports` but **never** `adapters`.
- `adapters` may import `ports` and `domain` but **never** `engines` or `pipelines`.
- Only `pipelines` (composition root) sees both `engines` and `adapters` at once.
- `config.schema` is importable by engines (typed config), but `config.loader` (which does IO) is importable only by `pipelines`/`cli`.

Enforced mechanically by **`import-linter`** contracts in `pyproject.toml`, run in CI and pre-commit. A violated arrow fails the build.

---

## 7. Domain model organization

The aggregate root is `CandidateRepresentation`. It is assembled left-to-right by the online stages, each attaching exactly one slice:

```
CandidateRepresentation
├── identity      (R1)  CandidateId + provenance
├── integrity     (R4)  IntegrityReport         ← IntegrityEngine
├── eligibility   (R4)  EligibilityReport       ← EligibilityEngine
├── career        (R2)  CareerProfile           ← features.career
├── semantic      (R3)  SemanticProfile         ← SemanticEngine
├── credibility   (R2)  CredibilityProfile      ← features.skills + LexiconEngine
├── behavioral    (R2)  BehavioralProfile       ← BehavioralEngine
├── logistics     (R2)  LogisticsProfile        ← LogisticsEngine
├── archetype     (R3)  ArchetypeAssignment     ← SemanticEngine
└── quality       (R5)  CandidateQualityVector  ← CQVAssembler
```

Principles:
- **Make illegal states unrepresentable.** `RelevanceTier` is an enum, not an int. A `Ranking` cannot be constructed with a duplicate rank — the constructor rejects it. Scores are a `NewType('Score', float)` so they can't be confused with raw similarities.
- **Frozen everywhere.** Mutation happens only through `with_*` copy methods, producing a new aggregate. This makes every stage a referentially transparent function `f(rep) -> rep'`.
- **Provenance is carried, not discarded.** The Reasoning Engine needs to cite real facts (spec Stage-4 "no hallucination"), so each profile keeps a handle back to the source facts it was derived from.
- `JobDescriptionSpec` is a **value object**, parsed once, frozen, shared read-only across all candidates.

---

## 8. Configuration architecture

Three-layer YAML composition, resolved deterministically at startup:

```
base.yaml  →  runtime/{online|offline}.yaml  →  profiles/{ci|local}.yaml
   (later layers override earlier; deep-merge; order is fixed)
```

- Every YAML maps to a **pydantic v2 model** in `config/schema.py`. Unknown keys are rejected (`extra="forbid"`) so a typo fails loudly instead of silently disabling a gate.
- Behaviour-defining config (`weights/`, `anchors/`, `gates/`, `integrity/`, `lexicon/`) is **separated from runtime/IO config** (`runtime/`). A reviewer changing scoring weights never risks touching thread counts.
- The **locked** scoring weights consumed at online time come from `artifacts/weights/scoring_weights.locked.yaml` (written by O8/O10), not from `configs/`. `configs/weights/` holds the *candidate* weights for the search; the artifact holds the *frozen, validated* result. This prevents "I tweaked the YAML and the leaderboard moved" drift.
- `config/determinism.py` is the single home for: global seed, `OMP_NUM_THREADS`/`MKL_NUM_THREADS`, numpy RNG construction, and onnxruntime `SessionOptions` (intra/inter-op threads, `CPUExecutionProvider` only). Determinism is configuration, owned in one file, asserted at startup.

---

## 9. Artifact management architecture

Artifacts are the contract between offline and online. They are treated like a compiled binary: built once, hash-pinned, never hand-edited.

- **`MANIFEST.json`** is the registry: for each artifact key → relative path, `sha256`, producing builder version, and schema version.
- `ArtifactStorePort.load_*` **verifies the sha256 on every load**. A corrupted or stale artifact aborts the run rather than silently degrading the ranking (which would be undetectable without a live leaderboard).
- Artifacts are **versioned by content hash, not by mtime.** R0 (Artifact Loading) records the manifest hash into the run report, so any submission is traceable to the exact artifact set that produced it.
- The candidate-embedding artifact is the architectural payoff of R3 being **"Semantic Lookup"** rather than "encode": O5 precomputes all 100K candidate vectors offline (`sentence-transformers`), stored in `candidate_vectors.parquet` keyed by `CandidateId`. Online, R3 is an O(1) gather — this is what buys the 5-minute budget. The `onnx_embedder` adapter exists as a **fallback path** for any `CandidateId` absent from the precomputed set (robustness for the ≤100 sandbox sample and for any pool delta), keeping the online run correct without depending on the lookup being total.
- `data/` and `artifacts/` are git-ignored; `MANIFEST.json` schema and a checked-in `artifacts/README.md` document how to rebuild them (`redstack build`). The repo is reproducible from `data/raw/candidates.jsonl` + code alone.

---

## 10. Testing architecture

```
tests/
├── unit/         # one engine / one feature transform at a time; pure, fast, no IO
├── property/     # hypothesis: ranking invariants, score monotonicity, integrity idempotence
├── golden/       # snapshot: fixed candidate fixtures -> exact ScoreBreakdown / CSV bytes
├── determinism/  # same input twice -> identical bytes; thread-count invariance
├── integration/  # full R0..R9 on the sample_candidates fixture, asserting validator passes
├── fixtures/     # the sample candidates, honeypot exemplars, golden labels, tiny onnx stub
└── conftest.py
```

Discipline:
- **Engines are unit-tested in isolation** because they're pure — no mocks needed beyond a fake `EmbeddingModelPort`/`SemanticVectorStorePort` for `SemanticEngine`.
- **Property tests guard the spec invariants directly:** for any scored set, `RankingEngine.rank` produces 100 unique ranks, non-increasing scores, and `candidate_id`-ascending tie-breaks. This is where we mathematically prevent the "common rejections" in spec §6.
- **The validator is a test oracle:** an integration test runs the *project's* `validate_submission` logic against produced CSVs, and a separate test runs the *organizer's* `validate_submission.py` byte-for-byte against a sandbox sample output.
- **Honeypot fixtures** (mirroring spec §7: experience exceeding company age, expert@0-months) assert the Integrity Engine drops them — a regression here is a disqualification risk, so it's gated in CI.
- **Determinism tests** run the online pipeline twice and diff the output bytes, and run once at 1 thread and once at N threads asserting identical rankings.
- Coverage gate in CI; `mypy --strict` and `ruff` are blocking.

---

## 11. CLI architecture

A single Typer app, three verbs, each a thin shell over a pipeline composition root. The CLI **contains no business logic** — it parses, builds a `RunContext`, and dispatches.

```
redstack build    --config configs/runtime/offline.yaml         # O1..O10 -> artifacts/
redstack rank     --candidates data/raw/candidates.jsonl \      # R0..R9  -> submission.csv + run_report.json
                  --out submission.csv
redstack validate --submission submission.csv                   # ValidationEngine over a finished CSV
```

`redstack rank` is the spec §10.3 single reproduce command. `scripts/reproduce.sh` is a literal wrapper so Stage-3 reviewers copy-paste one line. Exit codes are meaningful (0 = valid submission written; non-zero = invariant or budget violation), so the sandbox check fails loudly.

---

## 12. Offline vs Online pipeline separation

The separation is **physical and import-enforced**, not merely conventional.

| | Offline (`pipelines/offline`, O1–O10) | Online (`pipelines/online`, R0–R9) |
|---|---|---|
| Runtime budget | unbounded (pre-computation) | ≤ 5 min, ≤ 16 GB, CPU, no network |
| ML runtime | `sentence-transformers` (training/encoding), `scikit-learn` (KMeans for O7, weight search O8) | `onnxruntime` only, fallback path |
| Network | allowed at author time | **forbidden** |
| Writes | `artifacts/` | `submission.csv`, `run_report.json` |
| Reads | `data/`, `configs/` | `artifacts/`, `configs/runtime/online.yaml`, candidates file |

Enforcement: an `import-linter` "forbidden" contract bans `pipelines.online.*` (and anything it imports) from importing `sentence_transformers`, `sklearn`, `adapters.st_embedder`, or any networking module. The online package *cannot* accidentally pull in a 5-minute-busting dependency, because the build fails if it tries. `adapters/st_embedder.py` additionally raises on import if an "online" environment marker is set — defence in depth.

O-stage → artifact mapping:

```
O1 Census            -> data profile / schema-drift report (gates the rest)
O2 Feature Extract   -> structured feature tables (parquet)
O3 Integrity Calib   -> integrity_thresholds.json
O4 Lexicon Discovery -> lexicon.compiled.json
O5 Embedding Gen     -> candidate_vectors.parquet, encoder.onnx
O6 Anchor Authoring  -> anchor_vectors.npy (from configs/anchors)
O7 Archetype Disc.   -> centroids.npy (KMeans)
O8 Weight Search     -> scoring_weights.locked.yaml (optimized vs golden_labels)
O9 Validation Battery-> offline quality report (NDCG/MAP on golden set)
O10 Packaging        -> MANIFEST.json (hashes everything above)
```

---

## 13. Data flow diagram

```
OFFLINE (author-time, unbounded)
 data/raw/candidates.jsonl ─▶ O1 Census ─▶ O2 Features ─▶ O3 Integrity Calib
                                                   │
 configs/lexicon.seed ─▶ O4 Lexicon ───────────────┤
 features ─▶ O5 Embeddings (sentence-transformers) ─┤
 configs/anchors ─▶ O6 Anchor vectors ─────────────┤
 embeddings ─▶ O7 Archetypes (KMeans) ─────────────┤
 features + data/golden ─▶ O8 Weight Search ───────┤
                                                   ▼
                                          O9 Validation Battery
                                                   ▼
                                          O10 Packaging ──▶ artifacts/ + MANIFEST.json

ONLINE (reproduce-time, ≤5 min)
 artifacts/ ─▶ R0 Load(+verify hashes)
 candidates.jsonl ─▶ R1 Parse ─▶ R2 Feature Extract ─┐
                                                     ▼
        R3 Semantic Lookup (vector gather; onnx fallback) ─▶ enrich SemanticProfile/Archetype
                                                     ▼
        R4 Gates + Eligibility (IntegrityEngine drops honeypots; EligibilityEngine applies JD blocks)
                                                     ▼
        R5 Scoring (CQV × behavioral × logistics, integrity/eligibility-gated)
                                                     ▼
        R6 Ranking (stable sort, top-100, tie-break cand_id asc, invariants)
                                                     ▼
        R7 Reasoning (evidence-grounded, no LLM)
                                                     ▼
        R8 Submission (UTF-8 CSV, validator-exact)  ─▶ submission.csv
                                                     ▼
        R9 Run Report (manifest hash, timings, budget, honeypot rate) ─▶ run_report.json
```

The growing object on the online path is a single `CandidateRepresentation` per candidate; stages are pure `rep -> rep'` except R0 (load), R3 (port read), R8/R9 (sink writes).

---

## 14. Import-boundary rules

Codified as `import-linter` contracts (committed, CI-blocking):

1. **Layered contract:** `domain < ports < features < engines < pipelines < cli`. No upward imports.
2. **`domain` independence:** `domain` may import only stdlib + `pydantic` + `numpy`.
3. **Engines purity:** `engines` is forbidden from importing `adapters`, `config.loader`, `observability` IO, or any ML/network module.
4. **Adapter isolation:** `adapters` is forbidden from importing `engines` or `pipelines`.
5. **Online containment:** the `pipelines.online` subgraph is forbidden from importing `sentence_transformers`, `sklearn`, `adapters.st_embedder`, `requests`/`httpx`/`urllib3`, or `socket`.
6. **Composition root exception:** only `pipelines` may import `adapters`.
7. **Config split:** `config.loader` (IO) importable only by `pipelines`/`cli`; `config.schema` (pure) importable anywhere.

---

## 15. Performance considerations

The 5-minute / 16 GB / CPU budget is met structurally, not by luck:

- **Vectorize over candidates, not loop.** R2/R3/R5 operate on column arrays. The CQV is built as an `(N, d)` matrix; scoring is a single matrix–vector product plus elementwise multipliers. No per-candidate Python object churn in the hot path — `CandidateRepresentation` objects are materialized only for the top-K that need reasoning.
- **Two-tier execution.** Cheap deterministic gates (integrity, eligibility, lexicon, logistics) run first as vectorized masks to shrink the candidate set *before* the more expensive semantic combination, then only survivors flow into full scoring. Reasoning (R7) runs on the top 100 only.
- **Semantic = lookup, not encode.** R3 gathers precomputed rows from a memory-mapped parquet/npy keyed by `CandidateId`; the onnx fallback handles only the rare miss. This is the single largest budget lever.
- **Memory:** stream the JSONL (`candidate_jsonl` yields, never `read().splitlines()`), keep embeddings as `float32` memory-mapped (≈ 100K × 384 × 4 B ≈ 150 MB), avoid pandas object-dtype columns.
- **Thread pinning** in `config/determinism.py` serves both reproducibility and predictable latency (no oversubscription thrash).
- **Budget guard** in `observability/timing.py` records per-stage wall time into the run report; a CI integration test asserts the sample run stays well under budget, catching regressions before submission (there's no live leaderboard to catch them later).

---

## 16. Future extensibility considerations

Without violating the freeze, the structure leaves clean seams:

- **New engine** → add a module under `engines/`, expose a Protocol, insert one stage call in `pipelines/online/stages.py`. Nothing else changes because engines don't reference each other.
- **New embedding backend** → new adapter implementing `EmbeddingModelPort`; swapped in the composition root. Engines are untouched.
- **New artifact** → register a key in `MANIFEST.json` + a loader method; the verify-on-load path is inherited for free.
- **New JD** → re-author `configs/anchors`, `configs/gates`, re-run O6/O8. Code is JD-agnostic; the JD is data.
- **Alternative scoring weights** → produced by O8 as a new locked artifact; A/B by manifest hash, fully traceable.
- **Learning-to-rank upgrade** (XGBoost) → fits as an O8 variant producing a model artifact + a `ScoringEngine` strategy; the CQV is already the feature vector such a model would consume.

---

## 17. Modules that must remain pure and deterministic

Pure (no IO, no clock, no unseeded randomness; identical output for identical input):

- **All of `domain/`** and **all of `features/`**.
- **Engines:** `integrity`, `eligibility`, `lexicon`, `cqv`, `behavioral`, `logistics`, `scoring`, `ranking`, `reasoning`, `validation`. `semantic` is pure *given its port* (the impurity is the adapter behind the port).
- **`config/schema.py`** and the merge logic in `config/loader.py` (deterministic deep-merge; the only IO is reading files).

These are property-tested for idempotence and determinism. Any randomness (e.g. tie-break) must be resolved by **explicit total ordering on `candidate_id`**, never by an RNG, matching the validator.

---

## 18. Modules that may depend on external ML runtimes

Confined to `adapters/` and the offline pipeline:

- `adapters/onnx_embedder.py` → `onnxruntime` (online; CPU provider; fixed threads).
- `adapters/st_embedder.py` → `sentence-transformers` (**offline only**, import-guarded).
- `pipelines/offline/stages/embedding_gen.py`, `archetype_discovery.py`, `weight_search.py` → `sentence-transformers`, `scikit-learn`.

No other module may import these. The online run's only ML dependency is `onnxruntime`, and even that sits behind `EmbeddingModelPort` so the hot path stays test-double-friendly.

---

## 19. Recommended design patterns

- **Hexagonal / Ports & Adapters** — the spine: engines depend on Protocols, adapters implement them, pipelines wire them.
- **Pipeline / Pipes-and-Filters** — O1–O10 and R0–R9 as ordered pure stages over a threaded representation.
- **Aggregate + Value Objects (DDD)** — `CandidateRepresentation` root; frozen `*Profile` value objects; copy-on-write enrichment.
- **Strategy** — scoring weights / embedding backend / vector store are interchangeable behind interfaces.
- **Composition Root** — adapter wiring lives *only* in `pipelines/*/pipeline.py`; nothing self-constructs its dependencies.
- **Manifest / Content-addressed artifacts** — hash-verified build outputs.
- **Specification objects** — `JobDescriptionSpec`, eligibility rules, integrity rules as declarative data + a pure evaluator.
- **Result/Report objects over exceptions for business verdicts** — `IntegrityReport`, `ValidationFinding[]` make gate decisions inspectable and loggable rather than control-flow-by-exception.

## 20. Anti-patterns to avoid

- **God engine / "ScoringService that does everything."** Each of the 11 engines owns exactly one concern; integrity must not live inside scoring.
- **Mutable shared state.** No in-place mutation of `CandidateRepresentation`; no module-level singletons holding model handles. State flows through `RunContext`.
- **Hidden IO in the domain/engines.** No file reads, no `datetime.now()` in scoring/reasoning, no network anywhere online. The clock and RNG are seams, not globals.
- **Config-as-magic-dict.** Never pass raw `dict`/`Any` config around; always the typed pydantic model. Untyped config defeats `mypy --strict` and hides typos until submission day.
- **Embedding at online time by default.** R3 is lookup; encoding is the exception path. Re-encoding 100K online courts the 5-minute cliff.
- **Stringly-typed identifiers and scores.** Use `NewType`/enums so a similarity can't be passed where a `Score` is expected.
- **Templated, name-insertion reasoning.** Spec Stage-4 penalizes it explicitly; the Reasoning Engine assembles from *actual evidence* attached to the representation and varies by what's present, with honest acknowledgement of gaps.
- **Silent coercion in parsing.** Sentinel values (`-1`, empty `skill_assessment_scores`, inverted salary `min > max`) are handled explicitly — they are *signal*, not noise to be smoothed away.
- **Letting the online package import offline-only libs.** Guarded by import-linter; treat a violation as a build break, never a warning.
- **Hand-editing `artifacts/`.** Artifacts are compiled outputs; edits break the manifest hash and invalidate reproducibility.
```
