# REDSTACK v1.1 — Repository & File Layout

**Status: architecture frozen.** This document does not design, add, remove, or redesign any component. It is the single navigational contract that translates the frozen layer specifications — Architecture, Domain, Ports, Features, Engines, Adapters, Offline Pipeline, Online Pipeline — into a concrete repository and file structure. Every architectural decision has already been made elsewhere; this document only answers *where each decision lives on disk*.

**How to read this document.** §0 explains the organizing principles. §1 is the complete tree. §2–§15 walk every directory and file with a fixed schema: **purpose · owner layer · dependencies · import restrictions · implementation notes · testing location**. §16–§17 fix the dependency and import-boundary rules (import-linter-enforceable). §18 fixes the build order. §19 is the ownership matrix. §20 is the readiness checklist.

**Three reconciliations between frozen documents** (carried forward verbatim, never re-litigated here):
- **Offline stages.** Architecture sketched `O1–O10`; the Offline Pipeline spec refined this to `O0–O18` producing the *same* frozen artifact contracts. The refined `O0–O18` is authoritative for stage files; the coarse names survive only as artifact-key aliases.
- **Engine decomposition.** Architecture fixes **eleven physical engine modules** under `engines/`. The Engine Layer spec describes **fourteen logical engine services** (`Candidate*Engine`, `SubmissionGenerationEngine`, `RunReportEngine`). The eleven modules are the files on disk; §9 carries the logical→physical map so no Engine-Layer concern is lost.
- **`FeatureLayout` location.** Domain build-order places the `FeatureLayout` *type* in `domain/candidate/quality.py`; the Feature Layer *owns the populated registry*. Resolution (§6/§8): the frozen VO type lives in `domain`, the populated `FeatureRegistry` + ordered constant lives in `features/registry.py`, both pinned to one shared `layout_version`.

---

## §0. Repository Philosophy

The repository encodes five disciplines simultaneously. Each maps to a structural rule that import-linter and CI enforce mechanically, so a violation is a build break rather than a review comment.

**Hexagonal Architecture (Ports & Adapters).** The core (`domain`, `features`, `engines`) depends only on abstractions (`ports`). Concrete infrastructure (`adapters`) implements those abstractions. Only the application layer (`pipelines`) — the *composition root* — sees both at once and binds an adapter to a port. This is why heavy ML/IO can never leak into the hot path: an engine literally cannot import `onnxruntime`, because it cannot import `adapters`. The hexagon edge is `ports/`, a set of `typing.Protocol`s with zero runtime coupling.

**Domain-Driven Design.** `CandidateRepresentation` is an aggregate root assembled left-to-right by online stages; each slice is a frozen value object attached copy-on-write. Business verdicts ("honeypot", "ineligible") are *Report data objects* that flow through the pipeline, not exceptions. Invariant violations (duplicate rank, wrong CQV dim) *raise*. Illegal states are unrepresentable: a `Ranking` cannot be constructed with a duplicate rank; a `Score` cannot be confused with a `Similarity` (distinct `NewType`s).

**Pipeline Architecture (Pipes & Filters).** Both pipelines are ordered pure stages over a single threaded object: offline `O0…O18` over the artifact set; online `R0…R9` over the `CandidateRepresentation`. Every stage is `f(x) -> x'`, referentially transparent and independently testable. Ports appear online only at `R0/R1/R3/R8/R9`; `R2/R4/R5/R6/R7` are pure engine work — the property that keeps the 5-minute budget predictable.

**Deterministic Systems.** No wall clock (recency uses an injected `as_of`), no online RNG (ties resolve by ascending `candidate_id`), float32 with fixed reduction order, pinned BLAS/OMP threads (thread-count-invariant output), enum-by-value serialization. Determinism is owned in exactly one file (`config/determinism.py`) and asserted at startup. Identical inputs ⇒ identical `submission.csv` bytes and identical `run_report.json` `reproducible` block.

**Offline vs Online separation.** The split is *physical and import-enforced*, not conventional. Offline (`pipelines/offline`) may use `sentence-transformers`, `scikit-learn`, seeded RNG, and authoring-time network; it writes `artifacts/`. Online (`pipelines/online`) may use `onnxruntime` only, no network, ≤5 min / ≤16 GB; it reads `artifacts/` and writes `submission.csv` + `run_report.json`. An import-linter "forbidden" contract bans the online subgraph from importing `sentence_transformers`, `sklearn`, `adapters.st_embedder`, or any networking module — the online package *cannot* accidentally pull in a budget-busting dependency.

**Why `src/` layout (not flat).** It forces `uv pip install -e .` before tests run, guaranteeing the Stage-3 sandbox exercises the *installed* package, not a working-directory accident.

**The hard line on directories.** `configs/` is **intent** (human-authored), `artifacts/` is **derived fact** (machine-built, hash-pinned, gitignored), `data/` is **raw fact** (read-only at runtime, gitignored). No code writes into `configs/` or `data/`; only the offline pipeline writes `artifacts/`.

---

## §1. Complete Repository Tree

Every file known from the frozen layer documents appears below. `[GI]` marks gitignored, machine-produced, or raw-input paths.

```text
redstack/
├── pyproject.toml
├── uv.lock
├── .python-version
├── Makefile
├── README.md
├── ARCHITECTURE.md
├── submission_metadata.yaml
├── .gitignore
├── .pre-commit-config.yaml
│
├── configs/
│   ├── base.yaml
│   ├── runtime/
│   │   ├── online.yaml
│   │   └── offline.yaml
│   ├── weights/
│   │   └── scoring_weights.yaml
│   ├── lexicon/
│   │   └── lexicon.seed.yaml
│   ├── anchors/
│   │   └── jd_anchors.yaml
│   ├── gates/
│   │   └── eligibility_rules.yaml
│   ├── integrity/
│   │   └── honeypot_rules.yaml
│   └── profiles/
│       ├── ci.yaml
│       └── local.yaml
│
├── data/                                        [GI]
│   ├── raw/
│   │   └── candidates.jsonl
│   └── golden/
│       └── golden_labels.csv
│
├── artifacts/                                   [GI] (README.md committed)
│   ├── README.md
│   ├── MANIFEST.json
│   ├── dataset_profile.json                     (O0)
│   ├── canonical_maps.json                      (O1)
│   ├── validation_report.json                   (O2)
│   ├── integrity_rules.json                     (O3)
│   ├── honeypot_catalog.json                    (O3)
│   ├── calibration/
│   │   └── integrity_thresholds.json            (O3 + O12)
│   ├── risk_weights.json                        (O12)
│   ├── lexicon/
│   │   └── lexicon.compiled.json                (O4)
│   ├── concepts.json                            (O4 + O5)
│   ├── term_graph.json                          (O4)
│   ├── phrase_graph.json                        (O4)
│   ├── jd_concepts.json                         (O6)
│   ├── gates/
│   │   └── eligibility_rules.yaml               (O6, authored→packaged)
│   ├── archetypes/
│   │   └── centroids.npy                        (O7)
│   ├── archetypes.json                          (O7)
│   ├── gold_labels.json                         (O8)
│   ├── calibration_split.json                   (O8)
│   ├── weights/
│   │   └── scoring_weights.locked.yaml          (O9)
│   ├── calibration_report.json                  (O9)
│   ├── feature_importance.json                  (O10)
│   ├── behavioral_weights.json                  (O11)
│   ├── embeddings/
│   │   ├── candidate_vectors.parquet            (O13a)
│   │   └── anchor_vectors.npy                   (O13b)
│   ├── career_vectors.parquet                   (O13c, optional)
│   ├── model/
│   │   └── encoder.onnx                         (O13)
│   ├── embedding_manifest.json                  (O13)
│   ├── feature_snapshot.parquet                 (O14)
│   ├── feature_manifest.json                    (O14)
│   ├── ranking_calibration.json                 (O15)
│   ├── reasoning_templates.json                 (O16)
│   ├── offline_report.json                      (O18)
│   └── reproducibility_report.json              (O18)
│
├── src/redstack/
│   ├── __init__.py
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── ids.py
│   │   ├── enums.py
│   │   ├── errors.py
│   │   ├── provenance.py
│   │   ├── source.py
│   │   ├── jd.py
│   │   ├── candidate/
│   │   │   ├── __init__.py
│   │   │   ├── representation.py
│   │   │   ├── identity.py
│   │   │   ├── integrity.py
│   │   │   ├── eligibility.py
│   │   │   ├── career.py
│   │   │   ├── semantic.py
│   │   │   ├── credibility.py
│   │   │   ├── behavioral.py
│   │   │   ├── logistics.py
│   │   │   ├── archetype.py
│   │   │   └── quality.py
│   │   ├── scoring.py
│   │   ├── ranking.py
│   │   ├── reasoning.py
│   │   └── validation.py
│   │
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── _types.py
│   │   ├── embedding.py
│   │   ├── semantic_index.py
│   │   ├── artifact_store.py
│   │   ├── candidate_source.py
│   │   ├── submission_sink.py
│   │   ├── run_report_sink.py
│   │   └── rng.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── layout.py
│   │   ├── registry.py
│   │   ├── view.py
│   │   ├── store.py
│   │   ├── parsing.py
│   │   ├── normalize.py
│   │   ├── career.py
│   │   ├── skills.py
│   │   ├── education.py
│   │   ├── geography.py
│   │   ├── signals.py
│   │   ├── latents.py
│   │   └── honeypot.py
│   │
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── integrity.py
│   │   ├── eligibility.py
│   │   ├── lexicon.py
│   │   ├── semantic.py
│   │   ├── cqv.py
│   │   ├── behavioral.py
│   │   ├── logistics.py
│   │   ├── scoring.py
│   │   ├── ranking.py
│   │   ├── reasoning.py
│   │   └── validation.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── loader.py
│   │   └── determinism.py
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── onnx_embedder.py
│   │   ├── st_embedder.py
│   │   ├── vector_store_parquet.py
│   │   ├── artifact_store_fs.py
│   │   ├── candidate_jsonl.py
│   │   ├── submission_csv.py
│   │   ├── run_report_json.py
│   │   └── entropy.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── context.py
│   │   ├── online/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py
│   │   │   └── stages.py
│   │   └── offline/
│   │       ├── __init__.py
│   │       ├── pipeline.py
│   │       ├── context.py
│   │       ├── runner.py
│   │       ├── registry.py
│   │       ├── graph.py
│   │       └── stages/
│   │           ├── __init__.py
│   │           ├── census.py                    (O0)
│   │           ├── normalization.py             (O1)
│   │           ├── validation.py                (O2)
│   │           ├── honeypot_discovery.py        (O3)
│   │           ├── lexicon_discovery.py         (O4)
│   │           ├── vocab_expansion.py           (O5)
│   │           ├── jd_concepts.py               (O6)
│   │           ├── archetype_discovery.py       (O7)
│   │           ├── labeling.py                  (O8)
│   │           ├── weight_search.py             (O9)
│   │           ├── feature_importance.py        (O10)
│   │           ├── behavioral_calib.py          (O11)
│   │           ├── risk_calib.py                (O12)
│   │           ├── embedding_gen.py             (O13a/b/c)
│   │           ├── feature_snapshot.py          (O14)
│   │           ├── ranking_calib.py             (O15)
│   │           ├── reasoning_templates.py       (O16)
│   │           ├── packaging.py                 (O17)
│   │           └── reproducibility.py           (O18)
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── timing.py
│   │   └── run_report.py
│   │
│   └── cli/
│       ├── __init__.py
│       ├── app.py
│       ├── build.py
│       ├── rank.py
│       └── validate.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── property/
│   ├── golden/
│   ├── integration/
│   ├── determinism/
│   ├── contract/
│   └── fixtures/
│
├── scripts/
│   ├── reproduce.sh
│   └── make_sandbox_sample.py
│
└── docs/
    ├── architecture.md
    ├── runbook.md
    └── adr/
```

---

## §2. Root-Level Files

| File | Purpose | Owner | Change frequency |
|---|---|---|---|
| `pyproject.toml` | Single source of build + tooling config: package metadata, dependency groups (online vs offline extras), and **all** tool config — `ruff`, `mypy --strict`, `pytest`, coverage, and the **import-linter contracts** (§17). | Build/Platform | Low |
| `uv.lock` | Fully pinned, committed dependency lock. Guarantees the Stage-3 sandbox resolves the exact graph the team built against. | Build/Platform | Low (on dep bumps) |
| `.python-version` | Pins `3.12` for `uv`/`pyenv`. | Build/Platform | Rare |
| `Makefile` | Thin task aliases over `uv`/CLI (`make build`, `make rank`, `make validate`, `make test`, `make lint`). No business logic — every target shells to a CLI verb (§14). | Build/Platform | Low |
| `README.md` | Setup instructions + the **one** Stage-3 reproduce command (spec 10.3). Points to `scripts/reproduce.sh`. | Docs | Medium |
| `ARCHITECTURE.md` | The frozen Architecture document (the master). This layout document is its repository-level companion. | Architecture | Frozen |
| `submission_metadata.yaml` | Mirrors the portal submission metadata (team identity, `github_repo`, `sandbox_link`, `reproduce_command`, compute declaration, AI-tools declaration, methodology summary, declarations). Filled from `submission_metadata_template.yaml`. Stage-3 review verifies portal metadata against it. | Team Lead | Medium (pre-submission) |
| `.gitignore` | Ignores `data/` and `artifacts/` (raw + derived fact), plus the usual Python/venv noise. `artifacts/README.md` is force-included. | Build/Platform | Low |
| `.pre-commit-config.yaml` | Pre-commit gate: `ruff` (lint+format), `mypy --strict`, and **import-linter**. Mirrors the CI blocking checks so boundary violations fail before push. | Build/Platform | Low |

**Rule:** root files are infrastructure, not behavior. None contains scoring logic, thresholds, or weights — those live in `configs/` (intent) or `artifacts/` (derived).

---

## §3. Config Layer Layout — `configs/`

Declarative, versioned, human-authored knobs. Three-layer composition resolved deterministically at startup: `base.yaml → runtime/{online|offline}.yaml → profiles/{ci|local}.yaml` (later layers deep-merge over earlier; order fixed). Every file maps to a pydantic v2 model in `config/schema.py` with `extra="forbid"`, so a typo fails loudly instead of silently disabling a gate.

**Critical distinction:** `configs/` is *intent*. Behaviour-defining knobs that get *frozen* (scoring weights, integrity thresholds, compiled lexicon, anchors, gates) are **consumed online from `artifacts/`, not from `configs/`**. `configs/weights/` holds the *candidate* weights for the O9 search; `artifacts/weights/scoring_weights.locked.yaml` holds the *frozen, validated* result. This prevents "I tweaked the YAML and the leaderboard moved" drift.

| File | Owner | Consumed by | Generated vs authored | Versioning rule |
|---|---|---|---|---|
| `base.yaml` | Platform | `config.loader` (all runs) | Authored | Layer-0 defaults; change rarely; bumps invalidate nothing downstream (runtime only). |
| `runtime/online.yaml` | Platform | online pipeline (R0), `config.determinism` | Authored | Holds `as_of`, seed, thread/op pins, ingestion skip-vs-abort policy, FLOOR sentinel, score-presentation transform. IO/runtime only — never weights. |
| `runtime/offline.yaml` | Platform | offline pipeline | Authored | Holds offline seed, st model id/revision, KMeans `k`, search budget, batch sizes, data roots. |
| `weights/scoring_weights.yaml` | Modeling | O9 weight search (as candidate priors) | Authored seed | The *search input*, not the online contract. O9 emits the locked artifact. |
| `lexicon/lexicon.seed.yaml` | Modeling | O4 lexicon discovery | Authored seed | Human seed terms per concept; O4 mines + O5 expands into the compiled artifact. |
| `anchors/jd_anchors.yaml` | Modeling | O6 anchor authoring | Authored | Positive/negative `jd.*` anchor intents (the JD "between the lines"). Re-authored per JD. |
| `gates/eligibility_rules.yaml` | Modeling | O6 (→ packaged artifact), `EligibilityEngine` | Authored | JD hard disqualifiers + soft penalties as declarative predicates keyed by `EligibilityCode`. |
| `integrity/honeypot_rules.yaml` | Modeling | O3 calibration | Authored | Human-declared honeypot rule shapes; O3 calibrates thresholds against the census. |
| `profiles/ci.yaml` | Platform | CI runs | Authored | Smallest-footprint overrides (sample sizes, fast paths) for the CI sample. |
| `profiles/local.yaml` | Platform | developer runs | Authored | Local-dev overrides (paths, verbosity). |

**Authoring vs frozen boundary:** anchors, gates, lexicon-seed, honeypot-rules, and candidate weights are *authored* here and *compiled/calibrated* into `artifacts/` by the offline pipeline. The online run never reads `configs/weights|lexicon|anchors|gates|integrity` — it reads their frozen artifact counterparts (§5). It reads only `configs/runtime/online.yaml` + `configs/base.yaml` + the active profile.

---

## §4. Data Layer Layout — `data/`

Raw fact. Read-only at runtime, gitignored, never written by code.

| Path | Ownership | Immutability | Ingestion path | Lifecycle |
|---|---|---|---|---|
| `data/raw/candidates.jsonl` | Organizer-provided | Read-only; never modified in place | Streamed by `CandidateSourcePort` (gzip-aware, constant memory) — offline O0/O1/O2/O13a/O14, online R1 | Dropped in by the operator; hashed (`input_file_sha256`) into every run report; ≈52 MB gz / 465 MB raw at 100K rows. |
| `data/golden/golden_labels.csv` | Team (O8 labeling output, checked in here for the search) | Read-only at calibration time | Read by O8/O9/O15 weight + ranking calibration | Produced by the O8 labeling workspace, then treated as fixed input to the deterministic search; leakage-free train/val split lives in `artifacts/calibration_split.json`. |

**Rules:** no code path writes into `data/`. The directory is gitignored; `docs/runbook.md` documents how to populate it. The candidate pool and golden labels are *facts*, distinct from `configs/` (intent) and `artifacts/` (derived). The offline pipeline reads `data/`; the online pipeline reads only `data/raw/candidates.jsonl`.

---

## §5. Artifact Layer Layout — `artifacts/`

The contract between offline and online — treated like a compiled binary: built once, hash-pinned, never hand-edited. Gitignored (except the committed `artifacts/README.md` that documents the rebuild command). **`MANIFEST.json` is the registry**; the online `ArtifactStorePort` self-verifies it, then verifies each artifact's sha256 (streamed during load) and asserts cross-artifact coherence. Any failure aborts the run — **no degraded mode**.

**The seven Architecture-frozen keys** (the minimum online contract) plus the refined Offline-Pipeline artifacts. "Producer" is the offline stage; "consumer" is the online stage or engine.

| Artifact (path / manifest key) | Producer | Consumer | Schema owner | Validation rule |
|---|---|---|---|---|
| `MANIFEST.json` | O17 Packaging | R0 (all) | `OfflineArtifactRegistry` / Ports §8 | Self-hash valid; all required keys present; cross-artifact coherence (layout/dim/model_id/anchor⊆jd_concepts). |
| `model/encoder.onnx` | O13 | R3 fallback (`OnnxEmbeddingModelAdapter`) | `embedding_manifest` | `dim`/`model_id` match manifest; loads under `CPUExecutionProvider`. |
| `embeddings/candidate_vectors.parquet` | O13a | R3 (`ParquetSemanticVectorStoreAdapter`, mmap) | `embedding_manifest` | Unit-norm rows; `id` unique; `dim == embedding.dim`. |
| `embeddings/anchor_vectors.npy` | O13b | R3 (`SemanticEngine`) | `jd_concepts` | Keys ⊆ `jd_concepts`; `dim == embedding.dim`. |
| `archetypes/centroids.npy` | O7 | R3 (nearest-centroid) | `archetypes.json` | `(k, dim)` float32; `dim == embedding.dim`; `k` fixed. |
| `lexicon/lexicon.compiled.json` | O4 | R2 (`features.skills`, `LexiconEngine`) | lexicon schema | Concepts cover the JD families. |
| `calibration/integrity_thresholds.json` | O3 + O12 | R4 (`IntegrityEngine`/Risk) | integrity schema | Thresholds in range; honeypot-recall check. |
| `weights/scoring_weights.locked.yaml` | O9 | R5 (`ScoringEngine`) | `ScoringWeights` VO | Keys == `ScoreComponent` exactly; `layout_version` matches `feature_manifest` + `FeatureLayout`. |
| `dataset_profile.json` | O0 | (offline only; O3/O11/O12 priors) | census schema | Coverage sane; `N == 100K`. |
| `canonical_maps.json` | O1 | O13/O14 normalization parity | norm schema | Maps total; no empties. |
| `validation_report.json` | O2 | offline audit | — | Reject rate within bound. |
| `integrity_rules.json` | O3 | R4 (`IntegrityEngine` detectors) | rules schema | Every code ∈ `IntegrityFlag`. |
| `honeypot_catalog.json` | O3 | offline calibration / fixtures | — | Suspect ids ⊆ pool. |
| `risk_weights.json` | O12 | R4 (`CandidateRiskEngine`) | risk schema | Bounds ordered. |
| `concepts.json` | O4 + O5 | R2 competency features | concept schema | Each concept has anchor text. |
| `term_graph.json` / `phrase_graph.json` | O4 | features (competency `in_career`) | lexicon schema | Acyclic where expected. |
| `jd_concepts.json` | O6 | R3 (`CandidateFitEngine` latents) | jd schema | Polarity tagged; anchor set ⊆. |
| `gates/eligibility_rules.yaml` | O6 (authored→packaged) | R4 (`EligibilityEngine`) | gate schema | Every code ∈ `EligibilityCode`. |
| `archetypes.json` | O7 | R3/R7 (labels, fingerprints, target flags) | archetype schema | Ids contiguous. |
| `gold_labels.json` | O8 | O9/O15 (offline) | gold schema | Tiers ∈ 0..4; ids ⊆ pool. |
| `calibration_split.json` | O8 | O9/O15 (offline) | gold schema | Disjoint; no leakage. |
| `calibration_report.json` | O9 | offline audit | — | Cross-val NDCG, weight stability recorded. |
| `feature_importance.json` | O10 | R7 (`ReasoningEngine` clause selection) | importance schema | Features ⊆ layout. |
| `behavioral_weights.json` | O11 | R5 (multiplier curves/bounds) | bhv schema | Bounds ordered; behavioral cannot dominate relevance. |
| `embedding_manifest.json` | O13 | R0 coherence; R2 doc-composition recipe | emb schema | Recipe matches online normalization byte-for-byte. |
| `career_vectors.parquet` *(optional)* | O13c | R3 (optional finer matching) | `embedding_manifest` | Unit-norm; ids ⊆ pool. |
| `feature_snapshot.parquet` | O14 | O8/O9/O10 + O18 online-parity oracle | `feature_manifest` | No NaN; `D == layout dim`. |
| `feature_manifest.json` | O14 | R2/R5 (layout check), O9 | layout schema | Matches `FeatureLayout`; shared `layout_version`. |
| `ranking_calibration.json` | O15 | R6 (tie-break/monotone confirm; disabled online by default) | rank schema | Monotone; validator-pass; **order-preserving (never changes ranks)**. |
| `reasoning_templates.json` | O16 | R7 (`ReasoningEngine`) | reason schema | Every slot has an evidence kind. |
| `offline_report.json` | O18 | offline audit | `RunReport` shape | `reproducible` block deterministic across rebuilds. |
| `reproducibility_report.json` | O18 | ship gate | — | All parity/dry-run/determinism checks pass. |
| `artifacts/README.md` *(committed)* | Platform | humans | — | Documents `redstack build` rebuild path. |

**Versioning rule (all artifacts):** value-changing edit ⇒ **minor**; layout/order/dim change ⇒ **major**. `layout_version` is shared across `FeatureLayout` / `feature_manifest` / `scoring_weights.locked` and must agree (online raises `ArtifactContractError` otherwise). Artifacts are versioned by **content hash**, not mtime; R0 records the manifest hash into the run report so any submission is traceable to the exact artifact set.

---

## §6. Domain Package Layout — `src/redstack/domain/`

**PURE.** Imports only stdlib, `pydantic` v2, and `numpy` (the latter only for the CQV array type). No IO, no ML, no pandas, no clock, no RNG, no network, no project package except other `domain` modules. Every model is `frozen=True, extra="forbid"`. The only `Any`-accepting boundary is `RawCandidate.from_mapping`, which narrows immediately.

**Imports forbidden everywhere in `domain/`:** `ports`, `features`, `engines`, `adapters`, `pipelines`, `config.loader`, `observability`, `onnxruntime`, `sentence_transformers`, `pandas`, `pyarrow`, `sklearn`, networking, `datetime.now`, any RNG.

| File | Responsibility | Imports allowed | Public types exposed |
|---|---|---|---|
| `ids.py` | `NewType` nominal aliases (mypy-only) for value separation. | stdlib `typing` | `CandidateId, AnchorId, ArchetypeId, SkillName, Score, Similarity, UnitScore, Multiplier, Months, LpaAmount, FeatureIndex` |
| `enums.py` | All closed vocabularies; ordered enums expose explicit `ordinal`. Serialize by value. | stdlib `enum` | `CompanySize, InstitutionTier, WorkMode, Proficiency, LanguageProficiency, RelevanceTier, CareerTrack, Severity, BuildStage, ScoreComponent, IntegrityFlag, EligibilityCode, ReasoningPolarity, LocationFit, NoticeFit, SignalAvailability, ValidationCode, EvidenceKind` |
| `errors.py` | Exception hierarchy for *invariant/programming* failures (verdicts are data, not exceptions). | — | `DomainError, SchemaError(InvalidCandidateId, FieldSchemaViolation), InvariantViolation(RepresentationStageError, CQVInvariantError, ScoreInvariantError, RankingInvariantError), ProvenanceError, ArtifactContractError, IntegrityViolation, IneligibleCandidate` |
| `provenance.py` | The anti-hallucination mechanism as types. | `ids`, `enums`, stdlib `datetime` | `EvidenceRef, ProvenanceHandle` |
| `source.py` | Lossless, validated mirror of `candidate_schema.json`; the canonical source for all `EvidenceRef`s. Tolerant of *semantic* contradictions (preserved for honeypot detection), strict on *types*. | `ids, enums, errors`, `pydantic` | `RawCandidate, RawProfile, RawPosition, RawEducation, RawSkill, RawCertification, RawLanguage, RawSignals` (+ `from_mapping`) |
| `jd.py` | `JobDescriptionSpec` value object — parsed once, frozen, shared read-only across all candidates. | `ids, enums` | `JobDescriptionSpec` |
| `candidate/representation.py` | The aggregate root, threaded R1→R7; optional slices + monotonic `BuildStage`; COW `with_*()` + staged accessors. | all `domain` siblings | `CandidateRepresentation` |
| `candidate/identity.py` | Identity + provenance anchor; equality/hash by `candidate_id` only. | `ids, provenance` | `Identity` |
| `candidate/integrity.py` | Honeypot verdict (R4). | `ids, enums, provenance` | `IntegrityReport, IntegrityFinding` |
| `candidate/eligibility.py` | JD hard blocks + soft penalties (R4). | `ids, enums, provenance` | `EligibilityReport, EligibilityFinding` |
| `candidate/career.py` | Structured career facts + product-vs-services + recency derivations. | `ids, enums, provenance` | `CareerProfile, PositionFact, TenureStats, CareerRecency` |
| `candidate/semantic.py` | Dense-fit signals; references (never inlines) the 384-d vector. | `ids, provenance`, numpy typing | `SemanticProfile, VectorRef` |
| `candidate/credibility.py` | Skill-trust + anti keyword-stuffing layer. | `ids, enums, provenance` | `CredibilityProfile, SkillTrust` |
| `candidate/behavioral.py` | Normalized 23-signal families; `SignalAvailability` discriminates UNKNOWN from low. | `ids, enums` | `BehavioralProfile` |
| `candidate/logistics.py` | Location/notice/relocation/salary fit; preserves salary inversion (not corrected). | `ids, enums` | `LogisticsProfile, SalaryBand` |
| `candidate/archetype.py` | O7 cluster assignment; ties by `ArchetypeId`. | `ids, enums` | `ArchetypeAssignment` |
| `candidate/quality.py` | The fixed-length `CandidateQualityVector` + the `FeatureLayout` **type** (the populated constant/registry lives in `features/registry.py`). Only model granted `arbitrary_types_allowed` (for the ndarray). | `ids`, numpy | `CandidateQualityVector, FeatureLayout` (type) |
| `scoring.py` | Per-component breakdown + scored candidate; invariant `weighted == raw*weight`, `base == Σ weighted`, gating ⇒ FLOOR. | `ids, enums, provenance` | `ScoringWeights, ScoreComponentValue, GateOutcome, ScoreBreakdown, ScoredCandidate` |
| `ranking.py` | Ordered, invariant-checked collection — the validator's six rules become unrepresentable-if-violated. | `ids, scoring, reasoning` | `RankedCandidate, Ranking` |
| `reasoning.py` | Evidence-grounded reasoning models; a clause cannot construct without ≥1 `EvidenceRef`. | `ids, enums, provenance` | `ReasoningClause, CandidateReasoning` |
| `validation.py` | The spec's validator rules + Stage-4 reasoning checks as data. | `enums` | `ValidationFinding, ValidationReport` |

**Testing location:** `tests/unit/domain/`, `tests/property/domain/` (idempotence, monotonicity, invariant rejection), `tests/golden/domain/` (serialization stability).

---

## §7. Ports Package Layout — `src/redstack/ports/`

The hexagon edges as `typing.Protocol`s plus small frozen DTOs they own. May import **only** stdlib `typing`, `domain/`, and numpy typing. May **never** import `adapters`, `engines`, `pipelines`, `config.loader`, `observability`, or any ML/IO runtime. Engines depend on these abstractions; adapters implement them; only `pipelines` instantiates an adapter and binds it.

| File | Protocol(s) defined | Implementing adapter(s) | Consuming engine / stage |
|---|---|---|---|
| `_types.py` | Shared DTOs (no protocols). | — (imported by all ports) | `FloatVector, FloatMatrix, RawMapping, ArtifactKey, ArtifactLocator, SourceRecord(Ok/Malformed), BulkVectorResult, SubmissionReceipt, ReportReceipt, Manifest, RunReport` (structural) |
| `embedding.py` | `EmbeddingModelPort` (`dim`, `model_id`, `encode`). | `onnx_embedder` (online), `st_embedder` (offline) | `SemanticEngine` (R3 fallback); offline O5/O6/O13 |
| `semantic_index.py` | `SemanticVectorStorePort` (`dim`, `contains`, `get`, `get_many`, `view_all`). | `vector_store_parquet` | `SemanticEngine` (R3) |
| `artifact_store.py` | `ArtifactStorePort` (`manifest`, `verify_all`, `load_bytes/text/json/npy`, `locate`). | `artifact_store_fs` | pipeline R0; offline O9/O17 verify |
| `candidate_source.py` | `CandidateSourcePort` (`stream`, `count`). | `candidate_jsonl` | pipeline R1; offline O0/O1/O2/O5 |
| `submission_sink.py` | `SubmissionSinkPort` (`write` → `SubmissionReceipt`). | `submission_csv` | pipeline R8; offline O9 dry-run |
| `run_report_sink.py` | `RunReportSinkPort` (`write` → `ReportReceipt`); **owns** the `RunReport` structural Protocol so `ports` never imports `observability`. | `run_report_json` | pipeline R9; offline build report |
| `rng.py` | `DeterministicEntropyPort` (`seed`, `as_of`, `derive`, `numpy_generator`); the single RNG + `as_of` seam. | `entropy` (`OfflineEntropy` / `OnlineEntropy`) | engines via injected `as_of`; offline O7/O8 RNG; online RNG **disabled** (raises) |

**Verdicts-vs-failures discipline:** a missing candidate / malformed line is *data* (returned as a value); integrity/contract violations (hash mismatch, wrong dim, broken submission invariant) *raise*. **Testing location:** `tests/contract/` (one shared parametrized suite per port, run against every real adapter *and* its fake) + `tests/fixtures/` (the fakes).

---

## §8. Features Package Layout — `src/redstack/features/`

**PURE.** Imports only `domain/`, `config.schema`, stdlib, and numpy. **No ports, no IO, no ML runtime, no clock, no RNG.** Recency uses the injected `as_of`. Semantic-similarity values and behavioral raw signals arrive already-resolved from the engines that own the ports; this layer never calls `EmbeddingModelPort`/`SemanticVectorStorePort`. Every feature cell is `FeatureCell = (value, confidence, evidence)`; the bulk path stores `value` in `(N,D)` and `confidence` at group granularity `(N,G)`, with full per-feature confidence + evidence materialized only for survivors/top-K.

This package realizes the 30 feature groups, the 11 `jd.*` latents, the honeypot detectors, and the behavioral composites from the frozen Feature Layer, organized into the modules below. The five Architecture-named extractors (`parsing, career, skills, education, signals`) are the core; the infrastructure modules (`layout, registry, view, store`) are the Feature-Layer-owned scaffolding; `normalize, geography, latents, honeypot` carry the remaining frozen concerns.

| File | Extracted feature families / role | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `layout.py` | The ordered, versioned CQV index map binding `FeatureId → FeatureIndex`; pins `layout_version`. References the `domain.candidate.quality.FeatureLayout` type. | — (frozen constant) | `FEATURE_LAYOUT` ordered tuple, `layout_version` | `domain.candidate.quality`, `domain.ids` |
| `registry.py` | The populated `FeatureRegistry`: every `FeatureDefinition` (id, group, dtype, bounds, source slices, dependencies, version, tier A–D, polarity). Single source of truth; rejects unknown ids. | layout | `FeatureRegistry, FeatureDefinition, FeatureSchema, FeatureMetadata, FeatureVersion, FeatureManifest` | `layout`, `domain` |
| `view.py` | Read-only `FeatureView` accessor (`get`, `group_confidence`, `importance`) — the only surface engines use to read feature cells. | registry, computed cells | `FeatureView` | `registry`, `domain` |
| `store.py` | Snapshot/lineage/cache metadata + explainability chain. | registry, extractor outputs | `FeatureSnapshot, FeatureLineage, FeatureCache, FeatureProvenance, FeatureAuditRecord, FeatureImportance, FeatureContracts, FeatureValidation` | `registry`, `domain` |
| `parsing.py` | Raw dict → `RawCandidate` (tolerant of schema drift, never silently coerces); mints `EvidenceRef` paths. Realizes Ingestion/O2 Validation. | `RawMapping` | `RawCandidate`, `SchemaError` | `domain.source, domain.errors` |
| `normalize.py` | Canonical text/date/skill-token/company normalization (NFC, lexicon canonical map) + **composed embedding document** (fixed field order pinned by `embedding_manifest`). Realizes the Normalization engine. | `RawCandidate`, canonical maps | `NormalizedCandidate` intermediate | `domain`, `config.schema` |
| `career.py` | Career intelligence: `exp.*, sen.*, co.*, pvs.*, career.*` (progression, stability, title-inflation, product-density, hands-on, production-exposure). Descriptions dominate titles. | `RawCandidate.career_history`, `as_of` | `CareerProfile` + career CQV cells | `domain.candidate.career` |
| `skills.py` | Competency groups 8–16 (`retr/rank/recsys/ir/nlp/llm/mle/mlops/eval`) as trust-weighted evidence aggregates; credibility trust (endorsement×duration×assessment); the `jd.keyword_only` anti-stuffer primitive. | `skills[]`, assessment scores, descriptions, semantic sims | `CredibilityProfile` (structural) + competency cells | `domain.candidate.credibility` |
| `education.py` | `edu.*` tier/field/timeline; cross-links `hp` on impossible timelines. | `education[]` | education CQV cells | `domain.enums` |
| `geography.py` | `geo.*, reloc.*, notice.*, sal.*` location/hub/relocation/notice/salary cells. | `LogisticsProfile` inputs, JD hub set | logistics CQV cells | `domain.candidate.logistics` |
| `signals.py` | The 23 `redrob_signals` → `bhv.*` composites (Part 4) with **sentinel→UNKNOWN** discipline (`−1`/`{}` never become 0); single-ownership routing. | `RawSignals`, `as_of` | `BehavioralProfile` + behavioral cells | `domain.candidate.behavioral` |
| `latents.py` | The `jd.*` positive/negative latent composition (Part 2): `jd.retrieval_ranking, jd.production_ml, jd.product_company, jd.hybrid_retrieval, jd.eval_framework`; negatives `jd.keyword_only, jd.consulting_only, jd.title_chaser, jd.pure_researcher, jd.framework_enthusiast, jd.inactive` + per-latent confidence. | competency + career + behavioral cells | latent CQV cells | `registry`, `domain` |
| `honeypot.py` | The `hp.*` impossibility detectors + `hp.composite` (Part 5): timeline, skill-time, overlap, title-seniority, education-career, salary (soft), experience-inflation, keyword-stuffing, behavioral/signal/identity. | `RawCandidate`, `CareerProfile`, thresholds | `hp.*` cells (consumed by `IntegrityEngine`) | `domain` |

**Import restrictions:** no `ports`, `adapters`, `engines`, `pipelines`, IO, or ML. **Testing location:** `tests/unit/features/` (golden value tests on `sample_candidates`), `tests/property/features/` (range/no-NaN/idempotence), honeypot fire/don't-fire fixtures in `tests/fixtures/`.

---

## §9. Engines Package Layout — `src/redstack/engines/`

The eleven Architecture-frozen domain services. Stateless callables: take a `CandidateRepresentation` (+ typed config / injected port) and return an enriched one or a verdict. Engines depend **only** on `domain`, `ports` (injected), `features`, and `config.schema`; they **never** import `adapters`, `pipelines`, `observability`, or **each other** (the graph is a forest — the pipeline threads the representation between them). All are 100% pure except `semantic`, which is pure *given its ports*.

**Logical→physical reconciliation.** The Engine Layer spec's fourteen logical services map onto these eleven files (and, for IO-bound services, onto adapters + observability wired by the pipeline):

| Logical engine (Engine Layer) | Physical home |
|---|---|
| CandidateIngestionEngine | `features/parsing.py` + pipeline (via `CandidateSourcePort`) |
| CandidateNormalizationEngine | `features/normalize.py` |
| CandidateConsistencyEngine | `engines/integrity.py` (detection half) |
| CandidateEmbeddingEngine | `adapters/{onnx,st}_embedder.py` behind `EmbeddingModelPort`; thin use in `engines/semantic.py` |
| CandidateRetrievalEngine | `engines/semantic.py` |
| CandidateFeatureEngine | `features/*` + `engines/cqv.py` |
| CandidateBehaviorEngine | `engines/behavioral.py` |
| CandidateFitEngine | `engines/eligibility.py` (+ `lexicon.py`, `logistics.py` fit) |
| CandidateRiskEngine | `engines/integrity.py` (aggregation half) |
| CandidateScoringEngine | `engines/scoring.py` |
| CandidateRankingEngine | `engines/ranking.py` |
| CandidateReasoningEngine | `engines/reasoning.py` |
| SubmissionGenerationEngine | `engines/validation.py` + `adapters/submission_csv.py` |
| RunReportEngine | `observability/run_report.py` + `adapters/run_report_json.py` |

| File | Ownership / concern | Dependencies (injected ports) | Inputs | Outputs | Side effects | Runtime budget |
|---|---|---|---|---|---|---|
| `integrity.py` | `IntegrityEngine` — honeypot detection (O3 rules) + risk aggregation; `is_honeypot = (≥2 HARD) OR (composite ≥ threshold)`. | ArtifactStore (calibration) via pipeline | `CareerProfile`, `RawCandidate`, thresholds, `hp.*` cells | `IntegrityReport` | none (pure) | ≤ ~15–25 s (R4 share) |
| `eligibility.py` | `EligibilityEngine` — JD hard blocks + soft penalties from `gates/eligibility_rules.yaml`. | ArtifactStore (gates/lexicon) | representation + `JobDescriptionSpec` | `EligibilityReport` | none | within R4 ≤ 10 s |
| `lexicon.py` | `LexiconEngine` — compiled-lexicon symbolic matching; anti keyword-stuffing corroboration. | ArtifactStore (lexicon) | normalized tokens, descriptions | credibility/competency support | none | within R2 |
| `semantic.py` | `SemanticEngine` (Retrieval) — anchor cosine + nearest-centroid; **lookup-first, encode-fallback**. The only port-dependent engine. | SemanticVectorStore, EmbeddingModel | `CandidateId`s, anchors, centroids | `SemanticProfile, ArchetypeAssignment` | port reads only | ≤ 15 s (R3) |
| `cqv.py` | `CQVAssembler` — folds all populated slices into the `(N,D)` CandidateQualityVector aligned to `FeatureLayout`. | — | populated representation | `CandidateQualityVector` | none | within R2/R5 |
| `behavioral.py` | `BehavioralEngine` — availability/responsiveness/engagement → bounded **multiplier** inputs. | Entropy (`as_of`) | `BehavioralProfile` inputs | bounded multiplier | none | within R2 |
| `logistics.py` | `LogisticsEngine` — location/notice/relocation/salary → bounded **multiplier**. | — | `LogisticsProfile` inputs | bounded multiplier | none | within R2 |
| `scoring.py` | `ScoringEngine` — weighted CQV combination × behavioral × logistics, integrity/eligibility-gated; `base == Σ weighted`; floored ⇒ FLOOR. | ArtifactStore (weights) | folded CQV + gates + multipliers + weights | `ScoredCandidate` | none | ≤ 8 s (R5) |
| `ranking.py` | `RankingEngine` — stable sort `(−score, candidate_id)`, top-100, deterministic tie-break, invariant enforcement. | — | `ScoredCandidate[]` | `Ranking` | none | ≤ 2 s (R6) |
| `reasoning.py` | `ReasoningEngine` — evidence-grounded, template-free, **no-LLM/no-network** clause assembly over top-K. | — | `RankedCandidate` + re-hydrated raw | `CandidateReasoning`, `Ranking.with_reasoning` | none | ≤ 3 s (R7) |
| `validation.py` | `ValidationEngine` — the spec's validator rules as code (mirrors `validate_submission.py`) + Stage-4 reasoning checks; defence-in-depth at R9. | — | `Ranking` | `ValidationReport` | none | ≤ 1 s |

**Import restrictions (all):** allowed `domain, ports, features, config.schema`; forbidden `adapters, pipelines, observability` IO, `config.loader`, any ML/network module, cross-engine imports, `datetime.now`, online RNG. **Testing location:** `tests/unit/engines/` (pure, with domain fixtures + fake ports for `semantic`), `tests/property/engines/` (ranking invariants, score monotonicity, integrity idempotence), `tests/golden/engines/` (exact `ScoreBreakdown`).

---

## §10. Adapters Package Layout — `src/redstack/adapters/`

**INFRASTRUCTURE — impure.** The *only* layer permitted to touch the filesystem, read/write CSV/JSONL/parquet/numpy, and load ONNX or sentence-transformers. Each adapter implements **exactly one** port and contains **no** business logic. May import `domain`, `ports`, `config.schema`, and infrastructure libraries; may **never** import `engines` or `pipelines`. Cross-adapter dependency is always via an injected port. Instantiated only by the pipeline composition roots.

`adapter → port → pipeline usage`:

| File (class) | Implements (port) | Runtime / library | Pipeline usage |
|---|---|---|---|
| `candidate_jsonl.py` (`JsonlCandidateSourceAdapter`) | `CandidateSourcePort` | stdlib `io, gzip, json` | offline O0/O1/O2/O5; online R1 |
| `artifact_store_fs.py` (`FilesystemArtifactStoreAdapter`) | `ArtifactStorePort` | `pathlib, hashlib, json`, numpy, `config.schema` | offline O17 verify; online R0 |
| `onnx_embedder.py` (`OnnxEmbeddingModelAdapter`) | `EmbeddingModelPort` | `onnxruntime` (CPU EP), numpy | online R3 **fallback** |
| `st_embedder.py` (`SentenceTransformerEmbeddingAdapter`) | `EmbeddingModelPort` | `sentence-transformers, torch` | offline O13 **only — import-guarded** |
| `vector_store_parquet.py` (`ParquetSemanticVectorStoreAdapter`) | `SemanticVectorStorePort` | `pyarrow` / numpy mmap | online R3 |
| `submission_csv.py` (`CsvSubmissionSinkAdapter`) | `SubmissionSinkPort` | `csv, io, hashlib` | offline O9 dry-run; online R8 |
| `run_report_json.py` (`JsonRunReportSinkAdapter`) | `RunReportSinkPort` | `json, io, hashlib` | offline O18 report; online R9 |
| `entropy.py` (`OfflineEntropy` / `OnlineEntropy`) | `DeterministicEntropyPort` | numpy | offline O7/O8 RNG; online `as_of` only (RNG raises) |

**Key adapter contracts (implementation notes):** `artifact_store_fs` self-hashes the manifest, streams per-key sha256 during load, exposes `locate()` for parquet/ONNX (consumer mmaps), enforces path containment + safe deserialization (`allow_pickle=False`, no pickle), fail-fast with no degraded mode. `candidate_jsonl` is gzip-transparent, UTF-8 strict, `source_index` = enumeration order, blank-line skip, O(1) memory, `Malformed` as data. `vector_store_parquet` mmaps read-only with an in-memory id index; miss ⇒ `None`/`missing`. `submission_csv` writes fixed header/order/precision, RFC-4180 quoting, UTF-8 no-BOM, `\n`, atomic temp+rename, post-format invariant re-assert, `output_sha256`; passes the organizer `validate_submission.py`. `onnx_embedder` pins threads, fixed pooling+L2 norm, order-preserving, no network. `st_embedder` import-guards against online (`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`), pins revision, exports `encoder.onnx` with ε-parity verification.

**Testing location:** `tests/contract/` (shared port suites run identically against each adapter and its fake) + `tests/unit/adapters/` (adapter-specific: gzip-vs-plain parity, tampered-byte detection, path-traversal rejection, ε-parity, atomicity) using temp dirs.

---

## §11. Config Package Layout — `src/redstack/config/`

Typed config schema + deterministic YAML composition + the determinism policy. `config.schema` is pure and importable anywhere; `config.loader` does IO and is importable only by `pipelines`/`cli`.

| File | Responsibility | Ownership | Imports allowed | Imported by |
|---|---|---|---|---|
| `schema.py` | Pydantic v2 model for **every** YAML file (`extra="forbid"`); the typed view engines/pipelines receive. Untyped config is forbidden — `mypy --strict` must see the shape. | Platform | `domain`, `pydantic` | anywhere (engines, features, adapters, pipelines, cli) |
| `loader.py` | Deterministic deep-merge (`base ← runtime ← profile`), then validate-or-die. The merge logic is pure; the only IO is reading files. | Platform | `schema`, `pyyaml`, stdlib | `pipelines`, `cli` **only** |
| `determinism.py` | The single home for: global seed, `OMP_NUM_THREADS`/`MKL_NUM_THREADS`, numpy RNG construction, onnxruntime `SessionOptions` (intra/inter-op threads, `CPUExecutionProvider` only). Asserted at startup. | Platform | stdlib, numpy, `onnxruntime` types | `pipelines`, `cli` |

**Import restriction (enforced):** `config.loader` (IO) must be unreachable from `engines`/`features`/`domain`; only `config.schema` (pure) crosses into them. **Testing location:** `tests/unit/config/` (merge determinism, unknown-key rejection), `tests/determinism/` (thread-pin assertions).

---

## §12. Pipelines Package Layout — `src/redstack/pipelines/`

**APPLICATION layer — the composition roots.** This is the *only* place adapters are instantiated and bound to ports. Pipelines may import all of `domain, ports, engines, adapters, config, observability, features`. They sequence stages and thread the growing `CandidateRepresentation` (online) or artifact set (offline); engines never self-construct dependencies.

### Shared

| File | Responsibility |
|---|---|
| `context.py` | `RunContext` base: resolved config + bound ports + seeds + loaded manifest. The immutable carrier handed to stages. |

### Online — `pipelines/online/`

| File | Responsibility | Orchestration |
|---|---|---|
| `pipeline.py` | The R0…R9 orchestrator — the literal Stage-3 reproduce spine. Builds the immutable `OnlineRunContext` (binds `CandidateSource, SemanticVectorStore, EmbeddingModel(onnx), SubmissionSink, RunReportSink, OnlineEntropy`), then runs stages strictly sequentially R0→R9, copy-on-write. Fail-fast on any integrity/coherence failure; never returns a partially-bound context. | Composition root; ports touched only at R0/R1/R3/R8/R9. |
| `stages.py` | One pure callable per R-stage: `r0_load, r1_ingest, r2_features, r3_semantic, r4_gates, r5_score, r6_rank, r7_reason, r8_submit, r9_report`. Each is `f(ctx, x) -> x'`; R2/R4/R5/R6/R7 make **no port calls**. | Pure stage functions. |

### Offline — `pipelines/offline/`

| File | Responsibility |
|---|---|
| `pipeline.py` | `OfflinePipeline`: declares the O0–O18 stage set + dependency DAG; pure orchestration over injected stage callables (`plan() → run() → finalize()`). Owns no IO itself. |
| `context.py` | `OfflinePipelineContext`: the resolved build environment — config, bound ports (`CandidateSource, EmbeddingModel(st), ArtifactStore, OfflineEntropy`), `seed`, `as_of`, output roots, `FeatureRegistry`/`FeatureLayout`. Immutable. Records `config_hash, seed, as_of, code_version`. |
| `runner.py` | `OfflinePipelineRunner`: executes the plan with **resume** (skip up-to-date by checkpoint hash), checkpointing (`StageReceipt`s), per-stage timing/metrics, failure quarantine. Staleness keyed by `(input_hashes, stage_version, config_slice)`. |
| `registry.py` | `OfflineArtifactRegistry`: typed catalog of *expected* artifacts (keys, schemas, owners, versions, lineage, validators). Each produced artifact validated against it before manifesting. |
| `graph.py` | `OfflineExecutionGraph`: the O0–O18 dependency DAG + parallelization plan; deterministic topo order (ties by stage id). |
| `stages/census.py … stages/reproducibility.py` | One module per offline stage (O0–O18). Each: `purpose · inputs · outputs · deps · algorithm · artifacts`. Mapping: `census`(O0), `normalization`(O1), `validation`(O2), `honeypot_discovery`(O3), `lexicon_discovery`(O4), `vocab_expansion`(O5), `jd_concepts`(O6), `archetype_discovery`(O7), `labeling`(O8), `weight_search`(O9), `feature_importance`(O10), `behavioral_calib`(O11), `risk_calib`(O12), `embedding_gen`(O13a/b/c), `feature_snapshot`(O14), `ranking_calib`(O15), `reasoning_templates`(O16), `packaging`(O17), `reproducibility`(O18). |

**Online containment (import-linter, CI-blocking):** `pipelines.online.*` and anything it imports is forbidden from importing `sentence_transformers, sklearn, adapters.st_embedder, requests/httpx/urllib3, socket`. **Testing location:** `tests/integration/` (full R0…R9 on `sample_candidates`, organizer-validator pass), `tests/determinism/` (twice-run byte-diff; 1-thread vs N-thread), offline stage tests under `tests/unit/pipelines/offline/`.

---

## §13. Observability Package Layout — `src/redstack/observability/`

Cross-cutting. May import `domain` only (and stdlib). It **builds** an object conforming to the port-owned `RunReport` structural Protocol — `observability` never imports `ports` and `ports` never imports `observability`.

| File | Responsibility | Imports | Imported by |
|---|---|---|---|
| `logging.py` | Structured, deterministic logging (no timestamps in any reproducibility-relevant line; audit lines clearly separated). Per-stage metric capture surface. | stdlib | pipelines, cli |
| `timing.py` | Stage timing + the **hard budget guard**: records per-stage wall-ms into the run report, computes `within_budget` and `peak_rss_mb`. A CI integration test asserts the sample run stays well under the 5-min budget (no live leaderboard to catch regressions later). | stdlib, `resource` | pipelines |
| `run_report.py` | The `RunReport` model builder: assembles the `reproducible` block (`code_version, config_hash, manifest_hash, artifact_hashes, input_file_sha256, candidate_count, output_sha256, honeypot_count_top100, honeypot_rate, eligibility_summary, score_distribution_digest`), the `audit` block (`run_id, started_at, ended_at, host_label` — excluded from any repro hash), `timings`, and `budget`. Conforms structurally to `RunReportSinkPort`'s `RunReport`. | `domain` | pipelines R9 / offline O10/O18 |

**Testing location:** `tests/unit/observability/` (budget-guard math, reproducible/audit split), `tests/determinism/` (the `reproducible` block is byte-stable across runs; `audit` is ignored by determinism assertions).

---

## §14. CLI Package Layout — `src/redstack/cli/`

The thinnest layer — a single Typer app, three verbs, **no business logic**. Each verb parses argv, builds a `RunContext` (via `config.loader` + `config.determinism`), and dispatches to a pipeline composition root. Exit codes are meaningful (0 = valid submission/artifacts written; non-zero = invariant or budget violation) so the sandbox check fails loudly.

| File | Responsibility | Maps to |
|---|---|---|
| `app.py` | The Typer application object; wires the three verbs; global options (`--config`, `--profile`, `--verbose`). | — |
| `build.py` | `redstack build --config configs/runtime/offline.yaml` → runs O0…O18 → writes `artifacts/` + `MANIFEST.json`. | `pipelines/offline/pipeline.py` |
| `rank.py` | `redstack rank --candidates data/raw/candidates.jsonl --out submission.csv` → runs R0…R9 → writes `submission.csv` + `run_report.json`. **This is the spec 10.3 single reproduce command**; `scripts/reproduce.sh` is a literal wrapper. | `pipelines/online/pipeline.py` |
| `validate.py` | `redstack validate --submission submission.csv` → runs `ValidationEngine` over a finished CSV (structural + Stage-4 reasoning checks). | `engines/validation.py` |

**Import restrictions:** may import `pipelines, config, observability`; must not contain scoring/ranking/integrity logic. **Testing location:** `tests/integration/cli/` (each verb end-to-end on the sample; exit-code assertions).

---

## §15. Test Repository Layout — `tests/`

Mirrors the source package tree under each category. Discipline: engines/features/domain are pure ⇒ unit-tested with fixtures and (for `semantic`) fake ports — no mocks. Mocking happens **only** at the port boundary, and even there behavioral *fakes* are preferred over `unittest.mock`.

| Category | Purpose | Ownership | Expected coverage |
|---|---|---|---|
| `unit/` | One engine / feature transform / domain model at a time; pure, fast, no IO. Sub-trees mirror `src/` (`unit/domain/`, `unit/features/`, `unit/engines/`, `unit/config/`, `unit/observability/`, `unit/adapters/`). | Each layer's owner | Highest count; every public type/transform. |
| `property/` | Hypothesis-driven invariants: ranking produces 100 unique ranks + non-increasing scores + id-ascending tie-breaks; score monotonicity; integrity idempotence; feature range/no-NaN; COW stage monotonicity. | Domain + Engines owners | All spec invariants + stage guards. |
| `contract/` | One shared, parametrized behavioral suite per port, run against **every** implementation — the real adapter *and* its fake — so fakes cannot drift from reality. | Ports owner | All seven ports × {adapter, fake}. |
| `golden/` | Snapshot: fixed candidate fixtures → exact `ScoreBreakdown` / CSV bytes / reasoning text / run-report `reproducible` block. | Modeling + Platform | The ranking-critical paths + serialization. |
| `integration/` | Full R0…R9 on the `sample_candidates` fixture; asserts the organizer `validate_submission.py` passes byte-for-byte; CLI verbs end-to-end; offline build → online rank round-trip. | Pipelines owner | The two pipelines + CLI. |
| `determinism/` | Same input twice ⇒ identical `submission.csv` bytes + identical run-report `reproducible` block; 1-thread vs N-thread identical ranking; restart/seed reproducibility. | Platform | Both pipelines, end-to-end. |
| `fixtures/` | The canonical fakes (`InMemoryArtifactStore, InMemoryVectorStore, StubEmbeddingModel, ListCandidateSource, CapturingSubmissionSink, CapturingRunReportSink, FixedEntropy`), `sample_candidates`, honeypot exemplars (spec §7), golden labels, a tiny onnx stub, malformed-line fixtures. | Shared (versioned) | — (supports all categories) |
| `conftest.py` | Shared pytest configuration, fixture registration, profile selection (`profiles/ci.yaml`). | Platform | — |

---

## §16. Dependency Graph

Inward-only (hexagonal). The allowed import arrows, top to bottom:

```text
                 cli
                  │
                  ▼
              pipelines ───────────────┐ (composition root ONLY)
            /     │      \              ▼
           ▼      ▼       ▼          adapters
       config  observ.  engines         │
          │       │     /   │  \        ▼
          │       │    ▼    ▼   ▼      ports
          │       │ features │   └──────┤
          │       │    │     │          │
          ▼       ▼    ▼     ▼          ▼
              ─────────  domain  ─────────
                     (numpy, pydantic, stdlib)
```

**Allowed imports (per layer):**

| Layer | May import |
|---|---|
| `domain` | stdlib, `pydantic`, `numpy` (vectors only) |
| `ports` | `domain`, stdlib `typing`, numpy typing |
| `features` | `domain`, `config.schema`, stdlib, numpy |
| `engines` | `domain`, `ports`, `features`, `config.schema` |
| `config.schema` | `domain`, `pydantic` |
| `config.loader` | `config.schema`, `pyyaml`, stdlib (IO) |
| `adapters` | `domain`, `ports`, `config.schema`, infrastructure libs |
| `pipelines` | all of the above (+ instantiates `adapters`) |
| `observability` | `domain`, stdlib |
| `cli` | `pipelines`, `config`, `observability` |

**Forbidden imports (the load-bearing negatives):** `domain` imports nothing from the project; `engines` never imports `adapters`/`pipelines`/`observability`-IO/`config.loader`/ML/network/each other; `adapters` never imports `engines`/`pipelines`; `config.loader` unreachable from `engines`/`features`/`domain`; `pipelines.online.*` never imports `sentence_transformers`/`sklearn`/`adapters.st_embedder`/networking; only `pipelines` imports `adapters`.

---

## §17. Import Boundary Rules

Codified as `import-linter` contracts in `pyproject.toml`, run in CI and pre-commit. A violated arrow is a **build break**, never a warning.

1. **Layered contract:** `domain < ports < features < engines < pipelines < cli`. No upward imports.
2. **Domain purity:** `domain` may import only stdlib + `pydantic` + `numpy`.
3. **Engine purity (forbidden):** `engines` may not import `adapters`, `config.loader`, `observability` IO, or any ML/network module; engines may not import each other.
4. **Adapter isolation (forbidden):** `adapters` may not import `engines` or `pipelines`.
5. **Online containment (forbidden):** `pipelines.online` (and its transitive imports) may not import `sentence_transformers`, `sklearn`, `adapters.st_embedder`, `requests`/`httpx`/`urllib3`, or `socket`. Defence-in-depth: `adapters/st_embedder.py` raises on import when an "online" environment marker is set.
6. **Composition-root exception (allowed):** only `pipelines` may import `adapters`.
7. **Config split:** `config.loader` (IO) importable only by `pipelines`/`cli`; `config.schema` (pure) importable anywhere.
8. **Observability/ports independence:** `ports` may not import `observability`; `observability` may not import `ports` (the `RunReport` Protocol is satisfied structurally).

These eight contracts are themselves covered by a CI job that asserts import-linter exits clean.

---

## §18. Build Order

Implement strictly inward-out; each phase is independently testable before the next begins. Dependencies are why the order is fixed, not preference.

| Phase | Package | Depends on | Why this order |
|---|---|---|---|
| **1** | `domain` | nothing | The stable core. Order within: `ids → enums → errors → provenance → source → jd → candidate/* slices → quality (+FeatureLayout type) → scoring → ranking → reasoning → validation → representation`. |
| **2** | `ports` | `domain` | Defines the seams engines/adapters target. Order: `_types → per-port Protocols + error classes`. |
| **3** | `config` | `domain` | `schema` (pure) before `loader`; `determinism` alongside. Engines need `config.schema`. |
| **4** | test fakes + `contract/` suites | `ports` | Every adapter and engine needs a target to test against before it exists. |
| **5** | `adapters` | `ports`, `config.schema` | Order: `artifact_store_fs → candidate_jsonl → vector_store_parquet → onnx_embedder → submission_csv → run_report_json → entropy → st_embedder` (offline, last). Merge each only when it passes its contract suite identically to its fake. |
| **6** | `features` | `domain`, `config.schema` | Order: `layout → registry → view → store → primitive extractors (parsing, normalize, geography, education) → competency (skills) + anti-stuffer (latents) → career → signals → honeypot`. |
| **7** | `engines` | `domain, ports, features, config.schema` | Pure leaf engines first (`behavioral, logistics, lexicon, cqv, scoring, ranking, reasoning, integrity, eligibility, validation`), then the port-bound `semantic`. |
| **8** | `pipelines/offline` | engines + adapters + config | Build context/runner/registry/graph, then O0→O18 in DAG order (O0→O1→O2; O13a unblocks O7/O14; O4/O5/O6; O14; O3/O12/O7; O8→O9→O10→O11→O15→O16; O17→O18). |
| **9** | `pipelines/online` | engines + adapters + config | R0…R9 composition root + pure stage functions. |
| **10** | `observability` | `domain` | Timing/budget/run-report; wired by pipelines. |
| **11** | `cli` | pipelines, config, observability | The thin entrypoints (`build, rank, validate`) + `scripts/reproduce.sh`. |

Wire the execution graph (composition roots) **last** in phases 8–9 — the only place ports meet engines.

---

## §19. Ownership Matrix

| Module / file | Owner layer | Consumed by | Mutability |
|---|---|---|---|
| `domain/*` | Domain | ports, features, engines, adapters, pipelines | Frozen contracts; changes ripple widest — change least often |
| `ports/*` | Ports | engines (abstractions), adapters (impl), pipelines (bind) | Frozen Protocol surfaces |
| `features/layout.py`, `features/registry.py` | Features | engines (`cqv`, `scoring`), O8/O9 | `layout_version`-gated; major bump on order change |
| `features/{parsing,normalize,career,skills,education,geography,signals,latents,honeypot}.py` | Features | engines, pipelines | Versioned per feature; value-change ⇒ minor |
| `engines/*` (11 modules) | Engines | pipelines only | Behavior-stable; pure |
| `config/schema.py` | Config | everywhere | Additive; `extra="forbid"` |
| `config/loader.py`, `config/determinism.py` | Config | pipelines, cli | Low churn |
| `adapters/*` (8 modules) | Adapters | pipelines (instantiate) | Highest churn; swappable behind ports |
| `pipelines/**` | Application | cli | Composition roots; wiring only |
| `observability/*` | Observability | pipelines, cli | Cross-cutting |
| `cli/*` | CLI | end users / `scripts/` | Thin; argv→pipeline |
| `configs/**` | Modeling/Platform | loader, offline pipeline | Human-authored intent; per-JD re-authoring |
| `artifacts/**` | Offline pipeline | online pipeline, engines | **Write-once per build; immutable; gitignored** |
| `data/**` | Operator/Team | both pipelines | Read-only raw fact; gitignored |
| `tests/**` | Each layer's owner | CI | Co-evolves with code |

---

## §20. Implementation Readiness Checklist

- [ ] **Repository complete** — every directory in §1 exists with its `__init__.py`; `src/` layout installs editable (`uv pip install -e .`).
- [ ] **Architecture mapped** — every file in the frozen layer docs has exactly one home here; the three reconciliations (offline O0–O18, engine logical→physical, FeatureLayout location) are recorded and respected.
- [ ] **No undefined modules** — every module in §6–§14 has purpose, owner, dependencies, import restrictions, implementation notes, and a testing location.
- [ ] **No undefined files** — every artifact (§5), config (§3), and data path (§4) is accounted for with producer/consumer/validation.
- [ ] **All ports implemented** — each of the seven ports (§7) has a real adapter (§10) and a fake (§15), both passing the shared contract suite.
- [ ] **All engines placed** — the eleven engine modules (§9) exist; the fourteen logical engines map onto them (+ adapters/observability) with no orphan concern.
- [ ] **All pipelines placed** — offline `O0–O18` and online `R0–R9` each have a stage module and a composition root (§12); ports appear online only at R0/R1/R3/R8/R9.
- [ ] **Test locations defined** — every source module names its test category (§15); `contract/`, `determinism/`, and the budget-guard integration test exist.
- [ ] **Boundaries enforceable** — the eight import-linter contracts (§17) are written into `pyproject.toml` and gate CI + pre-commit.
- [ ] **Determinism owned** — `config/determinism.py` is the sole home for seeds + thread pins; no `datetime.now`/online RNG anywhere reachable from `pipelines.online`.
- [ ] **Reproduce path wired** — `redstack rank` ↔ `scripts/reproduce.sh` ↔ `submission_metadata.yaml:reproduce_command` agree on one command.

**Conclusion.** With this layout, the frozen architecture is fully translated into concrete files: every module, config, artifact, and test has a single, unambiguous home and a defined contract. A developer can clone the repository, read this document, and begin implementing any phase in §18 immediately — without making a single repository-level architectural decision.
