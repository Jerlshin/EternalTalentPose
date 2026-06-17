# REDSTACK v1.1 — Offline Pipeline Specification

**Scope:** `src/redstack/pipelines/offline/` only. Architecture, Domain, Ports, Engine, and Feature layers are frozen; this document does not alter them. No implementation code — architecture and execution design detailed enough that implementation can begin immediately.

**Goal:** transform the 100,000 raw candidates + the JD into production-ready, hash-pinned ranking artifacts, so the online pipeline can satisfy **≤5 min wall-clock, ≤16 GB RAM, CPU-only, no network**. Every expensive operation — embedding the pool, clustering, weight search, calibration — lives here.

**Reconciliation with the freeze.** The architecture doc sketched a coarse `O1–O10`. This document refines that sketch into `O0–O18`; it produces the **same frozen artifact contracts** every other layer consumes (`MANIFEST.json`, `model/encoder.onnx`, `embeddings/candidate_vectors.parquet`, `embeddings/anchor_vectors.npy`, `lexicon/lexicon.compiled.json`, `archetypes/centroids.npy`, `calibration/integrity_thresholds.json`, `weights/scoring_weights.locked.yaml`) and adds the artifacts the finer stages require. No frozen interface changes.

**Offline purity rules (differ deliberately from online):**
- Offline **may** use ML runtimes (`sentence-transformers`, `scikit-learn`), seeded RNG (`OfflineEntropy`), and network *at authoring time* (e.g., to draft anchors/lexicon with AI assistance — declared per the spec's AI-tools policy).
- But **committed artifacts are static, reviewed, deterministic, and hash-pinned.** The deterministic portions are re-derivable from data + code; O18 proves it. Nothing opaque (e.g., a live LLM call) is baked into an online-consumed artifact.
- **Reproducibility-first.** Every stage is seeded, content-addressed, and resumable; the build is reproducible from `data/raw/candidates.jsonl` + `job_description.md` + code + config.

---

# PART 1 — OFFLINE PIPELINE OVERVIEW

### `OfflinePipeline`
- **Responsibilities:** declare the ordered stage set (O0–O18) and their dependency DAG; own no IO itself (delegates to ports/adapters); pure orchestration over injected stage callables.
- **Lifecycle:** `plan()` (topo-sort the DAG, resolve which stages are stale) → `run()` (execute stale stages in dependency order) → `finalize()` (package + validate).
- **Inputs:** `OfflinePipelineContext`.
- **Outputs:** a fully populated `artifacts/` tree + `OfflinePipelineManifest` + `OfflinePipelineReport`.
- **Failure modes:** stage failure (fail-fast, quarantine partial output); DAG cycle (config error, caught at `plan()`); missing input (raises).
- **Reproducibility:** stages are pure-given-ports; the DAG and stage versions are fixed; identical inputs ⇒ identical artifacts (deterministic stages) / ε-stable artifacts (embedding stages).

### `OfflinePipelineContext`
- **Responsibilities:** the resolved environment for a build — config, bound ports (`CandidateSource`, `EmbeddingModel(st)`, `ArtifactStore`, `OfflineEntropy`), `seed`, `as_of`, output roots, the `FeatureRegistry`/`FeatureLayout` constant.
- **Lifecycle:** built once by the offline composition root; immutable thereafter.
- **Inputs:** `configs/runtime/offline.yaml`, `configs/*` (anchors/gates/lexicon seed), `data/raw`, `data/golden`.
- **Outputs:** none (a carrier).
- **Failure modes:** invalid config (`pydantic` validation); missing data path (raises).
- **Reproducibility:** records `config_hash`, `seed`, `as_of`, `code_version` for the report.

### `OfflinePipelineRunner`
- **Responsibilities:** execute the plan with **resume** (skip up-to-date stages), **checkpointing** (persist each stage's output + a `StageReceipt`), per-stage timing/metrics, and failure recovery.
- **Lifecycle:** `for stage in topo_order: if stale(stage): run + checkpoint else skip`.
- **Inputs:** the plan + context.
- **Outputs:** `StageReceipt`s, per-stage metrics → `OfflinePipelineReport`.
- **Failure modes:** stage raises → quarantine its partial output, mark downstream stale, abort (or `--continue-on-error` for non-critical stages); checkpoint corruption → recompute.
- **Reproducibility:** stage staleness keyed by `(input_hashes, stage_version)`; a clean rebuild and an incremental rebuild produce identical deterministic artifacts.

### `OfflinePipelineManifest`
- **Responsibilities:** the registry of produced artifacts with sha256, bytes, `schema_version`, producing stage, and `layout_version`/`model_id` where relevant; carries `manifest_sha256` (self-integrity). **This is the frozen `MANIFEST.json`** the online `ArtifactStorePort` verifies (Ports §8).
- **Inputs:** all artifact paths + stage receipts.
- **Outputs:** `artifacts/MANIFEST.json`.
- **Failure modes:** missing required artifact (raises at packaging); hash drift (caught at O18).
- **Reproducibility:** the manifest *is* the reproducibility contract for online.

### `OfflinePipelineReport`
- **Responsibilities:** the build audit — per-stage timings, metrics (census stats, honeypot counts, calibration NDCG, cluster quality), seeds, hashes, reproducibility verdict.
- **Outputs:** `artifacts/offline_report.json` (mirrors the port-owned `RunReport` shape; `reproducible` vs `audit` split).
- **Reproducibility:** the `reproducible` block is deterministic across rebuilds.

### `OfflineArtifactRegistry`
- **Responsibilities:** the typed catalog of *expected* artifacts — keys, schemas, owners, versions, lineage, validators (Part 10). The runner validates each produced artifact against it before manifesting.
- **Failure modes:** schema/validator failure → stage fails.
- **Reproducibility:** registry is a frozen, versioned code constant.

### `OfflineExecutionGraph`
- **Responsibilities:** the dependency DAG over O0–O18 + the parallelization plan (Part 11).
- **Reproducibility:** fixed topology; deterministic topo order (ties by stage id).

---

# PART 2 — OFFLINE STAGES (O0–O18)

Each: **purpose · inputs · outputs · dependencies · algorithms · complexity · artifacts.** `N = 100K`.

| Stage | Purpose |
|---|---|
| **O0 Dataset Census** | Profile the pool before anything else (Part 3). In: candidate stream. Out: `DatasetProfile`, distributions, outliers. Deps: —. Alg: streaming frequency/quantile aggregation (counters, t-digest). Cx: O(N). Artifacts: `dataset_profile.json`. |
| **O1 Candidate Normalization** | Canonicalize text/dates/skill tokens/companies (Feature Layer §Normalization). In: raw stream. Out: normalized records + canonical maps. Deps: O0. Alg: deterministic normalization (NFC, lexicon canonical map seed, date parse). Cx: O(N). Artifacts: `canonical_maps.json`. |
| **O2 Candidate Validation** | Validate against `candidate_schema.json` → `RawCandidate` (Ingestion engine, offline mode). In: normalized records. Out: validated set + reject log. Deps: O1. Alg: pydantic validation; structural-not-semantic (contradictions preserved for O3). Cx: O(N). Artifacts: `validation_report.json`. |
| **O3 Honeypot Discovery** | Discover anomaly patterns, calibrate detector thresholds (Part 4). In: validated set, census. Out: honeypot catalog + rules + thresholds. Deps: O0, O2. Alg: rule firing + distribution-based thresholding + severity classification. Cx: O(N·rules). Artifacts: `honeypot_catalog.json`, `integrity_rules.json`, `integrity_thresholds.json`. |
| **O4 Lexicon Discovery** | Mine domain terminology families from descriptions (Part 5). In: validated descriptions, seed lexicon. Out: lexicon + concept dictionary + graphs. Deps: O2. Alg: tokenize → TF-IDF/c-TF-IDF per concept → phrase mining (PMI/n-grams) → graph build. Cx: O(N·tokens). Artifacts: `lexicon.compiled.json`, `concepts.json`, `term_graph.json`, `phrase_graph.json`. |
| **O5 Semantic Vocabulary Expansion** | Expand the lexicon with embedding-nearest terms (catch synonyms stuffers won't use). In: lexicon, embeddings. Out: expanded `ConceptDictionary`. Deps: O4, O13a. Alg: embed seed terms (st), nearest-neighbour expansion with manual review gate. Cx: O(terms·dim). Artifacts: `concepts.json` (expanded). |
| **O6 JD Concept Extraction** | Author the JD's positive/negative **anchors** (the `jd.*` latents) as concept texts (Feature Layer Part 2). In: JD doc, concept dictionary. Out: anchor concept set + `JobDescriptionSpec` data. Deps: O5. Alg: concept authoring (human + AI-assisted, reviewed), polarity tagging. Cx: O(concepts). Artifacts: `jd_concepts.json`, `gates/eligibility_rules.yaml` (authored). |
| **O7 Archetype Discovery** | Cluster candidates into the 10 archetypes (Part 6). In: candidate embeddings. Out: catalog + fingerprints + centroids. Deps: O13a. Alg: KMeans (seeded via `OfflineEntropy`), silhouette/k-selection, fingerprinting. Cx: O(N·k·iter·dim). Artifacts: `archetypes.json`, `centroids.npy`. |
| **O8 Candidate Labeling Workspace** | Human-in-the-loop gold labeling (Part 7). In: features, evidence, archetypes. Out: gold/review/calibration datasets. Deps: O7, O14. Alg: stratified + active-learning sampling; reviewer UI; agreement. Cx: human-bounded (~hundreds–low-thousands). Artifacts: `gold_labels.json`, `review_queue.json`, `calibration_split.json`. |
| **O9 Weight Calibration** | Calibrate `ScoringWeights` to maximize the composite on gold (Part 8). In: feature snapshot, gold labels, archetypes. Out: scoring weights + report. Deps: O8, O14. Alg: seeded constrained search / regularized fit with tier priors; cross-validated on the calibration split. Cx: O(folds·search·N_gold). Artifacts: `scoring_weights.locked.yaml`, `calibration_report.json`. |
| **O10 Feature Importance Analysis** | Quantify per-feature contribution for reasoning + pruning. In: feature snapshot, weights, gold. Out: importance scores. Deps: O9. Alg: permutation importance + ablation NDCG deltas. Cx: O(features·folds). Artifacts: `feature_importance.json`. |
| **O11 Behavioral Calibration** | Calibrate behavioral multiplier bounds/curves (Feature Layer Part 4). In: behavioral features, gold, census. Out: behavioral weights/bounds. Deps: O14, (O8 optional). Alg: fit multiplier curves; bound to avoid behavioral domination of relevance. Cx: O(N). Artifacts: `behavioral_weights.json`. |
| **O12 Risk Calibration** | Calibrate honeypot threshold + risk/confidence weights (Feature Layer Part 5/§Risk). In: honeypot detectors, census, gold (honeypot exemplars). Out: risk weights + thresholds. Deps: O3, O14. Alg: threshold selection trading honeypot recall vs false-positive loss; confidence-shrink params. Cx: O(N). Artifacts: `risk_thresholds.json`, `risk_weights.json` (merged into `integrity_thresholds.json`). |
| **O13 Embedding Generation** | Produce all vectors + onnx export + vector stores (Part 9). Sub-stages: **O13a** candidate vectors (deps O1), **O13b** concept/anchor vectors (deps O6), **O13c** career-history vectors (deps O1, optional). In: composed docs. Out: vectors + manifest + onnx. Alg: st batched encode, L2-normalize, parquet/npy. Cx: O(N·dim) — the compute-dominant stage (minutes). Artifacts: `candidate_vectors.parquet`, `anchor_vectors.npy`, `career_vectors.parquet`?, `model/encoder.onnx`, `embedding_manifest.json`. |
| **O14 Candidate Representation Construction** | Build the canonical **feature snapshot** (the `(N,D)` CQV + group confidences) used by O8/O9/O10 and as the online correctness oracle (O18). In: validated set, lexicon, candidate+anchor vectors, behavioral signals, `FeatureLayout`. Out: feature snapshot. Deps: O4, O13a, O13b. Alg: run the pure Feature extractors over the pool (columnar). Cx: O(N·D). Artifacts: `feature_snapshot.parquet`, `feature_manifest.json`. |
| **O15 Ranking Calibration** | Calibrate the score normalization / monotone calibration curve; confirm tie-break behavior. In: scored gold + held-out. Out: calibration params. Deps: O9, O10, O11, O12. Alg: isotonic/monotone calibration; tie-break verification against `validate_submission.py`. Cx: O(N_gold·log). Artifacts: `ranking_calibration.json`. |
| **O16 Reasoning Template Construction** | Build the **evidence-slot** reasoning templates (deterministic, LLM-free at online time) from gold reference reasonings + feature-evidence patterns (Feature Layer Part 8). In: gold reasonings, feature importance, evidence schema. Out: clause templates + slot vocabularies. Deps: O8, O10. Alg: pattern extraction → evidence-slot templates per polarity/rank-band; **no online LLM**. Cx: O(gold). Artifacts: `reasoning_templates.json`. |
| **O17 Artifact Packaging** | Hash + register every artifact; write the manifest. In: all artifacts. Out: `MANIFEST.json`. Deps: all. Alg: streaming sha256, schema validation vs `OfflineArtifactRegistry`, manifest self-hash. Cx: O(artifact bytes). Artifacts: `MANIFEST.json`. |
| **O18 Reproducibility Validation** | Prove the build is reproducible and online-consumable. In: packaged artifacts. Out: repro verdict. Deps: O17. Alg: (1) reload via `ArtifactStorePort.verify_all`; (2) **online-vs-offline feature parity** (recompute features for a sample, diff vs snapshot); (3) deterministic dry-run ranking on the golden set; (4) re-run deterministic stages and diff hashes. Cx: O(sample). Artifacts: `reproducibility_report.json`. |

---

# PART 3 — DATASET CENSUS (O0)

- **`DatasetProfile`:** top-level counts (N, fields present), per-field coverage, schema-drift findings (does data match `candidate_schema.json`).
- **`DistributionCatalog`:** frequency tables + quantiles for: **skill frequencies** (and skill→proficiency/endorsement/duration joint dist), **title frequencies**, **company frequencies** (+ product/consulting tagging coverage), **industry**, **location/country**, **education** (tier, field, degree, year spans), **experience** (`years_of_experience` histogram, derived-vs-stated gap), **behavioral** (each of the 23 signals' distribution, sentinel rates for `−1`/`{}`, `last_active` recency vs `as_of`).
- **`OutlierCatalog`:** tail/anomaly candidates per field (impossible dates, inverted salary, expert-zero-duration, extreme tenure) — the raw material O3 turns into rules.
- **`CandidateStatistics`:** per-candidate quick stats (counts, spans) cached for downstream reuse.
- **Why first:** every threshold (honeypot, behavioral, risk) is set *relative to the observed distribution*, not hard-coded. The census makes calibration data-driven and the resulting thresholds defensible at Stage 5.

---

# PART 4 — HONEYPOT DISCOVERY (O3)

Spec: ~80 honeypots forced to tier 0; **>10% in top 100 ⇒ disqualification**. Strategy mirrors Feature Layer Part 5.

- **Discover anomalies:** run the structural impossibility detectors (timeline, skill-time, overlap, title-seniority, education-career, salary, experience-inflation, keyword-stuffing, behavioral-inconsistency, signal-impossibility, identity) over the pool using O0 distributions to define "impossible vs merely unusual".
- **Estimate frequency:** measure each anomaly's prevalence; expect a small categorical-impossibility cohort (~80) distinct from the large "noisy synthetic" mass (mismatched titles/descriptions, common inverted salary).
- **Classify severity:** `HARD` (categorically impossible: end<start, current+end_date, tenure≫experience, expert@0-months en masse) vs `SOFT` (unusual but possible).
- **Produce rules + thresholds:** `IntegrityRuleSet` (the codified detectors) + `RiskThresholds` (per-detector cut points + the composite `honeypot_threshold`), calibrated to **maximize honeypot recall while bounding real-candidate loss**. A hard gate requires **≥2 HARD impossibilities** or composite ≥ threshold; lone soft anomalies only dampen.
- **Outputs:** `HoneypotCatalog` (discovered exemplars + suspected ids + evidence), `IntegrityRuleSet` (→ `integrity_rules.json`), `RiskThresholds` (→ `integrity_thresholds.json`, consumed by `CandidateConsistencyEngine`/`CandidateRiskEngine`).
- **False-positive mitigation:** thresholds set on the census/outlier distributions; salary inversion is soft (common in pool); validated in O18's dry-run honeypot-rate check.

---

# PART 5 — LEXICON DISCOVERY (O4) + EXPANSION (O5)

Automatically discover concept terminology from descriptions so competency features depend on *meaning*, not the candidate's exact keyword choice (anti-stuffing + Tier-5 recall).

- **Concepts mined:** retrieval, ranking, recommendation, matching, search, evaluation, production-ML, MLOps (the JD's "absolutely need" families).
- **Algorithm (O4):** tokenize normalized descriptions → per-concept c-TF-IDF seeded from the human seed lexicon → phrase mining (PMI bigrams/trigrams) → build a `TermGraph` (term co-occurrence) and `PhraseGraph` (multi-word concept phrases).
- **Algorithm (O5):** embed seed + mined terms (st), expand each concept by embedding-nearest neighbours, **human-reviewed** to avoid drift; produces synonym coverage a stuffer wouldn't anticipate.
- **Outputs:** `LexiconCatalog` (concept → canonical terms + weights → `lexicon.compiled.json`), `ConceptDictionary` (`concepts.json`: concept → expanded vocab + anchor text), `TermGraph`, `PhraseGraph`.
- **Consumed by:** Feature Layer competency groups (8–16) for `*.in_career` and trust scoring; O6 anchor authoring.
- **Determinism:** TF-IDF/PMI deterministic; expansion is seeded + reviewed; committed vocab is static.

---

# PART 6 — ARCHETYPE DISCOVERY (O7)

- **Algorithm:** KMeans (seeded via `OfflineEntropy.numpy_generator("kmeans")`) over candidate embeddings; k chosen by silhouette/elbow targeting the **10 expected archetypes** — retrieval / ranking / recsys / data / ML / researcher / consultant / keyword-stuffer / product / startup engineer. Fingerprint each cluster from its dominant features (top skills, product density, behavioral profile).
- **Outputs:**
  - `ArchetypeCatalog` (`archetypes.json`): id → label, size, fingerprint, target-flag (is this a JD-desired archetype).
  - `ArchetypeFingerprint`: per-cluster feature signature (centroid in feature space + top discriminating features) — used for reasoning ("matches the product-ML-engineer archetype").
  - `ArchetypeCentroid` (`centroids.npy`): embedding-space centroids, consumed online for nearest-centroid assignment (`CandidateRetrievalEngine`, ties by `ArchetypeId`).
- **Use:** target archetypes (retrieval/ranking/recsys/product/startup) boost; negative archetypes (consultant/keyword-stuffer/researcher) flag for Fit/Risk. Labeling (O8) stratifies by archetype.
- **Determinism:** seeded init; fixed k; centroid order fixed; reproducible.

---

# PART 7 — LABELING WORKSPACE (O8)

Human-in-the-loop review producing the gold labels that calibration trusts — this is the "real engineering" that survives Stage 4/5.

- **Capabilities:** candidate review (raw + composed view); evidence review (the `FeatureCell` → `EvidenceRef` → raw-field chain); feature inspection (values + confidence + tier); archetype inspection; **fit-tier assignment** (`RelevanceTier 0–4`; honeypots → 0); reference **reason generation** (a human-written reasoning per labeled candidate, seeding O16 templates).
- **Sampling strategy (`ReviewDataset`):** stratified by archetype + honeypot-suspect + borderline (uncertainty), with **active learning** — label the most decision-relevant candidates first (near the top-100 cut, near eligibility boundaries) given limited human time (~hundreds to low-thousands of labels).
- **Quality:** multi-reviewer overlap on a subset → inter-rater agreement; disagreements adjudicated; reviewer + timestamp recorded.
- **Outputs:** `GoldLabelDataset` (candidate_id → tier + reasoning + reviewer + evidence), `ReviewDataset` (queue + sampling provenance), `CalibrationDataset` (the train/validation split for O9/O15, stratified, leakage-free).
- **Determinism:** labels are human facts (fixed once committed); splits are seeded.
- **Note:** the workspace is an offline tool (notebook/Streamlit), not part of the online path; its only online-visible outputs are calibrated weights, thresholds, and templates derived from the labels.

---

# PART 8 — WEIGHT CALIBRATION (O9, +O11, +O12)

- **Inputs:** the O14 feature snapshot (so calibration uses the *exact* features the online path computes), `GoldLabelDataset`, archetypes, behavioral signals.
- **Method:** optimize per-`ScoreComponent` `ScoringWeights` to maximize the challenge composite (`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`) on the calibration split, **cross-validated**, with **tier priors** from Feature Layer Part 10 as regularization and Tier-A floors. Output must be **linear weights** consistent with the frozen `ScoringEngine` (weighted CQV combination). Search is seeded (`OfflineEntropy`).
- **O11 Behavioral Calibration:** fit the behavioral **multiplier** curves/bounds → `behavioral_weights.json`; bounds prevent behavioral signals from dominating relevance (JD: they *modulate*, never define).
- **O12 Risk Calibration:** set the honeypot composite threshold + confidence-shrink params trading honeypot recall vs false-positive loss → merged into `integrity_thresholds.json`/`risk_weights.json`.
- **Outputs:** `ScoringWeights` (`scoring_weights.locked.yaml`, carrying `layout_version`), `BehavioralWeights`, `RiskWeights`, `CalibrationReport` (cross-val NDCG, weight stability across folds, ablation deltas, overfitting checks given the tiny gold set).
- **Determinism:** seeded; the locked weights are the frozen artifact `CandidateScoringEngine` binds to; `layout_version` must match `FeatureLayout` + `feature_manifest`.

---

# PART 9 — EMBEDDING GENERATION (O13)

| Embedding | Source doc | Stored as | Online use |
|---|---|---|---|
| **JD embeddings** | the JD document | (folded into anchors) | reference |
| **Concept/anchor embeddings** | O6 concept texts (positive + negative `jd.*`) | `anchor_vectors.npy` | R3 anchor similarity |
| **Candidate embeddings** | composed candidate doc (O1 recipe) | `candidate_vectors.parquet` keyed by `CandidateId` | R3 lookup |
| **Career-history embeddings** | per-role descriptions (optional, finer matching) | `career_vectors.parquet` | optional R3 detail |
| **Archetype embeddings** | O7 centroids | `centroids.npy` | R3 archetype assignment |

- **Similarity indexes / vector stores:** the candidate store (id→row mmap), anchor matrix, centroid matrix — consumed by `SemanticVectorStorePort`.
- **onnx export:** the same base model exported to `model/encoder.onnx` for the **online fallback** (`EmbeddingModelPort` onnx adapter). Cross-runtime cosine within ε (Ports §9).
- **Outputs:** `EmbeddingManifest` (`embedding_manifest.json`: `model_id`, `dim`, normalization, **doc-composition recipe** — must match O1/online normalization byte-for-byte, pooling), `EmbeddingRegistry` (which embedding sets exist + keys), `EmbeddingVersion` (semver tying vectors to model+recipe).
- **Determinism:** within-runtime bitwise; the doc-composition recipe is pinned so online fallback lands in the same space; the dominant compute cost (minutes for N·dim) lives entirely here.

---

# PART 10 — ARTIFACT REGISTRY

Each artifact: **schema · ownership · versioning · lineage · validation.**

| Artifact | Schema (key fields) | Owner stage | Version | Lineage (depends on) | Validation |
|---|---|---|---|---|---|
| `dataset_profile.json` | counts, distributions, outliers | O0 | profile-v | raw | coverage sane, N==100K |
| `canonical_maps.json` | token/company/skill canonical maps | O1 | norm-v | O0 | maps total, no empties |
| `validation_report.json` | accept/reject counts, reject log | O2 | — | O1 | reject rate within bound |
| `integrity_rules.json` | rule codes + predicates | O3 | rules-v | O0,O2 | every code in `IntegrityFlag` |
| `integrity_thresholds.json` | per-detector + composite thresholds, risk weights | O3,O12 | thr-v | O0,O3,O14 | thresholds in range; recall check |
| `honeypot_catalog.json` | exemplars + suspect ids | O3 | — | O2 | suspects ⊆ pool |
| `lexicon.compiled.json` | concept→terms+weights | O4 | lex-v | O2 | concepts cover JD families |
| `concepts.json` | concept→expanded vocab+anchor text | O5,O6 | concept-v | O4,O5 | each concept has anchor text |
| `term_graph.json`/`phrase_graph.json` | graphs | O4 | lex-v | O2 | acyclic-where-expected |
| `jd_concepts.json` | anchors + polarity | O6 | jd-v | O5 | polarity tagged; ⊆ anchor set |
| `gates/eligibility_rules.yaml` | predicates per `EligibilityCode` | O6 | gate-v | O6 | every code valid |
| `centroids.npy` | (k,dim) float32 | O7 | arch-v | O13a | dim==embedding.dim; k fixed |
| `archetypes.json` | id→label,fingerprint,target | O7 | arch-v | O7 | ids contiguous |
| `gold_labels.json` | id→tier,reasoning,reviewer | O8 | gold-v | O14,O7 | tiers∈0..4; ids⊆pool |
| `calibration_split.json` | train/val ids | O8 | gold-v | O8 | disjoint, no leakage |
| `scoring_weights.locked.yaml` | component→weight, layout_version | O9 | weights-v | O8,O14 | components==`ScoreComponent`; layout match |
| `behavioral_weights.json` | multiplier curves/bounds | O11 | bhv-v | O14 | bounds ordered |
| `feature_importance.json` | feature→importance | O10 | imp-v | O9,O14 | features⊆layout |
| `embedding_manifest.json` | model_id,dim,recipe,pooling | O13 | emb-v | O1,O6,O7 | recipe matches online |
| `candidate_vectors.parquet` | id,vector(dim) | O13a | emb-v | O1 | unit-norm; id unique |
| `anchor_vectors.npy` | (a,dim) | O13b | emb-v | O6 | keys⊆jd_concepts |
| `model/encoder.onnx` | onnx graph | O13 | emb-v | O13 | dim/model_id match |
| `feature_snapshot.parquet` | id, (D) values, (G) confidence | O14 | layout-v | O4,O13 | no NaN; D==layout dim |
| `feature_manifest.json` | layout_version, feature list | O14 | layout-v | O14 | matches `FeatureLayout` |
| `ranking_calibration.json` | calibration curve, tie-break check | O15 | rank-v | O9–O12 | monotone; validator-pass |
| `reasoning_templates.json` | clause templates, slots | O16 | reason-v | O8,O10 | every slot has evidence kind |
| `MANIFEST.json` | per-artifact sha256+meta, self-hash | O17 | manifest-v | all | self-hash valid; required keys present; cross-coherence (Ports §8) |
| `reproducibility_report.json` | parity/dry-run/determinism verdict | O18 | — | O17 | all checks pass |

**Versioning rule:** value-changing edit ⇒ minor; layout/order/dim change ⇒ major. `layout_version` is shared across `FeatureLayout`/`feature_manifest`/`scoring_weights` and must agree (online raises `ArtifactContractError` otherwise).

---

# PART 11 — EXECUTION GRAPH

**Dependency DAG (topo order):**
```
O0 ─▶ O1 ─▶ O2 ─┬─▶ O3 ──────────────┐
                └─▶ O4 ─▶ O5 ─▶ O6 ──┤
O1 ─▶ O13a ─┬─▶ O7 ──────────────────┤
O6 ─▶ O13b ─┘                        │
{O4,O13a,O13b} ─▶ O14 ───────────────┼─▶ O8 ─▶ O9 ─▶ O10 ─┐
                                     ├─▶ O11             │
                              {O3,O14}─▶ O12             │
                       {O9,O10,O11,O12} ─▶ O15 ──────────┤
                              {O8,O10} ─▶ O16 ───────────┤
                                                all ─▶ O17 ─▶ O18
```

- **Parallelization:** after O2, `O3 ∥ O4`; after embeddings, `O7 ∥ O14`; `O11 ∥ O12 ∥ O10` post-O9; O13 internally batched (the heavy stage). The runner schedules independent branches concurrently; merges are deterministic (id order, seeded).
- **Incremental rebuilds:** each stage's staleness = `hash(input_artifacts + stage_version + config_slice)`. A clean and an incremental build yield identical deterministic artifacts.
- **Cache invalidation:** lineage-driven — changing O1 normalization invalidates O13/O14 and everything downstream; the runner recomputes the transitive closure only.
- **Artifact invalidation:** a `schema_version`/`layout_version` bump invalidates the artifact and its dependents; partial/failed outputs are quarantined (never manifested).
- **Failure recovery:** stage-level checkpoints + `StageReceipt`s; on failure, quarantine, mark downstream stale, fail-fast (critical) or continue (non-critical, flagged). Re-run resumes from the last good checkpoint.
- **Resume capability:** `runner.run()` skips up-to-date stages by checkpoint hash; a re-invoked build does minimal work.

---

# PART 12 — OFFLINE OUTPUT CONTRACT (who consumes what)

| Consumer | Artifacts consumed |
|---|---|
| **Feature Layer** | `lexicon.compiled.json`, `concepts.json`, `term/phrase_graph.json`, `feature_manifest.json`, `feature_importance.json`, (candidate/anchor vectors for semantic feature values), `behavioral_weights.json` |
| **CandidateFitEngine** | `jd_concepts.json` + `anchor_vectors.npy` (latents), `gates/eligibility_rules.yaml`, `lexicon.compiled.json`, `archetypes.json` |
| **CandidateRiskEngine** | `integrity_rules.json`, `integrity_thresholds.json`, `risk_weights.json`, `honeypot_catalog.json` |
| **CandidateScoringEngine** | `scoring_weights.locked.yaml`, `behavioral_weights.json`, `ranking_calibration.json`, `feature_manifest.json` |
| **CandidateRankingEngine** | `ranking_calibration.json` (tie-break/monotone confirm) — scores only otherwise |
| **CandidateReasoningEngine** | `reasoning_templates.json`, `feature_importance.json`, `archetypes.json` (fingerprints) |
| **Online Pipeline (R0)** | `MANIFEST.json` (+ everything it references), `model/encoder.onnx`, `candidate_vectors.parquet`, `anchor_vectors.npy`, `centroids.npy` |

**The single online contract:** at R0 the online `ArtifactStorePort` loads `MANIFEST.json`, self-verifies it, verifies each referenced artifact's sha256, and asserts cross-artifact coherence (`layout_version` agreement across `feature_manifest`/`scoring_weights`; `embedding.dim` across vectors/anchors/centroids/onnx; anchor set ⊆ `jd_concepts`). Any failure aborts the run — no degraded mode (Ports §8).

---

## Build order for offline implementation

1. `OfflinePipelineContext` + `OfflineArtifactRegistry` + `OfflineExecutionGraph` + the runner with resume/checkpointing.
2. O0 → O1 → O2 (deterministic, the substrate everything else trusts).
3. O13a candidate embeddings (unblocks O7, O14) → O4/O5/O6 (lexicon → concepts → anchors) → O13b/O13c.
4. O14 feature snapshot (binds the `FeatureLayout`; the calibration substrate).
5. O3 + O12 (honeypot/risk calibration) and O7 (archetypes).
6. O8 labeling workspace → O9 weight calibration → O10 importance → O11 behavioral → O15 ranking calibration → O16 reasoning templates.
7. O17 packaging → O18 reproducibility validation (the gate that proves the artifact set is online-consumable and deterministic).

Each stage merged only when (a) its artifact passes the registry validator, (b) it is reproducible across two clean runs (deterministic stages) or ε-stable (embedding stages), and (c) O18's online-vs-offline feature parity holds on the `sample_candidates` fixture.
