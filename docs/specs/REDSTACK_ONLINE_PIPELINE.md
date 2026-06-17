# REDSTACK_ONLINE_PIPELINE.md

REDSTACK v1.1 — Online (Runtime Inference) Pipeline Specification

**Scope:** `src/redstack/pipelines/online/` only. Architecture, Domain, Ports, Engine, Feature, and Offline layers are frozen; this document changes none of them. It defines the runtime pipeline that executes inside the competition constraints and emits the final submission CSV. No code, no pseudocode.

**Cross-document reconciliation (Part 8 "no calibration layer" vs frozen offline `ranking_calibration.json`):** "No calibration layer" is interpreted as *no runtime calibration or learning*. R5 applies the **locked** `scoring_weights.locked.yaml` deterministically and R6 ranks on the **raw** `final_score`. The offline monotone curve in `ranking_calibration.json` is order-preserving and therefore ranking-neutral; online it is **disabled by default** and, if ever enabled, applies only as a presentation transform to the emitted `score` column — it can never change ranks. No silver-label / weak-supervision component runs online.

---

# PART 1 — ONLINE PIPELINE OVERVIEW

**Mission.** Given the released `candidates.jsonl` (100,000 candidates) and the offline artifact set, produce a validator-compliant top-100 `submission.csv` (`candidate_id,rank,score,reasoning`) plus a `run_report.json`, in a single, deterministic, reproducible run. This pipeline is the literal Stage-3 reproduce command.

**Runtime constraints (competition ceiling / internal target).** ≤ 5 min wall-clock ceiling; **internal target ≤ 150 s**. CPU only. No GPU. No network — no hosted LLM, no remote fetch, no telemetry egress.

**Memory constraints.** ≤ 16 GB ceiling; **internal target ≤ 4 GB peak RSS**. Achieved by streaming ingestion, memory-mapped vectors, a single columnar `(N, D)` feature matrix, and materializing rich per-candidate `CandidateRepresentation` objects only for gate survivors / the top-K reasoning set.

**Determinism requirements.** Identical inputs ⇒ identical `submission.csv` bytes and identical `run_report.json` `reproducible` block. No wall clock in logic (recency uses the injected `as_of`). No online RNG (`OnlineEntropy` raises on `numpy_generator`). All ties resolve by `candidate_id` ascending. float32 with fixed reduction order; BLAS/OMP threads pinned so output is **thread-count-invariant**.

**Artifact assumptions.** All artifacts referenced by `MANIFEST.json` are present, hash-consistent, schema-compatible, and cross-coherent (Ports §8 / Offline Part 12). The online pipeline **never** rebuilds an artifact; it loads, verifies, and applies. Required keys: `model/encoder.onnx`, `candidate_vectors.parquet`, `anchor_vectors.npy`, `centroids.npy`, `lexicon.compiled.json`, `concepts.json`, `jd_concepts.json`, `gates/eligibility_rules.yaml`, `integrity_rules.json`, `integrity_thresholds.json`, `risk_weights.json`, `scoring_weights.locked.yaml`, `behavioral_weights.json`, `feature_manifest.json`, `feature_importance.json`, `archetypes.json`, `reasoning_templates.json`, `ranking_calibration.json`, `embedding_manifest.json`.

**Reproducibility requirements.** The run records `code_version`, `config_hash`, `manifest_hash`, per-artifact hashes, `input_file_sha256`, `output_sha256`, `as_of`, and `seed` into `run_report.json`. A second run on the same inputs reproduces the `reproducible` block byte-for-byte.

**Success criteria.**
1. `submission.csv` passes the organizer `validate_submission.py` (exactly 100 rows; ranks 1–100 once each; ids unique + `^CAND_[0-9]{7}$`; score non-increasing by rank; ties by id ascending).
2. Honeypot rate in top-100 ≈ 0 (well under the 10% disqualification gate).
3. Every top-100 reasoning is evidence-backed (Stage-4 survivable).
4. Wall-clock ≤ 150 s, peak RSS ≤ 4 GB, CPU-only, no network.
5. Output is byte-deterministic across runs.

---

# PART 2 — COMPLETE ONLINE EXECUTION DAG

Stages R0–R9. Ports appear only at R0/R1/R3/R8/R9; R2/R4/R5/R6/R7 are pure engine work (no port calls), which keeps the hot path testable and the budget predictable.

| Stage | Depends on | Inputs | Outputs |
|---|---|---|---|
| **R0 Artifact Loading** | — | `MANIFEST.json` + all artifacts; `config` | bound, verified artifact handles; `OnlineRunContext` |
| **R1 Candidate Ingestion** | R0 | `candidates.jsonl(.gz)` via `CandidateSourcePort` | stream of `RawCandidate` + `Identity` (stage `PARSED`) |
| **R2 Feature Extraction** | R1, R0(lexicon, feature_manifest) | `RawCandidate`, `as_of`, lexicon, `FeatureLayout` | structural slices (Career/Credibility/Logistics/Behavioral); bulk `(N,D)` CQV values + `(N,G)` confidence (stage `FEATURED`) |
| **R3 Semantic Hydration** | R2, R0(vectors, anchors, centroids, onnx) | `CandidateId`s, composed docs | `SemanticProfile`, `ArchetypeAssignment`; semantic feature values folded into CQV (stage `SITUATED`) |
| **R4 Gates & Eligibility** | R3, R0(integrity_rules, thresholds, gates) | full structural+semantic representation | `IntegrityReport`, `EligibilityReport`; floor mask (stage `GATED`) |
| **R5 Scoring** | R4, R0(weights, behavioral_weights, importance) | folded CQV + gates + multipliers | `ScoredCandidate` + `ScoreBreakdown` (stages `VECTORIZED`→`SCORED`) |
| **R6 Ranking** | R5 | `ScoredCandidate[]` | `Ranking` (top-100, invariants enforced) (stage `RANKED`) |
| **R7 Reasoning** | R6, R0(reasoning_templates, importance, archetypes), R1(top-K raw) | `RankedCandidate` + re-hydrated raw | `Ranking.with_reasoning` (stage `EXPLAINED`) |
| **R8 Submission** | R7 | `Ranking` | `submission.csv` + `ValidationReport` + `SubmissionReceipt` |
| **R9 Run Report** | R8 + all stage metrics, R0(manifest hashes) | metrics, hashes, timings | `run_report.json` |

**True execution order:** R0 → R1 → R2 → R3 → R4 → R5 → R6 → R7 → R8 → R9 (strictly sequential at the stage boundary; the representation flows forward, copy-on-write).

**Parallelizable work (within a stage, deterministic merge by `source_index`):**
- R1 streams while R2 begins on completed batches (pipeline overlap of ingestion + feature extraction).
- R2 internal fan-out: structural-feature, behavioral, and document-composition computations are independent per candidate and run data-parallel.
- R3 cosine-vs-anchors and nearest-centroid are vectorized matmuls (BLAS-parallel, thread-pinned).
- R4 integrity and eligibility detectors are vectorized over the population (integrity ∥ eligibility, then the floor mask joins).
- R7 runs only over the top-100 (trivial) and may parallelize per-candidate clause assembly.

Determinism rule across all parallelism: thread count must not change output; reductions follow `FeatureLayout`/`ScoreComponent` order; all merges and ties order by `candidate_id`.

---

# PART 3 — R0 ARTIFACT LOADING

**Purpose.** Load and integrity-verify the entire artifact set, bind every port to its concrete adapter, and assemble the immutable `OnlineRunContext`.

**Manifest verification.** (1) Read `MANIFEST.json`; recompute and check `manifest_sha256` (self-integrity). (2) Assert `manifest_schema_version` is supported. (3) Assert every **required** key (Part 1) is present.

**Artifact loading order.** (a) manifest → (b) small JSON/YAML config-class artifacts (`feature_manifest`, `scoring_weights.locked`, `behavioral_weights`, `integrity_*`, `risk_weights`, `gates`, `jd_concepts`, `concepts`, `lexicon`, `archetypes`, `feature_importance`, `reasoning_templates`, `ranking_calibration`, `embedding_manifest`) → (c) `anchor_vectors.npy`, `centroids.npy` (small, loaded fully) → (d) `model/encoder.onnx` (onnx session init, `CPUExecutionProvider`, pinned threads) → (e) `candidate_vectors.parquet` opened **memory-mapped** via `SemanticVectorStorePort` (not read into RAM).

**Hash validation.** Streaming sha256 during each load, compared to the manifest entry. **Cross-artifact coherence** asserted: `layout_version` agreement across `feature_manifest` / `scoring_weights` / `FeatureLayout` constant; `embedding.dim` equal across `candidate_vectors` / `anchor_vectors` / `centroids` / onnx; `embedding.model_id` consistent between `candidate_vectors` and the onnx fallback; the anchor set ⊆ `jd_concepts`; centroid `dim` == `embedding.dim`; every gate code ∈ `EligibilityCode`; every integrity code ∈ `IntegrityFlag`; scoring weight keys == `ScoreComponent`.

**Corruption handling.** Any hash mismatch, missing key, schema-version incompatibility, or coherence failure → `ArtifactContractError` / `ManifestError` → **fail-fast, abort the run**. No degraded mode (with no live leaderboard, silent degradation is undetectable).

**Startup diagnostics.** Emit `manifest_hash`, per-artifact hashes, `layout_version`, `embedding.model_id`/`dim`, artifact byte totals, and the resolved `as_of`/`seed`/`config_hash` into the R0 metrics block (carried to R9).

**Output contract.** `OnlineRunContext` = { resolved config, bound ports (`CandidateSource`, `SemanticVectorStore`, `EmbeddingModel(onnx)`, `SubmissionSink`, `RunReportSink`, `OnlineEntropy`), loaded artifact objects, verified `Manifest`, `as_of`, `seed` }. Immutable.

**Failure contract.** Raises on any integrity/coherence failure; never returns a partially-bound context.

**Runtime budget.** ≤ 20 s (dominated by streaming-hashing the ~150 MB vector file + onnx; small artifacts negligible).

**Memory budget.** ≤ ~400 MB (onnx session + fully-loaded small matrices; the candidate vectors stay mmap'd).

---

# PART 4 — R1 CANDIDATE INGESTION

**Purpose.** Turn the raw input into validated, typed `RawCandidate` + `Identity`, lazily and in file order.

**JSONL streaming.** `CandidateSourcePort.stream()` yields `SourceRecord`s (gzip-aware, UTF-8, constant memory, `source_index` = enumeration order). The ingestion engine never materializes the full file.

**Schema validation.** Each `Ok` record is validated against `candidate_schema.json` by `features.parsing` into a `RawCandidate`; `Identity` is minted (id pattern check + `ProvenanceHandle(source_index)`). Semantic contradictions (inverted salary, expert-zero-duration, current-with-end-date) are **preserved**, not corrected — they are honeypot signal downstream.

**Malformed record handling.** A non-JSON line arrives as a `Malformed` record (from the port). Schema-invalid-but-parseable records raise `SchemaError` in parsing.

**Quarantine behavior.** Policy from config; **default full-run policy: abort** on any malformed/`SchemaError` (exactly 100,000 well-formed rows are expected; a deviation indicates wrong input). The sandbox/sample profile may switch to *skip-and-record*. Quarantined records are logged with `line_no` + reason and counted; they never enter the candidate set.

**Throughput expectations.** ~2,500–3,500 records/s (JSON decode + pydantic validation) → 100K in ~30–40 s. Overlaps with R2.

**Output contract.** Lazy `Iterator[IngestedCandidate]` (`RawCandidate` + `Identity` + `source_index`) at stage `PARSED`; `IngestionMetrics` (`rows_read, ok, malformed, schema_reject, duplicate_id`).

**Failure contract.** `CandidateSourceError` (IO/decompress) → abort. Per-record failures handled per policy. Duplicate `candidate_id` → recorded; default abort.

**Runtime budget.** ≤ 35 s.

**Memory budget.** O(1) streaming (≤ ~150 MB transient buffers); no full-file materialization.

---

# PART 5 — R2 FEATURE EXTRACTION

**Purpose.** Compute all structured features into the canonical, versioned representation, producing the bulk CQV matrix and the structural domain slices.

**Feature families (Feature Layer Parts 1–4).** Identity/Geography/Experience/Seniority/Education/Company/Product-vs-Service; the competency groups (retrieval, ranking, recsys, IR, NLP, LLM, ML-eng, MLOps, eval) as **trust-weighted evidence aggregates** (never keyword flags); Open-Source/Leadership/Startup/Founding; the logistics families (Salary/Relocation/Notice); the behavioral composites (the 23 signals → availability/recruitability/reliability/etc., sentinel `−1`/`{}` ⇒ `UNKNOWN`, never 0); Consistency/Risk primitives. Semantic feature values are **left as placeholders** here and filled at R3.

**Extraction order.** Normalization (canonical text/dates/skill tokens, **composed embedding document** using the recipe pinned in `embedding_manifest`) → primitive features (data-parallel) → derived features (career intelligence, credibility trust, behavioral composites) → latent placeholders. Fixed topological order (determinism).

**Provenance generation.** Each feature emits a `FeatureCell(value, confidence, evidence)`; `EvidenceRef`s point to the exact `RawCandidate` fields used. Bulk path stores `value` in `(N,D)` and `confidence` at group granularity `(N,G)`; full per-feature confidence + evidence are materialized **only for survivors/top-K** (memory).

**Evidence collection.** Evidence is attached but not yet serialized; it rides with survivors into R7.

**Intermediate representation creation.** Build `CareerProfile`, `CredibilityProfile` (structural), `LogisticsProfile`, `BehavioralProfile`; write each candidate's CQV row at its `source_index` in the shared `(N,D)` matrix. Advance `BuildStage` → `FEATURED`.

**Output:** `CandidateRepresentation` partially populated (career/credibility/logistics/behavioral); bulk `(N,D)` float32 CQV (semantic indices still placeholder) + `(N,G)` confidence.

**Invariants.**
- CQV row dim == `FeatureLayout.D`; `feature_manifest.layout_version` == `FeatureLayout` constant version.
- No `NaN`/`inf` in any populated CQV index (sentinels resolved to `UNKNOWN` flags + neutral values).
- All `UnitScore` features ∈ [0,1]; bounded floats within documented ranges.
- Recency features computed only against `as_of` (no wall clock).
- `competency = trust + in_career + semantic − stuffing_penalty`; a claimed-only skill with no corroboration yields competency ≈ 0.
- Single-ownership of the 23 signals (no double counting): behavioral families own 3–8,10,16–23; credibility owns 9,11; logistics owns 12–15.
- Stage advances `PARSED → FEATURED`; regression raises `RepresentationStageError`.

**Runtime budget.** ≤ 35 s (the heaviest pure-CPU stage; fully vectorized columnar).

**Memory budget.** ≤ ~1.2 GB peak: `(N,D)` float32 (≈ 100K×D×4) + `(N,G)` confidence + transient columnar arrays; per-candidate objects only for the (later) survivor set.

---

# PART 6 — R3 SEMANTIC HYDRATION

**Purpose.** Attach dense semantic fit and archetype assignment, folding semantic feature values into the CQV.

**Embedding lookup.** `SemanticVectorStorePort.view_all()` returns the mmap'd `(N,dim)` matrix + aligned id order; `get_many(ids)` for the active set. Candidate vectors are read by `CandidateId` (O(1) row index), **not** recomputed.

**Candidate vector retrieval.** Align store rows to the ingested candidates by `CandidateId` (preserve `source_index` order). Vectors are read-only, L2-normalized (offline guarantee).

**Anchor retrieval.** `anchor_vectors.npy` (positive + negative `jd.*` concepts) loaded at R0; cosine of each candidate vector vs every anchor is a single vectorized matmul → `positive_fit`, `negative_fit`, `net_semantic_fit`, `best_positive_anchor` (argmax, ties by `AnchorId` ascending).

**Archetype assignment.** Nearest-centroid over `centroids.npy` (vectorized), ties by `ArchetypeId` ascending → `ArchetypeAssignment` (id, distance, membership confidence, `is_target_archetype`).

**Delta handling.** The 100K pool is fully precomputed offline, so the store covers all released ids; `get_many` returns no misses in full reproduction. The id order from the store and the ingestion order are reconciled by `CandidateId`; any id present in input but absent from the store is a **miss** (next paragraph), not an error.

**Cold-start / missing embeddings (exact behavior).** For any `CandidateId` not found in the store (expected only for the ≤100 sandbox sample or a pool delta): compose its embedding document (R2 recipe) and encode via `EmbeddingModelPort` (onnx, CPU, offline) → a vector in the same space (cosine within ε of the offline space). The encoded vector is used for that candidate's similarities/archetype. If encoding **also** fails (`EmbeddingError`): the candidate's semantic features are set to `UNKNOWN` with zero net fit, `risk.uncertainty` raised, and it is recorded as a semantic miss — the run continues (the candidate simply cannot earn semantic credit and will rank low). A miss is **never** fatal to the run.

**Folding.** Write `semantic_*` feature values into the CQV at their `FeatureLayout` indices; attach `SemanticProfile` + `ArchetypeAssignment`; advance `BuildStage` → `SITUATED`.

**Invariants.** Similarities ∈ [−1,1]; `net_semantic_fit` ∈ [0,1]; anchor keys ⊆ artifact set; vector `dim` == `embedding.dim`; folding leaves no `NaN`.

**Runtime budget.** ≤ 15 s (mmap gather + two vectorized matmuls; fallback encode rare/none).

**Memory budget.** ≤ ~300 MB additional (mmap resident pages ~150 MB + anchor/centroid matrices + similarity buffers). The full vector matrix is **never** copied into RAM.

---

# PART 7 — R4 GATES AND ELIGIBILITY

**Purpose.** Produce the integrity and eligibility verdicts that gate scoring; build the floor mask.

**IntegrityEngine (via Consistency + Risk).** Apply `integrity_rules.json` detectors using `integrity_thresholds.json`: timeline impossibility, skill-time contradiction, employment overlap, title-seniority, education-career, experience inflation, keyword-stuffing, behavioral/signal impossibility, identity anomaly. `CandidateRiskEngine` aggregates findings into `honeypot_score`, contradiction/uncertainty, and finalizes `IntegrityReport`. `is_honeypot = (≥2 HARD impossibilities) OR (honeypot_score ≥ threshold)`. Inverted salary is a **soft** sanity flag only (common in the pool), never a hard honeypot.

**EligibilityEngine.** Apply `gates/eligibility_rules.yaml` predicates against the representation: HARD blocks (`PURE_RESEARCH_NO_PRODUCTION`, `LANGCHAIN_OPENAI_ONLY_RECENT`, `NO_PRODUCTION_CODE_18M`, `CONSULTING_FIRMS_ONLY_CAREER`, `PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP`, `CLOSED_SOURCE_5Y_NO_VALIDATION`); SOFT penalties (`TITLE_CHASER_SUB_18M_HOPS`, `NOTICE_OVER_30`, `OUTSIDE_INDIA_NO_SPONSOR`, `OUTSIDE_EXPERIENCE_BAND`). `is_eligible = (no hard blocks)`.

- **Exclusion rules (FLOOR):** `is_honeypot` OR `not is_eligible` ⇒ the candidate's `final_score` is forced to `FLOOR` at R5 (cannot occupy a top rank).
- **Cap rules:** floored candidates are partitioned to the filler tail; non-floored candidates fill ranks first (R6). The honeypot top-100 rate is thereby held ≈ 0 by construction.
- **Penalty rules (down-weight):** soft penalties reduce the relevant fit components / multipliers at R5 but never floor.

**Ordering requirements.** Integrity and Eligibility are computed independently (parallelizable), then **joined into a single floor mask**. Both must complete before R5; neither depends on the score. Within R4, detectors run before aggregation; `is_honeypot`/`is_eligible` are derived last.

**Invariants.**
- `is_eligible == (len(hard_blocks) == 0)`; `is_honeypot` per the rule above.
- Every finding (integrity or eligibility) carries ≥1 `EvidenceRef`.
- Findings sorted by `code` (determinism).
- The floor mask is a pure function of `(IntegrityReport, EligibilityReport)`.
- Stage advances `SITUATED → GATED`.

**Runtime budget.** ≤ 10 s (vectorized detectors). **Memory budget.** ≤ ~200 MB additional (boolean/float masks).

---

# PART 8 — R5 SCORING

**Purpose.** Deterministically apply the locked weights, gates, and bounded multipliers to produce `ScoredCandidate` + `ScoreBreakdown`. **No runtime calibration. No silver-label component.**

**CQV integration.** The folded `(N,D)` matrix is dotted with `ScoringWeights` (from `scoring_weights.locked.yaml`, keyed by `ScoreComponent`, summed in fixed `ScoreComponent` order) → `base_relevance` per candidate. Weight keys must equal `ScoreComponent` exactly (verified at R0).

**Fit score computation.** Positive `jd.*` latents (retrieval/ranking/recsys/production-ML/product-company/hybrid-retrieval/eval) and corroborated competency trust drive the positive components; `jd.keyword_only` subtracts (anti-stuffer); negative latents feed the eligibility/penalty path.

**Penalties.** Soft eligibility penalties reduce their associated components/multipliers (bounded). Contradiction/uncertainty from R4 reduce `confidence`.

**Multipliers.** `behavioral_multiplier` and `logistics_multiplier` (bounded by `behavioral_weights.json`) modulate `base_relevance` — they never *create* relevance. `archetype_adjustment` adds a bounded boost for target archetypes.

**Final score computation.** For each candidate:
- If floored (R4): `final_score = FLOOR` (a fixed sentinel, e.g. 0.0). No multipliers/adjustments applied.
- Else: `final_score = base_relevance × behavioral_multiplier × logistics_multiplier + archetype_adjustment`, then a deterministic **confidence shrink** toward a neutral prior proportional to uncertainty. **No calibration curve is applied** (the offline monotone curve is disabled by default and, being order-preserving, cannot change ranking even if enabled — it would only reshape the displayed `score`).

**Tie handling.** Equal `final_score`s are permitted; the `ScoredCandidate.tiebreak_key` is `candidate_id`, which R6 uses for the validator-mandated ascending tie-break.

**Numeric stability.** float32; component sum in fixed order; multipliers bounded; `FLOOR` is a finite constant; no division by zero (bounded denominators); no `NaN`/`inf` may appear (asserted). Reductions are thread-count-invariant (BLAS pinned).

**Normalization requirements.** `final_score` is mapped into a stable, monotone, order-preserving range for the CSV `score` column (a fixed affine/clamped transform that preserves the ranking exactly). This is presentation only — **not** a learned calibration.

**Output:** `CandidateQualityVector` folded; `ScoredCandidate(final_score, ScoreBreakdown(components+evidence, gates, multipliers, archetype_adjustment, final_score), tiebreak_key)`. Stages `VECTORIZED → SCORED`.

**Invariants.** `base_relevance == Σ component.weighted`; floored ⇒ `final_score == FLOOR`; `final_score` a deterministic function of inputs; multipliers within bounds; per-component evidence present.

**Runtime budget.** ≤ 8 s (matvec + elementwise). **Memory budget.** ≤ ~200 MB additional; `ScoredCandidate` objects materialized for the non-floored survivor set + the eventual top-100.

---

# PART 9 — R6 RANKING

**Purpose.** Deterministic top-100 selection into a spec-valid `Ranking`.

**Sorting procedure.** Stable sort all candidates by `(−final_score, candidate_id)` on **raw** scores (no calibration applied). Partition floored candidates to the tail.

**Deterministic tie breaks.** Equal scores ordered by `candidate_id` ascending (matches `validate_submission.py`).

**Cap enforcement.** Non-floored candidates fill ranks 1..100 first; floored candidates fill remaining slots only if fewer than 100 non-floored exist (degenerate edge case — recorded as a warning). This guarantees honeypots/ineligibles do not occupy real top ranks, keeping the top-100 honeypot rate ≈ 0.

**Top-100 selection.** Take the first 100; assign ranks 1..100; construct the domain `Ranking`, whose factory **enforces** the six invariants (count == 100; ranks 1–100 once; ids unique + pattern; non-increasing score; tie-break id ascending; sorted by `(−score, id)`) and raises `RankingInvariantError` on any violation.

**Output contract.** `Ranking` at stage `RANKED`; metrics (`cutoff_score` at rank 100, `top1`, `top10_min`, `tie_groups`, `floored_in_top100` (expect 0)).

**Runtime budget.** ≤ 2 s. **Memory budget.** holds the top-100; the rest is dropped.

---

# PART 10 — R7 REASONING GENERATION

**Purpose.** Produce evidence-grounded, Stage-4-survivable reasoning for each of the top-100, plus the audit trace.

**Provenance consumption.** For the top-100, re-hydrate `ProvenanceHandle.inline` (the `RawCandidate` retained from R1 for survivors). Each candidate's `ScoreBreakdown` components and gate findings carry `EvidenceRef`s.

**Evidence selection.** Rank features by `feature_importance.json`; select the top contributing features (with their `EvidenceRef`s) as the strengths; pull the candidate's specific facts (years, current title, named corroborated skills, product-company tenure, key signal values, archetype fingerprint).

**Concern selection.** Surface soft eligibility penalties and notable gaps (e.g., notice > 30, consulting density, low availability) as honest concerns — required wherever a gap exists.

**Diversity constraints.** Reasonings must be substantively distinct (Stage-4 "variation"): variation arises from *which evidence is present per candidate*, assembled from the `reasoning_templates.json` evidence-slot templates — **not** name-insertion templating. A diversity check flags near-identical outputs.

**Hallucination prevention.** Every `ReasoningClause` requires ≥1 `EvidenceRef` resolving to a real `RawCandidate` field — enforced at construction. A clause citing a non-existent fact raises `ProvenanceError`. Tone must match the rank band (top = net-positive, tail = measured).

**Interaction with ranking.** R7 consumes the finished `Ranking` and attaches reasoning via `Ranking.with_reasoning(...)`, re-asserting the structural invariants; **it never reorders ranks**.

**Why deterministic.** Rendered text is a pure function of the ordered, evidence-bound clause set and the static templates; there is no RNG and no LLM online. Identical evidence ⇒ identical text; differing evidence ⇒ different text (the source of legitimate variation).

**Runtime budget.** ≤ 3 s (top-100 only). **Memory budget.** top-100 representations + their raw records (small).

---

# PART 11 — R8 SUBMISSION GENERATION

**Purpose.** Emit the validator-compliant CSV and confirm validity before writing.

**CSV construction.** `SubmissionSinkPort.write(ranking)` produces header `candidate_id,rank,score,reasoning` + exactly 100 rows in rank order, with deterministic `score` precision and the raw-score-derived ranking preserved.

**Schema validation / validator integration.** Before writing, the `ValidationEngine` runs the structural checks (mirroring `validate_submission.py`) plus the Stage-4 reasoning checks (no empty/identical/templated/hallucinated/rank-inconsistent reasoning). On any HARD finding → abort (do not emit a rejectable file). After writing, the sink re-asserts non-increasing-by-rank + id-ascending tie-break at the emitted precision (`SubmissionContractError` otherwise).

**Encoding requirements.** UTF-8, no BOM; `\n` line endings; RFC-4180 quoting for any `reasoning` containing comma/quote/newline; no trailing whitespace.

**Output naming.** `<participant_id>.csv` (from config; the `.csv` extension and participant-id stem satisfy the validator's filename rule). Written atomically (temp + rename).

**Failure handling.** `ValidationReport.is_valid == False` or `SubmissionContractError` → abort, no file. `SubmissionWriteError` (IO) → abort. A `SubmissionReceipt` (`row_count`, `bytes_written`, `output_sha256`) is returned for R9.

**Runtime budget.** ≤ 1 s. **Memory budget.** negligible.

---

# PART 12 — R9 RUN REPORT GENERATION

**Purpose.** Emit `run_report.json` — the audit + reproducibility artifact.

**Structure (port-owned `RunReport`).**
- **`reproducible`** (deterministic; the only block compared by determinism tests): `code_version`, `config_hash`, `manifest_hash`, per-artifact `artifact_hashes`, `input_file_sha256`, `candidate_count`, `output_sha256`, `honeypot_count_top100`, `honeypot_rate`, `eligibility_summary` (per-code counts), `score_distribution_digest`, `as_of`, `seed`.
- **`audit`** (excluded from any repro hash): `run_id`, `started_at`, `ended_at`, `host_label`.
- **`timings`**: per-stage R0–R9 wall-ms + budget headroom.
- **`budget`**: `limit_seconds`, `used_seconds`, `within_budget`, `peak_rss_mb`.

**Score breakdowns.** For the top-100 (and a sample of the tail), the per-component `ScoreBreakdown` (component, raw, weight, weighted, gates, multipliers, final) — reconstructable.

**Gate statistics.** Per-`IntegrityFlag` and per-`EligibilityCode` fire counts; floored count; honeypot rate (for the Stage-3 ≤10% gate).

**Evidence traces.** For each top-100 candidate, the reasoning audit trace: clause → `EvidenceRef` → raw field, plus the feature-importance ranking used.

**Audit information.** Artifact set identity (manifest + per-artifact hashes), config, code, input, and output hashes — a complete provenance chain from submission back to inputs and artifacts.

**Reproducibility metadata.** Everything needed to re-derive the run; the `reproducible` block must reproduce byte-for-byte on a second run.

**Stage-5 defense support.** Because every ranked candidate's score is decomposed into evidence-backed components and every reasoning clause traces to a real field, the report lets the team walk an interviewer from any top-100 placement → its score components → the features → the candidate's actual profile facts, and demonstrate the honeypot/eligibility gates fired as designed — exactly the "explain and defend your architecture" requirement.

**Runtime budget.** ≤ 1 s. **Memory budget.** negligible.

---

# PART 13 — RESUME / RECOVERY STRATEGY

**Restart semantics.** The online run is a single, short (~130 s), fully deterministic process. The canonical recovery action is **re-run from scratch**: because the pipeline is deterministic and reads only static artifacts + the input file, a fresh run reproduces the identical result. No mid-run resume is required or offered for the default path.

**Checkpointing policy.** Minimal by design. R0 verification results are cached in-process. No persistent inter-stage checkpoints are written in the default profile (the cost of a full re-run is within budget, and checkpoints would add nondeterminism risk and IO). An **optional** debugging profile may persist the R3 hydrated vectors / R2 feature matrix to disk for faster iteration during development — never used for the official run.

**Partial run behavior.** Outputs are written **atomically** (temp + rename) only at R8/R9; a crash before R8 leaves **no** `submission.csv` (never a partial/rejectable file). A crash between R8 and R9 leaves a valid `submission.csv` and no/partial `run_report.json` → re-run regenerates both deterministically.

**Corruption handling.** Artifact corruption is caught at R0 (hash verify) → abort. Input corruption is caught at R1 (decode/schema) → abort per policy. There is no remote state to reconcile.

**No network.** All recovery is local and offline; no retries hit any external service.

---

# PART 14 — PERFORMANCE BUDGET

| Stage | Runtime (target) | Memory (peak additional) | Notes |
|---|---|---|---|
| R0 Artifact Loading | ≤ 20 s | ≤ 400 MB | streaming sha256 of ~150 MB vectors + onnx; vectors stay mmap'd |
| R1 Candidate Ingestion | ≤ 35 s | O(1) (≤ 150 MB transient) | JSONL streaming + schema validation; overlaps R2 |
| R2 Feature Extraction | ≤ 35 s | ≤ 1.2 GB | bulk `(N,D)` float32 + `(N,G)` confidence; heaviest pure stage |
| R3 Semantic Hydration | ≤ 15 s | ≤ 300 MB | mmap gather + 2 vectorized matmuls; fallback encode rare |
| R4 Gates & Eligibility | ≤ 10 s | ≤ 200 MB | vectorized detectors; floor mask |
| R5 Scoring | ≤ 8 s | ≤ 200 MB | matvec + elementwise; survivor `ScoredCandidate`s |
| R6 Ranking | ≤ 2 s | small | stable sort 100K + top-100 |
| R7 Reasoning | ≤ 3 s | small | top-100 only; re-hydrated raw |
| R8 Submission | ≤ 1 s | negligible | CSV + validation + atomic write |
| R9 Run Report | ≤ 1 s | negligible | JSON serialization |
| **Total** | **≈ 130 s (≤ 150 s ceiling)** | **≈ 2–3 GB peak (≤ 4 GB)** | CPU-only, deterministic, no network |

**How the budget is maintained.** (1) Heavy work is offline — online does lookup, not encode (R3) and apply, not calibrate (R5). (2) Vectorized columnar numeric paths; no per-candidate Python in the hot loop (rich objects only for survivors/top-K). (3) Memory-mapped vectors; a single `(N,D)` matrix; sentinel-resolved no-`NaN` features. (4) Pinned BLAS/OMP threads (predictable latency + thread-invariant determinism). (5) Stage timers + budget guard write `within_budget` and `peak_rss_mb` to R9; a CI integration test on the sample asserts the budget holds before any real submission.

---

# PART 15 — VALIDATION CHECKPOINTS

| Stage | Assertions | Invariants | Warning conditions | Fatal conditions |
|---|---|---|---|---|
| R0 | manifest self-hash; per-artifact sha256; required keys present | cross-artifact coherence (layout/dim/model_id/anchor⊆jd_concepts) | unusually large artifact bytes | any hash/schema/coherence failure → abort |
| R1 | id pattern; schema valid; UTF-8 | `source_index` monotonic; provenance id match | malformed/skip count > 0 (sandbox) | `CandidateSourceError`; malformed in full run; duplicate id |
| R2 | CQV dim == D; no NaN; ranges | layout_version match; single-signal-ownership; recency vs `as_of` only | high `UNKNOWN` density; many stuffing flags | `CQVInvariantError`; layout mismatch |
| R3 | similarities ∈ [−1,1]; net ∈ [0,1]; vector dim | anchors ⊆ artifact; no NaN after fold | nonzero store misses (sandbox/delta) | `VectorStoreError`; dim mismatch |
| R4 | findings carry evidence; codes valid | `is_eligible`/`is_honeypot` derivations; floor mask pure | hard-block rate or honeypot rate unusually high | invalid code in rules; missing thresholds |
| R5 | `base == Σ weighted`; floored ⇒ FLOOR; no NaN | deterministic final_score; multipliers bounded | many floored; multiplier saturation | `ScoreInvariantError`; weights/layout mismatch |
| R6 | 100 rows; ranks 1–100 once; ids unique+pattern; non-increasing; tie-break | `Ranking` factory invariants | `floored_in_top100 > 0`; large tie groups | `RankingInvariantError`; < 100 candidates |
| R7 | every clause has evidence; ≥1 JD link; ≤2 sentences | rank-band tone consistency; no reorder | low concern coverage; near-duplicate texts | `ProvenanceError`; missing top-K hydration |
| R8 | validator pass; UTF-8/no-BOM; quoting; precision | post-format monotonic + tie-break | none | `ValidationReport` invalid; `SubmissionContractError`; IO error |
| R9 | required fields present; `reproducible` stable | audit excluded from repro hash | `within_budget == false` | `ReportWriteError` |

---

# PART 16 — FAILURE MODES

| Failure | Cause | Detection | Impact | Mitigation |
|---|---|---|---|---|
| Artifact corruption | bad/tampered bytes, partial offline build | R0 streaming sha256 + manifest self-hash | run aborts at startup | fail-fast; rebuild artifacts offline; never run on unverified artifacts |
| Manifest incoherence | layout/dim/model_id/anchor mismatch across artifacts | R0 cross-artifact coherence checks | abort | offline O18 reproducibility validation must pass before shipping artifacts |
| Schema drift | input fields/enums diverge from `candidate_schema.json` | R1 pydantic validation | per-record `SchemaError` → abort (full run) | validate input provenance; sandbox profile can skip+log |
| Malformed input lines | corrupt/truncated JSONL | R1 `Malformed` records | abort (full run) per policy | atomic re-run; verify the input file hash |
| Missing embeddings | id absent from store (sandbox/pool delta) | R3 `get_many` miss list | candidate encoded via onnx fallback; if that fails, semantic = UNKNOWN, ranks low | fallback encoder (same model); never fatal; recorded in R9 |
| Overfiring gates | thresholds too aggressive; rampant synthetic mismatches | R4 stats (hard-block/honeypot rate) vs offline census expectation | real candidates wrongly floored → NDCG loss | conservative O3/O12 calibration (≥2 HARD for honeypot; soft salary inversion); warning when rates exceed census band |
| Ranking instability | float nondeterminism, thread variance, unstable sort | R6 invariant checks + determinism CI (1-thread vs N-thread) | nonreproducible ranks | float32 fixed-order reductions; pinned BLAS/OMP; stable sort; tie-break by id |
| Reasoning failure | dangling evidence; templated/identical outputs | R7 `ProvenanceError`; Stage-4 diversity/hallucination checks at R8 | invalid/penalized reasoning | clause-requires-evidence invariant; evidence-driven (not templated) assembly; diversity check |
| Validator failure | malformed CSV, wrong row count, bad tie-break/precision | R8 `ValidationEngine` + post-format re-assert | submission would be rejected | abort before write; `Ranking` enforces structure at construction; byte-stable sink |
| Budget overrun | slow IO/encode, oversubscribed threads | budget guard `within_budget`; CI sample timing | risk of exceeding the 5-min ceiling | lookup-not-encode (R3); pinned threads; columnar vectorization; CI gate on sample timing |

---

# PART 17 — FINAL ONLINE PIPELINE SPECIFICATION (one-page summary for `generate_submission.py`)

**Mission.** Read `candidates.jsonl`, apply the offline artifact set, emit a validator-compliant top-100 `<participant_id>.csv` + `run_report.json`, deterministically, in ≤ 150 s / ≤ 4 GB / CPU-only / no network.

**Order (strictly sequential; ports only at R0/R1/R3/R8/R9):**
- **R0** Load + verify all artifacts via `ArtifactStorePort` (manifest self-hash, per-artifact sha256, cross-artifact coherence); bind ports; build the immutable `OnlineRunContext` (`as_of`, no online RNG). Fail-fast on any integrity failure.
- **R1** Stream + validate candidates (`CandidateSourcePort` → `RawCandidate` + `Identity`); abort on malformed in full runs; constant memory.
- **R2** Extract all structural + behavioral features into a single `(N,D)` float32 CQV + `(N,G)` confidence; attach evidence; no `NaN`; competency = evidence-aggregate (anti-stuffer); layout_version must match.
- **R3** Hydrate semantics by **lookup** (`SemanticVectorStorePort`, mmap), cosine vs anchors + nearest-centroid; onnx **fallback** only for store misses; fold semantic values into the CQV.
- **R4** Gates: Integrity (honeypot: ≥2 HARD or composite ≥ threshold) + Eligibility (JD hard blocks / soft penalties) → floor mask; both carry evidence.
- **R5** Score: `base = Σ(component·locked_weight)`; floored ⇒ FLOOR; else `base × behavioral × logistics + archetype_adj`, deterministic confidence shrink; **no runtime calibration, no silver labels**; order-preserving score normalization for presentation only.
- **R6** Rank on **raw** scores: sort `(−score, candidate_id)`, non-floored first, take top-100, ranks 1..100; `Ranking` factory enforces all six validator invariants.
- **R7** Reasoning for top-100: evidence-bound clauses (every clause cites a real field), strengths by feature importance + honest concerns, deterministic + varied, no reorder; attach via `with_reasoning`.
- **R8** Submission: validate (structural + Stage-4) → abort if invalid → write UTF-8/RFC-4180 CSV atomically via `SubmissionSinkPort`; receipt carries `output_sha256`.
- **R9** Run report: `reproducible` (hashes, honeypot rate, eligibility summary, score digest) + `audit` + `timings` + `budget` via `RunReportSinkPort`; supports Stage-5 defense.

**Guarantees.** Deterministic + reproducible (no clock/RNG; ties by id; float32 fixed-order; thread-invariant). Honeypot top-100 rate ≈ 0 by floor-partitioning. Output passes `validate_submission.py`. Recovery = deterministic full re-run; outputs written atomically so no partial/rejectable file is ever produced.
