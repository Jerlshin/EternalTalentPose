# REDSTACK v1.1 — Engine Layer Specification

**Scope:** `src/redstack/engines/` only. Architecture, Domain Layer, and Ports Layer are frozen; this document does not alter them. No implementation code — architecture detailed enough that engine implementation can begin immediately.

**Engine Layer rules (inherited, restated as law):**
- Engines depend **only** on `domain/`, `ports/` (abstractions, injected), `features/` (pure transforms), and `config.schema` (typed config). They **never** import `adapters/`, `pipelines/`, or each other. The engine graph is a **forest**: the pipeline threads the growing `CandidateRepresentation` between engines; engines never call one another.
- Engines are **stateless callables**. All state is the representation passed in and returned (copy-on-write). No module-level mutable singletons; model/store handles arrive via injected ports.
- An engine is **pure given its ports**. "Impure" engines (Ingestion, Embedding, Retrieval, Submission, RunReport) touch the outside world only through an injected port; with a fake port they are deterministic and testable in-process.
- **No wall clock, no RNG online.** Recency uses the injected `as_of` from `DeterministicEntropyPort`; ties resolve by `candidate_id` ascending; online RNG is disabled.

**Confidence/uncertainty note (freeze-respecting):** the frozen `ScoreBreakdown` has no `confidence` field. `CandidateRiskEngine` computes confidence/uncertainty as an **engine-internal intermediate** that modulates `final_score` (shrink-toward-prior) and is emitted as a run-report metric. It is **not** persisted as a new domain field. Persisting it would require a domain unfreeze — flagged in §9/§10.

---

## §0. Engine → stage → domain map (anchor table)

| Engine | Online stage | Offline stage | Produces (domain slice / artifact) | Injected ports |
|---|---|---|---|---|
| 1 CandidateIngestionEngine | R1 | O1, O2, O5 | `Identity`, `RawCandidate` | CandidateSource |
| 2 CandidateNormalizationEngine | R2 | O2 | canonical intermediate (feeds slices) | — |
| 3 CandidateConsistencyEngine | R4 | O3 | `IntegrityFinding[]` (+ raw risk) | ArtifactStore (calibration) |
| 4 CandidateEmbeddingEngine | R3 (fallback) | O5, O6 | candidate/anchor vectors; fallback `FloatVector` | EmbeddingModel, ArtifactStore |
| 5 CandidateRetrievalEngine | R3 | — | `SemanticProfile`, `ArchetypeAssignment` | SemanticVectorStore, EmbeddingModel |
| 6 CandidateFeatureEngine | R2 | O2 | `CareerProfile`, `CredibilityProfile`(structural), `LogisticsProfile`, CQV feature row, `FeatureManifest` | ArtifactStore (lexicon) |
| 7 CandidateBehaviorEngine | R2 | — | `BehavioralProfile` | Entropy (`as_of`) |
| 8 CandidateFitEngine | R4 | — | `EligibilityReport`, fit components, logistics fit | ArtifactStore (gates/lexicon) |
| 9 CandidateRiskEngine | R4→R5 | — | `IntegrityReport` (final), confidence modifier | — |
| 10 CandidateScoringEngine | R5 | — | `CandidateQualityVector` (folded), `ScoredCandidate` | ArtifactStore (weights) |
| 11 CandidateRankingEngine | R6 | — | `Ranking` | — |
| 12 CandidateReasoningEngine | R7 | — | `CandidateReasoning`, `Ranking.with_reasoning` | — |
| 13 SubmissionGenerationEngine | R8 | O9 (dry-run) | `submission.csv`, `ValidationReport` | SubmissionSink |
| 14 RunReportEngine | R9 | O10 | `RunReport` | RunReportSink, ArtifactStore |

---

## §1. CandidateIngestionEngine

- **Purpose:** turn the raw input stream into validated, typed `RawCandidate`s + `Identity`, with ingestion metrics. The R1 / O-read entrypoint.
- **Inputs:** `Iterator[SourceRecord]` (from `CandidateSourcePort`); ingestion policy from `config.schema` (skip-vs-abort on malformed).
- **Outputs:** a lazy `Iterator[IngestedCandidate]` (`RawCandidate` + `Identity` + `source_index`); an `IngestionMetrics` summary.
- **Dependencies:** `CandidateSourcePort`; `features.parsing` (dict → `RawCandidate`); `domain.errors` (`SchemaError`).
- **Internal workflow:** for each `SourceRecord` → `Ok`: `features.parsing.validate(raw)` → `RawCandidate` → mint `Identity` (id pattern + `ProvenanceHandle(source_index)`); `Malformed`/`SchemaError`: apply policy (record + skip, or abort). Tally metrics.
- **Failure modes:** `CandidateSourceError` (IO, propagated); `SchemaError` per record (policy-driven); duplicate `candidate_id` across the stream (recorded; first wins or abort per policy).
- **Determinism:** file order preserved; identical file ⇒ identical sequence and metrics.
- **Threading:** single-consumer generator; not shared across threads; downstream sharding happens after ingestion.
- **Performance:** stream-parse 100K within ~30–60 s (dominated by JSON decode); O(1) per record.
- **Memory:** O(1) streaming; no full-file materialization.
- **Auditability:** emits `rows_read, ok_count, malformed_count, schema_reject_count, duplicate_id_count, parse_ms`; each rejection logged with `line_no` + reason.
- **Testability:** contract-tested against `ListCandidateSource`; property test on order/`source_index` monotonicity; golden test on a malformed-line fixture.

## §2. CandidateNormalizationEngine

- **Purpose:** deterministic canonicalization of free-text and dates **before** feature extraction, so downstream engines see stable, comparable inputs.
- **Inputs:** `RawCandidate`.
- **Outputs:** a `NormalizedCandidate` intermediate (canonical text tokens, parsed `date`s, normalized skill names via lexicon canonical map, company-name canonicalization, industry canonicalization). Not a persisted domain slice — an internal carrier consumed by Feature/Behavior/Retrieval engines.
- **Dependencies:** `features.parsing`/`features.skills` helpers; lexicon canonical map (passed in, sourced from artifact by the pipeline).
- **Internal workflow:** lowercase/strip/collapse whitespace; Unicode NFC normalization; canonical skill-token mapping; parse dates to `date`; normalize `CompanySize`/enums (already typed by domain); compose the **embedding document** text (deterministic field concatenation order) for the Embedding/Retrieval engines.
- **Failure modes:** unparseable date that passed schema (defensive) → `SchemaError`; unknown skill token → mapped to itself (never dropped).
- **Determinism:** pure function; fixed normalization rules; fixed document-composition order (critical — embedding determinism depends on identical text composition offline and online).
- **Threading:** stateless, reentrant; data-parallel safe.
- **Performance:** ~µs per candidate; ~10–20 s for 100K.
- **Memory:** per-candidate; no accumulation.
- **Auditability:** emits `normalized_count, unknown_skill_tokens, date_repair_count`.
- **Testability:** golden tests pinning canonical outputs (esp. the composed embedding document — offline/online must match byte-for-byte).

## §3. CandidateConsistencyEngine

- **Purpose:** detect profile contradictions and impossible profiles; produce `IntegrityFinding`s with evidence and a raw honeypot-risk estimate. (Offline O3 it *calibrates* the thresholds; online R4 it *applies* them.)
- **Inputs:** `CareerProfile`, `RawCandidate`, `LogisticsProfile` (salary), `education`/`skills`, behavioral `RawSignals`; calibrated thresholds (`integrity_thresholds.json`).
- **Outputs:** `tuple[IntegrityFinding, ...]` (each with `IntegrityFlag` + `EvidenceRef`s) and a `raw_honeypot_risk: UnitScore`. (Final `IntegrityReport` is assembled by `CandidateRiskEngine`.)
- **Dependencies:** `domain.candidate.integrity`, `domain.provenance`; thresholds via injected config/artifact.
- **Internal workflow:** run deterministic rule set — `TENURE_EXCEEDS_EXPERIENCE`, `ROLE_DURATION_DATE_MISMATCH`, `CURRENT_ROLE_HAS_END_DATE`, `EXPERT_SKILL_ZERO_USAGE`, `EDUCATION_TIMELINE_IMPOSSIBLE`, `EXPERIENCE_PREDATES_PLAUSIBLE_START`, `ASSESSMENT_FOR_ABSENT_SKILL`; **signal-consistency** checks (inverted salary → logistics sanity flag, not honeypot); aggregate fired rules into a raw risk score.
- **Failure modes:** missing calibration (`ArtifactContractError`, fatal at R0); no rules fire → empty findings (valid).
- **Determinism:** pure; findings sorted by `code`; thresholds fixed by artifact.
- **Threading:** stateless; data-parallel.
- **Performance:** vectorized over the population where possible; ~15–25 s for 100K.
- **Memory:** per-candidate; columnar in bulk.
- **Auditability:** emits per-`IntegrityFlag` fire counts and the honeypot-risk distribution digest; every finding carries evidence.
- **Testability:** honeypot fixtures (spec §7 exemplars) must fire the correct flags; property test that benign profiles fire none.

## §4. CandidateEmbeddingEngine

- **Purpose:** generate dense vectors. Offline (O5/O6) it builds candidate + anchor vector artifacts via sentence-transformers; online (R3) it serves as the **fallback** encoder for candidates absent from the store. Batching, caching, artifact integration.
- **Inputs:** composed embedding documents (from Normalization); offline: the full candidate stream + authored anchors.
- **Outputs:** offline → `candidate_vectors.parquet`, `anchor_vectors.npy`, exported `encoder.onnx` (written by the offline stage, not the engine directly); online → `FloatVector`s for miss ids.
- **Dependencies:** `EmbeddingModelPort` (st offline / onnx online); `ArtifactStorePort` (load encoder online).
- **Internal workflow:** batch documents; call `EmbeddingModelPort.encode`; assert unit-norm + dim; (offline) tile into the parquet/npy layout keyed by `CandidateId`; (online) cache per-`source_index` within the run to avoid re-encode.
- **Failure modes:** `EmbeddingError` (encode failure); dim/`model_id` mismatch vs manifest (`ArtifactContractError`); **network access is forbidden online** (onnx adapter must be offline-capable).
- **Determinism:** within a runtime, identical doc ⇒ identical vector; cross-runtime (st↔onnx) cosine within ε (Ports §9), not bitwise.
- **Threading:** one session per instance; batched encode; thread/op counts pinned for determinism.
- **Performance:** offline unbounded; online fallback rare (misses ≈ 0 in full reproduction).
- **Memory:** batch-bounded; offline writes stream to disk; online cache holds only miss vectors.
- **Auditability:** emits `encode_calls, encoded_docs, fallback_miss_count, encode_ms, model_id, dim`.
- **Testability:** contract-tested against `StubEmbeddingModel`; norm/shape/order asserted; offline/online document-composition parity test.

## §5. CandidateRetrievalEngine

- **Purpose:** the R3 semantic stage — anchor similarity, hybrid (dense + lexical) fit, archetype assignment, and recall optimization (two-tier shrink before heavy scoring).
- **Inputs:** candidate vectors (`SemanticVectorStorePort.view_all`/`get_many`); JD `anchor_vectors`; `centroids.npy`; the compiled lexicon (hybrid term match); `JobDescriptionSpec`.
- **Outputs:** `SemanticProfile` (anchor similarities, positive/negative fit, `net_semantic_fit`, `vector_ref`) and `ArchetypeAssignment` per candidate.
- **Dependencies:** `SemanticVectorStorePort`, `EmbeddingModelPort` (fallback); pure numpy for cosine/nearest-centroid; lexicon (artifact).
- **Internal workflow:** gather vectors (lookup-first; encode misses); cosine vs positive/negative anchors (vectorized matmul); compute `net_semantic_fit`; hybrid blend with lexical match score (anti keyword-stuffing — dense negative anchors + lexical trust); nearest-centroid archetype (ties by `ArchetypeId` asc); **recall tier:** mark candidates passing a coarse fit floor as scoring-eligible to shrink the heavy path.
- **Failure modes:** `VectorStoreError` (corrupt); anchor set ⊄ artifact (`ArtifactContractError`); store dim ≠ anchor dim (fatal).
- **Determinism:** fixed reduction/argmax order; ties by id; numpy float32 with pinned BLAS threads (thread-count-invariant output).
- **Threading:** stateless; vectorized single-threaded matmul (BLAS pinned) or data-parallel shards with deterministic concat.
- **Performance:** 100K × (few anchors + K centroids) matmul is seconds; full R3 target ~30–60 s including the rare fallback.
- **Memory:** mmap matrix (≈150 MB resident on touched pages); anchor/centroid matrices tiny.
- **Auditability:** emits `lookup_hits, fallback_misses, mean_positive_fit, mean_negative_fit, archetype_histogram, recall_tier_survivors`.
- **Testability:** fixed vectors → fixed similarities (golden); nearest-centroid tie determinism; hybrid-blend monotonicity property.

## §6. CandidateFeatureEngine

- **Purpose:** structured feature extraction into the canonical, **versioned** feature set: produces `CareerProfile`, structural `CredibilityProfile`, `LogisticsProfile`, the candidate's CQV feature row, and a `FeatureManifest` describing the layout/version.
- **Inputs:** `NormalizedCandidate` (career, education, skills, signals 12–15 for logistics); compiled lexicon (JD-relevant skill mapping); `FeatureLayout` constant.
- **Outputs:** `CareerProfile`, `CredibilityProfile` (skill-trust, keyword-stuffing, title-coherence), `LogisticsProfile`, a `(D,)` CQV feature row aligned to `FeatureLayout`, `FeatureManifest(layout_version, feature_groups)`.
- **Dependencies:** `features.career/skills/education/signals`; lexicon (artifact); `domain.candidate.quality.FeatureLayout`.
- **Internal workflow:** derive tenures/recency (`as_of`), product-vs-services + consulting classification, skill-trust = endorsement×duration×assessment coherence, keyword-stuffing score, logistics fit inputs; emit features into fixed-index groups (`career_*, skill_*, semantic_*, credibility_*, logistics_*, behavioral_*` — semantic/behavioral filled by their engines, this engine owns the structural groups); validate no NaN, ranges.
- **Failure modes:** `CQVInvariantError` (NaN/out-of-range/dim); layout-version mismatch (`ArtifactContractError`).
- **Determinism:** fixed `FeatureLayout` order; `as_of`-relative only; float32.
- **Threading:** stateless; data-parallel; bulk path writes into a shared `(N,D)` matrix at deterministic row indices.
- **Performance:** ~30–45 s for 100K (the heaviest pure-CPU stage).
- **Memory:** bulk `(N,D)` float32 matrix (≈ 100K×D×4); per-candidate objects only for survivors/top-K.
- **Auditability:** emits `feature_rows, nan_rejects, feature_group_coverage, layout_version`; the `FeatureManifest` is reported.
- **Testability:** golden feature rows for fixtures; layout-version pin; group-coverage invariant; NaN-rejection test.

## §7. CandidateBehaviorEngine

- **Purpose:** interpret the 23 Redrob signals into bounded availability / responsiveness / engagement / reliability / verification components → the behavioral **multiplier** inputs (per `redrob_signals_doc`).
- **Inputs:** `RawSignals` (single-ownership routing: signals 3–8,10,16–23 here; 9,11 → Credibility; 12–15 → Logistics); injected `as_of`.
- **Outputs:** `BehavioralProfile` (normalized component scores + `SignalAvailability` per family + retained raw for evidence).
- **Dependencies:** `features.signals`; `DeterministicEntropyPort.as_of`; `domain.candidate.behavioral`.
- **Internal workflow:** recency from `last_active_date` vs `as_of` (stale ⇒ low availability); responsiveness from response-rate + inverse response-time; engagement from views/searches/saves/applications/connections (bounded log-scale); reliability from interview/offer rates with **sentinel −1/{} ⇒ UNKNOWN, never 0**; verification from email/phone/linkedin/completeness. Recruiter-attractiveness composite from saves + views + search appearances. The final multiplier is computed here as bounded `Multiplier` inputs; the policy bounds live in config.
- **Failure modes:** missing `as_of` (config error); out-of-range rate (defensive `SchemaError`).
- **Determinism:** `as_of`-driven, no clock; pure.
- **Threading:** stateless; data-parallel.
- **Performance:** ~10 s for 100K.
- **Memory:** per-candidate; bulk columnar.
- **Auditability:** emits per-family score distributions, `unknown_github_count, unknown_offer_count, stale_active_count`.
- **Testability:** sentinel-handling unit tests (−1/{} ⇒ UNKNOWN); stale-recency boundary tests; multiplier-bounds property.

## §8. CandidateFitEngine

- **Purpose:** JD alignment + the explicit detectors, producing the `EligibilityReport` (hard blocks + soft penalties) and JD-fit score components, including logistics fit.
- **Inputs:** `CareerProfile`, `CredibilityProfile`, `SemanticProfile`, `LogisticsProfile`, `JobDescriptionSpec`; gate predicates (`configs/gates`); lexicon.
- **Outputs:** `EligibilityReport`; fit `ScoreComponentValue` inputs (`CAREER_FIT`, `EXPERIENCE_FIT`, `EDUCATION_FIT`, plus logistics fit multiplier inputs).
- **Dependencies:** `domain.candidate.eligibility`, `domain.jd`; gate config; lexicon.
- **Internal workflow:** detectors — **retrieval/ranking-experience** (career descriptions + skills + semantic), **production-ML-experience**, **consulting-only** (`CONSULTING_FIRMS_ONLY_CAREER`), **title-chaser** (sub-18m hops), **product-company** classification; map to HARD blocks (`PURE_RESEARCH_NO_PRODUCTION`, `LANGCHAIN_OPENAI_ONLY_RECENT`, `NO_PRODUCTION_CODE_18M`, `CONSULTING_FIRMS_ONLY_CAREER`, `PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP`, `CLOSED_SOURCE_5Y_NO_VALIDATION`) and SOFT penalties (`TITLE_CHASER_SUB_18M_HOPS`, `NOTICE_OVER_30`, `OUTSIDE_INDIA_NO_SPONSOR`, `OUTSIDE_EXPERIENCE_BAND`); logistics fit (location hub set, notice fit, work-mode).
- **Failure modes:** missing gate config (`ArtifactContractError`); JD anchor/skill reference missing (fatal).
- **Determinism:** pure; findings sorted by `code`; predicates are data.
- **Threading:** stateless; data-parallel.
- **Performance:** ~15 s for 100K.
- **Memory:** per-candidate; columnar.
- **Auditability:** emits per-`EligibilityCode` counts; `hard_block_rate, soft_penalty_rate`; every finding carries evidence.
- **Testability:** the JD-trap candidates (keyword-stuffer "Marketing Manager", consulting-only, CV-only) must hit the right codes; eligible-candidate negatives must pass.

## §9. CandidateRiskEngine

- **Purpose:** finalize the `IntegrityReport` and compute contradiction/uncertainty/**confidence** as the modifier on relevance.
- **Inputs:** `IntegrityFinding[]` + raw honeypot risk (from Consistency); `SignalAvailability` flags (from Behavior); eligibility soft penalties; `claimed_vs_assessed_gap`.
- **Outputs:** final `IntegrityReport(findings, honeypot_score, is_honeypot, rules_evaluated)`; an engine-internal `confidence: UnitScore` and `uncertainty` (see freeze note) that feeds Scoring.
- **Dependencies:** `domain.candidate.integrity`; calibrated `honeypot_threshold`.
- **Internal workflow:** aggregate findings → `honeypot_score`; `is_honeypot = any HARD or score ≥ threshold`; **contradiction score** from title-vs-description + claimed-vs-assessed gaps; **uncertainty** rises with UNKNOWN signal families and sparse evidence; **confidence = 1 − uncertainty** (bounded). Confidence is passed to Scoring as a shrink-toward-prior factor; **not persisted** as a domain field.
- **Failure modes:** missing threshold (fatal); contradictory inputs handled deterministically.
- **Determinism:** pure; fixed aggregation order.
- **Threading:** stateless; data-parallel.
- **Performance:** ~5–10 s for 100K.
- **Memory:** per-candidate; columnar.
- **Auditability:** emits `honeypot_count, honeypot_rate, mean_contradiction, mean_uncertainty, confidence_histogram`.
- **Testability:** honeypot-threshold boundary; high-UNKNOWN ⇒ low-confidence property; gating-consistency with Consistency outputs.

## §10. CandidateScoringEngine

- **Purpose:** fold the CQV, combine weighted components, apply gates and multipliers, fuse confidence, and emit a `ScoredCandidate` with a full `ScoreBreakdown`.
- **Inputs:** complete `CandidateRepresentation` (all slices); `ScoringWeights` (locked artifact); confidence/multiplier inputs.
- **Outputs:** folded `CandidateQualityVector`; `ScoredCandidate(final_score, breakdown, tiebreak_key=candidate_id)`.
- **Dependencies:** `domain.scoring`, `domain.candidate.quality`; `ArtifactStorePort` (weights, via pipeline at R0); pure numpy.
- **Internal workflow:** assemble CQV (`CQVAssembler` semantics); `base_relevance = Σ component.weighted` (summed in `ScoreComponent` order); **gate:** honeypot or ineligible ⇒ `final_score = FLOOR`; else `final_score = base × behavioral_multiplier × logistics_multiplier + archetype_adjustment`, then **confidence fusion** (shrink toward prior when confidence low); calibration/normalization to a stable scale (monotonic; preserves order). Build `ScoreBreakdown` with per-component evidence.
- **Failure modes:** weights/layout-version mismatch (`ArtifactContractError`); `ScoreInvariantError` (gating/formula contradiction); NaN in CQV (`CQVInvariantError`).
- **Determinism:** fixed summation/reduction order; float32; calibration is a fixed monotone transform; no RNG.
- **Threading:** stateless; bulk path is a single `(N,D)·w` matvec + elementwise multipliers (BLAS pinned, thread-invariant).
- **Performance:** matvec over 100K is ~seconds; full R5 ~15–20 s.
- **Memory:** operates on the bulk `(N,D)` matrix; per-candidate `ScoredCandidate` objects materialized for survivors/top-K only.
- **Auditability:** emits `score_distribution_digest, floored_count, mean_multiplier, component_contribution_means`; breakdown is fully reconstructable.
- **Testability:** gating invariant (honeypot/ineligible ⇒ FLOOR); `base == Σ weighted` property; calibration monotonicity; golden scores for fixtures.

## §11. CandidateRankingEngine

- **Purpose:** the R6 stage — deterministic sort, tie-break, top-K extraction into a spec-valid `Ranking`.
- **Inputs:** `Sequence[ScoredCandidate]`; `size` (default 100).
- **Outputs:** `Ranking` (invariant-checked at construction).
- **Dependencies:** `domain.ranking`.
- **Internal workflow:** sort by `(−final_score, candidate_id)`; take top `size`; assign ranks `1..size`; construct `Ranking` (the domain factory enforces count/unique-rank/monotonicity/tie-break and raises on violation).
- **Failure modes:** `RankingInvariantError` (should be impossible given valid scores — defence); fewer than `size` candidates (config/data error).
- **Determinism:** total order `(−score, id)`; ties strictly by id ascending; identical inputs ⇒ identical ranking.
- **Threading:** single-threaded stable sort (the only correct-by-construction option); trivial cost.
- **Performance:** sort 100K + slice 100 ≈ ms.
- **Memory:** holds the top-`size`; the rest is discarded.
- **Auditability:** emits `cutoff_score (rank-100), top1_score, top10_min_score, tie_groups_count`.
- **Testability:** property tests for all six `Ranking` invariants; tie-break determinism; equal-score id-ordering.

## §12. CandidateReasoningEngine

- **Purpose:** the R7 stage — produce evidence-grounded, Stage-4-survivable reasoning for each ranked candidate, plus an audit trace.
- **Inputs:** `RankedCandidate` + the **re-hydrated** top-K `CandidateRepresentation` (with `ProvenanceHandle.inline`); `ScoreBreakdown`.
- **Outputs:** `CandidateReasoning` (clauses + rendered ≤2 sentences); `Ranking.with_reasoning(...)`; an audit trace mapping each clause → `EvidenceRef`.
- **Dependencies:** `domain.reasoning`, `domain.provenance`.
- **Internal workflow:** select strengths/concerns from the breakdown and eligibility findings; **every clause requires ≥1 `EvidenceRef`** (no-hallucination guarantee enforced at construction); ensure ≥1 JD link; render with tone matching the rank band; variation arises from differing evidence, **not** templates or name-insertion.
- **Failure modes:** `ProvenanceError` (dangling evidence) — fatal for that clause; missing top-K hydration (pipeline error).
- **Determinism:** rendered text is a pure function of the ordered clause set; identical evidence ⇒ identical text.
- **Threading:** stateless; runs over the top-K only (small); data-parallel safe.
- **Performance:** top-K (≤ a few hundred) ⇒ milliseconds.
- **Memory:** holds only top-K representations + raw.
- **Auditability:** the audit trace **is** the deliverable; emits `clauses_per_candidate, concern_coverage, jd_link_coverage`.
- **Testability:** Stage-4 checks as tests — specific-facts, JD-connection, honest-concerns, no-hallucination, variation, rank-consistency; templated-output detector must flag synthetic templated input.

## §13. SubmissionGenerationEngine

- **Purpose:** the R8 stage — emit the validator-compliant CSV, integrate validation, and record submission metadata.
- **Inputs:** `Ranking` (with reasoning).
- **Outputs:** `submission.csv` (via `SubmissionSinkPort`); a `ValidationReport`; a `SubmissionReceipt` (row count, output sha256).
- **Dependencies:** `SubmissionSinkPort`; `domain.validation`; the `ValidationEngine` logic (structural + reasoning checks).
- **Internal workflow:** run `ValidationEngine` over the `Ranking` (defence-in-depth, mirrors `validate_submission.py` + Stage-4 reasoning checks); on pass, `SubmissionSinkPort.write(ranking)` (atomic, byte-stable, RFC-4180); capture the receipt.
- **Failure modes:** `SubmissionContractError`/`ValidationReport.is_valid == False` (abort — never emit a rejectable file); `SubmissionWriteError` (IO).
- **Determinism:** byte-for-byte reproducible CSV; deterministic score formatting; stable output sha256.
- **Threading:** single writer.
- **Performance:** trivial (100 rows).
- **Memory:** negligible.
- **Auditability:** emits `validation_findings, is_valid, output_sha256, row_count`.
- **Testability:** emitted CSV passes the organizer's `validate_submission.py`; byte-equality golden; reasoning-with-special-chars round-trip.

## §14. RunReportEngine

- **Purpose:** assemble the audit + reproducibility + metrics + artifact report (R9 online; O10 build report offline).
- **Inputs:** per-stage metrics from every engine; `ArtifactStorePort.manifest()` (hashes); timings + budget from the timing guard; `SubmissionReceipt`; config hash; input file hash.
- **Outputs:** a `RunReport` (the port-owned structural contract) written via `RunReportSinkPort`.
- **Dependencies:** `RunReportSinkPort`, `ArtifactStorePort`; `observability` metric collectors (the report object conforms to the port's `RunReport` Protocol).
- **Internal workflow:** populate `reproducible` (code/config/manifest/artifact/input/output hashes, candidate count, honeypot rate, eligibility summary, score digest), `audit` (run_id, wall-clock), `timings` (R0–R9 ms), `budget` (used/limit, within_budget, peak_rss). Write deterministically.
- **Failure modes:** `ReportWriteError`; missing manifest (fatal upstream at R0).
- **Determinism:** the `reproducible` block is deterministic; `audit` excluded from any repro hash.
- **Threading:** single writer.
- **Performance:** trivial.
- **Memory:** negligible.
- **Auditability:** the report **is** the audit artifact; ties a submission to exact artifacts/config/code/input/output.
- **Testability:** `reproducible` block byte-stable across identical runs; `audit` ignored by determinism assertions.

---

## §15. Engine Dependency Graph (data-dependency DAG — no direct imports)

```
Ingestion ─▶ Normalization ─▶ ┬─▶ Feature ──────────────┐
                              ├─▶ Behavior ──────────────┤
                              └─▶ Embedding ─▶ Retrieval ─┤
                                                          ▼
                              Consistency ──▶ Risk ◀───── (Behavior signals)
                              Feature/Retrieval/Behavior ─▶ Fit ─▶ Risk
                                                          ▼
                                                       Scoring ─▶ Ranking ─▶ Reasoning
                                                                                 ▼
                                                              SubmissionGeneration ─▶ RunReport
```
The arrows are **data** dependencies realized by the pipeline; engines do not import one another. Branches after Normalization (Feature ∥ Behavior ∥ Embedding→Retrieval) are independent and may run concurrently.

## §16. Engine Execution Graph (parallelizable phases)

```
Phase A (stream):    Ingestion → Normalization                    [pipelined, O(1) memory]
Phase B (fan-out):   { Feature | Behavior | Embedding→Retrieval } [data-parallel, deterministic merge]
Phase C (gates):     Consistency → (with Behavior/Fit) → Risk     [Fit ∥ Consistency, then Risk joins]
Phase D (rank):      Scoring → Ranking                            [bulk matvec, then single sort]
Phase E (explain):   Reasoning (top-K) → SubmissionGeneration     [tiny]
Phase F (report):    RunReport
```
Determinism rule across all phases: **thread-count must not change output.** Any data-parallel shard merges by `source_index`/`candidate_id` order, and float reductions follow `FeatureLayout`/`ScoreComponent` order.

## §17. Offline Build Flow (engine view)

```
O1 Census          Ingestion(stream) → population stats
O2 Feature Extract Ingestion → Normalization → Feature → feature tables
O3 Integrity Calib Consistency(calibration mode) → integrity_thresholds.json
O4 Lexicon Disc.   (features + seed) → lexicon.compiled.json
O5 Embedding Gen   Ingestion → Normalization → Embedding(st) → candidate_vectors + encoder.onnx
O6 Anchor Author   Embedding(st, anchors) → anchor_vectors.npy
O7 Archetype Disc. (vectors + Entropy.rng) → centroids.npy
O8 Weight Search   (features + golden + Entropy.rng) → scoring_weights.locked.yaml
O9 Validation Bat. SubmissionGeneration(dry-run on golden) + ArtifactStore.verify_all
O10 Packaging      RunReport(build) + manifest hashing
```

## §18. Online Ranking Flow (engine view)

```
R0 Load     (pipeline: ArtifactStore.verify, ports bound)
R1 Ingest   Ingestion
R2 Features  Normalization → { Feature | Behavior }
R3 Semantic  Embedding(fallback) → Retrieval
R4 Gates     Consistency → Fit → Risk
R5 Score     Scoring
R6 Rank      Ranking
R7 Reason    Reasoning (top-K)
R8 Submit    SubmissionGeneration
R9 Report    RunReport
```

## §19. Candidate Lifecycle Diagram (one candidate's state)

```
SourceRecord
   └─Ingestion→ RawCandidate + Identity        (BuildStage: PARSED)
   └─Normalization→ NormalizedCandidate
   └─Feature/Behavior→ Career/Credibility/Logistics/Behavioral   (FEATURED)
   └─Retrieval→ Semantic/Archetype                                (SITUATED)
   └─Consistency/Fit/Risk→ Integrity/Eligibility                  (GATED)
        ├─ is_honeypot or ineligible → score FLOOR (filler tail)
        └─ survivor →
   └─Scoring→ CQV + ScoredCandidate                               (VECTORIZED→SCORED)
   └─Ranking→ RankedCandidate (if top-100)                        (RANKED)
   └─Reasoning→ CandidateReasoning                                (EXPLAINED)
   └─Submission→ CSV row
```

## §20. Score Lifecycle Diagram

```
CQV features ─▶ Σ(component·weight) = base_relevance
                        │
   gate(integrity, eligibility): honeypot/ineligible ⇒ final = FLOOR
                        │ (survivor)
   × behavioral_multiplier × logistics_multiplier + archetype_adjustment
                        │
   confidence fusion (shrink toward prior when uncertainty high)   [engine-internal]
                        │
   calibration/normalization (monotone, order-preserving)
                        ▼
                  final_score = Score
                        ▼
   Ranking: sort (−score, id) → rank 1..100  (ties by id ascending)
```

## §21. Failure Recovery Strategy

| Failure class | Where | Strategy |
|---|---|---|
| Artifact integrity (hash/version/coherence) | R0 / `ArtifactStore` | **Fail-fast, abort.** No degraded run (no live leaderboard to catch silent loss). |
| Malformed input line | R1 / Ingestion | Tagged record; pipeline policy (default full-run: abort; sandbox: skip+log). |
| Schema-invalid candidate | R1 / parsing | `SchemaError`; policy-driven skip/abort; counted in metrics. |
| Vector store miss | R3 / Retrieval | **Recover:** encode fallback via `EmbeddingModelPort`. Not an error. |
| Embedding runtime failure | R3 | `EmbeddingError`; if a miss can't be encoded, that candidate is gated out (recorded), run continues. |
| Honeypot / ineligible | R4 | **Not a failure** — score FLOOR, candidate stays in the population as filler. |
| Score/Ranking invariant violation | R5/R6 | `ScoreInvariantError`/`RankingInvariantError` — abort (indicates a bug). |
| Submission would be invalid | R8 | Abort before write; never emit a rejectable file. |
| Budget exceeded | timing guard | Recorded in RunReport (`within_budget=false`); surfaced loudly in CI before submission. |

## §22. Deterministic Reproducibility Strategy

- **Inputs pinned:** `seed`, `as_of`, config hash, manifest hash, input file hash — all recorded in the RunReport `reproducible` block.
- **No nondeterministic sources:** no wall clock in logic, no online RNG (`OnlineEntropy` raises on RNG use), ties by id.
- **Float determinism:** float32 everywhere; reductions in `FeatureLayout`/`ScoreComponent` order; BLAS/OMP threads pinned so output is **thread-count-invariant** (asserted by determinism tests running 1-thread vs N-thread).
- **Stable iteration:** `Mapping`s consumed via sorted keys; findings/clauses/components pre-sorted by code/index.
- **Byte-stable outputs:** submission CSV and the RunReport `reproducible` block are byte-for-byte reproducible; their sha256s are recorded and diffed in CI.

## §23. Engine Ownership Matrix

| Engine | Owning concern | Purity | Stage | Ports |
|---|---|---|---|---|
| Ingestion | input typing | port-pure | R1/O-read | CandidateSource |
| Normalization | canonicalization | pure | R2/O2 | — |
| Consistency | contradiction detection | pure (calib) | R4/O3 | ArtifactStore |
| Embedding | text→vector | port-pure | R3/O5,O6 | EmbeddingModel, ArtifactStore |
| Retrieval | semantic/hybrid fit | port-pure | R3 | SemanticVectorStore, EmbeddingModel |
| Feature | structured features/CQV | pure | R2/O2 | ArtifactStore (lexicon) |
| Behavior | Redrob signals | pure | R2 | Entropy(`as_of`) |
| Fit | JD alignment/eligibility | pure | R4 | ArtifactStore (gates) |
| Risk | honeypot/confidence | pure | R4→R5 | — |
| Scoring | aggregation/calibration | pure | R5 | ArtifactStore (weights) |
| Ranking | sort/tie-break/top-k | pure | R6 | — |
| Reasoning | evidence/explanation | pure | R7 | — |
| SubmissionGeneration | CSV/validation | port-pure | R8/O9 | SubmissionSink |
| RunReport | audit/repro | port-pure | R9/O10 | RunReportSink, ArtifactStore |

## §24. Data Contract Matrix

| Engine | Consumes | Produces |
|---|---|---|
| Ingestion | `SourceRecord` | `RawCandidate`, `Identity` |
| Normalization | `RawCandidate` | `NormalizedCandidate` (intermediate) |
| Consistency | `CareerProfile`, `RawCandidate`, calibration | `IntegrityFinding[]`, raw risk |
| Embedding | embedding docs | candidate/anchor vectors, fallback `FloatVector` |
| Retrieval | vectors, anchors, centroids, lexicon | `SemanticProfile`, `ArchetypeAssignment` |
| Feature | `NormalizedCandidate`, lexicon, `FeatureLayout` | `CareerProfile`, `CredibilityProfile`, `LogisticsProfile`, CQV row, `FeatureManifest` |
| Behavior | `RawSignals`, `as_of` | `BehavioralProfile` |
| Fit | Career/Credibility/Semantic/Logistics, JD, gates | `EligibilityReport`, fit components |
| Risk | findings, SignalAvailability, gaps | `IntegrityReport`, confidence (internal) |
| Scoring | full representation, weights, confidence | CQV (folded), `ScoredCandidate` |
| Ranking | `ScoredCandidate[]` | `Ranking` |
| Reasoning | `RankedCandidate`, top-K rep | `CandidateReasoning`, `Ranking.with_reasoning` |
| SubmissionGeneration | `Ranking` | `submission.csv`, `ValidationReport`, `SubmissionReceipt` |
| RunReport | metrics, manifest, timings, receipt | `RunReport` |

## §25. Engine State Machine Definitions

- **Candidate aggregate (global):** `PARSED → FEATURED → SITUATED → GATED → VECTORIZED → SCORED → RANKED → EXPLAINED` (monotonic `BuildStage`; each engine advances it; regression raises `RepresentationStageError`).
- **Embedding (per batch):** `PENDING → BATCHED → ENCODED → {CACHED | LOOKED_UP}` (online: `LOOKED_UP` is the store hit; `ENCODED` only on miss).
- **Retrieval (per candidate):** `GATHERED → SCORED_VS_ANCHORS → ARCHETYPED → {RECALL_KEPT | RECALL_DROPPED}`.
- **Risk (per candidate):** `FINDINGS_IN → AGGREGATED → {HONEYPOT | CLEAN} → CONFIDENCE_SET`.
- **Submission (run-level):** `RANKING_IN → VALIDATED → {WRITTEN | ABORTED}` (no partial files).
- **RunReport (run-level):** `COLLECTING → ASSEMBLED → WRITTEN`.

## §26. Metrics emitted by every engine

| Engine | Key metrics |
|---|---|
| Ingestion | `rows_read, ok, malformed, schema_reject, duplicate_id, parse_ms` |
| Normalization | `normalized, unknown_skill_tokens, date_repairs, ms` |
| Consistency | per-`IntegrityFlag` fire counts, `raw_risk_digest, ms` |
| Embedding | `encode_calls, encoded_docs, fallback_misses, model_id, dim, ms` |
| Retrieval | `lookup_hits, fallback_misses, mean_pos_fit, mean_neg_fit, archetype_histogram, recall_survivors, ms` |
| Feature | `feature_rows, nan_rejects, group_coverage, layout_version, ms` |
| Behavior | `unknown_github, unknown_offer, stale_active, family_score_digests, ms` |
| Fit | per-`EligibilityCode` counts, `hard_block_rate, soft_penalty_rate, ms` |
| Risk | `honeypot_count, honeypot_rate, mean_contradiction, mean_uncertainty, confidence_histogram, ms` |
| Scoring | `score_digest, floored_count, mean_multiplier, component_means, ms` |
| Ranking | `cutoff_score, top1, top10_min, tie_groups, ms` |
| Reasoning | `clauses_per_candidate, concern_coverage, jd_link_coverage, ms` |
| SubmissionGeneration | `validation_findings, is_valid, output_sha256, row_count, ms` |
| RunReport | `manifest_hash, within_budget, used_seconds, peak_rss_mb` |

All metrics flow to `RunReportEngine`; timing metrics feed the budget guard.

## §27. Sequence diagrams (engine-level)

**Offline O1–O10**
```
O1  Ingestion.stream ───────────────▶ census tally
O2  Ingestion ▶ Normalization ▶ Feature ──▶ feature tables (fs)
O3  Consistency(calibrate) ─────────▶ integrity_thresholds.json
O4  (features+seed) ────────────────▶ lexicon.compiled.json
O5  Ingestion ▶ Normalization ▶ Embedding(st) ─▶ candidate_vectors.parquet + encoder.onnx
O6  Embedding(st, anchors) ─────────▶ anchor_vectors.npy
O7  (vectors)+Entropy.rng("kmeans")─▶ centroids.npy
O8  (features+golden)+Entropy.rng ──▶ scoring_weights.locked.yaml
O9  SubmissionGeneration(golden dry-run); ArtifactStore.verify_all
O10 RunReport(build) ▶ manifest hashing ▶ MANIFEST.json
```

**Online R0–R9**
```
R0  ArtifactStore.manifest()/verify_all(); bind ports; OnlineEntropy(as_of)
R1  Ingestion(CandidateSource.stream) ─▶ RawCandidate + Identity
R2  Normalization ─▶ { Feature | Behavior }  ─▶ Career/Credibility/Logistics/Behavioral
R3  Embedding(onnx fallback) ▶ Retrieval(VectorStore.view_all/get_many) ─▶ Semantic/Archetype
R4  Consistency ▶ Fit ▶ Risk ─▶ Integrity/Eligibility (+confidence)
R5  Scoring(weights) ─▶ ScoredCandidate[]
R6  Ranking ─▶ Ranking
R7  Reasoning(top-K, re-hydrated raw) ─▶ Ranking.with_reasoning
R8  SubmissionGeneration ▶ SubmissionSink.write ─▶ submission.csv + receipt
R9  RunReport ▶ RunReportSink.write ─▶ run_report.json
```

Note: ports appear only at R0/R1/R3/R8/R9; R2/R4/R5/R6/R7 are pure engine work with no port calls — the property that keeps the hot path testable and the 5-minute budget predictable.

---

## Build order for engine implementation

1. **Pure, port-free, leaf engines first** (testable with domain fixtures only): Normalization, Feature, Behavior, Consistency, Fit, Risk, Scoring, Ranking, Reasoning.
2. **Port-bound engines** against their fakes (Ports §16/§17): Ingestion, Embedding, Retrieval, SubmissionGeneration, RunReport.
3. Each engine merged only when (a) its unit/property/golden tests pass and (b) it advances `BuildStage` correctly in an integration test over `sample_candidates`.
4. Wire the execution graph (§16) in `pipelines/online` and `pipelines/offline` **last** — the composition roots, the only place ports meet engines.
