# REDSTACK v1.1 — Feature Layer Specification

**Scope:** `src/redstack/features/` only. Architecture, Domain, Ports, and Engine layers are frozen; this document does not alter them. No implementation code — architecture detailed enough that feature implementation can begin immediately.

**Feature Layer rules (inherited, restated as law):**
- `features/` is **pure**: imports only `domain/`, `config.schema`, stdlib, and numpy. **No ports, no IO, no ML runtime, no clock, no RNG.** Recency uses the injected `as_of`. Semantic-similarity *values* and behavioral *raw signals* arrive as already-resolved inputs from the engines that own the ports; this layer never calls `EmbeddingModelPort` or `SemanticVectorStorePort` itself.
- This layer **owns** the `FeatureLayout` constant (the ordered CQV schema referenced by the frozen `domain.candidate.quality`), the feature **definitions**, the pure **extractors**, and the feature **store metadata**. `CandidateFeatureEngine` orchestrates the extractors; `CandidateScoringEngine` consumes the CQV; the offline O8 weight search binds to the same layout.

**Three principles thread through every part:**
1. **Keyword-stuffing resistance.** A feature value is an *evidence aggregate*, never a keyword flag. Claimed skill → weak; claimed skill + appears in role descriptions + endorsement×duration trust + assessment coherence + dense anchor match → strong. Stuffers score near zero on competency features by construction.
2. **Determinism & reproducibility.** Every feature is a pure function of `(RawCandidate, JobDescriptionSpec, artifacts, as_of)`. Versioned, fixed layout order, float32, no nondeterministic source.
3. **Explainability by construction.** Every feature cell carries `EvidenceRef`s back to the raw fields that produced it. Reasoning may only cite features that have evidence — Stage-4 "no hallucination" becomes mechanical.

**Feature cell model.** Each feature emits a `FeatureCell = (value: float, confidence: UnitScore, evidence: tuple[EvidenceRef, ...])`. The bulk path stores `value` in the `(N, D)` CQV matrix and `confidence` at **group granularity** `(N, num_groups)`; full per-feature `confidence` + `evidence` are materialized only for gate survivors / top-K (memory, see Part 7).

**Feature ID convention.** `"<group>.<name>"` (e.g. `retr.built_retrieval_system`, `bhv.availability`, `hp.timeline_impossible`). `FeatureLayout` maps each id → fixed `FeatureIndex`.

---

# PART 1 — FEATURE TAXONOMY

Thirty groups. Each block: **purpose · inputs · outputs · extraction · failure modes · confidence · explainability.** All outputs are `FeatureCell`s unless noted; values are `UnitScore [0,1]` unless a range is given.

### 1. Identity (`id.*`)
- **Purpose:** identification + provenance anchor; not a scored signal.
- **Inputs:** `Identity`, `RawCandidate.candidate_id`.
- **Outputs:** `id.is_valid_id` (flag), metadata only (no CQV weight).
- **Extraction:** pattern check; dedup key.
- **Failure modes:** malformed id → upstream `SchemaError` (never reaches here).
- **Confidence:** 1.0 (structural).
- **Explainability:** evidence = the id field.

### 2. Geography (`geo.*`)
- **Purpose:** location fit vs JD hubs (Pune/Noida/Hyderabad/Mumbai/Delhi-NCR; outside-India no-sponsor).
- **Inputs:** `LogisticsProfile.location/country`, `willing_to_relocate`, JD hub set.
- **Outputs:** `geo.hub_match`, `geo.india_relocatable`, `geo.outside_india_no_sponsor` (neg).
- **Extraction:** map location→hub via canonical city table; relocation widens eligible set.
- **Failure modes:** unknown city → `UNKNOWN` (low confidence), never hard-block.
- **Confidence:** lower when city unrecognized.
- **Explainability:** evidence = location, country, relocate flag.

### 3. Experience (`exp.*`)
- **Purpose:** experience band fit (JD 5–9 "range not requirement").
- **Inputs:** `CareerProfile.stated/derived_experience_years`.
- **Outputs:** `exp.years`, `exp.in_band` (5–9 soft), `exp.derived_vs_stated_gap`.
- **Extraction:** band membership with soft shoulders; gap = |stated−derived|.
- **Failure modes:** stated≪derived feeds honeypot (`exp` group cross-links `hp.experience_inflation`).
- **Confidence:** lower when gap large.
- **Explainability:** evidence = years_of_experience + summed durations.

### 4. Seniority (`sen.*`)
- **Purpose:** senior-engineer judgment proxy (not title).
- **Inputs:** career titles, scope language, tenure, leadership signals.
- **Outputs:** `sen.level` (0..1), `sen.title_vs_scope_gap`.
- **Extraction:** scope/ownership cues in descriptions × tenure; **title alone is discounted** (title inflation handled in Part 3).
- **Failure modes:** title-description mismatch → low, flagged.
- **Confidence:** evidence-volume driven.
- **Explainability:** evidence = titles, description scope phrases, durations.

### 5. Education (`edu.*`)
- **Purpose:** education tier/field, low JD weight.
- **Inputs:** `education` (tier, field, years).
- **Outputs:** `edu.tier_score`, `edu.field_relevance`, `edu.timeline_valid` (→honeypot).
- **Extraction:** tier map; field relevance to ML/CS/IR; timeline sanity.
- **Failure modes:** impossible years → honeypot cross-link.
- **Confidence:** high (structured).
- **Explainability:** evidence = institution, degree, years, tier.

### 6. Company (`co.*`)
- **Purpose:** company quality/scale context.
- **Inputs:** `career_history.company`, `company_size`, `industry`.
- **Outputs:** `co.scale_progression`, `co.industry_relevance`.
- **Extraction:** size trajectory; industry vs target (AI/product).
- **Failure modes:** unknown company → neutral.
- **Confidence:** medium.
- **Explainability:** evidence = company, size, industry per role.

### 7. Product-vs-Service (`pvs.*`)
- **Purpose:** the JD-critical product-company-vs-services split.
- **Inputs:** per-role `is_product_company`, `is_consulting_firm`, `CareerTrack`.
- **Outputs:** `pvs.product_density` (frac of tenure at product cos), `pvs.consulting_density` (neg), `pvs.product_recent` (recency-weighted).
- **Extraction:** tenure-weighted product fraction; recency-weighted; consulting set (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini).
- **Failure modes:** misclassified small company → medium confidence.
- **Confidence:** higher with known companies.
- **Explainability:** evidence = per-role company + classification.

### 8–16. Technical competency groups (`retr`, `rank`, `recsys`, `ir`, `nlp`, `llm`, `mle`, `mlops`, `eval`)
- **Purpose:** the JD's "absolutely need" stack — modeled as **evidence-aggregated competency**, not keyword presence.
- **Inputs:** `skills[]` (name, proficiency, endorsements, duration), `skill_assessment_scores`, role descriptions, `SemanticProfile` anchor sims for the matching concept, `CredibilityProfile.skill_trust`.
- **Outputs per group:** `<g>.claimed` (raw keyword presence, near-zero weight), `<g>.trust` (endorsement×duration×assessment), `<g>.in_career` (appears in descriptions), `<g>.semantic` (anchor sim), `<g>.competency` (the fused, weighted output).
  - `retr.*` retrieval systems; `rank.*` ranking/LTR; `recsys.*` recommendation; `ir.*` BM25/hybrid/vector-DB; `nlp.*`; `llm.*` (fine-tuning LoRA/QLoRA, but *not* "LangChain-only"); `mle.*` production ML eng; `mlops.*` indexing/serving/drift; `eval.*` NDCG/MRR/MAP/A-B.
- **Extraction:** `competency = w1·trust + w2·in_career + w3·semantic − w4·stuffing_penalty`; **claimed-only with zero corroboration ⇒ competency ≈ 0**.
- **Failure modes:** stuffer with all keywords → low competency (intended); sparse profile → `UNKNOWN`.
- **Confidence:** rises with corroborating sources count.
- **Explainability:** evidence = the specific skills, durations, description spans, anchor ids.

### 17. Open Source (`oss.*`)
- **Purpose:** external validation (JD "see how you think").
- **Inputs:** `github_activity_score` (−1 sentinel), OSS mentions in descriptions.
- **Outputs:** `oss.activity`, `oss.has_external_validation`.
- **Extraction:** normalize github (−1⇒UNKNOWN, never 0); description OSS cues.
- **Failure modes:** no github linked ⇒ UNKNOWN, not penalty.
- **Confidence:** low when github absent.
- **Explainability:** evidence = github score, description cues.

### 18. Leadership (`lead.*`)
- **Purpose:** mentoring/scaling-team fit (JD grows 4→12).
- **Inputs:** description leadership cues, team-size language.
- **Outputs:** `lead.scope`, `lead.management_only` (neg if no hands-on).
- **Extraction:** leadership phrases × retained coding signal; management-only is a negative.
- **Failure modes:** conflate people-mgmt with eng leadership → use coding-recency cross-check.
- **Confidence:** medium.
- **Explainability:** evidence = description spans.

### 19. Startup Fit (`startup.*`)
- **Purpose:** scrappy shipper vs big-co specialist.
- **Inputs:** company sizes, tenure pattern, shipping language.
- **Outputs:** `startup.small_co_experience`, `startup.shipping_signal`.
- **Extraction:** time at <200-person cos; "shipped/launched" cues.
- **Failure modes:** big-co-only → low (JD-aligned).
- **Confidence:** medium.
- **Explainability:** evidence = sizes, descriptions.

### 20. Founding Engineer (`found.*`)
- **Purpose:** the JD's narrow ideal (own intelligence layer, ambiguity tolerance).
- **Inputs:** ownership language, breadth across stack, product+ML combo.
- **Outputs:** `found.ownership`, `found.breadth`.
- **Extraction:** end-to-end ownership cues × technical breadth.
- **Failure modes:** rare; most candidates low (expected — JD says ~10 great matches).
- **Confidence:** evidence-driven.
- **Explainability:** evidence = description spans.

### 21. Recruiter Availability (`avail.*`)
- **Purpose:** is the candidate actually reachable/available.
- **Inputs:** `open_to_work_flag`, `last_active_date` vs `as_of`.
- **Outputs:** `avail.open`, `avail.recency`, `avail.available` (composite).
- **Extraction:** recency decay from `as_of`; stale (>~180d) ⇒ low.
- **Failure modes:** future/again-invalid date → UNKNOWN.
- **Confidence:** high (direct signals).
- **Explainability:** evidence = open flag, last_active, as_of.

### 22. Engagement (`eng.*`)
- **Purpose:** platform activity intensity.
- **Inputs:** views/searches/saves/applications/connections (signals 5,6,10,17,18).
- **Outputs:** `eng.passive` (views/searches/saves), `eng.active` (applications), `eng.network` (connections), `eng.velocity`.
- **Extraction:** bounded log-scale per signal.
- **Failure modes:** zero everything ⇒ low, not error.
- **Confidence:** high.
- **Explainability:** evidence = each 30d count.

### 23. Responsiveness (`resp.*`)
- **Purpose:** will they reply to recruiters.
- **Inputs:** `recruiter_response_rate`, `avg_response_time_hours`.
- **Outputs:** `resp.rate`, `resp.speed`, `resp.reliable` (composite).
- **Extraction:** rate direct; inverse-bounded response time.
- **Failure modes:** none beyond range checks.
- **Confidence:** high.
- **Explainability:** evidence = rate, hours.

### 24. Salary Alignment (`sal.*`)
- **Purpose:** comp fit + band sanity.
- **Inputs:** `expected_salary_range_inr_lpa`, JD band (if any).
- **Outputs:** `sal.fit`, `sal.is_inverted` (→honeypot/logistics sanity, not hard).
- **Extraction:** overlap with target band; inversion flag (min>max).
- **Failure modes:** inverted band common in pool ⇒ **soft sanity flag, not honeypot**.
- **Confidence:** medium.
- **Explainability:** evidence = min/max.

### 25. Relocation (`reloc.*`)
- **Purpose:** mobility to hubs.
- **Inputs:** `willing_to_relocate`, location.
- **Outputs:** `reloc.willing`, `reloc.needed`.
- **Extraction:** needed = not already in a hub.
- **Failure modes:** none.
- **Confidence:** high.
- **Explainability:** evidence = relocate flag, location.

### 26. Notice Period (`notice.*`)
- **Purpose:** time-to-hire (JD ≤30 ideal, buyout ≤30, >30 higher bar).
- **Inputs:** `notice_period_days` (0–180).
- **Outputs:** `notice.fit`, `notice.over_30` (neg).
- **Extraction:** banded fit.
- **Failure modes:** none.
- **Confidence:** high.
- **Explainability:** evidence = notice days.

### 27. Behavioral composite (`bhv.*`)
- **Purpose:** the fused behavioral **multiplier** inputs (detailed in Part 4).
- **Inputs:** outputs of groups 21–23 + reliability/verification/freshness.
- **Outputs:** `bhv.availability, bhv.recruitability, bhv.trust, bhv.risk, …` (Part 4 list).
- **Extraction:** see Part 4.
- **Confidence:** `bhv.behavioral_confidence`.
- **Explainability:** evidence = constituent signals.

### 28. Risk (`risk.*`)
- **Purpose:** non-honeypot risk (uncertainty, contradiction).
- **Inputs:** signal-availability flags, claimed-vs-assessed gaps.
- **Outputs:** `risk.uncertainty`, `risk.contradiction`, `risk.confidence`.
- **Extraction:** UNKNOWN-density + gap aggregation (engine-internal confidence, Part 9).
- **Failure modes:** high-UNKNOWN ⇒ low confidence (intended).
- **Confidence:** meta-confidence.
- **Explainability:** evidence = which signals were UNKNOWN / which gaps.

### 29. Consistency (`cons.*`)
- **Purpose:** internal coherence checks feeding integrity.
- **Inputs:** title-vs-description, current-title-vs-history, skills-vs-roles.
- **Outputs:** `cons.title_role_coherence`, `cons.skill_role_coherence`, `cons.summary_coherence`.
- **Extraction:** semantic + lexical agreement between claimed title/skills and described work (the "Marketing Manager with AI keywords" trap).
- **Failure modes:** synthetic mismatched descriptions (rampant in pool) ⇒ low coherence (intended).
- **Confidence:** medium.
- **Explainability:** evidence = the conflicting title/description pair.

### 30. Honeypot (`hp.*`)
- **Purpose:** dedicated impossible-profile detection (Part 5).
- **Inputs:** all structural facts.
- **Outputs:** the 12 detectors + `hp.composite` (Part 5).
- **Extraction:** see Part 5.
- **Confidence:** per-detector + composite.
- **Explainability:** evidence = the impossible field pair(s).

---

# PART 2 — JD UNDERSTANDING FEATURES (latent families)

Latent concepts modeled as `value + confidence`, each backed by positive and negative evidence features. Namespace `jd.*`.

**Positive latents (what the JD wants):**
| Latent | Built from (positive evidence) | Confidence driver |
|---|---|---|
| `jd.retrieval_ranking` | `retr.competency`, `rank.competency`, `ir.competency`, role descriptions showing shipped search/ranking | corroboration count |
| `jd.production_ml` | `mle.competency`, `mlops.competency`, product-company tenure, "deployed to real users" cues | description specificity |
| `jd.product_company` | `pvs.product_density`, `pvs.product_recent` | known-company coverage |
| `jd.shipping_mentality` | `startup.shipping_signal`, `found.ownership`, velocity of shipped systems | cue density |
| `jd.eval_framework` | `eval.competency` (NDCG/MRR/MAP/A-B) | explicit metric mentions |
| `jd.hybrid_retrieval` | `ir.*` (BM25+dense), embeddings, vector-DB competency | named-tech + description |

**Negative latents (what the JD dislikes):**
| Latent | Built from (negative evidence) | Effect |
|---|---|---|
| `jd.keyword_only` | high `*.claimed` but low `*.trust`/`*.in_career`/`*.semantic` | dampens competency; primary anti-stuffer |
| `jd.consulting_only` | `pvs.consulting_density` ≈ 1, no product role ever | hard-eligibility input |
| `jd.title_chaser` | sub-18-month hops optimizing title ladder | soft penalty |
| `jd.pure_researcher` | research-only, no production deployment | hard-eligibility input |
| `jd.framework_enthusiast` | LangChain/tutorial-style cues, recent-LLM-only, no pre-LLM ML | soft/hard per JD |
| `jd.inactive` | low `avail.available`, stale activity, low responsiveness | behavioral down-weight |

**Confidence features:** each latent emits `jd.<x>.confidence` = f(evidence volume, source agreement, UNKNOWN density). Low confidence regresses the latent toward a neutral prior (so a sparse profile isn't confidently labeled either way). The Fit/Risk engines read both value and confidence.

**Anti-stuffing core:** `jd.keyword_only` is computed as `mean(claimed) − mean(trust·in_career·semantic)`; when a profile lists many AI skills with no corroboration this term is large and **subtracts** from every positive technical latent. This is the dense+symbolic counterpart that defeats the explicit JD trap.

---

# PART 3 — CAREER INTELLIGENCE FEATURES

Each: **rationale · extraction · expected distribution** (over the synthetic pool, which is heavy on mismatched-title traps, so most candidates score low on authenticity/depth).

| Feature | Rationale | Extraction | Expected distribution |
|---|---|---|---|
| `career.progression_quality` | upward, coherent trajectory signals judgment | title/scope monotonicity × company scale trend | right-skewed low (pool is noisy) |
| `career.stability` | JD wants 3+ yr intent, not hoppers | mean tenure, hop-rate inverse | bimodal (stable vs hoppers) |
| `career.promotion_velocity` | healthy growth vs title inflation | level deltas per year, capped | mostly low/moderate |
| `career.title_inflation` (neg) | titles outpacing scope | title level − scope-evidence level | long right tail (synthetic mismatches) |
| `career.role_consistency` | coherent role vs description | `cons.title_role_coherence` | **low overall** (pool deliberately mismatches titles & descriptions) |
| `career.experience_authenticity` | guards against inflated/contradictory tenure | stated-vs-derived years, date coherence | concentrated high, thin impossible tail (honeypots) |
| `career.company_progression` | scale/quality trajectory | size+industry trend | broad |
| `career.product_company_density` | JD-critical | tenure-weighted product fraction | bimodal |
| `career.consulting_density` (neg) | JD dislikes services-only | tenure at consulting set | spike at 0 and at 1 |
| `career.technical_depth` | real eng vs adjacent | competency trust × hands-on cues | right-skewed low |
| `career.hands_on_engineering` | JD "this role writes code", 18-mo rule | coding-recency cues vs architecture-only | bimodal |
| `career.research_only` (neg) | JD hard dislike | research roles, no deployment | small positive mass |
| `career.management_only` (neg) | no recent hands-on | mgmt cues without coding recency | small/moderate mass |
| `career.production_exposure` | shipped to real users | deployment language × product company | right-skewed low |

Extraction is tenure-weighted and recency-weighted throughout; **descriptions dominate titles** (the pool's titles are unreliable by design). Distributions are documented so the offline O9 validation battery can detect drift / miscalibration.

---

# PART 4 — BEHAVIORAL FEATURES (23 signals → 15 composites)

Built on `redrob_signals_doc`, honoring single-ownership routing (Engine §7) and **sentinel→UNKNOWN** (−1 / {} never become 0). Namespace `bhv.*`.

| Composite | Source signals | Definition (intuition) |
|---|---|---|
| `bhv.availability` | open_to_work, last_active (vs `as_of`) | reachable + recently active |
| `bhv.recruitability` | open + responsiveness + saves | likely to engage if contacted |
| `bhv.response_reliability` | recruiter_response_rate, avg_response_time | replies, and quickly |
| `bhv.interview_reliability` | interview_completion_rate | shows up to interviews |
| `bhv.market_demand` | profile_views, search_appearances, saved_by_recruiters | how much recruiters pull them |
| `bhv.market_momentum` | trend of views/saves/searches (30d) vs tenure-on-platform | rising vs cooling demand |
| `bhv.engagement_velocity` | applications + actions per active period | proactivity rate |
| `bhv.candidate_temperature` | availability × recency × applications | "hot" job-seeker vs passive |
| `bhv.recruiter_attractiveness` | saves + views + search appearances | demand-side desirability |
| `bhv.hiring_probability_proxy` | availability × responsiveness × interview/offer reliability | proxy for "can actually be hired" |
| `bhv.freshness` | last_active recency, signup recency | data currency |
| `bhv.trust` | verified_email + verified_phone + linkedin + completeness | identity/verification trust |
| `bhv.signal_consistency` | agreement across signals (e.g. high views but zero saves) | internal coherence of behavior |
| `bhv.behavioral_confidence` | inverse UNKNOWN density + signal_consistency | how much to trust the above |
| `bhv.behavioral_risk` | low availability + low reliability + inconsistency | down-weight magnitude |

**Sentinel discipline:** `github_activity_score=−1`, `offer_acceptance_rate=−1`, `skill_assessment_scores={}` ⇒ the contributing family is `UNKNOWN`, lowering `bhv.behavioral_confidence` rather than the score itself. The JD's "perfect-on-paper but inactive ⇒ not available" maps directly to `bhv.availability` × `bhv.hiring_probability_proxy` as a **multiplier** on relevance (never a relevance source).

---

# PART 5 — HONEYPOT FEATURES

Namespace `hp.*`. Spec §7: ~80 honeypots forced to tier 0; **>10% honeypot rate in top 100 ⇒ disqualification**. Therefore: detectors are calibrated offline (O3) and **a hard gate requires multiple corroborating impossibilities**; single soft anomalies only dampen, to protect recall (false positives drop real candidates and cost NDCG).

| Detector | Detection strategy | False-positive mitigation | Confidence |
|---|---|---|---|
| `hp.timeline_impossible` | role/education dates contradict (end<start, overlap-impossible, current+end_date) | tolerance window; ignore minor rounding | high when contradiction exact |
| `hp.skill_time_contradiction` | `Proficiency≥ADVANCED` with `duration_months∈{0,None}`, en masse | require count ≥ k, not single | scales with count |
| `hp.employment_overlap` | overlapping full-time roles beyond plausibility | allow brief transition overlaps | medium |
| `hp.title_seniority_anomaly` | seniority claimed ≫ tenure/scope possible | only flag extreme gaps | medium |
| `hp.education_career_anomaly` | degree timeline impossible vs career start | tolerance for non-linear paths | high when impossible |
| `hp.salary_anomaly` | inverted/implausible band | **soft only** (common in pool) | low |
| `hp.experience_inflation` | Σdurations/12 ≫ stated years | tolerance band from O3 | scales with gap |
| `hp.keyword_stuffing` | many high-proficiency skills, zero endorsements/duration/career-mention | require breadth + zero corroboration | scales with breadth |
| `hp.behavioral_inconsistency` | impossible signal combos (e.g. high saves, zero views) | only impossible, not merely unusual | medium |
| `hp.signal_impossibility` | out-of-contract signal values surviving parse | structural; rare | high |
| `hp.identity_anomaly` | provenance/id inconsistencies | structural | high |
| `hp.composite` | calibrated weighted aggregate of the above | **hard-gate only when ≥2 high-confidence impossibilities**; else soft dampen | the run's honeypot decision |

**Detection strategy (composite):** `hp.composite = calibrated_aggregate(detectors)`; `is_honeypot = (≥2 HARD impossibilities) OR (composite ≥ O3 threshold)`. **False-positive mitigation:** thresholds set on the O3 calibration set to keep the honeypot *recall* high while bounding real-candidate loss; salary inversion and lone soft anomalies never hard-gate. **Confidence scoring:** each detector emits confidence from how *categorically impossible* (vs merely unusual) the evidence is; `CandidateRiskEngine` finalizes. Honeypot decisions are reported (`honeypot_rate`) for the Stage-3 ≤10% gate.

---

# PART 6 — FEATURE STORE DESIGN

Pure, typed metadata objects (frozen pydantic). The store is **declarative**: definitions live in code+config, values are computed deterministically.

| Object | Purpose / key fields |
|---|---|
| `FeatureDefinition` | id, group, dtype, range/bounds, `source_slices`, `dependencies: tuple[FeatureId,...]`, `extractor_ref`, `version`, `tier (A–D)`, `polarity (+/−)`, doc. The unit of the taxonomy. |
| `FeatureRegistry` | the frozen set of all `FeatureDefinition`s; provides id↔index via `FeatureLayout`; rejects unknown ids; single source of truth, version-pinned. |
| `FeatureManifest` | per-run: `layout_version`, ordered feature list, group map, hash of the registry; written into the CQV/artifacts; checked by Scoring + O8 (must match weights' `layout_version`). |
| `FeatureVersion` | semver per feature + a global `layout_version`; bump rules: value-changing edit ⇒ minor, layout/order change ⇒ major. |
| `FeatureSnapshot` | the materialized `(N,D)` value matrix + `(N,num_groups)` confidence + id index for a given input+version; reproducible artifact. |
| `FeatureLineage` | DAG edges feature→dependencies→raw fields; powers invalidation + explainability. |
| `FeatureCache` | content-addressed cache keyed by `(candidate_hash, layout_version)`; online within-run only; offline persistent. |
| `FeatureAuditRecord` | per (candidate, feature): value, confidence, evidence refs, version, timestamp(audit-only). Materialized for survivors/top-K. |
| `FeatureImportance` | per-feature contribution (from O8 weights + ablation); read by Reasoning to pick which features to cite. |
| `FeatureMetadata` | human doc, expected distribution, owner, tier, polarity — for the validation battery + reviewers. |
| `FeatureSchema` | dtype/range/nullability contract per feature; enforced at extraction. |
| `FeatureValidation` | run-time assertions: no NaN, in-range, dependency satisfaction, layout match. |
| `FeatureContracts` | cross-feature invariants (e.g. `*.competency ≤ corroboration`; honeypot composite ≥ max single hard detector). |
| `FeatureProvenance` | the `feature → EvidenceRef[] → RawCandidate field` chain (Part 8). |

**Determinism:** registry + layout are frozen constants; snapshots are reproducible from `(input, layout_version)`; manifest hash ties a CQV to its exact feature set.

---

# PART 7 — FEATURE EXECUTION

- **Offline feature generation (O2):** full pure extraction over the 100K pool → persistent `FeatureSnapshot` (parquet, columnar). Heavy precompute (embeddings) is the Embedding engine's artifact, not this layer; this layer reads precomputed semantic *values* when folding.
- **Online feature generation (R2):** vectorized columnar extraction into the bulk `(N,D)` matrix; per-candidate objects only for survivors. Target ≤ ~45 s for structural features.
- **Feature caching:** content-addressed by `(candidate_hash, layout_version)`. Online: within-run memo for repeated subexpressions (e.g. canonicalized tokens). Offline: persistent snapshot reuse if input+version unchanged.
- **Feature persistence:** offline snapshots are artifacts (hash-pinned in MANIFEST); online emits no persistent feature store (only the CQV feeds Scoring + top-K audit records into the run report).
- **Feature recomputation / invalidation:** keyed by `FeatureVersion`/`layout_version` + input hash via `FeatureLineage`. A changed extractor bumps the feature version → its snapshot column invalidates → dependents recompute transitively.
- **Feature dependency graph:** raw fields → primitive features → derived → latent (`jd.*`) → composite (`bhv.*`, `hp.composite`). Acyclic; topologically ordered; the extraction order is fixed (determinism).
- **Feature execution graph:** primitives in parallel (data-parallel over candidates), then derived, then latents/composites; deterministic merge by `source_index`; thread-count-invariant reductions.
- **Materialization strategy:** **bulk = columnar values `(N,D)` float32 + group-confidence `(N,num_groups)`**; **survivors/top-K = full `FeatureCell`s with per-feature confidence + evidence**. This is the key memory decision — a full per-feature `(N,D)` confidence/evidence store would ~double the 150 MB footprint and is unnecessary for the 99.8% that won't be explained.

---

# PART 8 — EXPLAINABILITY

- **Traceability:** every `FeatureCell` carries `EvidenceRef`s (path+value into `RawCandidate`); `FeatureLineage` links feature→dependencies→raw fields. Any feature value can be walked back to source facts.
- **Auditability:** `FeatureAuditRecord` (value, confidence, evidence, version) is materialized for survivors/top-K and folded into the run report; for any ranked candidate the full feature derivation is reconstructable.
- **Reasoning generation:** `CandidateReasoningEngine` selects features by `FeatureImportance`, reads their `FeatureCell` evidence, and emits clauses **only** from features that have evidence — the no-hallucination guarantee is mechanical (a clause cannot cite a feature lacking `EvidenceRef`).
- **Stage-4 defendability:** because reasoning derives from importance-ranked, evidence-backed features (not free text), each top-100 explanation cites specific profile facts (years, named skills with trust, product-company tenure, signal values), connects to a JD requirement (the latent it supports), and acknowledges concerns (negative latents/soft penalties) — exactly the six Stage-4 checks. The audit trace lets a human (or the Stage-5 interview) trace any claim to a field.

**Explainability architecture (chain):** `RankedCandidate → ScoreBreakdown.component → FeatureImportance → FeatureCell(value,confidence,evidence) → EvidenceRef → RawCandidate.field`. This single chain serves reasoning, audit, and defense.

---

# PART 9 — FEATURE SCORING INTERFACE

Engines never touch raw arrays; they use a read-only `FeatureView` (typed accessor over a candidate's `FeatureCell`s + group confidences + importance), keyed by `FeatureId`.

| Engine | Reads | Uses for |
|---|---|---|
| `CandidateFitEngine` | positive `jd.*` latents + competency `*.trust`/`*.competency` + negative latents (`jd.consulting_only`, `jd.pure_researcher`, `jd.keyword_only`, `career.*` negatives) + logistics fit | eligibility hard-blocks/soft-penalties + fit `ScoreComponentValue`s |
| `CandidateRiskEngine` | `hp.*` detectors + `cons.*` + `risk.*` + `bhv.behavioral_confidence` | finalize `IntegrityReport` + engine-internal confidence/uncertainty |
| `CandidateScoringEngine` | the full CQV `(D,)` row + `ScoringWeights` (+ behavioral/logistics multipliers + confidence) | `base_relevance`, gating, `final_score` |
| `CandidateReasoningEngine` | top-K `FeatureCell` evidence + `FeatureImportance` | importance-ranked, evidence-backed clauses |
| `CandidateRankingEngine` | final scores only (no direct feature access) | sort/tie-break/top-k |

**Contract:** `FeatureView.get(feature_id) -> FeatureCell`; `FeatureView.group_confidence(group) -> UnitScore`; `FeatureView.importance(feature_id) -> float`. Read-only; deterministic; no side effects.

---

# PART 10 — FEATURE PRIORITY MATRIX

Tiers reflect the scoring weights (NDCG@10 = 0.50 dominates) and the DQ risk (honeypot >10%).

**Tier A — mission-critical (drive NDCG@10 + avoid DQ):**
- `jd.retrieval_ranking`, `jd.production_ml`, `jd.product_company`, `jd.hybrid_retrieval`, `jd.eval_framework` — the JD's "absolutely need", and the discriminators among near-duplicates.
- `jd.keyword_only` + competency `*.trust`/`*.competency` (8–16) — the anti-stuffer; without these the top-10 fills with keyword traps.
- `hp.composite` + the hard honeypot detectors — a single honeypot in top-10 is both a quality and a DQ risk.
- `pvs.product_density` (recency-weighted) — the product-vs-services split is the JD's loudest "between the lines" signal.
*Justification:* these decide who reaches the top-10/50 where the bulk of the composite is earned, and prevent disqualification.

**Tier B — strong signals:**
- `exp.in_band`, `career.production_exposure`, `career.technical_depth`, `career.hands_on_engineering`, `eval.competency`, `bhv.availability` × `bhv.hiring_probability_proxy` (the availability multiplier), negative latents `jd.consulting_only`/`jd.pure_researcher`/`jd.framework_enthusiast`.
*Justification:* materially reorder the borderline band (ranks ~10–60) and apply the JD's hard dislikes; strong but not solely decisive.

**Tier C — supporting signals:**
- `edu.*`, `geo.hub_match`/`reloc.*`/`notice.fit`, `eng.*`, `resp.*`, `bhv.trust`/`bhv.market_demand`, `co.*`, `sen.level`.
*Justification:* tie-breakers and modifiers; useful for the long tail (ranks 60–100) and as multiplier nudges, low standalone weight.

**Tier D — nice-to-have:**
- `oss.*`, `lead.*` nuance, `startup.*`/`found.*` (rare positives), `bhv.market_momentum`, `bhv.engagement_velocity`.
*Justification:* add color and reasoning richness; sparse in the pool; minimal NDCG impact but valuable for Stage-4 explanation quality.

---

## Build order for feature implementation

1. `FeatureLayout` + `FeatureRegistry` + `FeatureDefinition`/`FeatureSchema` (the frozen contract Scoring & O8 bind to).
2. Primitive extractors (Identity, Geography, Experience, Education, Company, Salary/Relocation/Notice, Engagement/Responsiveness/Availability) — pure, golden-tested.
3. Competency groups (8–16) with the trust/evidence aggregation + `jd.keyword_only` (anti-stuffer) — the highest-value, most-tested code.
4. Career-intelligence (Part 3) and `cons.*`.
5. Behavioral composites (Part 4) with sentinel discipline.
6. Honeypot detectors + `hp.composite` (Part 5), calibrated against O3 fixtures.
7. Latent JD families (Part 2) once their constituents exist.
8. Feature store metadata (Part 6), execution graph (Part 7), `FeatureView` (Part 9), explainability chain (Part 8).
9. Tiering (Part 10) is metadata on `FeatureDefinition`; the O8 weight search consumes it as priors.

Each feature merged only when (a) golden value tests on `sample_candidates` pass, (b) the honeypot fixtures fire/don't-fire correctly, and (c) its expected distribution (Part 3) holds on the offline census.
