# REDSTACK v1.1 — Domain Layer Specification

**Scope:** `src/redstack/domain/` only. Repository architecture is frozen; this document does not alter it. No implementation code — type design, invariants, and construction contracts detailed enough to begin coding immediately.

**Domain purity contract (inherited, restated as law):** `domain/` imports only stdlib, `pydantic` v2, and `numpy` (the latter only for the CQV vector type). No IO, no ML runtime, no pandas, no clock, no RNG, no network. Every "current time" is an injected `as_of` date, never `datetime.now()`. Every tie-break is a total order on `candidate_id`, never randomness.

**Reading guide:** §A–§E establish primitives (NewTypes, enums, value objects, provenance, errors). §F specifies the raw input model. §G specifies the ten `CandidateRepresentation` slices. §H–§J specify scoring, ranking, reasoning, validation. §K–§T are the cross-cutting strategies (immutability, COW, serialization, memory, determinism, type safety) and the construction flow.

---

## §A. NewTypes (`domain/ids.py`)

NewTypes are **nominal aliases checked statically by mypy only**; they carry no runtime validation. Runtime validation lives in the pydantic models and in the smart-constructor functions that mint them. The point is to make it a *type error* to pass a `Similarity` where a `Score` is expected.

| NewType | Base | Meaning | Minted by / validated where |
|---|---|---|---|
| `CandidateId` | `str` | `^CAND_[0-9]{7}$` | `Identity` constructor + `features.parsing` |
| `AnchorId` | `str` | JD anchor key (O6) | artifact load; `SemanticProfile` keys |
| `ArchetypeId` | `int` | cluster index (O7) | `ArchetypeAssignment` |
| `SkillName` | `str` | normalized skill token | `features.skills` (lowercased, lexicon-canonical) |
| `Score` | `float` | final ranking score | `ScoringEngine` only |
| `Similarity` | `float` | cosine, range `[-1.0, 1.0]` | `SemanticEngine` |
| `UnitScore` | `float` | bounded `[0.0, 1.0]` derived feature | feature/engine layers |
| `Multiplier` | `float` | bounded behavioral/logistics modifier `[m_min, m_max]` | Behavioral/Logistics engines |
| `Months` | `int` | non-negative duration | `features.career` |
| `LpaAmount` | `float` | INR lakhs/annum, `>= 0` | `LogisticsProfile` |
| `FeatureIndex` | `int` | position in the frozen CQV layout | `quality` module constant |

Rule: a NewType may only be *minted* (cast) inside the module that owns its validation. Downstream modules receive it already-typed and never re-cast `float`→`Score` ad hoc.

---

## §B. Enums (`domain/enums.py`)

All small fixed vocabularies are enums (or `Literal` where a model field is closed). **Never** compare these by raw string. Ordered enums expose an explicit `ordinal` for `<`/`>` comparison; ordering is part of the enum's contract, not an accident of definition order.

**Schema-mirroring enums** (values must equal the dataset strings exactly):

- `CompanySize` — *ordered*: `S_1_10 < S_11_50 < S_51_200 < S_201_500 < S_501_1000 < S_1001_5000 < S_5001_10000 < S_10001_PLUS`. Values are the literal bucket strings (`"1-10"`, …, `"10001+"`).
- `InstitutionTier` — `TIER_1, TIER_2, TIER_3, TIER_4, UNKNOWN` (ordered, tier_1 strongest).
- `WorkMode` — `REMOTE, HYBRID, ONSITE, FLEXIBLE`.
- `Proficiency` — *ordered*: `BEGINNER < INTERMEDIATE < ADVANCED < EXPERT`.
- `LanguageProficiency` — *ordered*: `BASIC < CONVERSATIONAL < PROFESSIONAL < NATIVE`.

**Domain judgement enums:**

- `RelevanceTier` — `IntEnum 0..4`. Ground-truth tiering vocabulary (honeypots are forced to tier 0). Domain holds *predicted* tiers only as an optional reasoning aid; the hidden truth is never in-repo.
- `CareerTrack` — `PRODUCT, SERVICES, MIXED, UNKNOWN` (product-vs-services is a load-bearing JD signal).
- `Severity` — `HARD, SOFT, INFO`.
- `BuildStage` — *ordered IntEnum*: `PARSED(0) < FEATURED(1) < SITUATED(2) < GATED(3) < VECTORIZED(4) < SCORED(5) < RANKED(6) < EXPLAINED(7)`. Tracks aggregate maturity (see §S).
- `ScoreComponent` — closed set of base-relevance components: `SKILL_MATCH, SEMANTIC_FIT, CAREER_FIT, EXPERIENCE_FIT, EDUCATION_FIT, CREDIBILITY, ARCHETYPE_FIT`. (Behavioral & logistics are *multipliers*, deliberately not components — see §H.)
- `IntegrityFlag` — honeypot rule codes (see §G.2).
- `EligibilityCode` — JD disqualifier / penalty codes (see §G.3).
- `ReasoningPolarity` — `STRENGTH, CONCERN, CONTEXT`.
- `LocationFit` — `PREFERRED_HUB, INDIA_RELOCATABLE, INDIA_NON_RELOCATABLE, OUTSIDE_INDIA_NO_SPONSOR`.
- `NoticeFit` — `SUB_30_IDEAL, BUYOUTABLE, OVER_30_HIGHER_BAR`.
- `SignalAvailability` — `PRESENT, UNKNOWN` (the sentinel discriminator; see §G.7).
- `ValidationCode` — mirrors `validate_submission.py` + Stage-4 reasoning checks (see §J).
- `EvidenceKind` — `PROFILE_FIELD, CAREER_FIELD, SKILL, EDUCATION, SIGNAL, DERIVED`.

Serialization rule: enums serialize **by value** (string/int), never by name or auto-int, so artifacts/run-reports are stable across refactors.

---

## §C. Value-Object conventions

Value objects (VOs) are frozen pydantic models with no identity of their own — equality is structural. Conventions applied to **every** model in this layer:

- `model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True)`.
- Collections are immutable: `tuple[...]`, `frozenset[...]`, or read-only `Mapping[...]`. No bare `list`/`dict`/`set` ever appears as a field type.
- No field defaults that hide missing data; "missing" is modeled explicitly (`X | None` or a `SignalAvailability` discriminator), never silently defaulted to `0`.
- Range-bounded floats use a validator that rejects out-of-range and `NaN`/`inf` at construction.

---

## §D. Provenance Architecture (`domain/provenance.py`)

Provenance is the mechanism that makes the Stage-4 **"no hallucination"** rule a *type-level guarantee* rather than a hope.

**`EvidenceRef`** (VO) — a typed pointer into real candidate data:
| Field | Type | Rule |
|---|---|---|
| `kind` | `EvidenceKind` | required |
| `path` | `str` | dotted path into the raw record (e.g. `career_history[0].title`) |
| `value` | `str \| int \| float \| bool` | the actual value at that path (JSON scalar) |
| `as_of` | `date \| None` | for time-relative evidence (recency) |

Invariant: an `EvidenceRef` may only be constructed from a value that actually exists in the `RawCandidate`; the minting helper (in `features`) verifies the path resolves. A dangling path raises `ProvenanceError`.

**`ProvenanceHandle`** (VO) — how a representation refers back to its source without paying 100K× the memory cost:
| Field | Type | Meaning |
|---|---|---|
| `candidate_id` | `CandidateId` | always present |
| `inline` | `RawCandidate \| None` | populated only for survivors / top-K |
| `source_index` | `int \| None` | row index into the columnar source store otherwise |

Exactly one of `inline` / `source_index` is non-null (constructor invariant). The bulk 100K path carries only `source_index`; the Reasoning Engine requires `inline`, so the pipeline re-hydrates raw records for the top-K before R7. This is the single most important memory decision in the layer (§Q).

Every `IntegrityFinding`, `EligibilityFinding`, `ScoreComponentValue`, and `ReasoningClause` carries one or more `EvidenceRef`s. The "no hallucination" invariant is then mechanical: **a `ReasoningClause` cannot be constructed without at least one `EvidenceRef`.**

---

## §E. Domain Error Hierarchy (`domain/errors.py`)

Two philosophies, deliberately separated:

- **Business verdicts are data, not exceptions.** "This candidate is a honeypot" / "ineligible" are `Report` objects (`IntegrityReport`, `EligibilityReport`) that flow through scoring and reasoning. They are inspectable, loggable, and never raised in the normal path.
- **Invariant/programming failures raise.** Constructing a `Ranking` with a duplicate rank is a bug; it raises immediately.

```
DomainError (base)
├── SchemaError                # raw input violates the candidate schema mirror
│   ├── InvalidCandidateId
│   └── FieldSchemaViolation
├── InvariantViolation         # a constructed object would break an invariant
│   ├── RepresentationStageError   # slice attached out of order / accessed before populated
│   ├── CQVInvariantError          # wrong dim, NaN, range, schema_version mismatch
│   ├── ScoreInvariantError        # gating/formula contradiction
│   └── RankingInvariantError      # count / unique-rank / monotonicity / tie-break
├── ProvenanceError            # EvidenceRef path does not resolve in RawCandidate
└── ArtifactContractError      # feature-layout / anchor-set / weights schema_version mismatch
                               #   (raised in domain factories, surfaced by the loader adapter)
```

`IntegrityViolation` and `IneligibleCandidate` exist as types for the rare cases an engine chooses to fail hard, but the default flow uses Reports.

---

## §F. Raw input model (`domain/source.py`)

A **lossless, validated mirror** of `candidate_schema.json`. It is a domain object (pure data + invariants) even though the dict→model conversion lives in `features.parsing`. It is the canonical source for all `EvidenceRef`s.

**`RawCandidate`** (aggregate of raw facts) — nested VOs mirror the schema exactly:

- `candidate_id: CandidateId`
- `profile: RawProfile` — `anonymized_name, headline, summary, location, country, years_of_experience (0..50), current_title, current_company, current_company_size: CompanySize, current_industry`.
- `career_history: tuple[RawPosition, ...]` (1..10) — each: `company, title, start_date: date, end_date: date | None, duration_months: Months, is_current: bool, industry, company_size: CompanySize, description`.
- `education: tuple[RawEducation, ...]` (0..5) — `institution, degree, field_of_study, start_year, end_year, grade: str | None, tier: InstitutionTier`.
- `skills: tuple[RawSkill, ...]` (0..n) — `name, proficiency: Proficiency, endorsements (>=0), duration_months: Months | None`.
- `certifications: tuple[RawCertification, ...]` — `name, issuer, year`.
- `languages: tuple[RawLanguage, ...]` — `language, proficiency: LanguageProficiency`.
- `redrob_signals: RawSignals` — all **23** signals, typed exactly (see §G.7 for the canonical list and sentinel handling).

**Validation rules at this layer are *tolerant of semantic contradictions but strict on types*.** Inverted salary bands, `is_current` with an `end_date`, expert-at-zero-months — these are **not** rejected here; they are preserved faithfully so the Integrity Engine can *detect* them downstream. Rejecting them at parse time would blind the honeypot filter. Only true type/shape violations raise `SchemaError`.

Construction rule: `RawCandidate.from_mapping(...)` (the only `Any`-accepting boundary in the whole layer) validates and narrows immediately; everything downstream is fully typed.

---

## §G. `CandidateRepresentation` slices

Each slice is a frozen VO. For every slice: **Purpose · Fields · Validation · Invariants · Construction · Relationships.** Producers are named per the frozen pipeline (R-stages / engines).

### §G.1 `Identity` (`candidate/identity.py`)
- **Purpose:** stable identification + provenance anchor.
- **Fields:** `candidate_id: CandidateId`; `anonymized_name: str`; `provenance: ProvenanceHandle`.
- **Validation:** `candidate_id` matches the canonical pattern; name non-empty.
- **Invariants:** `provenance.candidate_id == candidate_id`. Equality/hash by `candidate_id` **only**.
- **Construction:** R1, from `RawCandidate`.
- **Relationships:** referenced by every other slice indirectly via the aggregate; sole identity source for `ScoredCandidate`/`RankedCandidate`.

### §G.2 `IntegrityReport` (`candidate/integrity.py`)
- **Purpose:** honeypot / impossible-profile verdict (spec §7; mandate: drop anomalies before scoring).
- **Fields:** `findings: tuple[IntegrityFinding, ...]`; `honeypot_score: UnitScore`; `is_honeypot: bool`; `rules_evaluated: frozenset[IntegrityFlag]`.
- **`IntegrityFinding` VO:** `code: IntegrityFlag`, `severity: Severity`, `evidence: tuple[EvidenceRef, ...]` (≥1), `detail: str`.
- **`IntegrityFlag` vocabulary** (calibrated thresholds come from O3 `integrity_thresholds.json`; domain holds only the codes):
  - `TENURE_EXCEEDS_EXPERIENCE` — Σ career `duration_months` / 12 exceeds `years_of_experience` beyond tolerance.
  - `ROLE_DURATION_DATE_MISMATCH` — `duration_months` inconsistent with `end_date − start_date`.
  - `CURRENT_ROLE_HAS_END_DATE` — `is_current` yet `end_date` non-null.
  - `EXPERT_SKILL_ZERO_USAGE` — `Proficiency ≥ ADVANCED` with `duration_months in {0, None}`, especially en masse ("expert in 10 skills, 0 months").
  - `EDUCATION_TIMELINE_IMPOSSIBLE` — `end_year < start_year`, or graduation post-dating dependent career events impossibly.
  - `EXPERIENCE_PREDATES_PLAUSIBLE_START` — career start implausibly early vs first education.
  - `ASSESSMENT_FOR_ABSENT_SKILL` — `skill_assessment_scores` key not present in `skills`.
- **Validation:** `honeypot_score ∈ [0,1]`; findings sorted by `(code)` for determinism; each finding has ≥1 evidence ref.
- **Invariants:** `is_honeypot == (any finding.severity == HARD) or (honeypot_score ≥ calibrated_threshold)`. (`INVERTED_SALARY_BAND` is **not** a honeypot flag — see §G.8; it is logistics sanity, because inversion is common enough in the pool to be noise.)
- **Construction:** R4, by `IntegrityEngine` from `CareerProfile` + `RawCandidate`.
- **Relationships:** hard input to `ScoringEngine` gate (honeypot ⇒ score floored, §H) and to `Ranking` honeypot-rate guard.

### §G.3 `EligibilityReport` (`candidate/eligibility.py`)
- **Purpose:** apply JD hard disqualifiers and soft penalties (the JD's explicit "do NOT want" list).
- **Fields:** `hard_blocks: tuple[EligibilityFinding, ...]`; `soft_penalties: tuple[EligibilityFinding, ...]`; `is_eligible: bool`; `gates_passed: frozenset[EligibilityCode]`.
- **`EligibilityCode` vocabulary** (predicates authored as data in `configs/gates`, evaluated by the engine):
  - HARD: `PURE_RESEARCH_NO_PRODUCTION`, `LANGCHAIN_OPENAI_ONLY_RECENT`, `NO_PRODUCTION_CODE_18M`, `CONSULTING_FIRMS_ONLY_CAREER`, `PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP`, `CLOSED_SOURCE_5Y_NO_VALIDATION`.
  - SOFT: `TITLE_CHASER_SUB_18M_HOPS`, `NOTICE_OVER_30`, `OUTSIDE_INDIA_NO_SPONSOR`, `OUTSIDE_EXPERIENCE_BAND`.
- **Validation:** findings carry evidence; sorted by code.
- **Invariants:** `is_eligible == (len(hard_blocks) == 0)`. Soft penalties never set `is_eligible=False`; they feed scoring as down-weights.
- **Construction:** R4, by `EligibilityEngine` from `CareerProfile`, `CredibilityProfile`, `RawCandidate`, and the `JobDescriptionSpec`.
- **Relationships:** ineligible ⇒ score floored (§H); soft penalties ⇒ component/multiplier reduction; surfaced honestly in Reasoning (Stage-4 "honest concerns").

### §G.4 `CareerProfile` (`candidate/career.py`)
- **Purpose:** structured career facts + the product-vs-services and recency derivations the JD hinges on.
- **Fields:** `stated_experience_years: float`; `derived_experience_years: float`; `positions: tuple[PositionFact, ...]` (recent-first); `current_position: PositionFact | None`; `track: CareerTrack`; `tenure: TenureStats`; `recency: CareerRecency`; `title_consistency: UnitScore`.
  - **`PositionFact` VO:** `company, title, start_date, end_date | None, duration_months: Months, is_current, industry, company_size: CompanySize, is_product_company: bool, is_consulting_firm: bool, description_role_match: UnitScore`.
  - **`TenureStats` VO:** `position_count, mean_tenure_months, min_tenure_months, hop_rate` (fraction of roles < 18 months).
  - **`CareerRecency` VO:** `most_recent_start: date`, `is_currently_employed: bool`, `months_since_last_role: Months` (computed vs injected `as_of`).
- **Validation:** durations ≥ 0; `as_of`-relative fields require an `as_of` date; at most one `is_current`.
- **Invariants:** `positions` sorted by `start_date` desc; `current_position` is the unique `is_current` entry or `None`; `track` derived deterministically from per-position `is_product_company`/`is_consulting_firm`.
- **Construction:** R2, by `features.career`.
- **Relationships:** primary input to Integrity (date logic), Eligibility (consulting/production/recency), CQV (experience & career components), Reasoning.

### §G.5 `SemanticProfile` (`candidate/semantic.py`)
- **Purpose:** dense-fit signals against JD anchors; reference (not storage) of the candidate vector.
- **Fields:** `anchor_similarities: Mapping[AnchorId, Similarity]`; `positive_fit: Similarity`; `negative_fit: Similarity`; `net_semantic_fit: UnitScore`; `best_positive_anchor: AnchorId | None`; `vector_ref: VectorRef`.
  - **`VectorRef` VO:** `store_key: str`, `row_index: int`, `dim: int` — an index into the mmap'd `candidate_vectors.parquet`/`anchor_vectors.npy`. **The 384-d array is never inlined** (§Q).
- **Validation:** every similarity ∈ `[-1,1]`; `net_semantic_fit ∈ [0,1]`; `anchor_similarities` keys ⊆ artifact anchor set (`ArtifactContractError` otherwise).
- **Invariants:** `best_positive_anchor == argmax over positive anchors` (deterministic, ties by `AnchorId` ascending); `net_semantic_fit` is a documented bounded function of `positive_fit` and `negative_fit`.
- **Construction:** R3, by `SemanticEngine` via `SemanticVectorStorePort` (lookup-first) / `EmbeddingModelPort` (fallback) + anchor vectors.
- **Relationships:** `SEMANTIC_FIT` component of CQV; the negative-anchor term is the dense counterpart to the lexical keyword-stuffing trap.

### §G.6 `CredibilityProfile` (`candidate/credibility.py`)
- **Purpose:** trustworthiness of claimed skills/profile — the anti keyword-stuffing layer.
- **Fields:** `skill_trust: Mapping[SkillName, SkillTrust]`; `keyword_stuffing_score: UnitScore`; `claimed_vs_assessed_gap: UnitScore`; `title_description_coherence: UnitScore`; `relevant_skill_credibility: UnitScore`.
  - **`SkillTrust` VO:** `name: SkillName, proficiency: Proficiency, endorsements: int, duration_months: Months | None, assessment_score: UnitScore | None, trust: UnitScore, is_credible: bool`. `trust` combines endorsement weight × duration evidence × assessment coherence (the endorsement-and-duration trust multiplier).
- **Validation:** all `UnitScore`s ∈ `[0,1]`; assessment scores routed here from `redrob_signals.skill_assessment_scores` (single-ownership, §G.7).
- **Invariants:** `keyword_stuffing_score` high ⇒ many high-proficiency skills with zero endorsements/duration; `title_description_coherence` low when `current_title` contradicts career descriptions (the "Marketing Manager with all the AI keywords" trap).
- **Construction:** R2, by `features.skills` + `LexiconEngine`.
- **Relationships:** `CREDIBILITY` component; multiplicative trust dampener on `SKILL_MATCH`; feeds Eligibility (`LANGCHAIN_OPENAI_ONLY_RECENT`) and Reasoning.

### §G.7 `BehavioralProfile` (`candidate/behavioral.py`) — built on the 23 Redrob signals
- **Purpose:** normalize platform behaviour into bounded availability/responsiveness/engagement/reliability/verification components, per `redrob_signals_doc`. These feed a **multiplier**, never base relevance: a perfect-on-paper candidate inactive for 6 months with a low response rate is, for hiring, not available, and is down-weighted accordingly.
- **`RawSignals` VO** mirrors all 23 signals exactly. **Single-ownership routing** (each signal contributes to exactly one normalized family; cross-references documented, no double counting):

| # | Signal | Type / range | Owner family | Sentinel / normalization rule |
|---|---|---|---|---|
| 1 | `profile_completeness_score` | 0–100 | Verification | /100 |
| 2 | `signup_date` | date | Context | platform tenure vs `as_of` |
| 3 | `last_active_date` | date | **Availability** | recency vs `as_of`; stale (e.g. >180d) ⇒ low |
| 4 | `open_to_work_flag` | bool | **Availability** | direct |
| 5 | `profile_views_received_30d` | int≥0 | Engagement | bounded log-scale |
| 6 | `applications_submitted_30d` | int≥0 | Engagement | bounded |
| 7 | `recruiter_response_rate` | 0–1 | **Responsiveness** | direct |
| 8 | `avg_response_time_hours` | ≥0 | **Responsiveness** | inverse, bounded |
| 9 | `skill_assessment_scores` | dict→0–100 | → **Credibility** (§G.6) | `{}` ⇒ `UNKNOWN`, not 0 |
| 10 | `connection_count` | int≥0 | Engagement | bounded log-scale |
| 11 | `endorsements_received` | int≥0 | → **Credibility** (§G.6) | bounded |
| 12 | `notice_period_days` | 0–180 | → **Logistics** (§G.8) | — |
| 13 | `expected_salary_range_inr_lpa` | obj | → **Logistics** (§G.8) | inversion = sanity flag |
| 14 | `preferred_work_mode` | enum | → **Logistics** (§G.8) | — |
| 15 | `willing_to_relocate` | bool | → **Logistics** (§G.8) | — |
| 16 | `github_activity_score` | −1..100 | Engagement/→Credibility | **−1 ⇒ UNKNOWN**, never 0 |
| 17 | `search_appearance_30d` | int≥0 | Engagement | bounded |
| 18 | `saved_by_recruiters_30d` | int≥0 | Engagement (recruiter demand) | bounded |
| 19 | `interview_completion_rate` | 0–1 | **Reliability** | direct |
| 20 | `offer_acceptance_rate` | −1..1 | **Reliability** | **−1 ⇒ UNKNOWN**, never 0 |
| 21 | `verified_email` | bool | Verification | direct |
| 22 | `verified_phone` | bool | Verification | direct |
| 23 | `linkedin_connected` | bool | Verification | direct |

- **Derived fields:** `availability: UnitScore`, `responsiveness: UnitScore`, `engagement: UnitScore`, `reliability: UnitScore`, `verification: UnitScore`, each paired with a `SignalAvailability` so "unknown" (sentinel-driven) is distinguished from "low". Plus `raw: RawSignals` retained for evidence.
- **Validation:** rates ∈ `[0,1]`; dates valid; **every sentinel (`−1`, `{}`) maps to an explicit `UNKNOWN`, never silently to 0** — this is an invariant, because conflating "no GitHub" with "bad GitHub" would mis-rank engineers who simply didn't link an account.
- **Invariants:** each derived family ∈ `[0,1]` or `UNKNOWN`; recency computed only against injected `as_of`. The final behavioral **multiplier** is produced by `BehavioralEngine`, not stored here — domain holds normalized inputs only, keeping the multiplier policy in the engine layer.
- **Construction:** R2, by `features.signals`.
- **Relationships:** behavioral multiplier input to Scoring; availability/responsiveness surfaced in Reasoning.

### §G.8 `LogisticsProfile` (`candidate/logistics.py`)
- **Purpose:** location / notice / relocation / salary fit vs JD (owns signals 12–15).
- **Fields:** `location: str, country: str, location_fit: LocationFit, willing_to_relocate: bool, notice_period_days: int, notice_fit: NoticeFit, preferred_work_mode: WorkMode, work_mode_fit: UnitScore, salary: SalaryBand`.
  - **`SalaryBand` VO:** `min_lpa: LpaAmount, max_lpa: LpaAmount, is_inverted: bool`.
- **Validation:** `notice_period_days ∈ [0,180]`; `min_lpa, max_lpa ≥ 0`.
- **Invariants:** `is_inverted == (min_lpa > max_lpa)` (preserved, not corrected); `location_fit` derived from `(location, country, willing_to_relocate)` against the JD hub set (Pune/Noida/Hyderabad/Mumbai/Delhi-NCR); `notice_fit` thresholds: ≤30 ideal, buyout window, >30 higher bar.
- **Construction:** R2, by `features.signals` + `LogisticsEngine`.
- **Relationships:** logistics multiplier input to Scoring; Reasoning ("Bangalore-based, 30-day notice").

### §G.9 `ArchetypeAssignment` (`candidate/archetype.py`)
- **Purpose:** which O7 cluster the candidate belongs to.
- **Fields:** `archetype_id: ArchetypeId, distance: float, membership_confidence: UnitScore, secondary_archetype: ArchetypeId | None, label: str | None, is_target_archetype: bool`.
- **Validation:** `distance ≥ 0`; `confidence ∈ [0,1]`; `archetype_id` ∈ artifact centroid set (`ArtifactContractError` otherwise).
- **Invariants:** assignment is nearest-centroid under a fixed metric; ties broken by `ArchetypeId` ascending (determinism).
- **Construction:** R3, by `SemanticEngine` (nearest-centroid, pure numpy over `centroids.npy`).
- **Relationships:** `ARCHETYPE_FIT` component / boost for target archetypes; Reasoning context.

### §G.10 `CandidateQualityVector` (CQV) (`candidate/quality.py`)
- **Purpose:** the fixed-length numeric feature vector that `ScoringEngine` consumes — the boundary between rich typed profiles and vectorized scoring.
- **Fields:** `values: numpy float32 array (dim == D)`; `schema_version: str`; `layout_ref: FeatureLayout` (module-level frozen constant, **not** stored per instance).
  - **`FeatureLayout`** (module constant): an ordered, versioned tuple of `(FeatureName, FeatureIndex, source_slice, bounds)`. The single source of truth for feature order, shared by reference across all candidates.
- **Validation:** `len(values) == D`; **no `NaN`/`inf`** (all sentinels resolved upstream); each index within its documented bounds; `schema_version` matches `MANIFEST` (`CQVInvariantError`/`ArtifactContractError`).
- **Invariants:** feature order is **immutable and versioned**; identical inputs ⇒ identical vector (float32, fixed reduction order).
- **Construction:** R5, by `CQVAssembler`, folding all populated slices. **Bulk path:** the pipeline builds one `(N, D)` matrix in parallel to `FeatureLayout`; per-candidate `CQV` objects are materialized only for survivors/top-K (§Q).
- **Relationships:** dotted with `ScoringWeights` to yield base relevance.

---

## §H. ScoreBreakdown Architecture (`domain/scoring.py`)

- **`ScoringWeights` VO** (from `scoring_weights.locked.yaml`): `Mapping[ScoreComponent, float]`, `schema_version`. Frozen; weight set must equal `ScoreComponent` exactly (`ArtifactContractError` otherwise).
- **`ScoreComponentValue` VO:** `component: ScoreComponent, raw: UnitScore, weight: float, weighted: float, evidence: tuple[EvidenceRef, ...]`. Invariant: `weighted == raw * weight`.
- **`GateOutcome` VO:** `passed: bool, reason: EligibilityCode | IntegrityFlag | None`.
- **`ScoreBreakdown` VO:** `components: tuple[ScoreComponentValue, ...]` (one per `ScoreComponent`), `base_relevance: Score`, `integrity_gate: GateOutcome`, `eligibility_gate: GateOutcome`, `behavioral_multiplier: Multiplier`, `logistics_multiplier: Multiplier`, `archetype_adjustment: float`, `final_score: Score`.
- **`ScoredCandidate` VO:** `candidate_id: CandidateId, final_score: Score, breakdown: ScoreBreakdown, tiebreak_key: CandidateId`.

**Invariants (the scoring contract):**
1. `base_relevance == Σ component.weighted` (within float tolerance, summed in `FeatureLayout`/`ScoreComponent` order for determinism).
2. **Gating:** if `not integrity_gate.passed` *or* `not eligibility_gate.passed` ⇒ `final_score == FLOOR` (a fixed sentinel, e.g. `0.0`). A floored candidate can still be ranked into the filler tail but never the top.
3. `final_score` is a documented, deterministic function of `base_relevance`, the two bounded multipliers, and `archetype_adjustment`.
4. Multipliers ∈ their declared bounds; behavioral/logistics modulate, never create, relevance.
5. `tiebreak_key == candidate_id` (validator-mandated secondary order).

**Construction:** R5, by `ScoringEngine` from a `VECTORIZED`-stage `CandidateRepresentation` + `ScoringWeights`.

---

## §I. Ranking Models (`domain/ranking.py`)

- **`RankedCandidate` VO:** `candidate_id: CandidateId, rank: int, score: Score, scored: ScoredCandidate, reasoning: CandidateReasoning | None`.
- **`Ranking` aggregate:** `ordered: tuple[RankedCandidate, ...]`, `size: int` (default 100), `honeypot_count: int` (audit).

**Invariants enforced in the `Ranking` factory — this is where the validator's rules become unrepresentable-if-violated** (mirrors `validate_submission.py` exactly; violation raises `RankingInvariantError`):
1. `len(ordered) == size` (exactly 100).
2. `ranks == {1..size}`, each once.
3. `candidate_id`s unique, each matches `^CAND_[0-9]{7}$`.
4. `score` non-increasing by `rank`.
5. Tie-break: `score[i] == score[i+1] ⇒ candidate_id[i] < candidate_id[i+1]`.
6. `ordered` sorted by `(−score, candidate_id)`.

**Construction:** R6, by `RankingEngine.rank(scored_candidates)`: sort by `(−final_score, candidate_id)`, take top `size`, assign ranks `1..size`, then run the invariant check before returning. Reasoning is attached later (R7) via copy-on-write `with_reasoning(...)`.

**Relationships:** consumed by the CSV adapter (R8) and re-checked by `ValidationEngine` (R9, defence in depth).

---

## §J. Reasoning Models (`domain/reasoning.py`)

Designed against the six Stage-4 checks (specific facts, JD connection, honest concerns, no hallucination, variation, rank consistency).

- **`ReasoningClause` VO:** `polarity: ReasoningPolarity, fragment: str, evidence: tuple[EvidenceRef, ...]` (**≥1, enforced at construction** → no-hallucination guarantee), `jd_link: EligibilityCode | ScoreComponent | None` (which JD requirement this connects to).
- **`CandidateReasoning` VO:** `candidate_id: CandidateId, clauses: tuple[ReasoningClause, ...], rendered: str, rank_band: Literal["top","mid","tail"]`.

**Invariants:**
1. Every clause cites ≥1 resolvable `EvidenceRef` (no hallucination).
2. ≥1 `STRENGTH` clause for `top`/`mid` bands; ≥1 `CONCERN` clause whenever an eligibility soft-penalty or a known gap exists (honest concerns).
3. ≥1 clause carries a non-null `jd_link` (JD connection).
4. `rendered` ≤ 2 sentences; tone consistent with `rank_band` (rank consistency: `top` net-positive, `tail` measured).
5. **Determinism without templating:** `rendered` is a pure function of the *ordered clause set*; variation across candidates arises from *which evidence is present*, not from any randomness or name-insertion template. Two candidates with different evidence necessarily render differently; identical evidence renders identically.

**Construction:** R7, by `ReasoningEngine` from `RankedCandidate` + a top-K `CandidateRepresentation` whose `ProvenanceHandle.inline` is populated.

---

## §K. Validation Models (`domain/validation.py`)

- **`ValidationFinding` VO:** `code: ValidationCode, severity: Severity, message: str, location: str | None`.
- **`ValidationReport` VO:** `findings: tuple[ValidationFinding, ...], is_valid: bool, checks_run: frozenset[ValidationCode]`.
- **`ValidationCode`** mirrors `validate_submission.py` (`WRONG_ROW_COUNT, RANK_OUT_OF_RANGE, DUPLICATE_RANK, MISSING_RANK, DUPLICATE_ID, BAD_ID_FORMAT, SCORE_NOT_FLOAT, SCORE_INCREASING, TIEBREAK_VIOLATION`) **plus** Stage-4 reasoning checks (`EMPTY_REASONING, IDENTICAL_REASONING, TEMPLATED_REASONING, HALLUCINATION, RANK_REASONING_MISMATCH`).
- **Invariant:** `is_valid == (no finding with severity == HARD)`.
- **Relationship:** because `Ranking` already enforces structural rules at construction, `ValidationReport` is belt-and-suspenders for structure and the *primary* home for reasoning-quality verdicts.

---

## §L. Aggregate Root Design (`candidate/representation.py`)

- **`CandidateRepresentation`** is the aggregate threaded R1→R7.
- **Fields:** `identity: Identity` (required); `career, credibility, behavioral, logistics: <Profile> | None` (R2); `semantic, archetype: <…> | None` (R3); `integrity, eligibility: <Report> | None` (R4); `quality: CandidateQualityVector | None` (R5); `stage: BuildStage`.
- **Chosen state model:** *optional slices + a monotonic `BuildStage` discriminant*, with `require_career()` / `require_quality()` accessors that raise `RepresentationStageError` if accessed before population. (Considered and rejected for v1.1: distinct per-stage types — strictly safer but causes copy/type explosion across the pipeline; the discriminant approach gives most of the safety with far less ceremony, and the staged accessors localize the runtime check.)
- **Invariants:** `identity` always present; `stage` advances monotonically and matches the set of populated slices; at most one of each slice; `candidate_id` identical across every slice that carries one.
- **Equality/hash:** by `candidate_id` (the CQV's numpy array is excluded from hashing; representations are identity-equal).
- **Relationships:** the universal carrier — engines read the slices they need and return an enriched copy.

---

## §M. Construction Flow (R-stage → slice → producer → invariant)

| Stage | Slice attached | Producer | `BuildStage` after | Key invariant checked |
|---|---|---|---|---|
| R1 Parse | `identity` (+`RawCandidate`) | `features.parsing` | `PARSED` | id pattern; provenance id match |
| R2 Features | `career, credibility, behavioral, logistics` | `features.*` | `FEATURED` | ranges; single `is_current`; sentinel→UNKNOWN |
| R3 Semantic | `semantic, archetype` | `SemanticEngine` (ports) | `SITUATED` | similarity bounds; anchors/centroids ⊆ artifact |
| R4 Gates | `integrity, eligibility` | Integrity/Eligibility engines | `GATED` | honeypot/eligibility derivation; evidence present |
| R5 Vectorize | `quality` | `CQVAssembler` | `VECTORIZED` | dim==D; no NaN; schema_version |
| R5 Score | → `ScoredCandidate` | `ScoringEngine` | `SCORED` | gating; `base==Σ weighted` |
| R6 Rank | → `Ranking` | `RankingEngine` | `RANKED` | count/unique/monotonic/tie-break |
| R7 Reason | `reasoning` on `RankedCandidate` | `ReasoningEngine` | `EXPLAINED` | evidence≥1; rank consistency |

---

## §N. Immutable Design Strategy

- All models `frozen=True, extra="forbid"`. Collections are `tuple`/`frozenset`/read-only `Mapping`.
- No in-place mutation anywhere; the aggregate grows via copy-on-write.
- Inlined numpy arrays (CQV, top-K only) are set `writeable=False` to prevent aliased mutation.
- `arbitrary_types_allowed=True` is granted **only** to `CandidateQualityVector` (for the ndarray); every other model stays strictly pydantic-native.

## §O. Copy-on-Write Update Pattern

- Each slice attaches via a `with_<slice>()` method returning a **new** `CandidateRepresentation` (`model_copy(update={...})`), advancing `stage`.
- COW methods **reject regressions** (attaching `quality` before `career`/`semantic`/`integrity` exist raises `RepresentationStageError`) and **reject overwrite** (attaching a slice twice raises).
- `Ranking.with_reasoning(mapping)` produces a new `Ranking` with reasoning attached to each `RankedCandidate`; structural invariants are re-asserted.
- Because every transform is `f(rep) -> rep'`, each stage is referentially transparent and unit-testable in isolation.

## §P. Serialization Strategy

- Serialization is for **run reports, debugging, and artifact contracts only** — never the hot path. The submission CSV is produced by an adapter from `Ranking`; the domain knows nothing about CSV.
- `model_dump()` / `model_dump_json()` with: enums **by value**, deterministic key ordering, fixed float formatting (so run-report hashes are reproducible).
- `RawCandidate` ⇄ JSON round-trips losslessly (provenance depends on it).
- The CQV ndarray serializes (to list) **only** when dumping a top-K representation; bulk per-row vectors are never serialized.
- A `schema_version` accompanies any serialized CQV/weights/anchor-bearing object; load-time mismatch raises `ArtifactContractError`.

## §Q. Memory Efficiency Considerations (16 GB ceiling, 100K rows)

- **Vectors are referenced, not inlined.** `SemanticProfile.vector_ref` indexes the mmap'd store; 100K × 384 × float32 ≈ 150 MB stays out of Python objects.
- **Columnar bulk path.** The 100K flow operates on a single `(N, D)` float32 CQV matrix aligned to `FeatureLayout`; full `CandidateRepresentation`/`RawCandidate` objects are materialized **only** for gate survivors and the top-K reasoning set. `ProvenanceHandle` carries `source_index` in bulk and is re-hydrated to `inline` for the top-K only.
- **Shared immutables:** `FeatureLayout`, `ScoringWeights`, `JobDescriptionSpec`, and anchor sets are single module/run-level constants shared by reference, never copied per candidate.
- **Interning** of high-cardinality repeated strings (company names, skill tokens) is applied in `features` so domain VOs reference shared instances.
- Prefer `tuple` over `list` (lower overhead, hashable); avoid pandas object-dtype anywhere a numpy array suffices.

## §R. Determinism Considerations

- **No wall clock.** Every recency/tenure computation takes an injected `as_of: date` (from `configs/runtime/online.yaml`); the domain never calls `datetime.now()`.
- **No RNG.** All ties (anchors, archetypes, ranking) resolve by ascending `CandidateId`/`AnchorId`/`ArchetypeId`.
- **Stable iteration.** `Mapping` fields are consumed via sorted keys; `findings`/`clauses`/`components` are stored pre-sorted by code/index.
- **Stable float reduction.** Component and feature sums are reduced in `ScoreComponent`/`FeatureLayout` order; dtype is fixed float32; documented rounding for any display/format.
- **Enum-by-value** serialization keeps outputs stable across code reorders.

## §S. Modules that must remain pure & deterministic

**All of `domain/` is pure and deterministic, without exception.** It contains no IO, no ML, no clock, no randomness. The only `Any`-typed boundary is `RawCandidate.from_mapping`, which narrows immediately. Impurity (onnxruntime, parquet, filesystem) lives strictly in `adapters/`; the domain never imports it. `SemanticProfile` describes dense fit but holds only a `VectorRef` — the array itself is produced behind a port, outside the domain.

## §T. Type Safety Considerations

- `mypy --strict` across the layer; `Any` forbidden except the single parsing boundary.
- NewTypes separate `Score` / `Similarity` / `UnitScore` / `Multiplier` so they cannot be interchanged silently.
- Closed vocabularies are enums or `Literal`; read-only `Mapping`/`Sequence` in all signatures.
- Leaf models are `@final`; runtime pydantic validation backstops static guarantees (range, pattern, NaN).
- `ProvenanceHandle`, `EvidenceRef`, and the `ReasoningClause` evidence requirement turn the anti-hallucination rule into a compile-and-construct-time constraint rather than a review-time hope.

---

## Build order for implementation

1. `ids.py`, `enums.py`, `errors.py`, `provenance.py` (primitives — no intra-domain deps).
2. `source.py` (`RawCandidate` + nested raw VOs).
3. The ten slices under `candidate/` (each depends only on primitives + `source`).
4. `quality.py` + `FeatureLayout` constant.
5. `scoring.py` → `ranking.py` → `reasoning.py` → `validation.py`.
6. `representation.py` (aggregate root, COW, stage guards) last — it composes everything above.

Each step is independently unit- and property-testable before the next begins.
