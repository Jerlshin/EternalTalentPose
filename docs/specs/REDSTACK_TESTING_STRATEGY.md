# REDSTACK v1.1 — Testing Strategy

**Status: architecture frozen.** This is a testing *architecture specification*, not a coding guide. It introduces no new architecture, ports, engines, pipelines, or artifacts. It defines how the engineering team proves — to itself and to an external Stage-3/4/5 audit — that the frozen system is correct, deterministic, reproducible, performant, explainable, competition-compliant, honeypot-robust, and architecturally sound. Every authority cited (`REDSTACK_ARCHITECTURE.md`, the eight layer/pipeline/layout docs, `submission_spec`, `candidate_schema.json`, `sample_submission.csv`, `validate_submission.py`, the JD, `redrob_signals_doc`) is treated as an immutable contract that the tests *encode*, never reinterpret.

**The governing reality that shapes every decision below: there is no live leaderboard.** Scoring happens once, after submissions close. There is a three-submission cap. Silent degradation is therefore undetectable in production and unrecoverable at submission time. The test suite is the *only* feedback loop the team has. Consequently the suite is designed to catch a defect *before* it ships, not to diagnose one after — every category exists to close a specific failure mode that the competition would otherwise punish irreversibly.

**Test root (frozen, from `REDSTACK_REPOSITORY_LAYOUT.md §15):**

```text
tests/
├── unit/         property/      golden/
├── integration/  determinism/   contract/
├── fixtures/     conftest.py
```

A ninth logical category — **performance** — runs inside `integration/` and `determinism/` (it needs the full online pipeline assembled) but is specified separately (§14) because its pass/fail criteria are competition limits, not behavioral assertions.

---

## §1. Testing Philosophy

Six principles, each tied to a competition consequence.

**Deterministic-systems testing.** REDSTACK is a deterministic system by construction: no wall clock (recency uses an injected `as_of`), no online RNG (`OnlineEntropy` raises on `numpy_generator`), float32 with fixed reduction order, pinned BLAS/OMP threads, ties resolved by ascending `candidate_id`, enum-by-value serialization. Tests therefore assert *exact equality* (byte-for-byte CSV, identical run-report `reproducible` block), not statistical closeness. The single documented exception is the cross-runtime embedding boundary (sentence-transformers ↔ onnxruntime), where the contract is cosine agreement within ε (≥ 0.999), never bitwise — and the suite enforces exactly that boundary and no looser.

**Architecture-first testing.** The hexagonal boundaries are not style preferences; they are the mechanism that keeps the online run inside 5 minutes (an engine cannot import `onnxruntime` because it cannot import `adapters`). The suite treats the eight import-linter contracts (`REPOSITORY_LAYOUT §17`) as first-class test subjects: a CI job asserts import-linter exits clean, and contract tests prove engines depend only on port abstractions. A boundary violation is a *build break*, tested as such.

**Reproducibility-first testing.** Stage 3 reproduces the ranking step in a sandboxed Docker container matching the compute limits exactly; non-reproducibility is disqualification regardless of composite score. The suite makes reproducibility a tested property at three scopes: same-process twice, fresh-process restart, and the offline→online artifact handoff. The reproducibility certification gate (§19) is the release blocker.

**Offline-online parity.** The architecture's central performance lever is that R3 is *lookup*, not *encode*, and R5 *applies* locked weights rather than calibrating. This only works if the features the online path computes are identical to the features the offline calibration (O8/O9) trusted. Parity testing (the O18 obligation, mirrored in `determinism/` and `golden/`) recomputes features online for a sample and diffs against `feature_snapshot.parquet` — if they diverge, the locked weights are calibrated against features that no longer exist, and the ranking is silently wrong.

**Fail-fast philosophy.** Ports and adapters raise on integrity/contract violations (hash mismatch, wrong dim, broken submission invariant); they never degrade. The suite asserts the *raising*, not a fallback — a tampered artifact byte must produce `ArtifactContractError`, a sub-100 candidate set must produce `RankingInvariantError`, an unverified artifact must abort R0. Tests that assert "graceful degradation" would be testing a behavior the architecture forbids.

**Evidence-backed validation.** The anti-hallucination guarantee is structural: a `ReasoningClause` cannot be constructed without ≥1 resolvable `EvidenceRef`; a `FeatureCell` carries its `EvidenceRef`s; the explainability chain runs `RankedCandidate → ScoreBreakdown.component → FeatureImportance → FeatureCell → EvidenceRef → RawCandidate.field`. The reasoning suite (§11) tests that this chain holds for every emitted claim, because Stage 4 samples 10 rows and penalizes any claim not corresponding to the profile.

**Why these are required here specifically.** A normal product can ship a fix next sprint. REDSTACK gets one scored submission with no feedback and a hard reproduction gate. The philosophy converts every competition elimination criterion (format auto-reject, honeypot >10%, non-reproduction, failed reasoning checks, can't-defend interview) into an automated, blocking, pre-submission test.

---

## §2. Test Pyramid

A formal pyramid weighted toward fast, pure, deterministic tests at the base, with a thin cap of expensive whole-system tests. Distribution percentages are **targets by test count**, not by runtime.

| Layer | Objective | Scope | Failure criteria | Ownership | Target % |
|---|---|---|---|---|---|
| **Unit** | Prove each pure callable correct in isolation. | One domain model / feature transform / engine method; fakes for the one port `semantic` needs. | Any incorrect value, unhandled edge case, or unraised invariant. | Each layer's owner | **45%** |
| **Property** | Prove invariants hold over generated input space. | Ranking invariants, score monotonicity, integrity idempotence, feature range/no-NaN, COW stage monotonicity, tie-break totality. | A single counterexample (Hypothesis shrinks it). | Domain + Engines | **15%** |
| **Contract** | Prove every port implementation behaves identically. | One parametrized suite per port × {real adapter, fake}. | Any adapter diverging from its fake or the contract. | Ports | **10%** |
| **Golden** | Lock exact outputs against regression. | Fixed fixtures → exact `ScoreBreakdown`, CSV bytes, reasoning text, run-report `reproducible` block. | Any byte/value drift not explicitly re-blessed. | Modeling + Platform | **10%** |
| **Integration** | Prove stages compose end-to-end. | Full O0–O18 build and R0–R9 rank on `sample_candidates`; CLI verbs. | Pipeline error; organizer validator rejects output. | Pipelines | **10%** |
| **Determinism** | Prove identical inputs ⇒ identical outputs. | Twice-run byte diff; 1-thread vs N-thread; restart; seed. | Any difference in the `reproducible` region. | Platform | **5%** |
| **Performance** | Prove competition compute limits hold. | Online R0–R9 timing + RSS on the sample and a scaled pool. | Runtime > budget; RSS > budget; any network/GPU use. | Platform | **3%** |
| **Reproduction** | Prove Stage-3 sandbox reproducibility. | The single `redstack rank` command in a clean, network-less, CPU-only environment. | Output differs from reference, or run fails under limits. | Release owner | **2%** |

**Pyramid discipline.** Because engines/features/domain are pure, the base is broad and cheap and needs no mocks. Mocking happens *only* at the port boundary, and even there behavioral *fakes* (in-memory real implementations) are preferred over `unittest.mock` — so tests exercise real contract behavior, not call sequences. The expensive top (integration/determinism/performance/reproduction) is small but blocking: a green base with a red cap never ships.

---

## §3. Domain Layer Testing

**Authority:** `REDSTACK_DOMAIN_LAYER.md`. **Location:** `tests/unit/domain/`, `tests/property/domain/`, `tests/golden/domain/`. **Principle:** the domain makes illegal states unrepresentable; the suite proves that constructing an illegal state *raises*, and that legal states round-trip and order correctly. The domain is pure, so every test is fast and mock-free.

**Cross-cutting obligations (apply to every value object):**
- **Constructor validation** — out-of-range floats, `NaN`/`inf`, pattern violations, and shape errors raise at construction (`frozen=True, extra="forbid"`); unknown keys raise.
- **Illegal-state prevention** — the specific invariant of each type is proven unrepresentable (enumerated below).
- **Immutability** — mutation attempts fail; collections are `tuple`/`frozenset`/read-only `Mapping`; the CQV ndarray is `writeable=False`.
- **Equality semantics** — identity-bearing types (`Identity`, `CandidateRepresentation`, `ScoredCandidate`, `RankedCandidate`) compare/hash by `candidate_id`; value objects compare structurally.
- **Ordering semantics** — ordered enums sort by explicit `ordinal`, never definition order; tie domains order by ascending id.
- **Serialization stability** — `model_dump`/`model_dump_json` emit enums by value, deterministic key order, fixed float formatting; round-trips are lossless (`RawCandidate` especially); a `schema_version` mismatch raises `ArtifactContractError`.

**Per-target test matrix:**

| Target (module) | Invariant / illegal state proven to raise | Key positive assertions |
|---|---|---|
| `CandidateRepresentation` (`candidate/representation.py`) | Attaching a slice out of order or twice raises `RepresentationStageError`; accessing a slice before population raises; `stage` regression raises. | COW `with_*()` advances `BuildStage` monotonically (`PARSED→…→EXPLAINED`); `candidate_id` identical across slices; equality by id excludes the CQV array. |
| `Ranking` (`ranking.py`) | Any of the six validator rules violated raises `RankingInvariantError`: ≠100 rows, rank ∉ {1..100}, duplicate rank, duplicate/ill-formed id, score increasing by rank, tie not id-ascending. | `ordered` sorted by `(−score, candidate_id)`; `with_reasoning` re-asserts invariants. |
| `Score` / `Similarity` / `UnitScore` / `Multiplier` (`ids.py` + minting sites) | Passing a `Similarity` where a `Score` is expected is a mypy error (static); range/`NaN` violations raise at the minting model. | `Score` minted only by `ScoringEngine`; `Similarity` only by `SemanticEngine`; bounds enforced. |
| `IntegrityReport` (`candidate/integrity.py`) | A finding without ≥1 `EvidenceRef` raises; `honeypot_score ∉ [0,1]` raises. | `is_honeypot == (any HARD finding) OR (score ≥ threshold)`; findings sorted by code; `INVERTED_SALARY_BAND` is **not** a honeypot flag. |
| `EligibilityReport` (`candidate/eligibility.py`) | Finding without evidence raises. | `is_eligible == (len(hard_blocks)==0)`; soft penalties never set ineligible; codes ⊆ `EligibilityCode`. |
| `FeatureCell` (`features` model, exercised here via domain types it carries) | A cell value `NaN`/out-of-range raises; evidence path that doesn't resolve in `RawCandidate` raises `ProvenanceError`. | `(value, confidence, evidence)` triple; confidence ∈ [0,1]. |
| `CandidateQualityVector` (`candidate/quality.py`) | `len(values) ≠ D`, any `NaN`/`inf`, or `schema_version`≠manifest raises `CQVInvariantError`/`ArtifactContractError`. | float32; `FeatureLayout` order immutable + versioned; identical inputs ⇒ identical vector. |
| `ScoreBreakdown` / `ScoredCandidate` (`scoring.py`) | `weighted ≠ raw*weight` raises; `base ≠ Σ weighted` raises; floored candidate with `final_score ≠ FLOOR` raises `ScoreInvariantError`. | sum reduced in `ScoreComponent` order; multipliers within bounds; `tiebreak_key == candidate_id`. |
| All **IDs** (`ids.py`) | `CandidateId` not matching `^CAND_[0-9]{7}$` raises at the `Identity`/parsing boundary. | NewType separation holds under mypy; `Months ≥ 0`, `LpaAmount ≥ 0`. |
| All **enums** (`enums.py`) | Comparing by raw string is prevented (tests assert `ordinal` usage); unknown value rejected. | `CompanySize`, `Proficiency`, `InstitutionTier`, `LanguageProficiency`, `BuildStage`, `RelevanceTier` order correctly; serialize by value; values equal dataset strings exactly. |
| `CandidateReasoning` / `ReasoningClause` (`reasoning.py`) | A clause without ≥1 `EvidenceRef` raises (the no-hallucination guarantee at construction). | ≥1 clause carries a `jd_link`; `rendered` ≤ 2 sentences; tone matches `rank_band`. |
| `ValidationReport` (`validation.py`) | — | `is_valid == (no HARD finding)`; codes mirror `validate_submission.py` + Stage-4 checks. |
| `RawCandidate` (`source.py`) | True type/shape violations raise `SchemaError`; **semantic contradictions are preserved, not rejected** (inverted salary, `is_current`+`end_date`, expert@0-months) — tested explicitly so the honeypot path stays armed. | `from_mapping` is the only `Any` boundary; round-trips losslessly; all 23 signals typed exactly. |

**Property tests (`tests/property/domain/`):** generate arbitrary scored sets and assert `Ranking` always yields 100 unique ranks, non-increasing scores, id-ascending tie-breaks; generate arbitrary representations and assert COW never mutates in place and `stage` only advances; assert integrity/eligibility derivations are idempotent.

---

## §4. Feature Layer Testing

**Authority:** `REDSTACK_FEATURE_LAYER.md` (Parts 1–10). **Location:** `tests/unit/features/`, `tests/property/features/`, `tests/golden/features/`, with shared fixtures in `tests/fixtures/`. **Principle:** a feature value is an *evidence aggregate*, never a keyword flag — so the suite's central job is to prove that corroboration drives value and that stuffers score ≈ 0 by construction.

**Coverage obligations:**

**All 30 feature groups (Part 1).** Each group (`id, geo, exp, sen, edu, co, pvs, retr, rank, recsys, ir, nlp, llm, mle, mlops, eval, oss, lead, startup, found, avail, eng, resp, sal, reloc, notice, bhv, risk, cons, hp`) gets a golden value test on `sample_candidates`: a hand-constructed candidate with known fields produces a known cell value, confidence, and evidence set. Every group's documented **failure mode** is exercised (e.g. unknown city → `geo` low-confidence not hard-block; missing GitHub → `oss` UNKNOWN not penalty; inverted salary → `sal` soft flag not honeypot).

**All `jd.*` latent families (Part 2).** Positive latents (`jd.retrieval_ranking, jd.production_ml, jd.product_company, jd.shipping_mentality, jd.eval_framework, jd.hybrid_retrieval`) and negative latents (`jd.keyword_only, jd.consulting_only, jd.title_chaser, jd.pure_researcher, jd.framework_enthusiast, jd.inactive`). For each: a fixture that should activate it, a fixture that should not, and a low-evidence fixture proving the latent regresses toward the neutral prior (confidence-weighted). Negative latents are tested as *subtractive* on the positives they oppose.

**Competency aggregation.** The fused output `competency = w1·trust + w2·in_career + w3·semantic − w4·stuffing_penalty` is tested at its corners: claimed-only with zero corroboration ⇒ competency ≈ 0; claimed + in-career + trusted + semantic ⇒ high; trusted-but-not-semantic and semantic-but-not-trusted intermediate. The monotonicity property (more corroboration never lowers competency) is a Hypothesis test.

**Anti-keyword-stuffing behavior.** The decisive competition signal. A dedicated stuffer fixture family (§16) — many ADVANCED/EXPERT skills with zero endorsements, zero duration, zero career mentions, mismatched title — must produce: high `*.claimed`, near-zero `*.trust`/`*.in_career`/`*.semantic`, large `jd.keyword_only`, and net competency ≈ 0. This is asserted as a hard floor: a stuffer's fused competency must sit below a genuine practitioner's by a wide, fixture-pinned margin.

**Confidence propagation.** Every cell emits a confidence; the suite proves confidence *falls* with UNKNOWN density and evidence sparsity and *rises* with corroborating-source count, and that group-confidence `(N,G)` aggregates correctly from cell confidences. Latent confidence regressing a sparse profile toward neutral is tested as a value assertion.

**Evidence propagation.** Every non-structural cell carries `EvidenceRef`s whose `path` resolves in the source `RawCandidate`; a dangling path raises `ProvenanceError`. The full `FeatureLineage` chain (feature → dependencies → raw fields) is walked for a sample candidate and asserted acyclic and complete (every leaf terminates at a real field).

**Fixture requirements:** a labeled set covering each group's positive/negative/unknown cases; the stuffer family; a consulting-only career; an ideal-JD-match; sentinel-laden behavioral profiles (`github=−1`, `offer_acceptance=−1`, `skill_assessment={}`); impossible-timeline honeypots. Each fixture is a frozen `RawCandidate` with an expected-cells table.

**Edge cases:** empty `skills`/`education`; single-position careers; all-sentinel signals; maximum-length collections (10 positions, 5 educations); Unicode in names/companies; duplicate skill tokens; future-dated `last_active`.

**Confidence + evidence testing** are first-class (above), not afterthoughts — Stage 4 traceability and the no-hallucination guarantee both depend on them.

**Feature drift testing.** Because `feature_snapshot.parquet` (O14) is the calibration substrate *and* the online correctness oracle, the suite includes drift tests: (1) **distribution drift** — recomputed feature distributions over the census sample stay within the documented expected distributions (Part 3) per group; a shift beyond tolerance fails, catching an extractor regression before it poisons weights. (2) **layout drift** — `layout_version` of `FeatureLayout`, `feature_manifest.json`, and `scoring_weights.locked.yaml` agree; any disagreement fails (mirrors the R0/R5 `ArtifactContractError`). (3) **online-offline parity** — see §8 (O18) and §13; the same extractor over the same candidate yields the same cell online and offline.

---

## §5. Engine Layer Testing

**Authority:** `REDSTACK_ENGINE_LAYER.md` + `REDSTACK_REPOSITORY_LAYOUT.md §9` (eleven physical engine modules realizing fourteen logical services). **Location:** `tests/unit/engines/`, `tests/property/engines/`, `tests/golden/engines/`; engine composition exercised in `tests/integration/`. **Principle:** engines are stateless, pure (or pure-given-ports), and never call each other — so each is unit-tested in isolation with domain fixtures, and only `semantic` needs fake ports. Every engine test set has four parts: **unit** (correct verdict/value), **integration** (advances `BuildStage` correctly when threaded by the pipeline), **failure-path** (raises the right error / produces the right Report), **determinism** (identical input ⇒ identical output; findings/components pre-sorted).

| Engine (module) | Unit focus | Integration focus | Failure-path focus | Determinism focus |
|---|---|---|---|---|
| **Integrity** (`integrity.py`) | Each `IntegrityFlag` fires on its exemplar (tenure>experience, current+end_date, role/date mismatch, expert@0-months en masse, education-timeline, experience-predates-start, assessment-for-absent-skill); benign profile fires none. `is_honeypot` = (≥2 HARD) OR (composite ≥ threshold). | R4: attaches `IntegrityReport`, advances `SITUATED→GATED`. | Missing calibration ⇒ `ArtifactContractError` (fatal). | Findings sorted by code; threshold from artifact fixed. |
| **Eligibility** (`eligibility.py`) | Each HARD code (`PURE_RESEARCH_NO_PRODUCTION`, `LANGCHAIN_OPENAI_ONLY_RECENT`, `NO_PRODUCTION_CODE_18M`, `CONSULTING_FIRMS_ONLY_CAREER`, `PRIMARY_CV_SPEECH_ROBOTICS_NO_NLP`, `CLOSED_SOURCE_5Y_NO_VALIDATION`) and SOFT code (`TITLE_CHASER_SUB_18M_HOPS`, `NOTICE_OVER_30`, `OUTSIDE_INDIA_NO_SPONSOR`, `OUTSIDE_EXPERIENCE_BAND`) fires on the JD-derived exemplar; clean candidate passes all gates. | R4: attaches `EligibilityReport`. | Invalid gate code in rules ⇒ raise. | `is_eligible` derivation deterministic; findings sorted. |
| **Lexicon** (`lexicon.py`) | Compiled-lexicon symbolic match; synonym a stuffer wouldn't use is caught; anti-stuffing corroboration. | R2: feeds credibility/competency. | Missing lexicon artifact ⇒ raise. | Canonical map fixed by artifact. |
| **Semantic** (`semantic.py`) | Anchor cosine + nearest-centroid math (vectorized) against a fake `SemanticVectorStorePort`/`StubEmbeddingModel`; `best_positive_anchor` argmax ties by `AnchorId`; archetype ties by `ArchetypeId`. | R3: lookup-first; encode-fallback only on store miss; attaches `SemanticProfile`+`ArchetypeAssignment`, advances `FEATURED→SITUATED`. | Store miss ⇒ fallback encode; encode also fails ⇒ semantic UNKNOWN, zero net fit, recorded miss, **never fatal**; dim mismatch ⇒ `VectorStoreError`. | Same vectors ⇒ same similarities; ties deterministic. |
| **CQV** (`cqv.py`) | Folds all slices into the `(N,D)` vector aligned to `FeatureLayout`; no NaN. | R5: produces `CandidateQualityVector`. | Wrong dim / `schema_version` ⇒ `CQVInvariantError`. | Reduction in layout order; float32. |
| **Behavioral** (`behavioral.py`) | 23-signal families → bounded multiplier inputs; sentinel `−1`/`{}` ⇒ UNKNOWN (never 0); single-ownership routing (no double count). | R2: attaches `BehavioralProfile`. | — | `as_of`-only recency; no clock. |
| **Logistics** (`logistics.py`) | Location→`LocationFit` vs hub set (Pune/Noida/Hyderabad/Mumbai/Delhi-NCR); `notice_fit` bands (≤30 ideal / buyout / >30 higher bar); salary inversion preserved as soft flag. | R2: bounded multiplier. | — | Deterministic banding. |
| **Scoring** (`scoring.py`) | `base = Σ(component·locked_weight)` in fixed order; `final = base × behavioral × logistics + archetype_adj` then confidence shrink; floored ⇒ FLOOR; no multipliers on floored. | R5: `VECTORIZED→SCORED`; survivor `ScoredCandidate`s. | Weight keys ≠ `ScoreComponent` or layout mismatch ⇒ raise; any NaN ⇒ raise. | matvec thread-invariant; FLOOR finite constant. |
| **Ranking** (`ranking.py`) | Sort `(−score, candidate_id)`; non-floored fill ranks first; top-100; ranks 1..100; the six validator invariants enforced at construction. | R6: `SCORED→RANKED`. | <100 candidates ⇒ `RankingInvariantError`. | Stable sort; ties by id; 1-thread==N-thread. |
| **Reasoning** (`reasoning.py`) | Evidence-grounded clauses selected by `FeatureImportance`; ≥1 STRENGTH for top/mid, ≥1 CONCERN where a gap/soft-penalty exists, ≥1 `jd_link`; template-free, no-LLM, no-network. | R7: top-100 only over re-hydrated raw; `RANKED→EXPLAINED`; `with_reasoning`. | Dangling evidence ⇒ `ProvenanceError`; missing top-K hydration ⇒ raise. | Pure function of ordered clause set; identical evidence ⇒ identical text, different evidence ⇒ different text. |
| **Validation** (`validation.py`) | Mirrors `validate_submission.py` rule-for-rule + Stage-4 reasoning checks. | R8/R9 defence-in-depth over finished `Ranking`/CSV. | Produces `ValidationReport` with HARD findings on any violation. | Deterministic verdict. |

**Mocking rule:** only `semantic` uses fakes (the two embedding/store ports). Every other engine is tested with pure domain fixtures — no mocks, no `unittest.mock`.

---

## §6. Port Contract Testing

**Authority:** `REDSTACK_PORTS_LAYER.md §16–§17`. **Location:** `tests/contract/` (the suites) + `tests/fixtures/` (the fakes). **Principle:** each port has **one** abstract, parametrized behavioral suite that runs against **every** implementation — the real adapter *and* its in-memory fake. This is the mechanism that prevents fakes from drifting from reality: if the fake passes a contract the adapter fails (or vice-versa), the suite is red.

**Contract-suite design.** Each suite is parametrized over an implementation factory and asserts the port's documented behavior, exceptions, determinism, and (where applicable) the verdict-vs-failure discipline (missing data is a value; integrity violations raise).

| Port | Compliance criteria the suite asserts | Required fake |
|---|---|---|
| `CandidateSourcePort` | File order preserved; `source_index` monotonic == enumeration order; malformed line ⇒ `Malformed` record (**not** exception); gzip output identical to plain on identical content; blank lines skipped; IO/decompress failure ⇒ `CandidateSourceError`. | `ListCandidateSource` |
| `ArtifactStorePort` | Tampered byte ⇒ `ArtifactContractError`; missing key ⇒ error; incompatible `schema_version`/`layout_version` ⇒ error; manifest self-hash failure ⇒ `ManifestError`; path-escape (`..`/absolute) rejected; happy-path byte fidelity; verification idempotent + cached. | `InMemoryArtifactStore` |
| `EmbeddingModelPort` | Output shape `(n, dim)`; float32; per-row unit norm ± ε; order preserved regardless of `batch_size`; two calls identical (within-runtime bitwise); empty string valid (never raises); never returns zeros to mask failure; `EmbeddingError` on encode failure. | `StubEmbeddingModel` (hash-derived unit vectors) |
| `SemanticVectorStorePort` | `get` round-trips a known id; `get_many` preserves order and partitions found/missing; `view_all` id order aligns with rows; missing id ⇒ `None`/`missing` (never raises); read-only arrays; dim mismatch ⇒ `VectorStoreError`. | `InMemoryVectorStore` |
| `SubmissionSinkPort` | Emitted CSV passes the organizer `validate_submission.py`; byte-identical for identical `Ranking`; reasoning containing comma/quote/newline round-trips (RFC-4180); UTF-8 no-BOM, `\n`, fixed precision; post-format monotonic + tie-break re-asserted ⇒ `SubmissionContractError` rather than emit a rejectable file. | `CapturingSubmissionSink` |
| `RunReportSinkPort` | `reproducible` block byte-stable across runs with identical inputs; `audit` block ignored by the determinism assertion; required structural fields present. | `CapturingRunReportSink` |
| `DeterministicEntropyPort` | Same seed ⇒ identical streams; distinct labels ⇒ independent streams; `as_of` fixed from config; **online variant raises `EntropyDisabledError` on `numpy_generator`/`derive`**. | `FixedEntropy` |

**Fake adapter requirements.** Fakes are *behavioral* (in-memory real implementations), not mocks — they must pass the identical contract suite the real adapter passes. `StubEmbeddingModel` derives each vector deterministically from a hash of the input text then normalizes (reproducible, norm-1, no model download). Fakes live in `tests/fixtures/` and are versioned with the ports.

**Compliance criterion (the gate):** an adapter is mergeable only when it passes its shared contract suite *identically to its fake* (`PORTS_LAYER` build order) — plus its adapter-specific IO tests (§7).

---

## §7. Adapter Testing

**Authority:** `REDSTACK_ADAPTERS_LAYER.md §17–§20`. **Location:** `tests/contract/` (conformance) + `tests/unit/adapters/` (adapter-specific, real IO in temp dirs). **Principle:** adapters are infrastructure with no business logic, so testing splits cleanly into *conformance* (does it satisfy the port — already covered by the shared suite of §6, run against the real adapter) and *adapter-specific* (corruption, parity, failure injection that the fake cannot exercise because it has no real IO).

For every adapter: **contract conformance** (§6 suite) · **corruption tests** · **parity tests** · **failure-injection tests**.

| Adapter | Corruption tests | Parity tests | Failure-injection tests |
|---|---|---|---|
| `FilesystemArtifactStoreAdapter` | Flip one byte in any artifact ⇒ `ArtifactContractError`; corrupt the manifest self-hash ⇒ `ManifestError`; truncated file ⇒ error. | Streamed sha256 == reference sha256 on the byte. | Missing key; incompatible `schema_version`/`layout_version`; cross-artifact incoherence (dim/model_id/anchor⊄jd_concepts); path-traversal (`..`, absolute, escaping symlink) ⇒ rejected. |
| `JsonlCandidateSourceAdapter` | Malformed JSON line ⇒ `Malformed` at the correct `line_no`; bad UTF-8 byte ⇒ `Malformed`; truncated gzip ⇒ `CandidateSourceError`. | gzip and plain of identical content yield byte-identical `SourceRecord` sequences (incl. identical `Malformed` placement). | Unreadable path; mid-stream IO error; blank-line skipping; `source_index` monotonicity preserved across failures. |
| `ParquetSemanticVectorStoreAdapter` | Corrupt parquet ⇒ `VectorStoreError`; dim-mismatch column ⇒ error. | mmap round-trip == written vectors; `view_all` row order == id index order. | Missing id ⇒ `None`/`missing` (never raises); duplicate id at open ⇒ error. |
| `OnnxEmbeddingModelAdapter` | Corrupt/incompatible onnx ⇒ load error. | **ε-parity:** onnx vs sentence-transformers cosine ≥ 0.999 on a shared text set (the one documented non-bitwise boundary); shape/dtype/order/unit-norm match the contract. | No-network assertion (any socket attempt fails the test); `dim`/`model_id` mismatch vs manifest ⇒ `ArtifactContractError`; `EmbeddingError` surfaced not masked. |
| `SentenceTransformerEmbeddingAdapter` | — | onnx export ε-parity verification (same boundary, offline profile only). | **Import-guard:** importing under an online environment marker raises; pinned revision/threads enforced; offline-only (`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`). |
| `CsvSubmissionSinkAdapter` | Reasoning with comma/quote/newline ⇒ RFC-4180 quoted and round-trips; non-UTF-8 content rejected. | Output passes organizer `validate_submission.py`; byte-identical across two writes of the same `Ranking`. | Post-format monotonicity/tie-break violation ⇒ `SubmissionContractError` (no rejectable file written); IO failure mid-write ⇒ atomic temp+rename leaves no partial file. |
| `JsonRunReportSinkAdapter` | — | `reproducible` block byte-stable across runs; sorted keys, fixed floats, enums-by-value. | IO failure ⇒ `ReportWriteError`; atomic write; `audit` excluded from repro hash. |
| `DeterministicEntropyAdapter` | — | `OfflineEntropy` seeded streams reproducible across runs. | `OnlineEntropy.numpy_generator`/`derive` ⇒ `EntropyDisabledError`; `as_of` available online. |

**Discipline:** adapter-specific tests run real IO in temporary directories; they never touch `data/` or `artifacts/`. An adapter merges only when both the shared contract suite *and* its adapter-specific tests are green.

---

## §8. Offline Pipeline Testing

**Authority:** `REDSTACK_OFFLINE_PIPELINE.md` (O0–O18). **Location:** `tests/unit/pipelines/offline/` (per-stage), `tests/integration/` (DAG + manifest), `tests/golden/` (deterministic-stage artifacts), `tests/determinism/` (rebuild equality). **Principle:** each offline stage is pure-given-ports and validated against the `OfflineArtifactRegistry` before manifesting; the suite proves each stage's artifact passes its registry validator, the DAG resolves correctly, and the build is reproducible.

**Per-stage obligations** (inputs · outputs · validation strategy · artifact verification · failure criteria):

| Stage | Inputs → Outputs | Validation strategy | Artifact verification | Failure criteria |
|---|---|---|---|---|
| **O0 Census** | candidate stream → `dataset_profile.json` | coverage/quantile sanity; `N==100K` on full pool | registry schema | drift vs `candidate_schema.json` flagged |
| **O1 Normalization** | raw → `canonical_maps.json` | maps total, no empties; NFC determinism | schema | unparseable date that passed schema ⇒ `SchemaError` |
| **O2 Validation** | normalized → `validation_report.json` + `RawCandidate` set | structural-not-semantic (contradictions preserved) | reject-rate within bound | malformed beyond bound |
| **O3 Honeypot Discovery** | validated+census → `integrity_rules.json`, `honeypot_catalog.json`, `integrity_thresholds.json` | each rule code ∈ `IntegrityFlag`; thresholds in range; recall on synthetic impossibles | registry | suspects ⊄ pool |
| **O4 Lexicon Discovery** | descriptions+seed → `lexicon.compiled.json`, `concepts.json`, `term/phrase_graph.json` | concepts cover JD families; TF-IDF/PMI determinism | schema | concept coverage gap |
| **O5 Vocab Expansion** | lexicon+embeddings → expanded `concepts.json` | each concept has anchor text; seeded+reviewed | schema | drift beyond review gate |
| **O6 JD Concepts** | JD+concepts → `jd_concepts.json`, authored `gates/eligibility_rules.yaml` | polarity tagged; every code ∈ `EligibilityCode`; anchor ⊆ concepts | schema | invalid code |
| **O7 Archetype Discovery** | candidate vectors + seeded RNG → `centroids.npy`, `archetypes.json` | `dim==embedding.dim`; fixed `k`; contiguous ids; seeded init reproducible | registry | dim/k mismatch |
| **O8 Labeling** | features+archetypes → `gold_labels.json`, `calibration_split.json` | tiers ∈ 0..4; ids ⊆ pool; split disjoint, **no leakage** | schema | leakage detected |
| **O9 Weight Calibration** | snapshot+gold → `scoring_weights.locked.yaml`, `calibration_report.json` | keys == `ScoreComponent`; `layout_version` matches; cross-val stability | registry | layout mismatch; non-linear weights |
| **O10 Feature Importance** | snapshot+weights → `feature_importance.json` | features ⊆ layout | schema | unknown feature |
| **O11 Behavioral Calibration** | bhv features+gold → `behavioral_weights.json` | bounds ordered; behavioral cannot dominate relevance | schema | unbounded multiplier |
| **O12 Risk Calibration** | detectors+gold → `risk_weights.json`, merged `integrity_thresholds.json` | honeypot recall vs false-positive trade documented; ≥2-HARD gate | schema | recall below target |
| **O13a/b/c Embedding Gen** | composed docs → `candidate_vectors.parquet`, `anchor_vectors.npy`, `encoder.onnx`, `embedding_manifest.json` | unit-norm; id unique; anchors ⊆ jd_concepts; **doc-composition recipe matches online byte-for-byte**; onnx ε-parity | registry | recipe mismatch; non-unit vectors |
| **O14 Feature Snapshot** | validated+lexicon+vectors → `feature_snapshot.parquet`, `feature_manifest.json` | no NaN; `D==layout dim`; matches `FeatureLayout` | registry | NaN; dim mismatch |
| **O15 Ranking Calibration** | scored gold → `ranking_calibration.json` | monotone; **order-preserving (cannot change ranks)**; validator-pass on dry-run | schema | non-monotone curve |
| **O16 Reasoning Templates** | gold reasonings+importance → `reasoning_templates.json` | every slot has an evidence kind; **no online LLM baked in** | schema | slot without evidence kind |
| **O17 Packaging** | all artifacts → `MANIFEST.json` | streaming sha256; schema validation vs registry; manifest self-hash | self-hash valid; required keys present; cross-coherence | missing required artifact |
| **O18 Reproducibility** | packaged set → `reproducibility_report.json` | (1) `verify_all` reload; (2) **online-vs-offline feature parity** on sample; (3) deterministic dry-run ranking on golden; (4) re-run deterministic stages + diff hashes | all checks pass | any check fails |

**Artifact lineage verification.** A dedicated integration test walks the `OfflineExecutionGraph` DAG and asserts: lineage is acyclic; each artifact's recorded dependencies match the graph; a forced change to an upstream stage marks the correct transitive closure stale (incremental == clean rebuild for deterministic stages).

**Manifest verification.** The packaged `MANIFEST.json` is loaded through the *online* `ArtifactStorePort` (the consumer): self-hash valid, every required online key present (the full list from `ONLINE_PIPELINE` Part 1), cross-artifact coherence holds. This is the offline suite proving the online contract before any online test runs.

**Reproducibility verification.** Deterministic stages (O0–O4, O6, O7 seeded, O8 split seeded, O9 seeded, O10–O12, O14–O17) are run twice from clean state and their artifact hashes diffed for equality; embedding stages (O13) are checked for ε-stability, not bitwise. This is the offline half of §13/§19.

---

## §9. Online Pipeline Testing

**Authority:** `REDSTACK_ONLINE_PIPELINE.md` (R0–R9), Parts 14–17. **Location:** `tests/integration/` (full pipeline on `sample_candidates`), `tests/unit/pipelines/online/` (per-stage), `tests/determinism/`, performance harness (§14). **Principle:** ports appear only at R0/R1/R3/R8/R9; R2/R4/R5/R6/R7 are pure — so most stage logic is unit-tested purely, and the integration test proves the strictly-sequential R0→R9 composition produces a validator-passing CSV.

**Per-stage validation, state transition, memory, runtime:**

| Stage | Stage validation | State transition | Memory budget (test asserts ≤) | Runtime budget (test asserts ≤) |
|---|---|---|---|---|
| **R0 Artifact Loading** | manifest self-hash; per-artifact sha256; required keys; cross-coherence (layout/dim/model_id/anchor⊆jd_concepts/weight keys==`ScoreComponent`) | builds immutable `OnlineRunContext` | ~400 MB | 20 s |
| **R1 Ingestion** | id pattern; schema valid; UTF-8; semantic contradictions preserved | → `PARSED` | O(1) streaming (~150 MB transient) | 35 s |
| **R2 Feature Extraction** | CQV dim==D; no NaN; ranges; single-signal-ownership; recency vs `as_of` only; competency anti-stuffer | `PARSED→FEATURED` | ~1.2 GB | 35 s |
| **R3 Semantic Hydration** | similarities ∈[−1,1]; net ∈[0,1]; anchors⊆artifact; vector dim; no NaN after fold | `FEATURED→SITUATED` | ~300 MB (vectors stay mmap'd) | 15 s |
| **R4 Gates & Eligibility** | findings carry evidence; codes valid; derivations correct; floor mask pure | `SITUATED→GATED` | ~200 MB | 10 s |
| **R5 Scoring** | `base==Σ weighted`; floored⇒FLOOR; multipliers bounded; no NaN | `GATED→VECTORIZED→SCORED` | ~200 MB | 8 s |
| **R6 Ranking** | 100 rows; ranks 1–100 once; ids unique+pattern; non-increasing; tie-break | `SCORED→RANKED` | small | 2 s |
| **R7 Reasoning** | every clause has evidence; ≥1 jd_link; ≤2 sentences; rank-band tone; no reorder | `RANKED→EXPLAINED` | small (top-100 only) | 3 s |
| **R8 Submission** | validator pass; UTF-8/no-BOM; quoting; precision; post-format re-assert | — | negligible | 1 s |
| **R9 Run Report** | required fields; `reproducible` stable; `within_budget` recorded | — | negligible | 1 s |

**State-transition validation.** A dedicated test threads one candidate through R1→R7 and asserts `BuildStage` advances exactly `PARSED→FEATURED→SITUATED→GATED→VECTORIZED→SCORED→RANKED→EXPLAINED`, that each stage attaches exactly its slice, and that any attempt to skip or repeat a stage raises `RepresentationStageError`.

**Memory-budget validation.** The performance harness (§14) samples peak RSS per stage on a scaled pool and asserts each stage's additional peak stays within the table; the aggregate stays ≤ 4 GB internal target (≤ 16 GB ceiling). The `(N,D)` matrix and mmap'd vectors are confirmed to be the dominant consumers and confirmed *not* copied into RAM.

**Runtime validation.** The harness asserts each stage and the total stay within budget (≈130 s internal / ≤300 s ceiling); a regression beyond the per-stage target fails CI before submission.

**Critical correctness tests:**
- **Floor-mask correctness.** Every honeypot and every ineligible candidate receives `final_score == FLOOR` and is partitioned to the filler tail; non-floored candidates fill ranks first. A fixture with known honeypots/ineligibles asserts none reach the top and the top-100 honeypot rate is ≈ 0 (the §10 link).
- **Tie-break correctness.** Equal `final_score`s are resolved by ascending `candidate_id`, end-to-end, and survive CSV formatting at the emitted precision (rounding may create ties; id order within ties must hold — the validator permits this).
- **Ranking correctness.** On a fixture with a known total order, the produced ranks match; non-floored-before-floored holds; the `Ranking` factory invariants pass.
- **Reasoning correctness.** Every top-100 reasoning cites resolvable evidence, links to a JD requirement, acknowledges concerns where gaps exist, varies across candidates, and matches rank-band tone (full suite in §11).

**Cold-start / miss path.** A fixture id absent from the vector store triggers the onnx fallback (encoded in the same space); a fixture where encode also fails sets semantic UNKNOWN with zero net fit and records a miss — the run continues (never fatal). Both are asserted explicitly.

---

## §10. Honeypot Validation Suite

**Authority:** `submission_spec` §7, `REDSTACK_FEATURE_LAYER.md` Part 5, `REDSTACK_OFFLINE_PIPELINE.md` Part 4. **Location:** `tests/fixtures/honeypots/`, `tests/unit/features/honeypot/`, `tests/unit/engines/integrity/`, `tests/integration/`. **This section is critical: a top-100 honeypot rate > 10% is automatic Stage-3 disqualification regardless of composite score.** The dataset contains ~80 honeypots forced to relevance tier 0.

**Dedicated honeypot fixtures.** A versioned fixture family, one per impossibility class, each a frozen `RawCandidate` with the contradiction and an expected-detector table:
- **Experience inflation** — Σ career `duration_months`/12 ≫ `years_of_experience`; tenure at a company exceeding the company's plausible age (the spec's "8 years at a 3-year-old company").
- **Skill-time contradiction** — `Proficiency ≥ ADVANCED` with `duration_months ∈ {0, None}`, en masse (the spec's "expert in 10 skills, 0 years used").
- **Timeline impossibility** — `end_date < start_date`; `is_current` with non-null `end_date`; implausible employment overlap.
- **Education-career anomaly** — `end_year < start_year`; graduation impossibly post-dating dependent career events.
- **Behavioral/signal impossibility** — impossible signal combinations (high saves, zero views); out-of-contract signal values surviving parse.
- **Identity anomaly** — provenance/id inconsistency.
- **Keyword-stuffing honeypot** — many high-proficiency skills with zero endorsements/duration/career-mention (the link to §4's anti-stuffer).

**Synthetic impossible profiles.** Beyond the hand-built family, a generator (seeded, offline-only) produces parametrized impossible profiles across the classes to widen detector coverage; outputs are frozen into fixtures so tests stay deterministic.

**Detector recall evaluation.** At the **feature layer**, each `hp.*` detector must fire on its class's fixtures (recall = 1.0 on the categorical-impossibility set). At the **engine layer**, `CandidateRiskEngine` must mark `is_honeypot` when `(≥2 HARD impossibilities) OR (composite ≥ O3 threshold)`. The suite asserts the honeypot fixture set is classified honeypot at the calibrated threshold.

**False-positive evaluation.** Equally critical — false positives floor real candidates and cost NDCG. A "benign-but-unusual" fixture family (legitimately inverted salary, legitimately sparse profile, legitimately short tenure, missing GitHub) must **not** be classified honeypot: salary inversion is soft-only; lone soft anomalies dampen but never hard-gate; a single impossibility is insufficient (the ≥2-HARD rule). The suite pins a maximum false-positive rate on this family.

**Honeypot-rate monitoring.** The full R0–R9 integration run on a pool seeded with known honeypots computes the **top-100 honeypot rate** and asserts it ≈ 0 (well under 10%). The figure is the same `honeypot_rate` R9 writes to the run report; the test reads it from the report, mirroring exactly what Stage 3 measures.

**Acceptance criteria (blocking):**
1. Recall == 1.0 on the categorical-impossibility fixture set at the calibrated O3/O12 threshold.
2. False-positive rate on the benign-unusual family below the pinned bound.
3. Top-100 honeypot rate on the seeded integration pool **≈ 0, hard-capped well below 10%**.
4. Floor-mask correctness (§9) confirms every detected honeypot is floored out of the top.

The suite explicitly proves the system keeps honeypot incidence below the disqualification threshold — by floor-partitioning, not special-casing — which is the spec's stated expectation ("a good ranking system naturally avoids them").

---

## §11. Reasoning Validation Suite

**Authority:** `submission_spec` §3 (the six Stage-4 checks + the penalized list), `REDSTACK_DOMAIN_LAYER.md §J`, `REDSTACK_FEATURE_LAYER.md` Part 8. **Location:** `tests/unit/engines/reasoning/`, `tests/golden/reasoning/`, `tests/integration/` (the diversity check over a produced submission). **Principle:** Stage 4 samples 10 random rows and checks each reasoning against six criteria; the suite tests all six as hard properties, because the no-hallucination guarantee is structural and must be proven to hold for *every* emitted clause, not a sample.

| Stage-4 check | Test design | Failure (blocking) |
|---|---|---|
| **No hallucination** | Every `ReasoningClause` carries ≥1 `EvidenceRef` whose `path` resolves in that candidate's `RawCandidate` (construction-enforced; the test walks every clause of every top-100 reasoning and resolves each ref). No skill/employer/experience appears in reasoning that is absent from the profile. | Any clause without resolvable evidence; any claim not in the profile. |
| **Evidence grounding** | Each clause's fragment is derived from the cited `FeatureCell` evidence, selected by `FeatureImportance`; the explainability chain `RankedCandidate→component→importance→cell→EvidenceRef→field` is walked and asserted complete. | Broken chain; clause not traceable to a feature. |
| **Specific facts** | Reasoning references concrete profile facts (years, current title, named skills with trust, signal values, product-company tenure) — asserted by checking each top-K reasoning contains ≥1 evidence ref of a fact-bearing `EvidenceKind` (`PROFILE_FIELD`/`CAREER_FIELD`/`SKILL`/`SIGNAL`). | Generic praise with no fact ref. |
| **JD connection** | ≥1 clause per reasoning carries a non-null `jd_link` (an `EligibilityCode` or `ScoreComponent`/latent). | No JD link. |
| **Honest concerns** | Where a candidate has a soft eligibility penalty or a known gap, ≥1 `CONCERN` clause is present; a rank-95 candidate's reasoning is measured, not glowing. | Concern omitted where a gap exists. |
| **Variation** | Across the top-100 (and the Stage-4-style 10-row sample), rendered reasonings are substantively different — driven by *which evidence is present*, not a name-insertion template. A near-duplicate detector over the produced submission flags templated/identical strings. | All-identical strings; templated name-insertion; near-duplicate cluster beyond threshold. |
| **Rank consistency** | Tone matches `rank_band`: `top` net-positive, `tail` measured; a determinism test confirms identical evidence ⇒ identical text and different evidence ⇒ different text (so reasoning is a function of the ranking inputs, never independent of them). | Glowing tail / critical top; reasoning contradicting rank. |

**Penalized-behavior tests (mirroring the spec's explicit list):** empty reasoning, all-identical strings, name-insertion templates, hallucinated skills, and rank-contradicting tone each have a negative test asserting the system **cannot** produce them (empty fails the `≤2 sentence / ≥1 clause` invariant; identical/templated fail the variation detector; hallucination fails construction; contradiction fails rank-band tone).

**Traceability mandate.** A dedicated audit test reconstructs, for an arbitrary ranked candidate, the full derivation of every reasoning claim down to the raw field — proving the Stage-5 "defend-your-work" requirement is mechanically satisfiable (any claim can be traced to a field on demand).

---

## §12. Submission Compliance Testing

**Authority:** `validate_submission.py` (the literal oracle), `sample_submission.csv`, `submission_spec` §2–§3, §6. **Location:** `tests/contract/submission/`, `tests/golden/submission/`, `tests/integration/`. **Principle:** the team's `ValidationEngine` mirrors the organizer validator rule-for-rule, and a separate test runs the *organizer's actual* `validate_submission.py` byte-for-byte against produced output — defence in depth against any divergence between our mirror and theirs.

**Tests proving each rule (each a hard, blocking assertion):**
- **Exactly 100 data rows** (+1 header); 99 or 101 fails (the spec's most common rejection).
- **Header exactly** `candidate_id,rank,score,reasoning`, in order.
- **Valid candidate IDs** matching `^CAND_[0-9]{7}$`; every id exists in the candidate pool; typos rejected.
- **Unique ranks** — each integer 1–100 exactly once; ranks starting at 0 fails; missing rank fails.
- **Unique IDs** — no duplicate `candidate_id`.
- **Monotonic scores** — `score` non-increasing by rank (rank 1 ≥ rank 2 ≥ … ≥ rank 100); increasing-by-rank fails; all-identical scores is *permitted by the validator* but flagged by a separate quality warning (the spec notes "model isn't differentiating").
- **Deterministic tie-breaking** — equal scores ⇒ `candidate_id` ascending within the tie; verified at the *emitted* precision (rounding-induced ties must still hold id order).
- **score is a float**; non-float fails.
- **UTF-8 output**, no BOM; non-UTF-8 fails.
- **CSV compliance** — `.csv` extension; RFC-4180 quoting for reasoning containing comma/quote/newline; `\n` newlines; no trailing whitespace.

**Mirroring the competition validator.** Two layers: (1) the `ValidationEngine` unit tests assert our `ValidationCode` set and logic match `validate_submission.py` line-for-line (`WRONG_ROW_COUNT, RANK_OUT_OF_RANGE, DUPLICATE_RANK, MISSING_RANK, DUPLICATE_ID, BAD_ID_FORMAT, SCORE_NOT_FLOAT, SCORE_INCREASING, TIEBREAK_VIOLATION`). (2) An integration test invokes the **organizer's** `validate_submission.py` as an external oracle on the produced CSV and asserts "Submission is valid." Both must pass; a discrepancy between them is itself a failure (our mirror has drifted).

**Sample-format reference.** `sample_submission.csv` is used only as a format reference (the spec states it is not a quality ranking); a test confirms our CSV is structurally congruent (same header, 4 columns, 100 rows) while making no claim about score quality against it.

**Negative corpus.** A fixture corpus of known-bad CSVs (each violating exactly one rule) asserts the `ValidationEngine` and the organizer validator both reject each with the right finding — the §6 `SubmissionSinkPort` contract guarantees the sink raises `SubmissionContractError` rather than emit any of them.

---

## §13. Determinism Testing

**Authority:** `REDSTACK_ARCHITECTURE.md §15/§17`, `ONLINE_PIPELINE` determinism requirements, `ADAPTERS_LAYER §16`. **Location:** `tests/determinism/`. **Critical: Stage-3 reproduces the ranking in a clean sandbox; any nondeterminism is a reproduction failure.** Determinism tests compare the **`reproducible`** region only; the `audit` block (run_id, wall-clock timestamps, host label) is explicitly excluded.

| Test | Method | Acceptance criterion (exact) |
|---|---|---|
| **Repeated runs (same process)** | Run R0–R9 twice on identical inputs in one process. | `submission.csv` byte-identical; run-report `reproducible` block byte-identical; `output_sha256` equal. |
| **Hash equality** | Compare `output_sha256`, `manifest_hash`, per-artifact `artifact_hashes`, `config_hash`, `input_file_sha256` across runs. | All equal. |
| **Artifact equality** | Rebuild deterministic offline stages (O0–O12, O14–O17) from clean state twice. | Per-artifact sha256 equal for deterministic stages; O13 embedding artifacts ε-stable (cosine ≥ 0.999), not bitwise — the one documented exception. |
| **Ranking equality** | Diff the ordered `(candidate_id, rank, score)` triples across runs. | Identical ordering and scores. |
| **Reasoning equality** | Diff rendered reasoning strings across runs. | Identical (reasoning is a pure function of the ordered clause set). |
| **Single-thread vs multi-thread** | Run online once at 1 thread, once at N threads (BLAS/OMP). | Identical ranking + identical `reproducible` block (thread-count-invariant; pinned threads + fixed-order reductions). |
| **Restart reproducibility** | Run in two fresh processes (cold imports, fresh sessions). | `reproducible` block identical; proves no in-process global state leaks. |
| **Seed reproducibility** | Offline stages with the same `seed`/`as_of` (via `OfflineEntropy` substreams). | Identical seeded artifacts; distinct labels yield independent-but-reproducible streams; **online run uses no RNG** (a test asserts `OnlineEntropy.numpy_generator` raises `EntropyDisabledError`). |

**Determinism guards under test:** no wall clock anywhere reachable from `pipelines.online` (recency uses injected `as_of`); ties always by ascending `candidate_id`; float32 reductions in `FeatureLayout`/`ScoreComponent` order; enum-by-value serialization; mmap reads are read-only gathers. A static test (import-linter job, §17 of the layout) asserts the online subgraph cannot import a clock-driven or RNG-driven path.

**Offline-online parity (the parity half of determinism).** Recompute features online for the `sample_candidates` and diff against the offline `feature_snapshot.parquet` rows for the same ids; require exact equality for deterministic features and ε-equality for the semantic columns. Divergence means the locked weights are calibrated against features the online path no longer reproduces — a silent ranking corruption with no leaderboard to reveal it.

---

## §14. Performance Testing

**Authority:** `submission_spec` §3 (compute constraints), `ONLINE_PIPELINE` Part 14. **Competition ceiling:** ≤ 5 min wall-clock, ≤ 16 GB RAM, CPU-only, no GPU, no network, ≤ 5 GB intermediate disk. **Internal targets:** ≤ 150 s, ≤ 4 GB peak RSS. **Location:** a dedicated harness invoked from `tests/integration/` (sample) and a scaled-pool job in CI. **Principle:** the budget is met *structurally* (lookup-not-encode at R3, apply-not-calibrate at R5, columnar vectorization, mmap'd vectors); the harness proves the structure holds and catches regressions before submission, since Stage 3 reproduces under exactly these limits.

**Benchmark methodology.** Run R0–R9 on (a) the ≤100 `sample_candidates` (the sandbox-scale check) and (b) a scaled synthetic pool approximating 100K (CI may use a representative fraction with documented extrapolation, or the full pool on a 16 GB CPU runner for the pre-submission gate). Record per-stage wall-ms and peak RSS from `observability/timing.py` (the same numbers R9 writes). Repeat to establish variance; report median and worst-case.

**Profiling strategy.** Per-stage timers are always on (they feed the run report). For investigation, a profiling profile enables cProfile/py-spy sampling and a memory profiler — but profiling never runs in the determinism/repro gates (it perturbs timing). The harness attributes time to the expected dominant stages (R2 feature extraction, R0 hashing, R3 matmuls) and flags any inversion (e.g. R3 dominating would indicate accidental online encoding instead of lookup).

**Stress tests.** Worst-case inputs: maximum-size candidates (10 positions, 5 educations, large skill lists, long descriptions); a pool with a high store-miss rate forcing many onnx fallbacks (must still fit budget); maximum reasoning complexity on the top-100. Each asserts the budget still holds.

**Memory tests.** Assert peak RSS ≤ 4 GB internal / ≤ 16 GB ceiling; assert the candidate vector matrix stays mmap'd (resident ≈ touched pages, not the full ~150 MB copied); assert the `(N,D)` float32 matrix + `(N,G)` confidence are the dominant allocations and within the R2 ~1.2 GB envelope; assert rich per-candidate objects materialize only for survivors/top-K.

**Scalability tests.** Confirm the hot path is O(N) vectorized, not per-candidate Python: timing scales roughly linearly with pool size; doubling the pool does not super-linearly inflate R2/R5. Confirm R7 cost is bounded by top-100 (constant), not pool size.

**No-network / no-GPU enforcement.** A test runs the online pipeline with the network disabled (and asserts any socket attempt fails) and with no CUDA provider available; the onnx session must initialize under `CPUExecutionProvider` only. `st_embedder` import under an online marker must raise. These mirror the Stage-3 sandbox exactly.

**Pass/fail thresholds (blocking):**
1. Full-pool (or documented-extrapolation) wall-clock ≤ 150 s internal target; **hard fail > 300 s** (ceiling).
2. Peak RSS ≤ 4 GB internal; **hard fail > 16 GB**.
3. Intermediate disk ≤ 5 GB.
4. Zero network egress; zero GPU use; CPU-only session confirmed.
5. `within_budget == true` in the run report on the gating run.

A CI integration test on the sample asserts the budget holds *before any real submission* — there is no live leaderboard to catch a budget regression later.

---

## §15. Golden Dataset Strategy

**Authority:** `REDSTACK_ARCHITECTURE.md §10`, `REPOSITORY_LAYOUT §15`. **Location:** `tests/golden/` + `tests/fixtures/`. **Principle:** golden (snapshot) tests lock exact outputs so that any unintended change to scoring, feature extraction, serialization, or reasoning surfaces as a diff that must be explicitly re-blessed — the primary regression net for a system with one shot at scoring.

**Golden candidates.** A small, hand-curated, frozen set of `RawCandidate`s spanning the archetypes (ideal product-ML match, retrieval specialist, consulting-only, pure-researcher, keyword-stuffer, framework-enthusiast, honeypots, sentinel-laden behavioral). Each is committed with full provenance and an expected-feature table.

**Golden rankings.** For the golden candidate set under the locked artifacts, the exact `ScoredCandidate` ordering, `final_score`s (at fixed precision), floor partitioning, and the resulting `Ranking` are snapshotted. A change to weights, features, or gating that reorders the golden set fails until re-blessed with a documented reason.

**Golden reasonings.** The exact rendered reasoning string, clause set, evidence refs, and `jd_link`s for each golden top-K candidate are snapshotted — locking the no-hallucination, variation, and rank-consistency properties against drift.

**Golden artifacts.** The deterministic offline artifacts (manifest structure, `scoring_weights.locked.yaml`, `integrity_thresholds.json`, `feature_manifest.json`, `centroids.npy` shape/seed-stable content) are snapshotted by sha256 (deterministic stages) or ε-checked (embeddings). The golden `MANIFEST.json` cross-coherence is asserted.

**Regression prevention.** Golden tests are deliberately brittle: they assert *bytes/values*, not ranges. A diff is a forcing function — either a real regression (fix it) or an intended change (re-bless with justification in the commit). Because the competition gives no post-submission feedback, this brittleness is a feature: it makes silent drift impossible to merge unnoticed.

---

## §16. Fixture Architecture

**Authority:** `PORTS_LAYER §17`, `FEATURE_LAYER`, `submission_spec` §7, JD disqualifiers. **Location:** `tests/fixtures/`. **Principle:** fixtures are the shared, versioned ground truth every category builds on; they are frozen `RawCandidate`s (or fakes) with expected-output tables, owned centrally so a single change propagates consistently.

**Fixture families:**
- **Valid candidates** — representative, schema-clean profiles across experience bands and tracks; the baseline for golden/feature/engine tests.
- **Honeypots** — one per impossibility class (§10), plus the seeded synthetic generator's frozen outputs; the DQ-prevention corpus.
- **Keyword stuffers** — many ADVANCED/EXPERT skills, zero endorsements/duration/career-mention, mismatched title; the anti-stuffer corpus (§4) and a false-positive guard against over-flooring.
- **Consulting-only candidates** — entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini (the JD `CONSULTING_FIRMS_ONLY_CAREER` hard block) and the "currently consulting but prior product experience" exception (must *not* hard-block).
- **Ideal JD matches** — 6–8 years, 4–5 in applied ML at product companies, shipped retrieval/ranking, hybrid-retrieval + eval-framework experience, hub-located — the positive ceiling.
- **JD-disqualifier exemplars** — one per HARD/SOFT `EligibilityCode`: pure researcher; <12-month LangChain→OpenAI without pre-LLM ML; senior with no production code 18m; CV/speech/robotics without NLP; closed-source 5y no validation; sub-18-month title-chaser; >30-day notice; outside-India-no-sponsor; outside the 5–9 band.
- **Malformed candidates** — non-JSON lines, bad UTF-8, schema-invalid-but-parseable (for `SchemaError`), and semantically-contradictory-but-type-valid (preserved, not rejected).
- **Edge-case behavioral signals** — all-sentinel (`github=−1`, `offer_acceptance=−1`, `skill_assessment={}`); stale `last_active`; impossible combinations (high saves, zero views); boundary notice (0, 30, 180) and salary (inverted, zero).
- **Port fakes** — `InMemoryArtifactStore`, `InMemoryVectorStore`, `StubEmbeddingModel`, `ListCandidateSource`, `CapturingSubmissionSink`, `CapturingRunReportSink`, `FixedEntropy` (all passing the §6 contract suites), plus a tiny onnx stub and a known-bad-CSV negative corpus (§12).

**Ownership and versioning.** Fixtures are owned by the layer they primarily serve but committed centrally; each carries a version. A fixture change that alters an expected output requires re-blessing the dependent golden snapshots in the same commit. Honeypot and JD-disqualifier fixtures are owned jointly by Modeling (Stage-3/4 risk) and are change-controlled — they encode competition elimination criteria.

---

## §17. CI/CD Testing Pipeline

**Authority:** `REPOSITORY_LAYOUT §17` (import-linter contracts), `ARCHITECTURE §10`. **Principle:** fast, cheap, high-signal checks run first and gate the slow ones; a red base never spends CI minutes on integration. Every stage below is **blocking** unless explicitly noted.

**Execution order:**

| # | Stage | Blocking? | What it runs / asserts | Artifact retention |
|---|---|---|---|---|
| 1 | **lint** | Yes | `ruff` lint + format check. | — |
| 2 | **mypy** | Yes | `mypy --strict` across the package; `Any` forbidden except the single parsing boundary. | — |
| 2b | **import-linter** | Yes | The eight boundary contracts (§17 layout): layered, domain purity, engine purity, adapter isolation, online containment, composition-root exception, config split, ports/observability independence. | the contract report |
| 3 | **unit** | Yes | `tests/unit/` (all layers). Fast, pure. | junit xml |
| 4 | **property** | Yes | `tests/property/` Hypothesis suites; a found counterexample is committed to the regression corpus. | counterexample DB |
| 5 | **contract** | Yes | `tests/contract/` — every port suite × {adapter, fake}. | junit xml |
| 6 | **golden** | Yes | `tests/golden/` snapshots; a diff fails until re-blessed. | golden diffs |
| 7 | **integration** | Yes | `tests/integration/` — full offline build + online R0–R9 on `sample_candidates`; organizer `validate_submission.py` must pass; CLI verbs. | sample `submission.csv`, `run_report.json` |
| 8 | **determinism** | Yes | `tests/determinism/` — twice-run byte diff, 1-thread vs N-thread, restart, seed; offline-online parity. | both run reports for diffing |
| 9 | **performance** | Yes (pre-submission gate); nightly on full pool | timing + RSS + no-network/no-GPU on sample (every PR) and scaled/full pool (nightly + release). | per-stage timing report |

**Blocking failures.** Any non-green stage blocks merge. Determinism and performance are blocking on the release branch and on the pre-submission gate specifically (they are the Stage-3 predictors).

**Coverage requirements.** See §18 — coverage is a CI gate keyed to architectural risk, not a flat percentage.

**Artifact retention.** CI retains the sample `submission.csv`, both determinism run reports, the per-stage timing report, golden diffs, and Hypothesis counterexamples for each run, so a failure is diagnosable without re-running. The `reproducibility_report.json` from the release gate (§19) is retained with the submission.

**CI reporting.** Each run publishes: pass/fail per stage, coverage by layer, the budget headroom (`within_budget`, used_seconds, peak_rss_mb), the top-100 honeypot rate on the seeded integration pool, and the import-linter verdict — the dashboard a release owner reads before authorizing a submission.

---

## §18. Coverage Strategy

**Principle (stated by the prompt and adopted):** coverage is measured by **architectural risk**, not raw percentages. A 100%-line module with untested invariants is less safe than a 90%-line module whose every invariant and contract is proven. Targets below are floors, differentiated by where a defect would be unrecoverable (no leaderboard, one submission).

**Coverage dimensions:**
- **Line coverage** — global floor **90%**; `domain/` and `engines/` floor **95%** (pure, no excuse for gaps); `adapters/` may sit lower on hard-to-hit IO error branches, compensated by failure-injection tests (§7).
- **Branch coverage** — floor **85%** global; **100%** on the gating/floor logic in `ScoringEngine` and the invariant checks in `Ranking`/`CQV` (every `raise` path executed by a test).
- **Invariant coverage** — **100%**: every domain invariant (§3 matrix) has at least one test proving it raises on violation and one proving the legal case. Tracked explicitly as a checklist, not inferred from line coverage.
- **Contract coverage** — **100%**: every port × {adapter, fake} runs the full shared suite; every documented exception path is asserted.

**Risk-weighted emphasis (must be exhaustively covered regardless of percentage):** the floor-mask + honeypot path (DQ risk); the anti-stuffer competency fusion (top-10 quality); the six `Ranking` validator invariants (auto-reject risk); the six Stage-4 reasoning checks; the offline-online feature parity; determinism of the `reproducible` block; the R0 artifact-coherence checks. A gap in any of these blocks release even if global coverage is green.

**Measurement.** Coverage runs over unit+property+contract+golden+integration combined (determinism/performance excluded — they re-exercise covered paths). The CI gate fails on a drop below any floor or on an uncovered risk-weighted path.

---

## §19. Reproducibility Certification

**Authority:** `submission_spec` §5 (Stage 3), `OFFLINE_PIPELINE` O18, `ONLINE_PIPELINE` Part 17, `PORTS_LAYER §8/§13`. **This is the final gate: nothing is submitted until it passes.** Stage 3 reproduces the ranking step in a clean, sandboxed, CPU-only, network-less, 16 GB / 5 min container; failure here is disqualification regardless of composite score. Certification proves, mechanically, that the artifact set + code + the single command reproduce the exact submission inside the limits.

**The certification gate — all must pass (blocking, in order):**

1. **Artifact integrity.** Load the full artifact set through the online `ArtifactStorePort`: `MANIFEST.json` self-hash valid; every required key present (the full `ONLINE_PIPELINE` Part 1 list); each artifact's streamed sha256 matches; cross-artifact coherence holds (`layout_version` agreement across `feature_manifest`/`scoring_weights`/`FeatureLayout`; `embedding.dim` across vectors/anchors/centroids/onnx; `model_id` consistency; anchor set ⊆ `jd_concepts`; weight keys == `ScoreComponent`; integrity/eligibility codes valid). Any failure aborts — no degraded mode. (This is the `reproducibility_report.json` from O18, re-verified at release.)

2. **Offline-online parity.** Recompute features online for the `sample_candidates` (and a held-out slice) and diff against `feature_snapshot.parquet`: exact equality for deterministic features, ε-equality (cosine ≥ 0.999) for semantic columns. Proves the locked weights are applied to the same features they were calibrated on.

3. **Submission reproducibility.** Run the single Stage-3 command (`redstack rank --candidates … --out …`, via `scripts/reproduce.sh`) twice — once in-process, once in a fresh, network-disabled, CPU-only, 16 GB-capped environment matching the sandbox — and assert: identical `submission.csv` bytes; identical run-report `reproducible` block; identical `output_sha256`; 1-thread and N-thread runs identical. The output passes the organizer `validate_submission.py`.

4. **Compute-limit compliance.** The gating run completes ≤ 150 s internal target (hard ceiling 300 s), peak RSS ≤ 4 GB internal (ceiling 16 GB), ≤ 5 GB intermediate disk, zero network, CPU-only; `within_budget == true` in the report.

5. **Honeypot gate.** Top-100 honeypot rate on the full-pool gating run ≈ 0, hard-capped well below 10% (read from the run report, the same figure Stage 3 computes).

6. **Reasoning gate.** All six Stage-4 checks pass on the produced top-100 (every claim evidence-traceable; variation/honesty/rank-consistency hold) — proven by §11 run against the actual submission.

7. **Boundary + metadata integrity.** import-linter clean (online containment guaranteed); `submission_metadata.yaml:reproduce_command` matches the certified command; the GitHub repo + sandbox link are reachable; declarations honest.

**Release definition.** A submission may be released **only** when gates 1–7 are green on the candidate artifact set and code commit, the full CI pipeline (§17) is green, coverage floors (§18) hold, and the `reproducibility_report.json` is retained alongside the submission. Because there is no leaderboard and a three-submission cap, this gate is the team's only assurance — it is treated as the literal precondition for upload, owned by the release owner, and never waived.

---

*This testing strategy is internally consistent with all nine frozen REDSTACK documents and the competition artifacts. It introduces no new architecture, ports, engines, pipelines, or artifacts; it specifies only how the frozen system is proven correct, deterministic, reproducible, performant, explainable, compliant, honeypot-robust, and architecturally sound — to the standard of an external Stage-3/4/5 audit.*
