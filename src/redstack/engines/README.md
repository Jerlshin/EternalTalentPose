# `engines/` — The Domain Services

Eleven physical modules, stateless and (with one exception) pure. Each engine takes a `CandidateRepresentation` plus typed configuration (or, for `semantic.py`, an injected port) and returns an enriched copy or a verdict. **Engines never call each other** — the dependency graph between them is a forest of data dependencies, not direct imports; only the pipeline threads the growing representation from one engine to the next. This package may import `domain/`, `ports/` (by injection), `features/`, and `config.schema` — never `adapters/`, `pipelines/`, `observability/` IO, or any other engine. See [`/ARCHITECTURE.md` §6](../../../ARCHITECTURE.md#6-the-engines) and [`docs/specs/REDSTACK_ENGINE_LAYER.md`](../../../docs/specs/REDSTACK_ENGINE_LAYER.md).

## File inventory

| File | Engine | Online stage | Reads | Produces | Pure? |
|---|---|---|---|---|---|
| [`integrity.py`](integrity.py) | `IntegrityEngine` | R4 | `CareerProfile`, raw record, calibrated thresholds, the `hp.*` feature cells | `IntegrityReport` (honeypot verdict) | Yes |
| [`eligibility.py`](eligibility.py) | `EligibilityEngine` | R4 | representation + `JobDescriptionSpec` + gate rules | `EligibilityReport` (hard blocks, soft penalties) | Yes |
| [`lexicon.py`](lexicon.py) | `LexiconEngine` | R2 | normalized tokens, descriptions, the compiled lexicon | competency/credibility support (lexical corroboration) | Yes |
| [`semantic.py`](semantic.py) | `SemanticEngine` | R3 | candidate id, the vector store, anchor vectors, archetype centroids | `SemanticProfile`, `ArchetypeAssignment` | Pure *given its ports* — the only engine that touches a port |
| [`cqv.py`](cqv.py) | `CQVAssembler` | R5 | the fully populated representation | `CandidateQualityVector` | Yes |
| [`behavioral.py`](behavioral.py) | `BehavioralEngine` | R2 | `BehavioralProfile` inputs | bounded behavioral multiplier | Yes |
| [`logistics.py`](logistics.py) | `LogisticsEngine` | R2 | `LogisticsProfile` inputs | bounded logistics multiplier | Yes |
| [`scoring.py`](scoring.py) | `ScoringEngine` | R5 | folded CQV, gate verdicts, multipliers, the locked weights | `ScoredCandidate` + `ScoreBreakdown` | Yes |
| [`ranking.py`](ranking.py) | `RankingEngine` | R6 | `ScoredCandidate[]` | `Ranking` (invariant-checked at construction) | Yes |
| [`reasoning.py`](reasoning.py) | `ReasoningEngine` | R7 | `RankedCandidate` + the re-hydrated top-K representation | `CandidateReasoning`, attached via `Ranking.with_reasoning` | Yes |
| [`validation.py`](validation.py) | `ValidationEngine` | R8/R9 (defense-in-depth) | a finished `Ranking` | `ValidationReport` | Yes |

## Why this is a forest, not a web

The pipeline (`pipelines/online/stages.py`) is the only thing that calls more than one engine. Each engine's signature only ever names domain types and ports — never another engine. This means any single engine can be unit-tested with nothing but domain fixtures (and, for `semantic.py`, a fake port) — no mocking framework needed anywhere in this package, because nothing here calls out to infrastructure directly.

## The gating contract

`IntegrityEngine` and `EligibilityEngine` run independently at R4 and are joined into a single floor mask before R5. A candidate that is a detected honeypot **or** ineligible has `final_score` forced to a fixed floor sentinel at scoring time — no partial penalty, no chance of slipping into the ranked top by a strong semantic match alone. `BehavioralEngine` and `LogisticsEngine` produce *multipliers* that modulate an already-computed relevance score; neither can be a source of relevance on its own — a behaviorally inactive but otherwise perfect-on-paper candidate is down-weighted, never disqualified outright.

## Performance

Every engine here operates on either a single representation (called per-candidate during reasoning, which only ever covers the top-100) or a vectorized columnar batch (`integrity`, `eligibility`, `cqv`, `behavioral`, `logistics`, `scoring`, `ranking` over the full pool) — there is no per-candidate Python object churn in the hot path. See [`/ARCHITECTURE.md` §5.2](../../../ARCHITECTURE.md#52-online-pipeline--r0-through-r9) for the per-stage compute budget.
