# `domain/candidate/` — The `CandidateRepresentation` Slices

Ten frozen value-object slices, one per facet of a candidate's profile, each attached to the `CandidateRepresentation` aggregate (defined in [`representation.py`](representation.py)) at a specific online-pipeline stage. See [`/ARCHITECTURE.md` §7](../../../../ARCHITECTURE.md#7-the-domain-model) for the aggregate diagram and [`docs/specs/REDSTACK_DOMAIN_LAYER.md` §G](../../../../docs/specs/REDSTACK_DOMAIN_LAYER.md) for the exhaustive per-slice specification.

| File | Slice | Attached at | Purpose |
|---|---|---|---|
| [`representation.py`](representation.py) | `CandidateRepresentation` (the aggregate root) | — | Threads every other slice; enforces stage-ordering and single-attachment via copy-on-write `with_*()` builders. |
| [`identity.py`](identity.py) | `Identity` | R1 (Ingestion) | Stable candidate ID + provenance anchor; equality and hashing are by `candidate_id` only. |
| [`career.py`](career.py) | `CareerProfile` | R2 (Feature Extraction) | Structured career facts: tenure stats, recency, title trajectory, and the product-vs-services classification the job description hinges on. |
| [`credibility.py`](credibility.py) | `CredibilityProfile` | R2 | Per-skill trust (endorsement × duration × assessment coherence), keyword-stuffing score, and title-vs-description coherence — the anti-keyword-stuffing layer. |
| [`behavioral.py`](behavioral.py) | `BehavioralProfile` | R2 | Normalized platform engagement signals (availability, responsiveness, engagement, reliability, verification), each paired with a `SignalAvailability` discriminator so "unknown" (a sentinel value) is never silently treated as "low." |
| [`logistics.py`](logistics.py) | `LogisticsProfile` | R2 | Location fit, notice period, relocation willingness, and salary-band sanity (an inverted band is preserved as a flag, not corrected). |
| [`semantic.py`](semantic.py) | `SemanticProfile` | R3 (Semantic Hydration) | Anchor similarities and net semantic fit. Holds a `VectorRef` (a row index into the memory-mapped vector store) — the dense embedding itself is never inlined here. |
| [`archetype.py`](archetype.py) | `ArchetypeAssignment` | R3 | Which offline-discovered cluster the candidate belongs to, with membership confidence; ties broken by ascending archetype ID. |
| [`integrity.py`](integrity.py) | `IntegrityReport` | R4 (Gates & Eligibility) | The honeypot verdict: findings, a composite honeypot score, and `is_honeypot`. A finding without at least one evidence reference cannot be constructed. |
| [`eligibility.py`](eligibility.py) | `EligibilityReport` | R4 | Job-description hard blocks and soft penalties, each with a stable code and supporting evidence. |
| [`quality.py`](quality.py) | `CandidateQualityVector` (+ the `FeatureLayout` type) | R5 (Scoring) | The fixed-length, versioned numeric feature vector `ScoringEngine` consumes — the boundary between rich typed profiles and vectorized, columnar scoring. The populated feature-layout registry itself lives in `src/redstack/features/registry.py`; this module owns only the frozen value-object type. |

## Cross-slice invariants

- Every slice that carries a `candidate_id` carries the *same* `candidate_id` as every other slice on the same representation — enforced by the aggregate, not by each slice individually.
- `IntegrityReport` and `EligibilityReport` together determine the score floor mask applied at R5 (`ScoringEngine`): a honeypot or an ineligible candidate is floored, never partially penalized.
- `SemanticProfile.vector_ref` and `ArchetypeAssignment` are populated together at R3 by the same engine (`SemanticEngine`), since both come from the same precomputed-vector lookup.

See [`/ARCHITECTURE.md` §6](../../../../ARCHITECTURE.md#6-the-engines) for which engine populates each slice.
