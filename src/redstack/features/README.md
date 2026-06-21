# `features/` — Pure Feature Extraction

**Purity contract.** This package imports only `domain/`, `config.schema`, the standard library, and `numpy`. No ports, no IO, no ML runtime, no clock, no RNG. Recency calculations take an injected `as_of` date. This layer never calls `EmbeddingModelPort` or `SemanticVectorStorePort` directly — semantic similarity values arrive already-resolved from `SemanticEngine`, which owns those ports. See [`/ARCHITECTURE.md` §1](../../../ARCHITECTURE.md#1-what-redstack-does) and [`docs/specs/REDSTACK_FEATURE_LAYER.md`](../../../docs/specs/REDSTACK_FEATURE_LAYER.md).

## The central design rule

A feature value is an **evidence aggregate**, never a keyword flag. A claimed skill with no corroboration is worth nothing; a claimed skill that also appears in a role description, carries endorsements weighted by time-on-skill, and has a corroborating semantic-anchor match is worth a great deal. This is why a keyword-stuffed profile scores near zero on competency features by construction, not by a special-cased penalty.

Every feature is emitted as a `FeatureCell(value, confidence, evidence)` — a value, a confidence score, and one or more references to the exact raw fields the value was derived from. A `FeatureCell` whose evidence path doesn't resolve in the source record cannot be constructed (`ProvenanceError`).

## File inventory

| File | Role |
|---|---|
| [`layout.py`](layout.py) | The ordered, versioned index map binding every feature id to a fixed position in the candidate quality vector. Defines `SourceSlice`, builds the frozen layout spec and group ordering, and exposes `index_of` / `group_of` / `group_column`. The single source of truth for feature order. |
| [`registry.py`](registry.py) | The populated `FeatureRegistry`: every `FeatureDefinition` (id, group, dtype, bounds, dependencies, tier A–D, polarity). Rejects unknown feature ids. Builds on `layout.py`. |
| [`view.py`](view.py) | Shared numeric helpers (`clamp_unit`, `bounded_log_scale`, `inverse_bounded`, `recency_unit`, `days_between`, `mean_of`) and the `FeatureCell` / `FeatureView` types extractors use to emit and engines use to read feature values. |
| [`store.py`](store.py) | Feature store metadata: `FeatureProvenance`, `FeatureLineage`, `FeatureSnapshot`, `FeatureAuditRecord`, `FeatureImportance`, `FeatureContracts` (cross-feature invariants), `FeatureValidation`, `FeatureCache`. |
| [`evidence.py`](evidence.py) | `resolve_path` / `mint` — resolves a dotted/indexed path (e.g. `career_history[0].title`) into a raw candidate record and mints an `EvidenceRef`, raising `ProvenanceError` immediately if the path doesn't resolve. The mechanical core of the no-hallucination guarantee. |
| [`parsing.py`](parsing.py) | `validate()` — raw dict → typed `RawCandidate`, tolerant of schema drift, never silently coercing. Also defines `FeatureCell`, `make_cell`, `clamp_unit`, `mint_evidence`, and a local `resolve_path` used during parsing-time evidence minting. |
| [`normalize.py`](normalize.py) | Canonical text/date/skill-token/company normalization; `compose_embedding_document` — builds the exact text document that is later embedded, in a fixed field order pinned by `embedding_manifest.json` so the offline and online normalization paths produce byte-identical documents. |
| [`career.py`](career.py) | `extract_career` / `extract_pvs` — tenure, recency, title-trajectory, and the product-vs-services classification the job description hinges on; descriptions dominate titles by design, since titles in the candidate pool are not reliable signal. |
| [`skills.py`](skills.py) | `CompetencyConcept`, `CompetencyLexicon`, and `extract()` — the trust-weighted competency aggregation (endorsement × duration × assessment coherence) for each technical concept group, and the anti-keyword-stuffing primitive. |
| [`education.py`](education.py) | `extract_education` — tier/field/timeline features, cross-linking to the honeypot detectors on an impossible timeline. |
| [`geography.py`](geography.py) | `extract_geography` — location/hub/relocation/notice/salary-overlap features against the job description's target hub set and salary band. |
| [`signals.py`](signals.py) | `extract()` — turns the candidate's platform engagement signals into the availability/engagement/responsiveness/reliability/verification composites, honoring the sentinel-value discipline (`-1`, `{}` map to an explicit "unknown," never silently to zero). |
| [`latents.py`](latents.py) | `extract()` — the job-description latent composition (the positive/negative `jd.*` signals): builds each latent from its constituent upstream cells and computes `jd.keyword_only` as the anti-stuffer subtractive term. |
| [`honeypot.py`](honeypot.py) | `extract()` — the eleven impossible-profile detectors (timeline, skill-time contradiction, employment overlap, title-seniority anomaly, education-career anomaly, salary anomaly, experience inflation, keyword stuffing, behavioral inconsistency, signal impossibility, identity anomaly) plus the calibrated composite risk score. |
| [`extraction.py`](extraction.py) | The orchestrator: `build_career_profile`, `build_credibility_profile`, `build_logistics_profile`, `build_behavioral_profile` assemble the domain profile slices from the extractors above; `build_cells` / `extract_row` / `fold_semantic` assemble and fold a candidate's full feature-cell set into the columnar layout. |

## Extraction order (fixed, for determinism)

Normalization → primitive extractors (career, education, geography, signals — independent, data-parallel) → competency/credibility (skills) → honeypot detectors → latent composition (depends on the competency and career outputs) → semantic folding (`fold_semantic`, populated by `SemanticEngine`'s output, not computed here).

## Memory discipline

The full candidate pool's feature values live in one columnar `(N, D)` float32 matrix aligned to the layout in `layout.py`; per-feature confidence and evidence are fully materialized only for gate survivors and the eventual top-100 — materializing them for all 100,000 candidates would roughly double the matrix's memory footprint for no benefit, since 99.8% of candidates never need an explained reason.
