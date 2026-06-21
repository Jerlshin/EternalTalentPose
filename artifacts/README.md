# `artifacts/` — Offline Build Output

Gitignored, hash-pinned, write-once per build, and treated like a compiled binary — never hand-edited. This is the entire contract between the offline build and the online ranking run: the online run reads **only** from this directory (plus `configs/runtime/online.yaml` and the candidate file) and never recomputes anything found here. Rebuild with `redstack build` (`make build`). See [`/ARCHITECTURE.md` §8](../ARCHITECTURE.md#8-persistence-and-the-artifact-contract).

## The manifest

`MANIFEST.json` is the registry: for every artifact, its relative path, SHA-256 hash, producing build's code version, and schema version, plus a self-hash over the manifest's own canonical serialization. At load time (`redstack rank`, stage R0), the online run:

1. Verifies the manifest's self-hash.
2. Verifies every referenced artifact's SHA-256, streamed during the read (so verification costs essentially nothing extra).
3. Asserts cross-artifact coherence — embedding dimensions agree across the vector store, the anchor vectors, the archetype centroids, and the ONNX model; the feature-layout version agrees between the feature manifest and the locked scoring weights; every gate and integrity code referenced is a valid enum member.

**Any failure aborts the run.** There is no degraded mode — a ranking produced from a corrupted or incoherent artifact set would be silently wrong with no way to detect it afterward, which this repository's design treats as strictly worse than a loud failure at startup.

## Inventory

| Path | Producing stage | Consumed by |
|---|---|---|
| `MANIFEST.json` | O17 | R0 (all stages, transitively) |
| `dataset_profile.json` | O0 | offline only (O3, O11, O12 priors) |
| `canonical_maps.json` | O1 | offline normalization parity |
| `validation_report.json` | O2 | offline audit |
| `integrity_rules.json`, `honeypot_catalog.json` | O3 | R4 (`IntegrityEngine`) |
| `calibration/integrity_thresholds.json` | O3 + O12 | R4 (`IntegrityEngine`) |
| `risk_weights.json` | O12 | R4 |
| `lexicon/lexicon.compiled.json`, `concepts.json`, `term_graph.json`, `phrase_graph.json` | O4 + O5 | R2 (competency features), `LexiconEngine` |
| `jd_concepts.json` | O6 | R3 (`SemanticEngine` latents) |
| `gates/eligibility_rules.yaml` | O6 (authored, then packaged) | R4 (`EligibilityEngine`) |
| `archetypes/centroids.npy`, `archetypes.json` | O7 | R3 / R7 (archetype assignment, labels, reasoning fingerprints) |
| `gold_labels.json`, `calibration_split.json` | O8 | offline only (O9, O15) |
| `weights/scoring_weights.locked.yaml` | O9 | R5 (`ScoringEngine`) |
| `calibration_report.json` | O9 | offline audit |
| `feature_importance.json` | O10 | R7 (`ReasoningEngine` evidence selection) |
| `behavioral_weights.json` | O11 | R5 (multiplier bounds) |
| `embeddings/candidate_vectors.parquet`, `embeddings/anchor_vectors.npy`, `model/encoder.onnx`, `embedding_manifest.json` | O13 | R3 (`SemanticEngine`, lookup + fallback encode) |
| `feature_snapshot.parquet`, `feature_manifest.json` | O14 | offline (O9/O10) + R2 layout-version check |
| `ranking_calibration.json` | O15 | R6 (tie-break/monotonicity confirmation; the presentation curve is disabled online by default) |
| `reasoning_templates.json` | O16 | R7 (`ReasoningEngine`) |
| `reproducibility_report.json` | O18 | release/audit gate |
| `submission.csv`, `run_report.json` *(produced by `redstack rank`, not `redstack build`)* | R8/R9 | the ranking output itself |

## Versioning rule

A value-changing edit to an artifact's *content* (e.g. recalibrated weights with the same component set) is a minor version bump; a change to an artifact's *shape* (layout/order/dimension) is a major bump. `layout_version` is shared across the feature layout, the feature manifest, and the locked scoring weights — a mismatch raises `ArtifactContractError` at load time rather than silently scoring against the wrong feature positions.

## Local checkpoints

`_checkpoints/` (if present) holds per-stage `StageReceipt`s used by the offline runner's resume logic — these are build-process bookkeeping, not part of the artifact contract the online run reads, and are safe to delete to force a clean rebuild.
