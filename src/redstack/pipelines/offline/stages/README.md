# `pipelines/offline/stages/` — O0 Through O18

One module per offline build stage. Every stage is a pure transform over its declared inputs (upstream artifacts and `configs/`), producing one or more artifacts registered in `OfflineArtifactRegistry`. See [`/ARCHITECTURE.md` §5.1](../../../../../ARCHITECTURE.md#51-offline-pipeline--o0-through-o18) for the narrative description and [`docs/specs/REDSTACK_OFFLINE_PIPELINE.md` Part 2](../../../../../docs/specs/REDSTACK_OFFLINE_PIPELINE.md) for the exhaustive per-stage specification (algorithm, complexity, artifact schema).

## Stage inventory

| Stage | File | Purpose | Key artifact(s) |
|---|---|---|---|
| O0 | [`census.py`](census.py) | Profile the candidate pool — coverage, distributions, outliers — before any threshold is set. | `dataset_profile.json` |
| O1 | [`normalization.py`](normalization.py) | Canonicalize text, dates, skill tokens, company names. | `canonical_maps.json` |
| O2 | [`validation.py`](validation.py) | Validate the normalized pool against the candidate schema; structural rejects only. | `validation_report.json` |
| O3 | [`honeypot_discovery.py`](honeypot_discovery.py) | Discover and calibrate impossible-profile detection thresholds against the census. | `integrity_rules.json`, `honeypot_catalog.json`, `calibration/integrity_thresholds.json` |
| O4 | [`lexicon_discovery.py`](lexicon_discovery.py) | Mine domain terminology from role descriptions, seeded from `configs/lexicon/lexicon.seed.yaml`. | `lexicon/lexicon.compiled.json`, `term_graph.json`, `phrase_graph.json` |
| O5 | [`vocab_expansion.py`](vocab_expansion.py) | Expand the lexicon with embedding-nearest synonyms. | `concepts.json` (expanded) |
| O6 | [`jd_concepts.py`](jd_concepts.py) | Package the job description's semantic anchors and eligibility rules from `configs/anchors/` and `configs/gates/`. | `jd_concepts.json`, `gates/eligibility_rules.yaml` |
| O7 | [`archetype_discovery.py`](archetype_discovery.py) | Cluster the candidate pool into archetypes via seeded k-means. | `archetypes.json`, `archetypes/centroids.npy` |
| O8 | [`labeling.py`](labeling.py) (+ [`_labeling_seed.py`](_labeling_seed.py)) | Human-in-the-loop gold-label ingestion (`ReviewTag`, `GoldLabelSeed` — read from `data/golden/golden_labels.csv`) and the leakage-free calibration split. | `gold_labels.json`, `calibration_split.json` |
| O9 | [`weight_search.py`](weight_search.py) | Calibrate `ScoringWeights` against the gold labels, cross-validated; skipped (seed weights passed through) when the build runs with `--no-golden-labels`. | `weights/scoring_weights.locked.yaml`, `calibration_report.json` |
| O10 | [`feature_importance.py`](feature_importance.py) | Quantify per-feature contribution, used by the reasoning engine to select evidence. | `feature_importance.json` |
| O11 | [`behavioral_calib.py`](behavioral_calib.py) | Fit the bounded behavioral-multiplier curves. | `behavioral_weights.json` |
| O12 | [`risk_calib.py`](risk_calib.py) | Set the honeypot composite threshold and confidence-shrink parameters. | merged into `calibration/integrity_thresholds.json`, `risk_weights.json` |
| O13 | [`embedding_gen.py`](embedding_gen.py) | The compute-dominant stage: encode every candidate and anchor document; export the ONNX twin used by the online fallback encoder. | `embeddings/candidate_vectors.parquet`, `embeddings/anchor_vectors.npy`, `model/encoder.onnx`, `embedding_manifest.json` |
| O14 | [`feature_snapshot.py`](feature_snapshot.py) | Run every feature extractor over the full pool into the canonical `(N, D)` matrix — both the calibration substrate and the online correctness oracle. | `feature_snapshot.parquet`, `feature_manifest.json` |
| O15 | [`ranking_calib.py`](ranking_calib.py) | Fit the order-preserving score-presentation curve; confirm tie-break behavior. | `ranking_calibration.json` |
| O16 | [`reasoning_templates.py`](reasoning_templates.py) | Build evidence-slot reasoning templates from gold reference reasonings (data file: [`reasoning_templates.json`](reasoning_templates.json) — the committed seed/schema, distinct from the generated artifact of the same name under `artifacts/`). | `reasoning_templates.json` |
| O17 | [`packaging.py`](packaging.py) | Hash every artifact and write the manifest — the single contract the online run verifies against. | `MANIFEST.json` |
| O18 | [`reproducibility.py`](reproducibility.py) | Prove the build is reproducible and online-consumable: reload and verify, recompute features online for a sample and diff against the snapshot, dry-run rank the golden set. | `reproducibility_report.json` |

## Dependency shape

Stages form a DAG, not a line — `archetype_discovery` (O7) and `feature_snapshot` (O14) both depend on the embeddings from O13 but not on each other and can run concurrently; `behavioral_calib` (O11), `risk_calib` (O12), and `feature_importance` (O10) all depend on O9 but not on each other. [`../graph.py`](../graph.py) encodes the exact edges and the deterministic topological order the runner follows.
