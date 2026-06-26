# `data/golden/` — Gold-Label Calibration Data

| Path | Used by |
|---|---|
| `golden_labels.csv` | Offline stages O8–O10 (gold-label weight calibration) and O15 (ranking calibration), when `redstack build` is run **without** `--no-golden-labels`. |

This file is a human-labeled relevance dataset — one row per labeled candidate, a relevance tier (0–4, with honeypots forced to tier 0), and the reviewer's reference reasoning. It is treated as a **fixed input** to a deterministic calibration search once committed, with a leakage-free train/validation split recorded separately in `artifacts/calibration_split.json` so the same candidate never appears in both halves.

If this file is absent, `redstack build` raises `GoldLabelSeedMissingError` unless `--no-golden-labels` is passed, in which case the build skips the search entirely and packages the seed weights from [`configs/weights/scoring_weights.yaml`](../../configs/weights/README.md) unchanged as the locked online weights.

See [`/ARCHITECTURE.md` §5.1](../../ARCHITECTURE.md#51-offline-pipeline--o0-through-o18) (stages O8–O9) and [`docs/runbook.md`](../../docs/runbook.md).
