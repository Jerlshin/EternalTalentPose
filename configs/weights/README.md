# `configs/weights/` — Scoring Weight Seed

[`scoring_weights.yaml`](scoring_weights.yaml) is the **search input**, not the online contract. It holds the human-authored starting weights for each score component, keyed exactly to `redstack.domain.enums.ScoreComponent`:

| Component | Seed weight |
|---|---|
| `semantic_fit` | 0.30 |
| `skill_match` | 0.25 |
| `career_fit` | 0.20 |
| `credibility` | 0.10 |
| `experience_fit` | 0.08 |
| `education_fit` | 0.04 |
| `archetype_fit` | 0.03 |

Logistics fit and behavioral fit are deliberately **absent** from this list — they are bounded multipliers applied by `ScoringEngine`, never weighted-summed components. The job description's eligibility rules are gates, not weighted components, either.

## Two ways offline stage O9 uses this file

| Mode | Behavior |
|---|---|
| `redstack build` (default, gold labels present) | O9 treats these weights as a **starting point** for a cross-validated search against `data/golden/golden_labels.csv`, and emits the calibrated result to `artifacts/weights/scoring_weights.locked.yaml`. |
| `redstack build --no-golden-labels` | O9 **skips the search** and emits these weights through unchanged as the locked online weights. In this mode, editing this file directly changes ranking behavior on the next build. |

The online ranking run never reads this file — it reads only `artifacts/weights/scoring_weights.locked.yaml`, the frozen, validated result. See [`/ARCHITECTURE.md` §6](../../ARCHITECTURE.md#6-the-engines) (`ScoringEngine`) and [`/ARCHITECTURE.md` §9](../../ARCHITECTURE.md#9-configuration-architecture).
