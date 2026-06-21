# `configs/` — Declarative System Behavior

Everything in this directory is **human-authored intent**, never machine-generated, and never read directly by the online ranking run except for plain runtime/IO knobs. See [`/ARCHITECTURE.md` §9](../ARCHITECTURE.md#9-configuration-architecture) for why that distinction matters.

> **The hard rule.** `configs/` is *intent*; `artifacts/` is *derived fact*. Behavior-defining configuration here (scoring weights, the lexicon seed, semantic anchors, eligibility gates, honeypot rule shapes) is **authored** in this directory but **compiled or calibrated** by the offline build into its frozen counterpart under `artifacts/`. The online ranking run never reads `configs/weights/`, `configs/lexicon/`, `configs/anchors/`, `configs/gates/`, or `configs/integrity/` — only their `artifacts/` counterparts. It reads only `configs/base.yaml`, `configs/runtime/online.yaml`, and the active profile.

## Resolution order

Configuration is resolved as a deterministic three-layer deep merge, later layers overriding earlier ones:

```text
base.yaml  →  runtime/{online|offline}.yaml  →  profiles/{ci|local}.yaml
```

Every YAML file maps to a `pydantic` v2 model in `src/redstack/config/schema.py` with `extra="forbid"` — an unrecognized key fails the load loudly instead of silently doing nothing.

## Inventory

| Path | Layer | Authored by | Consumed by | Frozen counterpart in `artifacts/` |
|---|---|---|---|---|
| [`base.yaml`](base.yaml) | 0 | Platform | every run | — |
| [`runtime/online.yaml`](runtime/README.md) | 1 (online) | Platform | `redstack rank` (R0) | — |
| [`runtime/offline.yaml`](runtime/README.md) | 1 (offline) | Platform | `redstack build` | — |
| [`weights/scoring_weights.yaml`](weights/README.md) | seed | Modeling | offline stage O9 | `artifacts/weights/scoring_weights.locked.yaml` |
| [`lexicon/lexicon.seed.yaml`](lexicon/README.md) | seed | Modeling | offline stage O4 | `artifacts/lexicon/lexicon.compiled.json` |
| [`anchors/jd_anchors.yaml`](anchors/README.md) | authored | Modeling | offline stage O6 | `artifacts/embeddings/anchor_vectors.npy` |
| [`gates/eligibility_rules.yaml`](gates/README.md) | authored | Modeling | offline stage O6 | `artifacts/gates/eligibility_rules.yaml` (packaged) |
| [`integrity/honeypot_rules.yaml`](integrity/README.md) | authored | Modeling | offline stage O3 | `artifacts/calibration/integrity_thresholds.json` |
| [`profiles/ci.yaml`](profiles/README.md) | 2 | Platform | CI runs | — |
| [`profiles/local.yaml`](profiles/README.md) | 2 | Platform | developer runs | — |

See each subdirectory's own `README.md` for the file's exact schema and the stage that consumes it.
