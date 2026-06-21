# `configs/runtime/` — Per-Mode Runtime Knobs

Layer-1 configuration: **IO and runtime parameters only, never scoring weights or business rules.** This is the layer that lets a reviewer change thread counts or batch sizes without any risk of touching the scoring outcome.

## Files

| File | Mode | Key fields |
|---|---|---|
| [`online.yaml`](online.yaml) | `redstack rank` (R0–R9) | `as_of` — the injected reference date all recency calculations use instead of the system clock. `top_k` — the validator-fixed top-100 cut. `malformed_record_policy` — `skip` or `abort` on an unparseable input line. `score_presentation` — the order-preserving display transform and decimal precision applied to the emitted `score` column (never alters ranking). |
| [`offline.yaml`](offline.yaml) | `redstack build` (O0–O18) | `as_of`, `seed` — the offline build's injected clock and RNG seed. `st_model_id` / `st_model_revision` — the pinned sentence-transformers model and revision used for embedding generation (O13). `kmeans_k` — the fixed archetype cluster count (O7). `search_budget` — the gold-label weight-search iteration budget (O9). `embedding_batch_size`, `feature_batch_size` — batching knobs for O13/O14. |

## Why this split exists

`base.yaml` (one layer up) holds defaults shared by both modes; this directory holds the knobs specific to *which* pipeline is running. Neither file ever defines a scoring weight, an eligibility rule, or an integrity threshold — those live in [`../weights/`](../weights/README.md), [`../gates/`](../gates/README.md), and [`../integrity/`](../integrity/README.md), authored separately so a reviewer auditing "did anyone change the ranking logic?" only has to look in one place.

See [`/ARCHITECTURE.md` §9](../../ARCHITECTURE.md#9-configuration-architecture) and [`/ARCHITECTURE.md` §10](../../ARCHITECTURE.md#10-determinism-and-performance) (the `as_of`/seed determinism guarantees these files supply).
