# `configs/profiles/` — Layer-2 Environment Overrides

The final, optional merge layer — small overrides applied on top of `base.yaml` and the active `runtime/` file for a specific execution environment.

| Profile | Purpose | Overrides |
|---|---|---|
| [`ci.yaml`](ci.yaml) | Continuous-integration runs | `logging.level: WARNING`; shrinks `offline.search_budget`, `offline.embedding_batch_size`, and `offline.feature_batch_size` so the offline pipeline's CI exercise runs in a fraction of the time, on a representative sample rather than the full candidate pool. |
| [`local.yaml`](local.yaml) | Developer-machine runs | `logging.level: DEBUG`; sets `logging.deterministic: false` to allow human-readable, timestamped local log output (this flag controls log formatting only — it has no effect on ranking determinism, which is governed entirely by `config/determinism.py`). |

Select a profile with `--profile ci` or `--profile local` on either CLI verb; omit the flag to run with only `base.yaml` and the mode's `runtime/` file. See [`/ARCHITECTURE.md` §9](../../ARCHITECTURE.md#9-configuration-architecture) for the full three-layer resolution order.
