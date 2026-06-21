# `config/` — Typed Schema, Loader, and Determinism Policy

This package owns three distinct concerns, deliberately split across three files so the *pure* parts (the typed schema) can be imported from anywhere, while the *IO* parts (the loader) are reachable only from the orchestration and CLI layers. See [`/ARCHITECTURE.md` §9](../../../ARCHITECTURE.md#9-configuration-architecture).

## File inventory

| File | Owns | Importable from |
|---|---|---|
| [`schema.py`](schema.py) | A `pydantic` v2 model for every YAML file in `configs/`, plus the runtime enums (`RunMode`, `Profile`, `LogLevel`, `MalformedRecordPolicy`, `ScoreTransformKind`, `AnchorPolarity`, `HoneypotSeverity`). All models are frozen and reject unknown keys (`extra="forbid"`), so a configuration typo fails the load instead of silently doing nothing. Includes `RedstackConfig` (the fully-merged root), `DeterminismConfig`, `BudgetConfig`, `PathsConfig`, `OnlineRuntimeConfig`, `OfflineRuntimeConfig`, `ScoringWeightsConfig`, `LexiconSeedConfig`, `JdAnchorsConfig`, `EligibilityRulesConfig`, `HoneypotRulesConfig`, `IntegrityThresholds`, `ScoringPolicy`, `BehavioralPolicy`, `LogisticsPolicy`, and more. | anywhere — `engines`, `features`, `domain`-adjacent code, `pipelines`, `cli` |
| [`loader.py`](loader.py) | `load_config()` — the deterministic three-layer deep merge (`base → runtime/<mode> → profile`) and validate-or-die. Also `load_scoring_weights`, `load_lexicon_seed`, `load_jd_anchors`, `load_eligibility_rules`, `load_honeypot_rules` (typed readers for the individual seed files), `deep_merge`, and `config_fingerprint` (the config hash recorded in every run report). Raises `ConfigLoadError` on any load/validation failure. | only `pipelines/` and `cli/` |
| [`determinism.py`](determinism.py) | `DeterminismPolicy`, `apply_determinism()`, `assert_determinism()`, `pin_determinism()`, `make_rng()`, `build_onnx_session_options()` / `onnx_session_options()` — the single home for the global seed, `OMP_NUM_THREADS`/`MKL_NUM_THREADS` pinning, numpy RNG construction, and ONNX Runtime session thread/provider pinning. | `pipelines/`, `cli/` |

## Why `config.loader` is restricted

`config.schema` is pure (no IO) and is therefore safe for `engines/` and `features/` to depend on for typed config access. `config.loader` *does* IO (reading YAML off disk) and is restricted to `pipelines/`/`cli/` by an import-linter contract — this is what guarantees an engine can never accidentally trigger a filesystem read by importing the "wrong half" of the config package.

## Why determinism is owned in exactly one file

Every guarantee in [`/ARCHITECTURE.md` §10](../../../ARCHITECTURE.md#10-determinism-and-performance) — pinned thread counts, a single seeded RNG construction path, a single ONNX Runtime session configuration — is asserted from this one module, called once at startup by the CLI before any pipeline runs. Concentrating it here means there is exactly one place to audit when proving the system's determinism guarantees hold.
