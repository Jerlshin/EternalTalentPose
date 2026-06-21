# `cli/` — The `redstack` Command-Line Entrypoints

The thinnest layer in the codebase: a single Typer application, three verbs, **no business logic**. Each verb parses arguments, resolves and validates configuration, applies the determinism policy, and dispatches to a pipeline composition root. See [`/ARCHITECTURE.md` §4](../../../ARCHITECTURE.md#4-layer-reference).

## File inventory

| File | Verb | What it does |
|---|---|---|
| [`app.py`](app.py) | — | The Typer application object; registers `build`, `rank`, `validate` as its three commands. |
| [`build.py`](build.py) | `redstack build` | Resolves `configs/` (`--config`, `--profile`), applies the determinism policy, then calls `pipelines.offline.compose.run_offline_build()` (the default — runs the real O8–O10 gold-label calibration search, raising `GoldLabelSeedMissingError` loudly if `data/golden/golden_labels.csv` isn't present) or `run_offline_build_with_locked_heuristics()` (`--no-golden-labels` — skips the search, packages the seed weights from `configs/weights/scoring_weights.yaml` unchanged). `--force O9,O15` forces specific stage ids to recompute regardless of checkpoint state. |
| [`rank.py`](rank.py) | `redstack rank` | Resolves `configs/`, applies the determinism policy, then calls `pipelines.online.compose.run_online_rank()` with `--input` (default `data/raw/candidates.jsonl`), `--output` (default `artifacts/submission.csv`), and `--participant-id`. Prints the row count, top-100 honeypot rate, and budget compliance; exits non-zero if the run fell outside budget. |
| [`validate.py`](validate.py) | `redstack validate` | Re-validates an already-written CSV against the structural submission rules directly (header match, exact row count, rank-equals-position, candidate-ID pattern and uniqueness, non-blank reasoning, non-increasing score, ascending-id tie-break) — independent of any pipeline run, useful for re-checking a file after it's been copied or handed off. `--expected-size` overrides the required row count (default 100). |

## Why both `build` and `rank` accept `--config` as a directory or a file

Both verbs resolve `--config` by walking up from the given path until they find a directory containing `base.yaml` — so `--config configs`, `--config configs/base.yaml`, and `--config configs/runtime/offline.yaml` are all accepted and resolve to the same configuration root. This matches every invocation style used by the [`Makefile`](../../../Makefile) and the project's [`CLAUDE.md`](../../../CLAUDE.md) without requiring the caller to know the exact resolution rule.

## Exit codes

Every verb exits `0` on success. `build` exits `1` if any stage failed. `rank` exits `1` if the run fell outside its compute budget. `validate` exits `1` if any structural violation was found. None of the three ever exits non-zero silently — every failure path prints the specific violation(s) to stderr first.

See [`README.md` § Build, rank, validate](../../../README.md#build-rank-validate) for example invocations and [`/ARCHITECTURE.md` §5](../../../ARCHITECTURE.md#5-two-pipelines-one-codebase) for what each verb runs under the hood.
