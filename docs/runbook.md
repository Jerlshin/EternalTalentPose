# Runbook

How to go from an empty checkout to a finished, validated ranking. See [`/ARCHITECTURE.md`](../ARCHITECTURE.md) for *why* each step exists; this document is only the *how*.

## 1. Install

```bash
uv sync --frozen --group core --group offline --group dev   # full graph: hot path + offline ML + tooling
uv pip install -e .                                            # src/ layout: install before tests run
```

Use `--group core --group dev` instead if you only intend to run the online ranking step against an already-built `artifacts/` directory — it skips `sentence-transformers`/`scikit-learn`, which are offline-only and forbidden from the online import graph.

## 2. Populate `data/`

`data/` is gitignored and read-only at runtime — nothing in the pipeline writes into it. Two files are needed:

| Path | Required for | Notes |
|---|---|---|
| `data/raw/candidates.jsonl` | both `redstack build` and `redstack rank` | One JSON object per line, conforming to [`docs/guide/candidate_schema.json`](guide/candidate_schema.json). [`docs/guide/sample_candidates.json`](guide/README.md) shows the expected shape. |
| `data/golden/golden_labels.csv` | the real O8–O10 gold-label weight-calibration search | Only needed when running `redstack build` *without* `--no-golden-labels`. Without it, the build uses the locked-heuristic seed weights from `configs/weights/scoring_weights.yaml` as-is. |

See [`data/README.md`](../data/README.md) for the exact expected schema and lifecycle of both files.

## 3. Build the offline artifact set

```bash
make build
# == uv run redstack build --config configs/runtime/offline.yaml --no-golden-labels
```

This runs offline stages O0–O18 and writes a complete, hash-pinned `artifacts/` tree plus `artifacts/MANIFEST.json`. Drop `--no-golden-labels` (the default) to run the real gold-label calibration search — this requires `data/golden/golden_labels.csv` to be present, and raises loudly if it is not. Use `--force O9,O15` (comma-separated stage ids) to force specific stages to recompute even if their checkpoints are up to date. See [`src/redstack/pipelines/offline/README.md`](../src/redstack/pipelines/offline/README.md) for the full stage list.

## 4. Run the online ranking pass

```bash
make rank
# == uv run redstack rank --input data/raw/candidates.jsonl --output artifacts/submission.csv
```

This runs online stages R0–R9 against the already-built `artifacts/` set and writes `artifacts/submission.csv` plus `artifacts/run_report.json`. It exits non-zero if the run falls outside its compute budget. `scripts/reproduce.sh` wraps this exact invocation for a clean, single-command reproduction. See [`src/redstack/pipelines/online/README.md`](../src/redstack/pipelines/online/README.md) for the full stage list.

## 5. Validate the output

```bash
make validate
# == uv run redstack validate --submission artifacts/submission.csv
```

Runs `ValidationEngine` over the finished CSV: row count, rank uniqueness and range, candidate-ID format and uniqueness, score monotonicity, and the ascending-tie-break rule. This is the same logic the online run already applies before writing the file (R8) — `redstack validate` lets you re-check a file independently, e.g. after copying it elsewhere.

## 6. Re-running after a change

- Changed a `configs/` file under `weights/`, `lexicon/`, `anchors/`, `gates/`, or `integrity/`? Re-run `make build` — these are *authored intent*; the online run only ever reads their *compiled* counterparts in `artifacts/`.
- Changed `configs/runtime/online.yaml`? No rebuild needed — it's read directly by `redstack rank`.
- Changed candidate data only? Re-run `make rank`; no rebuild needed unless you're also recalibrating weights against new gold labels.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `redstack build` raises `GoldLabelSeedMissingError` | Running without `--no-golden-labels` but `data/golden/golden_labels.csv` is missing. Either add the file or pass `--no-golden-labels`. |
| `redstack rank` aborts at startup with an artifact/manifest error | `artifacts/` is stale, partially built, or was hand-edited. Re-run `make build`. Artifacts are content-hash-verified on every load — any corruption or staleness fails loudly rather than degrading the ranking silently. |
| `redstack rank` reports `within_budget=False` | The run exceeded its internal compute budget. Check `run_report.json`'s `timings` block for the offending stage; see [`/ARCHITECTURE.md` §10](../ARCHITECTURE.md#10-determinism-and-performance). |
****