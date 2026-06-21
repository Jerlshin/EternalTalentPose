# `scripts/` — Operational Glue

Small, standalone scripts that sit outside the `src/redstack` package and are not subject to its import boundaries — each is explicitly justified below where it touches something the package layers would otherwise forbid.

## Inventory

| Script | Purpose |
|---|---|
| [`reproduce.sh`](reproduce.sh) | A literal one-line wrapper around `uv run python -m redstack.cli.app rank --input <candidates> --output <submission>`, with `OMP_NUM_THREADS`/`MKL_NUM_THREADS` pinned defensively for thread-count invariance. This is the canonical single command referenced by `submission_metadata.yaml:reproduce_command` — running it twice on the same input must produce byte-identical output. Accepts the input and output paths as positional arguments, defaulting to `data/raw/candidates.jsonl` and `artifacts/submission.csv`. |
| [`make_sandbox_sample.py`](make_sandbox_sample.py) | Streams the first 500 records of `data/raw/candidates.jsonl` into `data/raw/sandbox_sample.jsonl` under constant memory — a small, deterministic slice for local development and demos that doesn't require the full candidate pool to be present. |
| [`profile_submission_analytics.py`](profile_submission_analytics.py) | A development triage and validation dashboard for an already-produced `submission.csv`. Re-runs the *real* R0–R7 online stages restricted to just the candidates already in the submission (every per-candidate computation in R2–R5 is population-independent, so this reproduces byte-identical results to the full run, much faster — R3 is a pure lookup against the precomputed vector store already built by the offline pipeline). Reads `artifacts/submission.csv`, `data/raw/candidates.jsonl`, and `artifacts/run_report.json`; writes `artifacts/debug_top100_lean.json` and `artifacts/debug_dashboard.md`. |
| [`build_dashboard.py`](build_dashboard.py) | Renders the markdown triage dashboard into an interactive, self-contained HTML page for visual review. Reads `artifacts/debug_top100_lean.json` and `artifacts/run_report.json`, streams just the matching 100 records out of `data/raw/candidates.jsonl` (one pass, no full-file load), and fills in [`dashboard_template.html`](dashboard_template.html) (CSS + vanilla JS, candidate data embedded as a JSON island — zero network calls or external assets). Writes `artifacts/debug_dashboard.html`; deterministic — rerunning produces a byte-identical file as long as the inputs haven't changed. |

## Why this directory is exempt from the online containment rule

`profile_submission_analytics.py` imports adapters directly and is **not** part of `src/redstack/pipelines/online` or `src/redstack/engines` — [`/CLAUDE.md`](../CLAUDE.md)'s online containment rule binds the online *runtime* package, not ad-hoc analysis tooling that a developer runs by hand outside any compute-budgeted path. It deliberately carries zero parallel reimplementation of scoring/eligibility logic — every number it prints is read directly off the live engine objects, specifically to avoid the drift risk of a second, hand-maintained copy of that math falling out of sync with the real pipeline.

Run any of these with `uv run python scripts/<name>.py` (or, for the shell script, directly: `scripts/reproduce.sh`).
