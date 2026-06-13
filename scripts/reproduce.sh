#!/usr/bin/env bash
# Stage-3 single reproduce command (spec 10.3). A literal wrapper around the
# `redstack rank` CLI verb — agrees with submission_metadata.yaml:reproduce_command.
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
uv run redstack rank \
  --candidates "${1:-data/raw/candidates.jsonl}" \
  --out "${2:-submission.csv}"
