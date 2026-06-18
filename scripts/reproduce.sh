#!/usr/bin/env bash
# Stage-3 single reproduce command (spec 10.3). A literal wrapper around the
# `redstack rank` CLI verb — agrees with submission_metadata.yaml:reproduce_command.[cite: 5]
set -euo pipefail

# Injected environment defensive pins to guarantee thread-count invariance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run the runtime engine targeting our input facts and output artifacts[cite: 1]
uv run python -m redstack.cli.app rank \
  --input "${1:-data/raw/candidates.jsonl}" \
  --output "${2:-artifacts/submission.csv}"