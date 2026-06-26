#!/usr/bin/env python3
"""
Root-level entrypoint proxy for auto-evaluator compliance (submission spec §10.3).

Judge invocation:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Forwards to:
    python -m redstack.cli.app rank --input <candidates> --output <out>
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Resolve repo root and src/ once at import time so bare `python rank.py` works
# without uv or an editable install — PYTHONPATH is injected into the subprocess.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"

# ---------------------------------------------------------------------------
# Deterministic thread-count pins — must be set before importing numpy / BLAS.
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RedStack candidate ranker — submission spec §10.3 entrypoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        required=True,
        metavar="PATH",
        help="Path to the input candidates.jsonl file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Destination path for the output submission.csv file.",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "redstack.cli.app",
        "rank",
        "--input",
        args.candidates,
        "--output",
        args.out,
    ]
    env = os.environ.copy()
    src = str(_SRC_DIR)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing_pp}" if existing_pp else src
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
