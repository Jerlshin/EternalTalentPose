#!/usr/bin/env python3
"""
Cross-platform task runner for RedStack.
Pure stdlib only — no POSIX binaries, no shell assumptions.
Works identically on macOS, Linux, and native Windows (PowerShell/CMD).

Usage:
    python scripts/run.py build
    python scripts/run.py rank
    python scripts/run.py validate
    python scripts/run.py clean
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Deterministic single-thread execution caps — applied BEFORE any subprocess.
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run(*args: str) -> None:
    result = subprocess.run([PYTHON, "-m"] + list(args), cwd=REPO_ROOT)
    sys.exit(result.returncode)


def cmd_build() -> None:
    artifact_dirs = [
        "artifacts/embeddings",
        "artifacts/models",
        "artifacts/gates",
        "artifacts/weights",
        "artifacts/calibration",
        "artifacts/lexicon",
        "artifacts/archetypes",
    ]
    for d in artifact_dirs:
        Path(REPO_ROOT / d).mkdir(parents=True, exist_ok=True)
    _run(
        "redstack.cli.app",
        "build",
        "--config",
        "configs/runtime/offline.yaml",
        "--no-golden-labels",
    )


def cmd_rank() -> None:
    Path(REPO_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    _run(
        "redstack.cli.app",
        "rank",
        "--input",
        "data/raw/candidates.jsonl",
        "--output",
        "artifacts/submission.csv",
    )


def cmd_validate() -> None:
    _run(
        "redstack.cli.app",
        "validate",
        "--submission",
        "artifacts/submission.csv",
    )


def cmd_clean() -> None:
    noise_dirs = [
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "dist",
        "build",
    ]
    for name in noise_dirs:
        target = REPO_ROOT / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            print(f"removed {target}")

    for pycache in REPO_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)

    print("clean done.")


COMMANDS = {
    "build": cmd_build,
    "rank": cmd_rank,
    "validate": cmd_validate,
    "clean": cmd_clean,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python scripts/run.py [{' | '.join(COMMANDS)}]", file=sys.stderr)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
