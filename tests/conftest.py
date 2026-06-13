"""Shared pytest configuration and fixture registration.

Sprint 0 bootstrap: establishes deterministic test execution (thread pins
applied before any numpy/onnxruntime import) and the profile selection hook.
Concrete fixtures (the canonical fakes, sample_candidates, honeypot
exemplars, golden labels) land in tests/fixtures/ as the contract suites
are implemented — none are defined yet.
"""
from __future__ import annotations

import os

# Determinism is owned in config/determinism.py at runtime; tests assert the
# same invariant by pinning before collection so byte-identical output is
# reproducible regardless of the host's core count.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# The CI profile (configs/profiles/ci.yaml) drives smallest-footprint runs.
os.environ.setdefault("REDSTACK_PROFILE", "ci")

collect_ignore_glob: list[str] = []
