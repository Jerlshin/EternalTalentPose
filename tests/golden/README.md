# `tests/golden/` — Byte-Exact Snapshot Tests

Golden tests lock an exact output so that any unintended change to scoring, feature extraction, serialization, or reasoning surfaces as a diff that must be explicitly re-blessed with a documented reason — never silently absorbed. They are deliberately brittle on purpose: a system with one scored ranking run and no live feedback loop benefits from a regression net that asserts bytes and values, not ranges.

## What belongs here

Per [`docs/specs/REDSTACK_TESTING_STRATEGY.md` §15](../../docs/specs/REDSTACK_TESTING_STRATEGY.md):

- **Golden candidates** — a small, hand-curated, frozen set of raw candidate records spanning the system's archetypes (an ideal job-description match, a keyword-stuffer, a consulting-only career, a pure researcher, an honeypot, a sentinel-laden behavioral profile), each committed with an expected-feature table.
- **Golden rankings** — for the golden candidate set under the locked artifacts, the exact `ScoredCandidate` ordering and `final_score`s at fixed decimal precision.
- **Golden reasonings** — the exact rendered reasoning string, clause set, and evidence references for each golden top-K candidate, locking the no-hallucination and rank-consistency properties against drift.
- **Golden artifacts** — the deterministic offline artifacts (manifest structure, locked scoring weights, integrity thresholds, feature manifest) snapshotted by content hash; embedding artifacts are checked for cosine-similarity stability instead of bit-exact equality, since they depend on a third-party model's floating-point behavior.

This directory currently holds only its package marker (`__init__.py`) — these are the snapshots to add as the corresponding pipeline stages stabilize. See [`tests/fixtures/README.md`](../fixtures/README.md) for the candidate fixtures these snapshots are built against.
