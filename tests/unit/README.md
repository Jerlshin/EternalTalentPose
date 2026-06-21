# `tests/unit/` — Isolated, Pure Tests

One domain model, feature transform, or engine method at a time. No IO, no mocking framework — domain and feature code is pure by construction, and the one engine that depends on a port (`SemanticEngine`) is tested against a fake from [`../fixtures/`](../fixtures/README.md), never a mock. This is the largest category in the [test pyramid](../README.md#test-pyramid-target-distribution-by-test-count).

## Current inventory

| File | Covers |
|---|---|
| [`test_eligibility_engine.py`](test_eligibility_engine.py) | `engines/eligibility.py` — each hard-block and soft-penalty code fires on its exemplar profile; a clean candidate passes every gate. |
| [`test_integrity_engine.py`](test_integrity_engine.py) | `engines/integrity.py` — each integrity flag fires on its exemplar impossibility; the honeypot verdict derivation (`is_honeypot`) is correct at the calibrated threshold. |
| [`test_lexicon_engine.py`](test_lexicon_engine.py) | `engines/lexicon.py` — compiled-lexicon matching and the anti-keyword-stuffing corroboration logic. |
| [`test_offline_compose.py`](test_offline_compose.py) | `pipelines/offline/compose.py` — the offline composition root wires its adapters and ports correctly. |
| [`test_validation_engine.py`](test_validation_engine.py) | `engines/validation.py` — the structural and reasoning-quality validation rules. |

## What belongs here

Per [`docs/specs/REDSTACK_TESTING_STRATEGY.md` §3–§5](../../docs/specs/REDSTACK_TESTING_STRATEGY.md), this category's full intended scope mirrors the source tree (`domain/`, `features/`, `engines/`, `config/`, `observability/`, `adapters/`) — every public type and pure transform gets a constructor-validation test (illegal states raise) and a positive-case test. New unit tests should be added alongside the corresponding source module under a matching subpath, e.g. a new test for `features/career.py` belongs at `tests/unit/features/test_career.py`.
