# `tests/integration/` — Full-Pipeline Tests

Exercises the offline build and the online ranking run end to end, against a small sample candidate pool, asserting the produced submission passes external structural validation. This is the only category that runs both an offline build and an online rank in the same test process.

## Current inventory

| File | Covers |
|---|---|
| [`test_online_pipeline.py`](test_online_pipeline.py) | Runs the full R0–R9 online ranking sequence end to end and asserts the produced ranking is structurally valid. |

## What belongs here

Per [`docs/specs/REDSTACK_TESTING_STRATEGY.md` §8–§9](../../docs/specs/REDSTACK_TESTING_STRATEGY.md), this category's full intended scope also includes: a full `O0`–`O18` offline build against a small fixture pool with manifest verification; an offline-build-then-online-rank round trip (proving the artifact handoff works end to end); and CLI-verb-level tests invoking `redstack build`/`rank`/`validate` as subprocesses and asserting exit codes. New integration tests should run against the candidate fixtures in [`../fixtures/`](../fixtures/README.md), never against `data/raw/candidates.jsonl` directly.
