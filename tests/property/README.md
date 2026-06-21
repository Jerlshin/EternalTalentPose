# `tests/property/` — Invariant Checks Over Generated Input

[Hypothesis](https://hypothesis.readthedocs.io/)-driven property tests: instead of asserting one example is correct, these assert a property holds for *every* input Hypothesis can generate, and shrink any counterexample found to a minimal reproduction.

## What belongs here

Per [`docs/specs/REDSTACK_TESTING_STRATEGY.md` §2–§3](../../docs/specs/REDSTACK_TESTING_STRATEGY.md), the properties this category is responsible for proving:

- **Ranking invariants** — for any generated set of scored candidates, `RankingEngine.rank()` always produces exactly 100 unique ranks, non-increasing scores by rank, and ties broken by ascending candidate ID (mirroring the six structural rules `domain.ranking.Ranking`'s constructor enforces).
- **Score monotonicity** — more corroborating evidence for a competency feature never lowers its fused value; the weighted-component sum always reconstructs `base_relevance` exactly.
- **Integrity idempotence** — re-running the integrity detectors on the same representation always yields the same verdict.
- **Feature range / no-NaN** — every feature cell's value stays within its documented bounds and is never `NaN`/`inf` across the generated input space.
- **Copy-on-write stage monotonicity** — `CandidateRepresentation`'s `BuildStage` marker only ever advances, never regresses, no matter what sequence of `with_*()` calls is generated.

This directory currently holds only its package marker (`__init__.py`); these are the properties to add tests for as the corresponding engine/domain logic is hardened. A found counterexample should be committed to the regression corpus, not discarded.

See [`tests/README.md`](../README.md) and [`docs/specs/REDSTACK_TESTING_STRATEGY.md` §3](../../docs/specs/REDSTACK_TESTING_STRATEGY.md).
