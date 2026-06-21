# `tests/` — Test Architecture

RedStack's test suite is the only feedback loop a ranking run gets before it's produced — there is no live, continuously-scored environment to catch a regression after the fact. The suite is therefore organized as a formal pyramid, weighted toward fast, pure, deterministic checks at the base, with a thin cap of expensive whole-pipeline tests. See [`/ARCHITECTURE.md` §11](../ARCHITECTURE.md#11-quality-gates) and [`docs/specs/REDSTACK_TESTING_STRATEGY.md`](../docs/specs/REDSTACK_TESTING_STRATEGY.md) for the exhaustive testing strategy this directory implements.

## Layout

| Directory | Purpose | README |
|---|---|---|
| [`unit/`](unit/README.md) | One domain model, feature transform, or engine method at a time — pure, fast, no IO. | [`unit/README.md`](unit/README.md) |
| [`property/`](property/README.md) | Hypothesis-driven invariant checks over generated input. | [`property/README.md`](property/README.md) |
| [`contract/`](contract/README.md) | One shared, parametrized suite per port, run against both the real adapter and its fake. | [`contract/README.md`](contract/README.md) |
| [`golden/`](golden/README.md) | Byte/value-exact snapshots that fail on any unblessed drift. | [`golden/README.md`](golden/README.md) |
| [`integration/`](integration/README.md) | The full offline build and online ranking run, end to end, against a sample candidate pool. | [`integration/README.md`](integration/README.md) |
| [`determinism/`](determinism/README.md) | Repeat-run, restart, and thread-count-invariance checks. | [`determinism/README.md`](determinism/README.md) |
| [`fixtures/`](fixtures/README.md) | The shared, versioned ground truth (fakes and candidate fixtures) every other category builds on. | [`fixtures/README.md`](fixtures/README.md) |
| [`conftest.py`](conftest.py) | Pins `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to `1` and the active config profile to `ci` before collection, so every test run is thread-count-deterministic regardless of host. | — |

## Test pyramid (target distribution, by test count)

| Layer | Objective | Target % |
|---|---|---|
| Unit | Prove each pure callable correct in isolation | 45% |
| Property | Prove invariants hold over generated input space | 15% |
| Contract | Prove every port implementation behaves identically to its fake | 10% |
| Golden | Lock exact outputs against regression | 10% |
| Integration | Prove stages compose end-to-end | 10% |
| Determinism | Prove identical inputs produce identical outputs | 5% |
| Performance / Reproduction | Prove compute-budget and reproduction guarantees hold | 5% |

## Mocking discipline

Engines and features are pure, so the base of the pyramid needs no mocks at all. Mocking happens **only** at the port boundary, and even there the suite prefers behavioral fakes (in-memory real implementations, in [`fixtures/`](fixtures/README.md)) over `unittest.mock` — a test exercises real contract behavior, not an assertion about call sequences.

## Running the suite

```bash
uv run pytest                                  # everything
uv run pytest -m "unit or property"            # fast suites only
uv run pytest -m "integration or determinism"  # full-pipeline suites
```

The marker set (`unit`, `property`, `contract`, `golden`, `integration`, `determinism`) is declared and enforced (`--strict-markers`) in `pyproject.toml`.
