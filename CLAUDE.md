# CLAUDE.md — System Steering Constraints for RedStack v1.1

## 1. Core Architecture & Boundary Rules
- **Hexagonal Isolation:** Maintain strict separation between layers (`domain` ──▶ `ports` ◀── `engines`/`features` ◀── `adapters`/`pipelines`). 
- **Online Containment Rule:** Modules under `src/redstack/pipelines/online/` and `src/redstack/engines/` must never import heavy training runtimes, text-vectorizers, or network frameworks (`sentence-transformers`, `scikit-learn`, `adapters.st_embedder`, `socket`, `requests`, `httpx`). A violation is treated as a critical build break.
- **Memory Optimization Boundary:** The online runtime pipeline (`R0…R9`) must process rows using constant $O(1)$ streaming memory scale-outs over `candidates.jsonl.gz` to honor the $\le 16$ GB RAM ceiling. Candidate profile re-hydration can occur *only* during step `R7` for the 100 survivors.
- **Hidden Side Effects:** No hidden file system side effects, network access, or volatile system clock dependencies (`datetime.now()`) inside `domain/` or `engines/`. Use injected seeds and the `ctx.as_of` context seam.

## 2. Code Style & Implementation Invariants
- **Data Primitive Mapping:** All judgment and structural enum values are strictly **lowercase snake_case** strings (e.g., `pure_research_no_production`, `expert_skill_zero_usage`, `tier_1`) to match the competition dataset schema. Never serialize or validate via UPPERCASE names.
- **Strict Determinism:** All operations must be fully reproducible. Thread allocations are restricted (`config/determinism.py`), component aggregation reductions follow a fixed `ScoreComponent` sequence loop, and ties inside `R6` (Ranking) are broken totally and deterministically by ordering values by `(-score, candidate_id)` in ascending order.
- **Anti-Hallucination Guarantees:** Every `ReasoningClause` inside the explanation generator must be bound to an explicit fact. Selectors use bracket-index paths (e.g., `career_history[0].end_date`) validated through `features/evidence.py`. If a path dangles, `mint()` must immediately raise a hard `ProvenanceError`.
- **Value Object Mutability:** `CandidateRepresentation` and `Ranking` objects are pure, immutable aggregate roots. Modify properties exclusively through clean copy-on-write constructor builders (e.g., `.with_features()`, `.with_score()`, `.with_reasoning()`).

## 3. Build, Test, and Quality Gate Commands
Always execute commands using the `uv` toolchain runner or your managed virtual environment to ensure dependency encapsulation:

### Code Formatting & Linting Rules
- **Run Style Audits:** `uv run ruff check src/`
- **Run Style Auto-Fixer:** `uv run ruff check src/ --fix`
- **Check Layout Code Formatting:** `uv run ruff format src/ --check`
- **Execute Layout Formatting:** `uv run ruff format src/`

### Static Analysis & Type Checking
- **Enforce Strict Static Typing:** `uv run mypy --strict src/`
- **Expose Complete Typing Reports:** `uv run mypy --strict src/ --report-html typecover/`

### Verification & Testing Suites
- **Run Complete Test Topology:** `uv run pytest`
- **Execute Isolated Testing Branches:**
  - *Unit Infrastructure tests:* `uv run pytest tests/unit/`
  - *Integration Component flows:* `uv run pytest tests/integration/`
  - *Determinism Verification suites:* `uv run pytest tests/determinism/`

### End-to-End Execution Lifecycles
- **Execute Offline Compilation Pipeline:** `uv run python -m redstack.cli.app build --config configs/base.yaml`
- **Execute Online Real-Time Inference Pass:** `uv run python -m redstack.cli.app rank --input data/raw/candidates.jsonl --output artifacts/submission.csv`