# Reference Assets

This directory holds the external reference material RedStack's domain model, validation logic, and offline calibration are built against. These files are inputs to the system's design, not outputs of it — they are not generated or modified by any pipeline stage.

## Inventory

| File | What it is | Consumed by |
|---|---|---|
| `candidate_schema.json` | The canonical JSON Schema every raw candidate record must conform to. `src/redstack/domain/source.py` (`RawCandidate`) is a lossless, validated mirror of this schema. | `features/parsing.py`, offline stage O2 (Validation) |
| `sample_candidates.json` | A small set of example candidate records in the schema above, used as a format reference and as a seed for test fixtures. | test fixtures, local development |
| `sample_submission.csv` | A format reference for the ranking output — header, column order, and row shape. It makes **no claim about ranking quality**; it exists only to pin the structural format. | `tests/` (structural-congruence checks) |
| `validate_submission.py` | The external, authoritative validator for a finished `submission.csv`: exactly 100 rows, ranks 1–100 each used once, candidate IDs unique and pattern-matching, score non-increasing by rank, ties broken by ascending candidate ID. `src/redstack/engines/validation.py` (`ValidationEngine`) mirrors this logic rule-for-rule as an internal defense-in-depth check, and the integration test suite runs this script directly against produced output as an external oracle. | `engines/validation.py` (mirrored), `tests/integration/` (run directly) |
| `submission_metadata_template.yaml` | A template for the repository-root `submission_metadata.yaml` — team identity, the canonical reproduce command, compute declaration (CPU-only / no network / RAM and wall-clock limits), AI-tooling declaration, and a methodology summary. | filled in once per release at the repository root |
| `job_description.docx` | The source job description that `configs/anchors/jd_anchors.yaml` and `configs/gates/eligibility_rules.yaml` are authored from (offline stage O6). | offline stage O6 (JD Concept Extraction), human authors |
| `redrob_signals_doc.docx` | The specification of the 23 platform engagement signals (availability, responsiveness, engagement, reliability, verification) carried in each candidate's `redrob_signals` field. `src/redstack/features/signals.py` and `domain/candidate/behavioral.py` implement this spec exactly, including its sentinel-value (`-1`, `{}`) discipline. | `features/signals.py`, `domain/candidate/behavioral.py` |
| `README.docx` | The original reference documentation this guide directory's `README.md` supersedes for day-to-day use. | historical reference |

## How to use this directory

- Treat every file here as **read-only input**, never edit them in place — they are the contract RedStack is built to satisfy, not artifacts it produces.
- When a feature extractor, a domain model, or the validation engine needs to change, check here first to confirm the change doesn't drift from the schema, the signal specification, or the validator.
- See [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) for how these contracts map onto the domain model (§7), the validation engine (§6), and the behavioral feature pipeline (§6, `BehavioralEngine`).
