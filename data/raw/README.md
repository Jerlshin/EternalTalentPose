# `data/raw/` — The Candidate Pool

| Path | Required for | Format |
|---|---|---|
| `candidates.jsonl` | `redstack build` and `redstack rank` | One JSON object per line, conforming to [`docs/guide/candidate_schema.json`](../../docs/guide/candidate_schema.json). Gzip (`.jsonl.gz`) is also accepted — `CandidateSourcePort` is gzip-transparent. |
| `sandbox_sample.jsonl` *(generated, optional)* | local development only | A small (first 500 records), deterministic slice produced by [`scripts/make_sandbox_sample.py`](../../scripts/README.md) for fast local iteration without the full pool. |

## Ingestion guarantees

The file is streamed, never fully materialized in memory — `JsonlCandidateSourceAdapter` yields one record at a time, in file order, preserving a stable `source_index` per record. A line that isn't valid JSON is reported as a tagged record rather than raising; whether that aborts the run or is skipped-and-logged is a configuration policy (`configs/runtime/online.yaml:malformed_record_policy`), not a property of this file.

See [`/ARCHITECTURE.md` §5.2](../../ARCHITECTURE.md#52-online-pipeline--r0-through-r9) (stage R1) and [`src/redstack/adapters/README.md`](../../src/redstack/adapters/README.md) (`candidate_jsonl.py`).
