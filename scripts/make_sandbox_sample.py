from __future__ import annotations

import json
from pathlib import Path

# Path definitions anchoring our raw and golden targets
RAW_DATA_PATH = Path("data/raw/candidates.jsonl")
OUTPUT_SAMPLE_PATH = Path("data/raw/sandbox_sample.jsonl")


def build_sandbox_sample() -> None:
    """Extracts a clean, deterministic slice of records for local runs."""
    if not RAW_DATA_PATH.exists():
        print(f"Error: Raw fact dataset missing at {RAW_DATA_PATH}. Aborting script.")
        return

    print(f"Reading raw pool from {RAW_DATA_PATH}...")
    sample_records: list[dict] = []

    # Sandbox sample must stay <=100 candidates (submission spec Sec 10.5).
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if len(sample_records) < 100:
                    sample_records.append(record)
                else:
                    break
            except json.JSONDecodeError:
                continue

    OUTPUT_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_SAMPLE_PATH, "w", encoding="utf-8") as outfile:
        for record in sample_records:
            outfile.write(json.dumps(record) + "\n")

    print(f"Successfully compiled {len(sample_records)} records into {OUTPUT_SAMPLE_PATH}!")


if __name__ == "__main__":
    build_sandbox_sample()