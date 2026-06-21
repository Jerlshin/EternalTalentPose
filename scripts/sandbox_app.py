

from __future__ import annotations

import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from redstack.config.determinism import apply_determinism
from redstack.config.loader import ConfigLoadError, load_config
from redstack.config.schema import RunMode
from redstack.domain.errors import DomainError
from redstack.pipelines.online import compose
from redstack.pipelines.online.pipeline import OnlinePipelineResult

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_ROOT = REPO_ROOT / "configs"
DEFAULT_SAMPLE_PATH = REPO_ROOT / "data" / "raw" / "sandbox_sample.jsonl"
FULL_POOL_RUN_REPORT = REPO_ROOT / "artifacts" / "run_report.json"

# Spec 10.5: "Accept a small candidate sample (<=100 candidates)". This is a
# hard cap on the demo's input, independent of the official top_k=100 cut
# used against the real 100k-candidate pool.
MAX_SANDBOX_CANDIDATES = 100
HONEYPOT_DISQUALIFY_RATE = 0.10
WALL_CLOCK_BUDGET_SECONDS = 300.0
RAM_BUDGET_GB = 16.0


# --------------------------------------------------------------------------- #
# Step 1 -- parse an arbitrary uploaded JSONL into candidate records.        #
# No record count or shape is assumed; malformed lines are reported, never   #
# raised, mirroring JsonlCandidateSourceAdapter's SourceMalformed semantics. #
# --------------------------------------------------------------------------- #
def parse_candidate_jsonl(raw_bytes: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """Decode JSONL bytes into candidate dicts; never raises on bad input."""
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    seen_ids: set[str] = set()
    text = raw_bytes.decode("utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(obj, dict):
            problems.append(f"line {line_no}: not a JSON object")
            continue
        cid = obj.get("candidate_id")
        if not isinstance(cid, str) or not cid:
            problems.append(f"line {line_no}: missing/blank candidate_id")
            continue
        if cid in seen_ids:
            problems.append(f"line {line_no}: duplicate candidate_id {cid!r}, skipped")
            continue
        seen_ids.add(cid)
        records.append(obj)
    return records, problems


def _safe_get(obj: object, *path: str, default: str = "—") -> str:
    """Defensive nested dict lookup over an *untrusted* uploaded record."""
    node: object = obj
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    if node is None or node == "":
        return default
    return str(node)


# --------------------------------------------------------------------------- #
# Step 2 -- run the real production pipeline end-to-end on the sample.      #
# --------------------------------------------------------------------------- #
def run_sandbox_ranking(
    records: list[dict[str, Any]],
) -> tuple[OnlinePipelineResult, list[dict[str, str]], dict[str, Any]]:
    """Rank ``records`` via the real composition root; return result + CSV + report.

    Raises:
        ConfigLoadError: the configs root failed to resolve/validate.
        DomainError: any real pipeline-stage failure (malformed input under
            the abort policy, an artifact-contract breach, a validation
            rejection, ...). Never caught here -- the caller renders it.
    """
    effective_top_k = min(MAX_SANDBOX_CANDIDATES, len(records))

    resolved = load_config(CONFIGS_ROOT, RunMode.ONLINE, None)
    apply_determinism(resolved.determinism)
    online = resolved.online
    if online is None:
        raise ConfigLoadError("loaded config has no 'online' runtime block")

    # The only override: rank exactly what this sample has, capped at 100.
    # The committed configs/runtime/online.yaml (top_k=100) is untouched --
    # the official `make rank` path against the real 100k pool is unaffected.
    resolved = resolved.model_copy(
        update={"online": online.model_copy(update={"top_k": effective_top_k})}
    )

    with tempfile.TemporaryDirectory(prefix="redstack_sandbox_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "sandbox_input.jsonl"
        with input_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        output_path = tmp_dir / "sandbox_submission.csv"
        report_path = tmp_dir / "sandbox_run_report.json"

        result = compose.run_online_rank(
            resolved,
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            participant_id="sandbox",
        )

        with output_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        report = json.loads(report_path.read_text(encoding="utf-8"))

    return result, csv_rows, report


# --------------------------------------------------------------------------- #
# Step 3 -- the executive triage dashboard, sourced only from the above.    #
# --------------------------------------------------------------------------- #
def render_dashboard(
    result: OnlinePipelineResult,
    csv_rows: list[dict[str, str]],
    report: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
) -> None:
    reproducible = report.get("reproducible", {})
    budget = report.get("budget", {})

    st.subheader("Compute-constraint compliance (this sandbox run)")
    cols = st.columns(4)
    cols[0].metric(
        "Wall-clock used",
        f"{budget.get('used_seconds', 0.0):.2f}s",
        help=f"Official budget ceiling: {WALL_CLOCK_BUDGET_SECONDS:.0f}s "
        "(measured on the full 100k-candidate pool at Stage 3, not this sample).",
    )
    cols[1].metric(
        "Peak RSS (this process)",
        f"{result.peak_rss_mb:.1f} MB",
        help=f"Official ceiling: {RAM_BUDGET_GB:.0f} GB. Reads "
        "`resource.ru_maxrss`, reported in KiB on Linux (every hosting target "
        "spec 10.5 lists -- HF Spaces / Streamlit Cloud / Replit / Colab / "
        "Binder/Docker -- runs Linux) but in bytes on macOS, so this figure "
        "reads ~1024x too large on a local Mac dev machine; trust it only "
        "when this app is actually hosted on one of those platforms.",
    )
    cols[2].metric("Rows ranked", str(result.row_count))
    honeypot_rate = result.honeypot_rate_top100
    honeypot_ok = honeypot_rate <= HONEYPOT_DISQUALIFY_RATE
    cols[3].metric(
        "Honeypot rate",
        f"{honeypot_rate:.1%}",
        delta="OK" if honeypot_ok else "OVER 10% GATE",
        delta_color="normal" if honeypot_ok else "inverse",
    )
    st.caption(
        "CPU-only / network-off are architecture guarantees, not just runtime "
        "measurements: `redstack.pipelines.online` and `redstack.engines` are "
        "barred at the import-linter level (`make imports`) from "
        "sentence-transformers, scikit-learn, torch, sockets, requests, and "
        "httpx (CLAUDE.md Online Containment Rule) -- the ranking step "
        "physically cannot reach the network or a GPU runtime."
    )

    if FULL_POOL_RUN_REPORT.is_file():
        full_report = json.loads(FULL_POOL_RUN_REPORT.read_text(encoding="utf-8"))
        full_budget = full_report.get("budget", {})
        full_repro = full_report.get("reproducible", {})
        with st.expander(
            "Last official full-pool reproduction (`make rank`, the real Stage-3 "
            "subject) -- for comparison, not part of this sandbox run"
        ):
            st.write(
                {
                    "candidate_count": full_repro.get("candidate_count"),
                    "used_seconds": full_budget.get("used_seconds"),
                    "within_budget": full_budget.get("within_budget"),
                    "honeypot_rate_top100": full_repro.get("honeypot_rate"),
                }
            )

    st.subheader("Hard-gate yield rates (full uploaded sample)")
    eligibility_summary: dict[str, int] = reproducible.get("eligibility_summary", {})
    sample_size = reproducible.get("candidate_count", len(records_by_id)) or 1
    if eligibility_summary:
        st.dataframe(
            [
                {
                    "gate": code,
                    "fired": count,
                    "% of sample": f"{100 * count / sample_size:.1f}%",
                }
                for code, count in sorted(
                    eligibility_summary.items(), key=lambda kv: -kv[1]
                )
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No hard/soft eligibility gates fired on this sample.")

    st.subheader("Top-cohort YOE")
    top_cut = max(1, round(0.10 * len(csv_rows)))
    top_rows = [r for r in csv_rows if int(r["rank"]) <= top_cut]
    yoe_values = [
        records_by_id[r["candidate_id"]].get("profile", {}).get("years_of_experience")
        for r in top_rows
        if r["candidate_id"] in records_by_id
        and isinstance(records_by_id[r["candidate_id"]].get("profile"), dict)
        and isinstance(
            records_by_id[r["candidate_id"]]["profile"].get("years_of_experience"),
            int | float,
        )
    ]
    avg_yoe = sum(yoe_values) / len(yoe_values) if yoe_values else 0.0
    st.metric(f"Average YOE, top {top_cut} of {len(csv_rows)}", f"{avg_yoe:.2f} years")

    st.subheader("Triage table")
    table_rows = []
    for row in csv_rows:
        record = records_by_id.get(row["candidate_id"], {})
        table_rows.append(
            {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "candidate_id": row["candidate_id"],
                "title @ company": (
                    f"{_safe_get(record, 'profile', 'current_title')} @ "
                    f"{_safe_get(record, 'profile', 'current_company')}"
                ),
                "YOE": _safe_get(record, "profile", "years_of_experience"),
                "location": _safe_get(record, "profile", "location"),
                "reasoning": row.get("reasoning", ""),
            }
        )
    st.dataframe(table_rows, hide_index=True, width="stretch")

    st.download_button(
        "Download ranked CSV",
        data="\n".join(
            [",".join(csv_rows[0].keys())]
            + [",".join(f'"{v}"' for v in r.values()) for r in csv_rows]
        )
        if csv_rows
        else "",
        file_name="sandbox_submission.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# App body.                                                                  #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="REDSTACK Sandbox", layout="wide")
st.title("REDSTACK — Hosted Sandbox & Triage Dashboard")
st.caption(
    "Submission spec Section 10.5: a hosted sample-reproducibility check, not "
    "the full Stage-3 100k-pool reproduction. Accepts up to "
    f"{MAX_SANDBOX_CANDIDATES} candidates."
)

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Candidate sample (.jsonl)", type=["jsonl", "ndjson", "txt"]
    )
    sample_n = st.slider(
        "...or use the bundled sample, first N candidates",
        min_value=1,
        max_value=MAX_SANDBOX_CANDIDATES,
        value=MAX_SANDBOX_CANDIDATES,
        disabled=uploaded is not None,
        help="Drag this below 100 to exercise the spec 10.5 'fewer than 100 "
        "profiles' path explicitly.",
    )
    run_clicked = st.button("Run ranking", type="primary")

if run_clicked:
    if uploaded is not None:
        records, problems = parse_candidate_jsonl(uploaded.getvalue())
    elif DEFAULT_SAMPLE_PATH.is_file():
        records, problems = parse_candidate_jsonl(DEFAULT_SAMPLE_PATH.read_bytes())
        records = records[:sample_n]
    else:
        records, problems = [], [f"bundled sample missing at {DEFAULT_SAMPLE_PATH}"]

    if problems:
        with st.expander(f"{len(problems)} line(s) skipped while parsing input"):
            for problem in problems:
                st.write(f"- {problem}")

    if len(records) > MAX_SANDBOX_CANDIDATES:
        st.warning(
            f"Input has {len(records)} candidates; spec 10.5 caps the sandbox "
            f"at {MAX_SANDBOX_CANDIDATES} -- using the first "
            f"{MAX_SANDBOX_CANDIDATES}."
        )
        records = records[:MAX_SANDBOX_CANDIDATES]

    if not records:
        st.error("No valid candidate records found in the input -- nothing to rank.")
        st.stop()

    records_by_id = {r["candidate_id"]: r for r in records}
    st.info(f"Ranking {len(records)} candidate(s) (top_k = min(100, n))...")

    started = time.perf_counter()
    try:
        result, csv_rows, report = run_sandbox_ranking(records)
    except (DomainError, ConfigLoadError) as exc:
        st.error(f"Ranking failed: {type(exc).__name__}: {exc}")
        st.stop()
    wall_elapsed = time.perf_counter() - started

    st.success(f"Ranking complete in {wall_elapsed:.2f}s -- {result.row_count} row(s).")
    render_dashboard(result, csv_rows, report, records_by_id)
else:
    st.info("Upload a sample or use the slider, then click **Run ranking**.")
