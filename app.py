import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RedStack Talent Ranker",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# Theming — dark glass-panel aesthetic, ported from artifacts/debug_dashboard.html
# ==============================================================================
CUSTOM_CSS = """
<style>
:root{
  --rs-bg-panel:#151b23; --rs-bg-elevated:#1c2530; --rs-bg-hover:#212b38;
  --rs-border:#2a3441; --rs-text:#e6edf3; --rs-text-dim:#8b98a9; --rs-text-faint:#5b6878;
  --rs-accent:#34d399; --rs-accent-dim:#1f5f48; --rs-amber:#fbbf24;
  --rs-crimson:#f87171; --rs-blue:#60a5fa;
  --rs-glass-bg:rgba(21,27,35,.68); --rs-glass-border:rgba(255,255,255,.06);
  --rs-shadow:0 1px 2px rgba(0,0,0,.4), 0 12px 28px -8px rgba(0,0,0,.5);
  --rs-chip-clear-bg:#0f2e22; --rs-chip-clear-fg:#5eead4;
  --rs-chip-amber-bg:#3a2c0d; --rs-chip-amber-fg:#fcd34d;
  --rs-chip-crimson-bg:#3a1414; --rs-chip-crimson-fg:#fca5a5;
  --rs-chip-blue-bg:#132743; --rs-chip-blue-fg:#93c5fd;
}
html, body, [class*="css"]{
  font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.stApp{
  background:
    radial-gradient(1100px 560px at 8% -8%, rgba(52,211,153,.07), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(96,165,250,.06), transparent 60%),
    #0d1117;
}
code, .rs-mono{font-family:"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;}

/* ---------- top bar ---------- */
.rs-topbar{
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;
  padding:18px 22px; background:var(--rs-glass-bg); backdrop-filter:blur(16px) saturate(180%);
  border:1px solid var(--rs-glass-border); border-radius:14px; box-shadow:var(--rs-shadow);
  margin-bottom:20px;
}
.rs-topbar h1{margin:0; font-size:22px; font-weight:800; letter-spacing:-.01em; color:var(--rs-text);}
.rs-topbar h1 span{color:var(--rs-text-dim); font-weight:500; margin-left:10px; font-size:15px;}
.rs-subtitle{margin:5px 0 0; font-size:12.5px; color:var(--rs-text-faint);}
.rs-subtitle code{color:var(--rs-text-dim);}

/* ---------- KPI cards ---------- */
.rs-kpi-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:6px 0 22px;}
.rs-kpi-card{
  background:var(--rs-glass-bg); backdrop-filter:blur(14px) saturate(160%);
  border:1px solid var(--rs-glass-border); border-left:4px solid var(--rs-accent-card,var(--rs-blue));
  border-radius:10px; padding:14px 16px; box-shadow:var(--rs-shadow); transition:transform .15s ease;
}
.rs-kpi-card:hover{transform:translateY(-2px);}
.rs-kpi-card .rs-kpi-label{font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--rs-text-faint); font-weight:700;}
.rs-kpi-card .rs-kpi-value{font-size:26px; font-weight:800; margin:6px 0 2px; letter-spacing:-.02em; color:var(--rs-text);}
.rs-kpi-card .rs-kpi-value small{font-size:13px; color:var(--rs-text-dim); font-weight:500;}
.rs-kpi-card .rs-kpi-sub{font-size:11.5px; color:var(--rs-text-dim);}
.rs-kpi-card.status-good{--rs-accent-card:var(--rs-accent);}
.rs-kpi-card.status-warn{--rs-accent-card:var(--rs-amber);}
.rs-kpi-card.status-bad{--rs-accent-card:var(--rs-crimson);}
.rs-kpi-card.status-info{--rs-accent-card:var(--rs-blue);}

/* ---------- section heading inside panels ---------- */
.rs-section-title{font-size:15px; font-weight:700; color:var(--rs-text); margin:0 0 4px;}
.rs-section-hint{font-size:11.5px; color:var(--rs-text-faint); font-weight:500; margin-left:6px;}

/* ---------- generic bar rows (gate chart + stage timeline) ---------- */
.rs-bar-row{display:flex; align-items:center; gap:12px; padding:7px 0;}
.rs-bar-label{width:250px; flex:0 0 250px; font-size:12.5px; color:var(--rs-text-dim);}
.rs-bar-label b{color:var(--rs-text); font-weight:700;}
.rs-bar-label .rs-bar-sub{display:block; font-size:11px; color:var(--rs-text-faint);}
.rs-bar-track{flex:1; background:var(--rs-bg-elevated); border-radius:5px; height:16px; overflow:hidden; position:relative;}
.rs-bar-fill{height:100%; border-radius:5px; transition:width .4s ease;}
.rs-bar-value{width:150px; flex:0 0 150px; text-align:right; font-size:11.5px; color:var(--rs-text-dim); font-variant-numeric:tabular-nums;}
.rs-stage-idle{display:flex; align-items:center; gap:10px; padding:6px 0; font-size:12.5px; color:var(--rs-text-dim);}
.rs-stage-idle .rs-dot{width:7px; height:7px; border-radius:50%; background:var(--rs-text-faint); flex:0 0 7px;}
.rs-stage-idle b{color:var(--rs-text); font-family:"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;}

/* ---------- badges/chips ---------- */
.rs-badge{
  font-size:10.5px; border-radius:999px; padding:3px 10px; font-weight:700; white-space:nowrap;
  letter-spacing:.01em; display:inline-flex; align-items:center; gap:4px;
}
.rs-badge-clear{background:var(--rs-chip-clear-bg); color:var(--rs-chip-clear-fg);}
.rs-badge-amber{background:var(--rs-chip-amber-bg); color:var(--rs-chip-amber-fg);}
.rs-badge-crimson{background:var(--rs-chip-crimson-bg); color:var(--rs-chip-crimson-fg);}
.rs-badge-blue{background:var(--rs-chip-blue-bg); color:var(--rs-chip-blue-fg);}

/* ---------- artifact rows ---------- */
.rs-artifact-row{
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:10px 4px; border-bottom:1px solid var(--rs-border); font-size:13px;
}
.rs-artifact-row:last-child{border-bottom:none;}
.rs-artifact-name{font-weight:600; color:var(--rs-text);}
.rs-artifact-meta{font-size:11.5px; color:var(--rs-text-faint);}

/* ---------- reasoning blockquote ---------- */
.rs-reasoning{
  margin:0; padding:14px 18px; background:var(--rs-bg-panel); border-left:3px solid var(--rs-accent);
  border-radius:8px; font-style:italic; color:var(--rs-text); font-size:13.5px; line-height:1.65;
}

/* ---------- Streamlit widget re-skin ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--rs-glass-bg) !important; border:1px solid var(--rs-glass-border) !important;
  border-radius:12px !important; box-shadow:var(--rs-shadow);
}
div[data-testid="stExpander"]{
  background:var(--rs-glass-bg); border:1px solid var(--rs-glass-border) !important; border-radius:10px;
}
div[data-testid="stTabs"] button[role="tab"]{font-weight:600;}
div[data-testid="stStatusWidget"]{
  border-radius:10px; border:1px solid var(--rs-glass-border) !important;
}
div[data-testid="stFileUploaderDropzone"]{
  background:var(--rs-bg-elevated); border-radius:10px;
}
button[data-testid="stBaseButton-primary"]{
  border-radius:8px; font-weight:700; letter-spacing:.01em;
}
button[data-testid="stBaseButton-secondary"]{
  border-radius:8px; font-weight:600;
}
div[data-testid="stDataFrame"]{
  border-radius:10px; overflow:hidden; border:1px solid var(--rs-border);
}
.rs-footer{text-align:center; color:var(--rs-text-faint); font-size:11.5px; padding:22px 0 8px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# Fixed paths & pipeline invocation constants — UNCHANGED from the original app
# ==============================================================================
REPO_ROOT = Path.cwd()

DATA_DIR = REPO_ROOT / "data" / "raw"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "candidates.jsonl"
OUTPUT_FILE = ARTIFACT_DIR / "submission.csv"
RUN_REPORT = ARTIFACT_DIR / "run_report.json"

# Descriptive labels only — sourced from src/redstack/pipelines/online/stages.py
# and pipeline.py comments. Purely for display; no bearing on execution.
STAGE_INFO: list[tuple[str, str, str]] = [
    ("R0", "Artifact Loading", "Load policy/anchor/archetype artifacts; bind ports"),
    ("R1", "Candidate Ingestion", "Stream + validate candidates.jsonl"),
    ("R2", "Feature Extraction", "Bulk structural feature extraction (pure)"),
    ("R3", "Semantic Hydration", "Vector-store lookup + ONNX fallback"),
    ("R4", "Gates & Eligibility", "Integrity + eligibility hard gates"),
    ("R5", "Scoring", "Locked weights, gates, bounded multipliers"),
    ("R6", "Ranking", "Deterministic floor-partitioned ranking"),
    ("R7", "Reasoning", "Evidence-bound reasoning for the top-100"),
    ("R8", "Submission Generation", "Validate + atomically write submission.csv"),
    ("R9", "Run Report", "Reproducibility, audit & timing report"),
]

ARTIFACT_CHECKLIST: list[tuple[str, Path, str]] = [
    ("submission.csv", OUTPUT_FILE, "Final ranked candidate submission"),
    ("run_report.json", RUN_REPORT, "Reproducibility, budget & gate audit"),
]


# ==============================================================================
# Presentation helpers — read-only; never touch pipeline behavior or outputs
# ==============================================================================
def _esc(value: Any) -> str:
    return html.escape(str(value))


def _get(d: dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes, sec = divmod(seconds, 60)
    if minutes:
        return f"{int(minutes)}m {sec:04.1f}s"
    return f"{sec:.1f}s"


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def render_kpi_grid(cards: list[dict[str, str]]) -> None:
    parts = []
    for c in cards:
        parts.append(
            f'<div class="rs-kpi-card status-{c["status"]}">'
            f'<div class="rs-kpi-label">{_esc(c["label"])}</div>'
            f'<div class="rs-kpi-value">{c["value"]}</div>'
            f'<div class="rs-kpi-sub">{_esc(c["sub"])}</div></div>'
        )
    st.markdown(f'<div class="rs-kpi-grid">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_bar_rows(rows: list[dict[str, Any]]) -> None:
    """Render a list of {label, sub, value_label, pct, color} as div-bar rows."""
    parts = []
    for r in rows:
        parts.append(
            '<div class="rs-bar-row">'
            f'<div class="rs-bar-label"><b>{_esc(r["label"])}</b>'
            f'<span class="rs-bar-sub">{_esc(r.get("sub", ""))}</span></div>'
            '<div class="rs-bar-track">'
            f'<div class="rs-bar-fill" style="width:{r["pct"]:.2f}%;background:{r["color"]};"></div>'
            "</div>"
            f'<div class="rs-bar-value">{_esc(r["value_label"])}</div>'
            "</div>"
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def status_badge(good: bool, good_text: str, bad_text: str) -> str:
    cls = "rs-badge-clear" if good else "rs-badge-crimson"
    text = good_text if good else bad_text
    return f'<span class="rs-badge {cls}">{_esc(text)}</span>'


# ==============================================================================
# Session state — lets search/sort/filter widgets rerun without losing results
# ==============================================================================
if "run_result" not in st.session_state:
    st.session_state.run_result = None

if st.session_state.run_result is None and OUTPUT_FILE.exists():
    # Artifacts already on disk from a prior run (e.g. server restart) —
    # surface them read-only so the dashboard never looks empty for a demo.
    st.session_state.run_result = {
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "elapsed": None,
        "command": None,
        "timestamp": OUTPUT_FILE.stat().st_mtime,
        "hydrated": True,
    }

run_report: dict[str, Any] | None = None
if RUN_REPORT.exists():
    try:
        run_report = json.loads(RUN_REPORT.read_text())
    except (json.JSONDecodeError, OSError):
        run_report = None

# ==============================================================================
# Header
# ==============================================================================
run_meta_line = "No pipeline run yet — upload a candidates file to get started."
if run_report is not None:
    run_id = _get(run_report, "audit", "run_id", default="—")
    code_version = _get(run_report, "reproducible", "code_version", default="—")
    manifest_hash = _get(run_report, "reproducible", "manifest_hash", default="—")
    reproducible_ok = bool(_get(run_report, "budget", "within_budget", default=False))
    manifest_short = manifest_hash[:12] if isinstance(manifest_hash, str) else manifest_hash
    run_meta_line = (
        f"Run <code>{_esc(run_id)}</code> · {_esc(code_version)} · "
        f"manifest <code>{_esc(manifest_short)}</code> · "
        f"budget honored: {'yes' if reproducible_ok else 'no'}"
    )

st.markdown(
    f"""
    <div class="rs-topbar">
      <div>
        <h1>🚀 RedStack <span>Evidence-Grounded Semantic Candidate Ranking</span></h1>
        <p class="rs-subtitle">{run_meta_line}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Upload & run panel
# ==============================================================================
with st.container(border=True):
    st.markdown('<p class="rs-section-title">📤 Upload &amp; Run</p>', unsafe_allow_html=True)
    st.caption("Upload a `candidates.jsonl` file and run the deterministic RedStack ranking pipeline.")

    uploaded_file = st.file_uploader("Upload candidates.jsonl", type=["jsonl"], label_visibility="collapsed")

    if uploaded_file is not None:

        INPUT_FILE.write_bytes(uploaded_file.getbuffer())

        st.success("Candidate file uploaded successfully.")
        st.caption(f"Input file: `{INPUT_FILE}`")

        run_clicked = st.button("▶ Run RedStack Ranking", type="primary", use_container_width=False)

        if run_clicked:

            with st.status("Executing online rank pipeline (R0 → R9)…", expanded=True) as status_box:

                env = os.environ.copy()

                src_dir = str(REPO_ROOT / "src")
                existing_pythonpath = env.get("PYTHONPATH", "")

                env["PYTHONPATH"] = (
                    f"{src_dir}{os.pathsep}{existing_pythonpath}"
                    if existing_pythonpath
                    else src_dir
                )

                env["OMP_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                env["OPENBLAS_NUM_THREADS"] = "1"
                env["VECLIB_MAXIMUM_THREADS"] = "1"
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONDONTWRITEBYTECODE"] = "1"

                command = [
                    sys.executable,
                    "-m",
                    "redstack.cli.app",
                    "rank",
                    "--input",
                    str(INPUT_FILE),
                    "--output",
                    str(OUTPUT_FILE),
                ]

                st.write(f"`{' '.join(command)}`")

                run_started = time.perf_counter()

                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                )

                elapsed = time.perf_counter() - run_started

                success = result.returncode == 0 and OUTPUT_FILE.exists()
                status_box.update(
                    label=(
                        f"Pipeline completed in {_fmt_duration(elapsed)}"
                        if success
                        else f"Pipeline failed (exit code {result.returncode})"
                    ),
                    state="complete" if success else "error",
                    expanded=not success,
                )

            st.session_state.run_result = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed": elapsed,
                "command": command,
                "timestamp": time.time(),
                "hydrated": False,
            }

# ==============================================================================
# Results
# ==============================================================================
result = st.session_state.run_result

if result is None:
    st.info("Results, execution statistics, and the candidate table will appear here after a run.")
else:
    success = result["returncode"] == 0 and OUTPUT_FILE.exists()

    if success:
        if result.get("hydrated"):
            st.info(
                "📦 Showing artifacts already present in `artifacts/` from a previous run "
                f"(last modified {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))})."
            )
        else:
            st.success("Ranking completed successfully.")

        # ---------------- Execution summary ----------------
        with st.container(border=True):
            st.markdown('<p class="rs-section-title">📊 Execution Summary</p>', unsafe_allow_html=True)
            cols = st.columns(4)
            cols[0].metric("Status", "Success ✅")
            cols[1].metric("Duration", _fmt_duration(result.get("elapsed")))
            cols[2].metric("Return Code", str(result["returncode"]))
            budget_ok = _get(run_report, "budget", "within_budget", default=None)
            cols[3].metric(
                "Budget",
                "Within limit" if budget_ok else ("Exceeded" if budget_ok is False else "—"),
            )

        # ---------------- KPI grid ----------------
        candidate_count = _get(run_report, "reproducible", "candidate_count", default=None)
        used_seconds = _get(run_report, "budget", "used_seconds", default=None)
        limit_seconds = _get(run_report, "budget", "limit_seconds", default=None)
        peak_rss = _get(run_report, "budget", "peak_rss_mb", default=None)
        honeypots = _get(run_report, "reproducible", "honeypot_count_top100", default=None)
        code_version = _get(run_report, "reproducible", "code_version", default="—")
        run_id = _get(run_report, "audit", "run_id", default="—")

        kpi_cards = [
            {
                "label": "Candidates Scored",
                "value": f"{candidate_count:,}" if candidate_count is not None else "—",
                "sub": "full pool processed end-to-end",
                "status": "info",
            },
            {
                "label": "Pipeline Runtime",
                "value": f"{used_seconds:.1f}s" if used_seconds is not None else "—",
                "sub": f"limit {limit_seconds:.0f}s" if limit_seconds is not None else "budget limit n/a",
                "status": "good" if budget_ok else ("bad" if budget_ok is False else "info"),
            },
            {
                "label": "Peak RSS",
                "value": f"{peak_rss:.0f} MB" if peak_rss is not None else "—",
                "sub": "≤ 16,384 MB streaming ceiling",
                "status": "good" if (peak_rss is not None and peak_rss <= 16384) else "info",
            },
            {
                "label": "Honeypots (Top-100)",
                "value": f"{honeypots} / 100" if honeypots is not None else "—",
                "sub": "verified via IntegrityEngine",
                "status": "good" if honeypots == 0 else ("bad" if honeypots else "info"),
            },
            {
                "label": "Code Version",
                "value": f"<small>{_esc(code_version)}</small>",
                "sub": "resolved run configuration",
                "status": "info",
            },
            {
                "label": "Run ID",
                "value": f"<small>{_esc(run_id)}</small>",
                "sub": "audit trail identifier",
                "status": "info",
            },
        ]
        render_kpi_grid(kpi_cards)

        # ---------------- Pipeline stage visualization ----------------
        with st.container(border=True):
            st.markdown(
                '<p class="rs-section-title">🧬 Pipeline Stages <span class="rs-section-hint">'
                "R0 → R9 · online runtime</span></p>",
                unsafe_allow_html=True,
            )
            timings_ms = _get(run_report, "timings", default=None)
            if timings_ms:
                max_ms = max(timings_ms.get(code, 0.0) for code, _, _ in STAGE_INFO) or 1.0
                rows = []
                for code, name, desc in STAGE_INFO:
                    ms = timings_ms.get(code)
                    if ms is None:
                        continue
                    pct_of_max = (ms / max_ms) * 100
                    seconds = ms / 1000.0
                    color = (
                        "var(--rs-crimson)"
                        if pct_of_max > 60
                        else "var(--rs-amber)"
                        if pct_of_max > 30
                        else "var(--rs-blue)"
                    )
                    rows.append(
                        {
                            "label": f"{code} · {name}",
                            "sub": desc,
                            "pct": pct_of_max,
                            "color": color,
                            "value_label": f"{seconds:.3f}s",
                        }
                    )
                render_bar_rows(rows)
            else:
                for code, name, desc in STAGE_INFO:
                    st.markdown(
                        f'<div class="rs-stage-idle"><span class="rs-dot"></span>'
                        f"<b>{code}</b>&nbsp;{_esc(name)} — {_esc(desc)}</div>",
                        unsafe_allow_html=True,
                    )

        # ---------------- Hard-gate fired rates ----------------
        eligibility_summary = _get(run_report, "reproducible", "eligibility_summary", default=None)
        if eligibility_summary and candidate_count:
            with st.container(border=True):
                st.markdown(
                    '<p class="rs-section-title">🚧 Hard-Gate Fired Rates <span class="rs-section-hint">'
                    f"full pool · N = {candidate_count:,} · source: run_report.json</span></p>",
                    unsafe_allow_html=True,
                )
                gate_rows = sorted(
                    eligibility_summary.items(), key=lambda kv: kv[1], reverse=True
                )
                max_pct = max((fired / candidate_count) * 100 for _, fired in gate_rows) or 1.0
                rows = []
                for gate_code, fired in gate_rows:
                    pct = (fired / candidate_count) * 100
                    color = (
                        "var(--rs-crimson)"
                        if pct > 50
                        else "var(--rs-amber)"
                        if pct > 20
                        else "var(--rs-blue)"
                    )
                    rows.append(
                        {
                            "label": gate_code,
                            "sub": "",
                            "pct": (pct / max_pct) * 100,
                            "color": color,
                            "value_label": f"{fired:,} · {pct:.2f}%",
                        }
                    )
                render_bar_rows(rows)

        # ---------------- Candidate exploration ----------------
        df = pd.read_csv(OUTPUT_FILE)

        with st.container(border=True):
            st.markdown(
                '<p class="rs-section-title">🏆 Candidate Exploration <span class="rs-section-hint">'
                "search · filter · sort</span></p>",
                unsafe_allow_html=True,
            )

            ctrl_cols = st.columns([2, 1, 1])
            search_term = ctrl_cols[0].text_input(
                "Search", placeholder="Search by candidate ID or reasoning text…", label_visibility="collapsed"
            )
            sort_choice = ctrl_cols[1].selectbox(
                "Sort by",
                ["Rank (best first)", "Score (high → low)", "Score (low → high)"],
                label_visibility="collapsed",
            )
            score_min, score_max = float(df["score"].min()), float(df["score"].max())
            if score_min < score_max:
                selected_range = ctrl_cols[2].slider(
                    "Score range",
                    min_value=score_min,
                    max_value=score_max,
                    value=(score_min, score_max),
                    label_visibility="collapsed",
                )
            else:
                ctrl_cols[2].caption(f"Score range: {score_min:.6f} (all rows equal)")
                selected_range = (score_min, score_max)

            filtered = df[(df["score"] >= selected_range[0]) & (df["score"] <= selected_range[1])]
            if search_term:
                needle = search_term.strip().lower()
                mask = (
                    filtered["candidate_id"].str.lower().str.contains(needle, na=False)
                    | filtered["reasoning"].str.lower().str.contains(needle, na=False)
                )
                filtered = filtered[mask]

            if sort_choice == "Score (high → low)":
                filtered = filtered.sort_values("score", ascending=False)
            elif sort_choice == "Score (low → high)":
                filtered = filtered.sort_values("score", ascending=True)
            else:
                filtered = filtered.sort_values("rank", ascending=True)

            st.caption(f"Showing {len(filtered)} of {len(df)} candidates")

            st.dataframe(
                filtered,
                hide_index=True,
                use_container_width=True,
                height=min(36 * (len(filtered) + 1) + 4, 640),
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                    "candidate_id": st.column_config.TextColumn("Candidate ID", width="small"),
                    "score": st.column_config.ProgressColumn(
                        "Score", format="%.6f", min_value=score_min, max_value=max(score_max, score_min + 1e-9)
                    ),
                    "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
                },
            )

            if not filtered.empty:
                with st.expander("🔎 Inspect a single candidate's full reasoning"):
                    pick = st.selectbox(
                        "Candidate",
                        filtered["candidate_id"] + " — rank " + filtered["rank"].astype(str),
                    )
                    picked_id = pick.split(" — ")[0]
                    picked_row = df[df["candidate_id"] == picked_id].iloc[0]
                    st.markdown(
                        f'<blockquote class="rs-reasoning">{_esc(picked_row["reasoning"])}</blockquote>',
                        unsafe_allow_html=True,
                    )

        # ---------------- Artifacts & downloads ----------------
        with st.container(border=True):
            st.markdown('<p class="rs-section-title">📦 Artifacts</p>', unsafe_allow_html=True)

            for name, path, desc in ARTIFACT_CHECKLIST:
                exists = path.exists()
                icon = "✅" if exists else "⬜"
                meta = f"{_fmt_bytes(path.stat().st_size)}" if exists else "not generated"
                st.markdown(
                    f'<div class="rs-artifact-row">'
                    f'<div><span class="rs-artifact-name">{icon} {_esc(name)}</span><br>'
                    f'<span class="rs-artifact-meta">{_esc(desc)}</span></div>'
                    f'<div class="rs-artifact-meta">{_esc(meta)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            dl_cols = st.columns(2)
            with dl_cols[0]:
                st.download_button(
                    label="📥 Download Submission",
                    data=OUTPUT_FILE.read_bytes(),
                    file_name="team_EternalTalentPose.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl_cols[1]:
                if RUN_REPORT.exists():
                    st.download_button(
                        label="📥 Download Run Report",
                        data=RUN_REPORT.read_bytes(),
                        file_name="run_report.json",
                        mime="application/json",
                        use_container_width=True,
                    )

        # ---------------- Logs (only meaningful for a fresh run) ----------------
        if not result.get("hydrated") and (result["stdout"] or result["stderr"]):
            with st.expander("🧾 Execution Logs"):
                log_tabs = st.tabs(["Standard Output", "Standard Error"])
                with log_tabs[0]:
                    st.code(result["stdout"] or "(empty)")
                with log_tabs[1]:
                    st.code(result["stderr"] or "(empty)")

    else:
        st.error(f"Ranking pipeline failed (exit code {result['returncode']}).")

        with st.container(border=True):
            st.markdown('<p class="rs-section-title">📊 Execution Summary</p>', unsafe_allow_html=True)
            cols = st.columns(3)
            cols[0].metric("Status", "Failed ❌")
            cols[1].metric("Duration", _fmt_duration(result.get("elapsed")))
            cols[2].metric("Return Code", str(result["returncode"]))

        log_tabs = st.tabs(["Standard Output", "Standard Error"])
        with log_tabs[0]:
            st.code(result["stdout"] or "(empty)")
        with log_tabs[1]:
            st.code(result["stderr"] or "(empty)")

st.markdown(
    '<div class="rs-footer">RedStack Talent Ranker · deterministic offline-built, online-served candidate ranking</div>',
    unsafe_allow_html=True,
)
