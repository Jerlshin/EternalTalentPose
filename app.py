import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RedStack Talent Ranker",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 RedStack")
st.subheader("Evidence-Grounded Semantic Candidate Ranking")

st.write(
    "Upload a `candidates.jsonl` file and run the deterministic RedStack ranking pipeline."
)

REPO_ROOT = Path.cwd()

DATA_DIR = REPO_ROOT / "data" / "raw"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "candidates.jsonl"
OUTPUT_FILE = ARTIFACT_DIR / "submission.csv"
RUN_REPORT = ARTIFACT_DIR / "run_report.json"

uploaded_file = st.file_uploader(
    "Upload candidates.jsonl",
    type=["jsonl"],
)

if uploaded_file is not None:

    INPUT_FILE.write_bytes(uploaded_file.getbuffer())

    st.success("Candidate file uploaded successfully.")
    st.caption(f"Input file: `{INPUT_FILE}`")

    if st.button("Run RedStack Ranking", type="primary"):

        with st.spinner("Running ranking pipeline..."):

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

            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

        if result.returncode == 0 and OUTPUT_FILE.exists():

            st.success("Ranking completed successfully.")

            df = pd.read_csv(OUTPUT_FILE)

            st.subheader("Top Ranked Candidates")
            st.dataframe(df.head(100), use_container_width=True)

            st.download_button(
                label="📥 Download Submission",
                data=OUTPUT_FILE.read_bytes(),
                file_name="team_EternalTalentPose.csv",
                mime="text/csv",
            )

            if RUN_REPORT.exists():

                st.download_button(
                    label="📥 Download Run Report",
                    data=RUN_REPORT.read_bytes(),
                    file_name="run_report.json",
                    mime="application/json",
                )

        else:

            st.error("Ranking pipeline failed.")

            if result.stdout:

                with st.expander("Standard Output"):
                    st.code(result.stdout)

            if result.stderr:

                with st.expander("Standard Error", expanded=True):
                    st.code(result.stderr)