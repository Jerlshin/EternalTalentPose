import streamlit as st
import pandas as pd
import subprocess
import os
import sys

# 1. Page Configuration
st.set_page_config(page_title="RedStack Ranker", layout="wide")
st.title("🚀 RedStack Talent Ranker")
st.markdown("Automated Ranking Sandbox (Aligned with Stage 3 constraints)")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload candidates.jsonl", type="jsonl")

if uploaded_file is not None:
    # Ensure local directory structure exists
    os.makedirs("data/raw", exist_ok=True)
    input_path = "data/raw/input.jsonl"
    output_path = "submission.csv"
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Run Ranking Pipeline"):
        with st.spinner("Executing pipeline..."):
            # 3. Execution Environment Setup
            # Set PYTHONPATH to the current directory so 'src' is importable
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd()
            
            try:
                # 4. Invoke run.py using the exact signature from reproduce.sh
                process = subprocess.run(
                    ["python", "run.py", "--candidates", input_path, "--out", output_path],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # 5. Display Results
                if os.path.exists(output_path):
                    df = pd.read_csv(output_path)
                    st.success("Ranking complete!")
                    st.dataframe(df.head(100))
                    
                    # 6. Download Button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Ranked CSV",
                            data=f,
                            file_name="team_EternalTalentPose.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("Pipeline finished but no CSV was created.")
                    
            except subprocess.CalledProcessError as e:
                # 7. Enhanced Error Debugging
                st.error("Pipeline failed!")
                st.subheader("Error Output (stderr):")
                st.code(e.stderr)
                st.subheader("Standard Output (stdout):")
                st.code(e.stdout)