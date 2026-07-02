import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="RedStack Ranker", layout="wide")
st.title("🚀 RedStack Talent Ranker")
st.markdown("Upload your `candidates.jsonl` to rank top talent using your pre-trained model.")

# File Uploader
uploaded_file = st.file_uploader("Upload candidates.jsonl", type="jsonl")

if uploaded_file is not None:
    # Ensure local directory structure exists for the pipeline
    os.makedirs("data/raw", exist_ok=True)
    input_path = "data/raw/input.jsonl"
    output_path = "submission.csv"
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Run Ranking Pipeline"):
        with st.spinner("Executing RedStack engine..."):
            # Execute your existing run.py logic
            # Based on your reproduce.sh, we call run.py
            try:
                subprocess.run(
                    ["python", "run.py", "--candidates", input_path, "--out", output_path],
                    check=True
                )
                
                if os.path.exists(output_path):
                    df = pd.read_csv(output_path)
                    st.success("Ranking complete!")
                    st.dataframe(df.head(100))
                    
                    # Download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Ranked CSV",
                            data=f,
                            file_name="team_EternalTalentPose.csv",
                            mime="text/csv"
                        )
            except subprocess.CalledProcessError as e:
                st.error(f"Pipeline failed: {e}")