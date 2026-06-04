import streamlit as st
import polars as pl
import os
import subprocess

st.set_page_config(page_title="Redrob Ranker", layout="wide")

st.title("🏆 Redrob Intelligent Candidate Discovery")
st.write("Rank 100,000 candidates for the Senior AI Engineer role in under 5 minutes.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Control Panel")
    input_file = st.text_input("Input Dataset Path", "candidates.jsonl.gz")
    output_file = st.text_input("Output CSV Path", "submission.csv")
    
    if st.button("Generate Synthetic Data (Test)"):
        with st.spinner("Generating 100,000 JSONL.GZ candidates..."):
            subprocess.run(["python", "generate_dummy_data.py"], check=True)
            st.success(f"Generated test dataset at candidates.jsonl.gz")
            
    if st.button("Run Ranking Pipeline", type="primary"):
        if not os.path.exists(input_file):
            st.error(f"File not found: {input_file}")
        else:
            with st.spinner("Running high-speed CPU ranking pipeline..."):
                try:
                    result = subprocess.run(
                        ["python", "rank.py", "--input", input_file, "--output", output_file],
                        capture_output=True, text=True, check=True
                    )
                    st.success("Pipeline completed successfully!")
                    with st.expander("View Logs"):
                        st.code(result.stdout)
                except subprocess.CalledProcessError as e:
                    st.error("Pipeline failed!")
                    st.code(e.stderr)

with col2:
    st.header("Top Candidates")
    if os.path.exists(output_file):
        df = pl.read_csv(output_file).to_pandas()
        st.dataframe(
            df,
            column_config={
                "score": st.column_config.NumberColumn("Score", format="%.3f"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large")
            },
            hide_index=True,
            use_container_width=True
        )
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Submission CSV",
            data=csv_data,
            file_name=output_file,
            mime='text/csv',
        )
    else:
        st.info("Run the pipeline to generate results.")
