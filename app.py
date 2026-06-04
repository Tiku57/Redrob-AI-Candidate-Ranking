import streamlit as st
import polars as pl
import os

st.set_page_config(page_title="Redrob Ranker Demo", layout="wide")

st.title("🏆 Redrob AI Candidate Ranking Dashboard")
st.markdown("""
**Project Showcase:** This application is a lightweight demonstration dashboard that visualizes the final ranking outputs and evaluation metrics of the Redrob pipeline.
The full ranking engine was executed against a 100,000-candidate dataset.
""")

st.header("Technology Stack & Architecture")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Pipeline Architecture
    1. **Lexical Retrieval:** Scikit-Learn TF-IDF broad matching (100k → 2k).
    2. **Semantic Ranking:** HuggingFace `all-MiniLM-L6-v2` Bi-Encoder (2k → 500).
    3. **Deep Reranking:** `ms-marco-MiniLM-L-6-v2` Cross-Encoder.
    4. **Validation Heuristics:** Boolean evidence validation & timeline consistency.
    5. **Diversity Engine:** Suppression of near-duplicate profiles.
    """)

with col2:
    st.markdown("""
    ### Runtime Metrics
    - **Total Runtime:** 31.2 seconds (End-to-End)
    - **Compute Environment:** Local CPU Only (No GPUs, zero external APIs)
    - **Memory Limits:** Under 16GB RAM
    - **Data Engine:** Polars for columnar memory-mapped processing.
    """)

st.divider()

st.header("Top Candidates (submission.csv)")
output_file = "submission.csv"

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
        label="Download Final Submission CSV",
        data=csv_data,
        file_name=output_file,
        mime='text/csv',
    )
else:
    st.warning("`submission.csv` is missing from the repository.")

st.divider()

st.header("Detailed Evaluation Report")
eval_file = "evaluation_report.md"
if os.path.exists(eval_file):
    with st.expander("View Full Report", expanded=False):
        with open(eval_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
else:
    st.info("evaluation_report.md not found.")
