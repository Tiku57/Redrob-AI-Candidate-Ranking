# Redrob AI Candidate Ranking

An AI-powered candidate ranking engine built for recruiter-quality hiring. This project dynamically parses unstructured Job Descriptions (JDs), evaluates candidates holistically using multi-stage semantic retrieval, and produces highly accurate, recruiter-trusted shortlists at scale.

## Problem Statement
Recruiters often rely on keyword-based Applicant Tracking Systems (ATS) that miss highly qualified candidates, reward superficial keyword stuffing, and fail to holistically evaluate true capability. This engine solves that by combining deep semantic understanding, evidence validation, consistency scoring, behavioral signals, and diversity-aware ranking into a cohesive evaluation formula.

## Architecture Overview
The pipeline processes 100,000 JSONL candidates locally on the CPU in under **35 seconds**. It is built with zero external API dependencies to ensure privacy, deterministic results, and strict hardware compliance.

### The Pipeline
1. **Lexical Retrieval (Stage 1):** Scikit-Learn TF-IDF rapidly filters 100k records down to a highly relevant 2,000-candidate pool.
2. **Dense Retrieval (Stage 2):** HuggingFace `all-MiniLM-L6-v2` Bi-Encoder maps profiles into dense vector space, isolating the top 500 semantic matches.
3. **Deep Reranking (Stage 3):** `ms-marco-MiniLM-L-6-v2` Cross-Encoder performs deep token-level relevance assessment on the finalists.
4. **Validation Engines:** 
   - **Evidence Scoring:** Checks claimed advanced skills against proximate action verbs in candidate history.
   - **Consistency Engine:** Filters mathematically impossible career timelines (Quality Risk).
5. **Diversity Engine:** Suppresses near-duplicate profiles via dynamic penalties.

## Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/Tiku57/Redrob-AI-Candidate-Ranking.git
cd Redrob-AI-Candidate-Ranking

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install -r requirements.txt
```

## Usage Instructions

To run the full ranking pipeline over your candidate dataset:

```bash
python rank.py
```
This command parses `job_description.md` and evaluates all candidates inside `candidates.jsonl.gz`. 
It will generate two artifacts:
- `submission.csv`: The finalized rank-ordered list of candidates with recruiter reasoning.
- `evaluation_report.md`: A comprehensive breakdown of system variance, quality risk metrics, and validation checks.

## Results & Performance
- **Speed:** End-to-end runtime of ~31.5 seconds on a standard CPU.
- **Constraints:** Requires <16GB RAM. Completely local execution.
- **Quality:** 100% of the Top-20 candidates strictly meet the 5-9 YOE requirements. Near-duplicate synthetic profiles are completely suppressed.

## Technology Stack
- **Python** 
- **Polars & NumPy:** Ultra-fast, memory-efficient columnar and matrix operations.
- **Scikit-Learn:** Optimized sparse vectorization (TF-IDF).
- **Sentence Transformers:** Local inference for Dense Encoders.
