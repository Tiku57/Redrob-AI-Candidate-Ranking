# 🚀 Redrob AI Candidate Ranking Engine

An AI-powered candidate ranking system designed to identify the most relevant candidates for a given Job Description (JD). The solution combines lexical retrieval, semantic search, reranking, skill validation, profile quality checks, and explainable scoring to generate high-quality candidate shortlists at scale.

---

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Runtime](https://img.shields.io/badge/Runtime-31.2s-success)
![Candidates](https://img.shields.io/badge/Candidates-100K-orange)
![Compute](https://img.shields.io/badge/CPU%20Only-✓-green)

---

## 🎯 Problem Statement

Traditional Applicant Tracking Systems (ATS) rely heavily on keyword matching, often overlooking qualified candidates and rewarding keyword stuffing.

This project addresses that challenge by combining semantic understanding, evidence-based skill validation, experience alignment, and profile consistency checks to produce more reliable candidate rankings.

---

## ✨ Key Features

- 📄 Dynamic Job Description parsing
- 🔍 Multi-stage candidate retrieval and ranking
- 🤖 Semantic matching using transformer models
- ✅ Skill evidence validation
- 📈 Experience-fit scoring
- 🛡️ Profile consistency and quality checks
- 👥 Near-duplicate profile handling
- 💡 Explainable ranking decisions
- ⚡ CPU-only execution with no external API dependencies

---

## 🏗️ Architecture Overview

The system processes candidate profiles through a three-stage ranking pipeline.

### Stage 1: Lexical Retrieval

TF-IDF retrieval rapidly filters the candidate pool and selects the most relevant profiles.

### Stage 2: Semantic Retrieval

A Bi-Encoder (`all-MiniLM-L6-v2`) computes dense embeddings and ranks candidates based on semantic similarity to the Job Description.

### Stage 3: Cross-Encoder Reranking

A Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) performs deeper relevance assessment on the top candidates to improve ranking quality.

### 🔎 Validation & Quality Layer

Additional ranking signals include:

- Experience alignment
- Skill match scoring
- Evidence validation
- Profile consistency checks
- Behavioral signals
- Duplicate profile penalties

---

## 🔄 Ranking Pipeline

```text
Job Description
       │
       ▼
   JD Parser
       │
       ▼
 TF-IDF Retrieval
   (100K → 2K)
       │
       ▼
Bi-Encoder Ranking
    (2K → 500)
       │
       ▼
Cross-Encoder
  Reranking
       │
       ▼
Validation Layer
• Skill Evidence
• Experience Fit
• Consistency Checks
• Quality Validation
       │
       ▼
  Final Ranking
       │
       ▼
Top 100 Candidates
```

---

## 📂 Project Structure

```text
Redrob-AI-Candidate-Ranking/
├── app.py
├── rank.py
├── parser.py
├── jd_parser.py
├── scorer.py
├── skill_engine.py
├── consistency_engine.py
├── diversity_engine.py
├── evaluator.py
├── feature_extractor.py
├── honeypot_detector.py
├── disqualifier.py
├── job_description.md
├── submission.csv
├── evaluation_report.md
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Tiku57/Redrob-AI-Candidate-Ranking.git
cd Redrob-AI-Candidate-Ranking
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Launch Dashboard

The Streamlit application serves as a lightweight dashboard for viewing ranking outputs and evaluation metrics.

```bash
streamlit run app.py
```

### Run Full Ranking Pipeline

```bash
python rank.py
```

The pipeline:

1. Parses the Job Description
2. Loads candidate profiles
3. Computes ranking scores
4. Applies validation checks
5. Generates explanations
6. Produces final outputs

Generated artifacts:

```text
submission.csv
evaluation_report.md
```

> Note: Large candidate datasets are excluded from the repository due to size constraints.

---

## 📊 Results

### Performance

- ⚡ Processes 100,000 candidate profiles
- ⏱️ End-to-end runtime: ~31.2 seconds
- 💻 CPU-only execution
- 🧠 Memory usage below 16 GB

### Ranking Quality

- ✅ Experience-fit integrated into ranking
- ✅ Skill evidence validation included
- ✅ Profile quality checks applied
- ✅ Near-duplicate profile suppression
- ✅ Explainable candidate recommendations

---

## 🛠️ Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Data Processing | Polars, NumPy |
| Retrieval | Scikit-Learn (TF-IDF) |
| Semantic Search | Sentence Transformers |
| Reranking | MS MARCO Cross-Encoder |
| Dashboard | Streamlit |
| Version Control | Git & GitHub |

---

## 📁 Outputs

### 📄 submission.csv

Final ranked candidate shortlist with scores and recruiter-facing reasoning.

### 📈 evaluation_report.md

Detailed evaluation metrics, ranking analysis, feature importance, validation checks, and runtime statistics.

---

## 🏆 Hackathon Highlights

- 🚀 Ranked 100,000 candidate profiles
- ⚡ Completed ranking in ~31.2 seconds
- 🤖 Multi-stage AI-powered retrieval pipeline
- 💡 Explainable candidate recommendations
- 💻 Fully CPU-based execution
- 🔒 No external API dependencies

---

## 👨‍💻 Author

**Aaditya Sattawan**

GitHub: https://github.com/Tiku57
