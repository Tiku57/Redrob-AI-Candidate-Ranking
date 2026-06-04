# 🚀 Redrob AI Candidate Ranking

An AI-powered candidate ranking system designed to identify the most relevant candidates for a given Job Description (JD). The solution combines lexical retrieval, semantic search, reranking, skill validation, and profile quality checks to generate explainable candidate shortlists at scale.

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
- 👥 Near-duplicate candidate handling
- 💡 Explainable ranking decisions
- ⚡ CPU-only execution with no external API dependencies

---

## 🏗️ Architecture Overview

The system processes candidate profiles through a three-stage ranking pipeline.

### Stage 1: Lexical Retrieval

TF-IDF retrieval rapidly filters the candidate pool and selects the most relevant profiles.

### Stage 2: Semantic Retrieval

A Bi-Encoder (`all-MiniLM-L6-v2`) computes dense embeddings and ranks candidates based on semantic similarity to the JD.

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

Run the ranking pipeline:

```bash
python rank.py
```

The pipeline will:

1. Parse the Job Description
2. Load candidate profiles
3. Compute ranking scores
4. Generate explanations
5. Produce final outputs

Generated files:

```text
submission.csv
evaluation_report.md
```

---

## 📊 Results

### Performance

- ⚡ Processes 100,000 candidate profiles
- ⏱️ End-to-end runtime: ~31.5 seconds
- 💻 CPU-only execution
- 🧠 Memory usage below 16 GB

### Ranking Quality

- ✅ Experience-fit integrated into ranking
- ✅ Skill evidence validation included
- ✅ Profile quality checks applied
- ✅ Duplicate candidate penalties improve ranking diversity

---

## 🛠️ Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Data Processing | Polars, NumPy |
| Retrieval | Scikit-Learn (TF-IDF) |
| Semantic Search | Sentence Transformers |
| Reranking | Cross-Encoder (MS MARCO MiniLM) |
| Interface | Streamlit |
| Version Control | Git & GitHub |

---

## 📁 Outputs

### 📄 submission.csv

Final ranked candidate shortlist.

### 📈 evaluation_report.md

Detailed evaluation metrics, ranking analysis, and validation statistics.

---

## 🏆 Hackathon Highlights

- 🚀 Ranked 100,000 candidate profiles
- ⚡ Completed ranking in ~31.5 seconds
- 🤖 Multi-stage AI-powered retrieval pipeline
- 💡 Explainable candidate recommendations
- 💻 Fully CPU-based execution
- 🔒 No external API dependencies
