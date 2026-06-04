import time
import argparse
import polars as pl
from parser import parse_candidates
from consistency_engine import add_consistency_scores
from skill_engine import evaluate_skills
from diversity_engine import apply_diversity_penalty
from scorer import compute_scores, generate_reasoning
from jd_parser import parse_jd
from evaluator import generate_report

def run_pipeline(input_file: str, jd_file: str, output_file: str):
    start_time = time.time()
    print(f"Starting pipeline for {input_file} using JD {jd_file}...")
    
    print("0. Parsing Job Description...")
    jd_rules = parse_jd(jd_file)
    print(f"   Extracted min_yoe={jd_rules['min_yoe']}, max_yoe={jd_rules['max_yoe']}")
    
    print("1. Parsing candidates...")
    df = parse_candidates(input_file, consulting_firms=jd_rules["consulting_firms"])
    initial_count = len(df)
    
    print("2. Applying Consistency & Skill Engines...")
    df = add_consistency_scores(df, jd_rules)
    df = evaluate_skills(df, jd_rules)
    
    print("3. Scoring and Ranking (Stage 1 Lexical -> Stage 2 Bi-Encoder -> Stage 3 Cross-Encoder)...")
    ranked_df = compute_scores(df, semantic_query=jd_rules["semantic_query"], min_yoe=jd_rules['min_yoe'], max_yoe=jd_rules['max_yoe'], top_k_stage1=2000, top_k_stage2=500)
    
    top_100 = ranked_df.head(100)
    
    print("4. Generating reasoning...")
    rows = top_100.to_dicts()
    reasonings = [generate_reasoning(row) for row in rows]
    
    top_100 = top_100.with_columns(
        rank=pl.Series(range(1, len(top_100) + 1)),
        reasoning=pl.Series(reasonings)
    )
    
    final_cols = ["candidate_id", "rank", "final_score", "reasoning"]
    final_output = top_100.select(final_cols).rename({"final_score": "score"})
    
    print(f"5. Saving to {output_file}...")
    final_output.write_csv(output_file)
    
    elapsed = time.time() - start_time
    print("6. Generating Evaluation Report...")
    generate_report(ranked_df, top_100, "evaluation_report.md", runtime_secs=elapsed)
    
    print(f"Pipeline completed in {elapsed:.2f} seconds.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking")
    parser.add_argument("--candidates", default="candidates.jsonl.gz", help="Input dataset")
    parser.add_argument("--jd", default="job_description.md", help="Job Description Markdown file")
    parser.add_argument("--out", default="submission.csv", help="Output CSV for submission")
    args = parser.parse_args()
    run_pipeline(args.candidates, args.jd, args.out)
