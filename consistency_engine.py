import polars as pl
import numpy as np

def add_consistency_scores(df: pl.DataFrame, jd_rules: dict) -> pl.DataFrame:
    """
    Computes probabilistic scores based on candidate profile heuristics to replace hard-filtering.
    Provides `consistency_score` (0.0 to 1.0) and `honeypot_risk` (0.0 to 1.0).
    """
    dur_months = df["total_job_duration_months"].to_numpy()
    yoe = df["years_of_experience"].to_numpy()
    yoe_months = yoe * 12
    grad_year = df["graduation_year"].to_numpy()
    
    # 1. Timeline Consistency: Overlapping jobs allowing for 1.5x YOE + 4 years of concurrent/part-time roles
    impossible_overlap = (dur_months > (yoe_months * 1.5) + 48).astype(float)
    
    # Impossible YOE check (e.g. 50+ years)
    impossible_yoe = (yoe > 50).astype(float)
    
    # Education Consistency: Graduated too recently for claimed experience
    years_since_grad = 2024 - grad_year
    # Ignore missing grad year (grad_year == 0 means years_since_grad == 2024, so it's > 40)
    # Only penalize if grad_year > 0 and experience significantly exceeds post-grad time + 6 years of college jobs
    impossible_grad = ((grad_year > 0) & ((years_since_grad + 6) < yoe)).astype(float)
    
    # 2. Skill Consistency: Claiming many advanced AI skills with very low actual experience
    advanced_skills = df["advanced_ai_skills"].to_numpy()
    unsupported_skills = ((advanced_skills >= 4) & (yoe < 1.0)).astype(float)
    
    # 3. Career Progression (Title velocity)
    def compute_progression_penalty(titles, durations):
        # Relaxed check: Intern to Principal in < 3 years
        has_intern = any('intern' in str(t).lower() for t in titles)
        has_senior_exec = any(any(x in str(t).lower() for x in ['principal', 'vp', 'head']) for t in titles)
        if has_intern and has_senior_exec and sum(durations) < 36:
            return 0.3
        return 0.0

    prog_penalties = df.select(["titles_list", "durations_list"]).map_rows(
        lambda row: (compute_progression_penalty(row[0], row[1]),)
    ).to_series().to_numpy()
    
    # 4. Honeypot Risk Score: High probability of synthetic or impossible profile
    honeypot_risk = np.clip(impossible_overlap + impossible_yoe + unsupported_skills + impossible_grad + prog_penalties, 0, 1)
    
    # 5. Career Consistency vs JD
    is_pure_consulting = df["is_pure_consulting"].cast(float).to_numpy()
    is_pure_research = df["is_pure_research"].cast(float).to_numpy()
    
    # Title chasers (less than 1 yrs average duration, but over 4 yrs experience)
    avg_dur = df["avg_job_duration"].to_numpy()
    title_chasers = ((avg_dur < 12) & (yoe > 4.0)).astype(float)
    
    # 6. Final Consistency Score
    consistency = 1.0 - (0.1 * is_pure_consulting) - (0.1 * is_pure_research) - (0.1 * title_chasers) - (0.5 * honeypot_risk) - prog_penalties
    consistency = np.clip(consistency, 0.05, 1.0)
    
    df = df.with_columns(
        honeypot_risk=pl.Series(honeypot_risk),
        consistency_score=pl.Series(consistency)
    )
    
    return df

