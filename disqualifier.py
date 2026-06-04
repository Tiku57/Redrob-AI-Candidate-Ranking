import polars as pl

def apply_disqualifiers(df: pl.DataFrame, jd_rules: dict = None) -> pl.DataFrame:
    """
    Applies JD-specific disqualifications to drop candidates from the ranking.
    """
    jd_rules = jd_rules or {}
    min_yoe = jd_rules.get("min_yoe", 3)
    max_yoe = jd_rules.get("max_yoe", 50)
    # 1. Marketing / Sales Trap
    df = df.filter(
        ~pl.col("current_title").str.contains(r"\b(marketing|sales|account executive)\b")
    )
    
    # 2. Title Chasers
    df = df.filter(
        ~((pl.col("avg_job_duration") < 18) & (pl.col("years_of_experience") > min_yoe))
    )
    
    # 3. Pure Consulting Trap (computed precisely in parser.py)
    df = df.filter(
        ~pl.col("is_pure_consulting")
    )
    
    # 4. Pure Research Trap (computed precisely in parser.py)
    df = df.filter(
        ~pl.col("is_pure_research")
    )
    
    # 5. Honeypots (Subtly impossible profiles)
    # E.g. claiming far more job duration months than mathematically possible given their years of experience
    df = df.filter(
        pl.col("total_job_duration_months") <= ((pl.col("years_of_experience") * 12) + 36) # 3 year leniency for overlaps
    )
    
    # Another honeypot: Impossible experience, plus strict bounds from JD
    # We allow a small leniency on max_yoe so we don't drop someone overqualified entirely unless specified
    df = df.filter(
        (pl.col("years_of_experience") >= 0) & (pl.col("years_of_experience") <= max_yoe + 10)
    )
    
    return df
