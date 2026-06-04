import polars as pl

def filter_anomalies(df: pl.DataFrame) -> pl.DataFrame:
    """
    Identifies and removes honeypots and keyword stuffers.
    Operates using vectorized Polars functions for CPU efficiency.
    """
    # 1. Honeypot Detection
    # Filter out candidates with impossible experience or missing core fields
    filtered_df = df.filter(
        # Realistic experience limit (e.g. max 50 years)
        (pl.col("years_experience") <= 50) & 
        (pl.col("years_experience") >= 0) &
        # Exclude completely empty text
        (pl.col("combined_text").str.len_chars() > 20)
    )
    
    # 2. Keyword Stuffer Detection
    # Heuristic: exclude texts that are absurdly long and repetitive, or lack spaces.
    # In Polars we do a fast word count proxy by splitting on space
    filtered_df = filtered_df.filter(
        pl.col("combined_text").str.split(" ").list.len() < 5000 
    )
    
    return filtered_df
