import polars as pl
import hashlib
import numpy as np

def apply_diversity_penalty(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes a text hash for candidate profiles and applies a compounding diversity penalty 
    (-0.05 per duplicate) to subsequent clones in the dataframe, preventing synthetic clones 
    from dominating the top ranks.
    """
    texts = df["combined_text"].to_list()
    hashes = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    
    # We must iterate in order of current scores to penalize lower-ranked clones more,
    # or just keep a running count. Assuming df is mostly sorted by semantic score.
    
    seen_counts = {}
    penalties = []
    
    for h in hashes:
        count = seen_counts.get(h, 0)
        penalty = count * 0.05  # 5% deduction for each subsequent clone
        penalties.append(penalty)
        seen_counts[h] = count + 1
        
    return df.with_columns(
        diversity_penalty=pl.Series(penalties),
        text_hash=pl.Series(hashes)
    )
