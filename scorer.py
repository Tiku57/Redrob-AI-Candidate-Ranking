import numpy as np
import polars as pl
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from feature_extractor import extract_lexical_features
from diversity_engine import apply_diversity_penalty

def compute_scores(df: pl.DataFrame, semantic_query: str, min_yoe: int, max_yoe: int, top_k_stage1: int = 2000, top_k_stage2: int = 500) -> pl.DataFrame:
    # 1. Stage 1: Lexical Score (TF-IDF)
    tfidf_matrix, query_vec = extract_lexical_features(df, semantic_query)
    stage1_scores = tfidf_matrix.dot(query_vec.T).toarray().flatten()
    df = df.with_columns(stage1_score=pl.Series(stage1_scores))
    
    top_candidates = df.top_k(top_k_stage1, by="stage1_score")
    
    # 2. Compute Experience Fit Score early to ensure Top 500 has diverse archetypes
    yoe = top_candidates["years_of_experience"].to_numpy()
    exp_fit_score = np.ones_like(yoe, dtype=float)
    for i, y in enumerate(yoe):
        if min_yoe <= y <= max_yoe:
            exp_fit_score[i] = 1.0
        elif y == min_yoe - 1 or y == max_yoe + 1:
            exp_fit_score[i] = 0.8
        elif y == min_yoe - 2 or y == max_yoe + 2:
            exp_fit_score[i] = 0.5
        elif y == min_yoe - 3 or y == max_yoe + 3:
            exp_fit_score[i] = 0.2
        else:
            exp_fit_score[i] = 0.05
            
    top_candidates = top_candidates.with_columns(exp_fit_score=pl.Series(exp_fit_score))
    
    # 3. Stage 2: Bi-Encoder Semantic Score
    print("      -> Bi-Encoder Ranking (Stage 2)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode(semantic_query, normalize_embeddings=True)
    
    candidate_texts = top_candidates["combined_text"].to_list()
    candidate_embs = model.encode(candidate_texts, normalize_embeddings=True, show_progress_bar=False)
    semantic_scores = np.dot(candidate_embs, query_emb)
    
    # Apply experience fit penalty during early retrieval to ensure cohort diversity
    stage2_mixed = semantic_scores * (exp_fit_score ** 4)
    
    top_candidates = top_candidates.with_columns(
        semantic_score=pl.Series(semantic_scores),
        stage2_mixed=pl.Series(stage2_mixed)
    )
    top_candidates_s2 = top_candidates.top_k(top_k_stage2, by="stage2_mixed")
    
    # Extract the filtered exp_fit_score for the top 500
    exp_fit_score_s2 = top_candidates_s2["exp_fit_score"].to_numpy()
    
    # Fix 2: Continuous Seniority Bonus
    yoe_s2 = top_candidates_s2["years_of_experience"].to_numpy()
    seniority_bonus = np.clip((yoe_s2 - min_yoe) / (max_yoe - min_yoe + 1e-5), 0, 1) * 0.05
    exp_fit_score_s2 = exp_fit_score_s2 + seniority_bonus
    
    # 4. Stage 3: Cross-Encoder Reranking
    print("      -> Cross-Encoder Reranking (Stage 3)")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    stage3_candidates = top_candidates_s2["combined_text"].to_list()
    cross_inputs = [[semantic_query, doc] for doc in stage3_candidates]
    cross_scores = cross_encoder.predict(cross_inputs, show_progress_bar=False)
    
    top_candidates_s2 = top_candidates_s2.with_columns(cross_score=pl.Series(cross_scores))
    
    # 5. Engagement Score Calculation
    int_comp = top_candidates_s2["interview_completion_rate"].to_numpy()
    off_acc = top_candidates_s2["offer_acceptance_rate"].to_numpy()
    prof_comp = top_candidates_s2["profile_completeness"].to_numpy()
    
    resp_rate = top_candidates_s2["recruiter_response_rate"].to_numpy()
    resp_norm = np.clip(resp_rate, 0, 1)
    
    gh_score = top_candidates_s2["github_activity_score"].to_numpy()
    gh_norm = np.clip(gh_score / 100.0, 0, 1)
    
    saved_30d = top_candidates_s2["saved_by_recruiters_30d"].to_numpy()
    saved_norm = np.clip(saved_30d / 10.0, 0, 1)
    
    search_app = top_candidates_s2["search_appearances"].to_numpy()
    search_norm = np.clip(search_app / 50.0, 0, 1)
    
    endorse = top_candidates_s2["endorsements"].to_numpy()
    endorse_norm = np.clip(endorse / 50.0, 0, 1)
    
    conn_count = top_candidates_s2["connection_count"].to_numpy()
    conn_norm = np.clip(conn_count / 500.0, 0, 1)
    
    # Compute recency norm to differentiate candidates with identical static engagement metrics
    last_active = top_candidates_s2["last_active_date"].to_list()
    today = datetime.now()
    days_since = []
    for d in last_active:
        try:
            days = (today - datetime.strptime(d, "%Y-%m-%d")).days
        except Exception:
            days = 300
        days_since.append(days)
    days_since = np.array(days_since)
    recency_norm = np.clip(1.0 - (days_since / 300.0), 0, 1)
    
    engagement_raw = (int_comp + off_acc + prof_comp + resp_norm + gh_norm + saved_norm + search_norm + endorse_norm + conn_norm + recency_norm) / 10.0
    
    # Fix 1: Cohort Min-Max Scaling for Engagement
    eng_min, eng_max = engagement_raw.min(), engagement_raw.max()
    engagement_score = (engagement_raw - eng_min) / (eng_max - eng_min + 1e-5)
    
    # 6. Normalization for Final Formula
    c_min, c_max = cross_scores.min(), cross_scores.max()
    cross_norm = (cross_scores - c_min) / (c_max - c_min + 1e-5)
    
    sem_array = top_candidates_s2["semantic_score"].to_numpy()
    s_min, s_max = sem_array.min(), sem_array.max()
    sem_norm = (sem_array - s_min) / (s_max - s_min + 1e-5)
    
    # 7. Base Relevance Scoring
    req_score = top_candidates_s2["req_score"].to_numpy()
    evidence_score = top_candidates_s2["evidence_score"].to_numpy()
    consistency = top_candidates_s2["consistency_score"].to_numpy()
    skill_penalty = top_candidates_s2["skill_to_role_penalty"].to_numpy()
    
    consistency_adj = np.clip(consistency - skill_penalty, 0.05, 1.0)
    
    # Final Formula (Feature Re-weighting: Weighted Relevance Scoring)
    raw_relevance = (0.45 * cross_norm) + (0.15 * sem_norm) + (0.15 * req_score) + (0.10 * evidence_score) + (0.10 * engagement_score) + (0.05 * consistency_adj)
    
    # Secondary ranking signal for stable ordering before applying diversity penalties
    secondary_rank_score = (0.4 * req_score) + (0.3 * evidence_score) + (0.2 * engagement_score) + (0.1 * consistency_adj)
    
    # Multiplicative application of experience fit ensures out-of-range candidates do not outrank matched ones
    base_final_scores = (raw_relevance * exp_fit_score_s2) + (0.01 * secondary_rank_score)
    
    # Penalty for invalid or low-quality candidate profiles
    honeypot_risk = top_candidates_s2["honeypot_risk"].to_numpy()
    demotion_mask = (honeypot_risk > 0.8) | (exp_fit_score_s2 < 0.5)
    base_final_scores[demotion_mask] -= 1000.0
    
    top_candidates_s2 = top_candidates_s2.with_columns(
        engagement_score=pl.Series(engagement_score),
        cross_norm=pl.Series(cross_norm),
        sem_norm=pl.Series(sem_norm),
        consistency_adj=pl.Series(consistency_adj),
        base_final_score=pl.Series(base_final_scores),
        exp_fit_score=pl.Series(exp_fit_score_s2)
    )
    
    # Sort descending by base final score before diversity application
    ranked = top_candidates_s2.sort(["base_final_score", "candidate_id"], descending=[True, False])
    
    # 8. Diversity Engine (Post-ranking)
    # Applying here means near-duplicate profiles are penalized correctly based on their final score rank
    ranked = apply_diversity_penalty(ranked)
    
    # Cap penalty at 0.25 max
    div_penalty = ranked["diversity_penalty"].to_numpy()
    div_penalty = np.clip(div_penalty, 0.0, 0.25)
    
    final_scores = ranked["base_final_score"].to_numpy() - div_penalty
    
    ranked = ranked.with_columns(
        diversity_penalty=pl.Series(div_penalty),
        final_score=pl.Series(final_scores)
    )
    
    # Final sort
    ranked = ranked.sort(["final_score", "candidate_id"], descending=[True, False])
    return ranked

def generate_reasoning(row: dict) -> str:
    """
    Generates reasoning explicitly referencing components.
    """
    title = str(row.get("current_title", "Professional")).title()
    yoe = row.get("years_of_experience", 0)
    matched_req = str(row.get("matched_req", ""))
    matched_pref = str(row.get("matched_pref", ""))
    
    int_comp = row.get("interview_completion_rate", 0)
    notice = row.get("notice_period_days", 90)
    
    risk = row.get("honeypot_risk", 0)
    consistency = row.get("consistency_adj", 1.0)
    
    all_matched = [m.strip().title() for m in (matched_req + ", " + matched_pref).split(",") if m.strip()]
    matched_str = "• " + "\n• ".join(all_matched[:4]) if all_matched else "• Standard ML Stack"
    
    reasoning = (
        f"**Match:**\n{matched_str}\n\n"
        f"**Experience:**\n{yoe} years\n\n"
        f"**Risk:**\nQuality Risk: {risk:.2f}, Consistency: {consistency:.2f}"
    )
    return reasoning
