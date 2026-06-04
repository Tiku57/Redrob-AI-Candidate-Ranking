import polars as pl
import re

def evaluate_skills(df: pl.DataFrame, jd_rules: dict) -> pl.DataFrame:
    """
    Computes required and preferred skill scores, extracts matched/missing skills, 
    and applies a penalty for claiming advanced skills without supporting evidence.
    """
    req_skills = set(jd_rules.get("required_skills", []))
    pref_skills = set(jd_rules.get("preferred_skills", []))
    
    # Advanced technical skills requiring context validation
    advanced_buzzwords = {'rlhf', 'distributed training', 'langgraph', 'vector databases', 'agentic systems', 'qdrant', 'milvus', 'rag', 'vector db'}
    
    # Synonym dictionary for better semantic matching
    SYNONYMS = {
        "retrieval": ["vector search", "semantic search", "embedding search"],
        "ranking": ["ranking systems", "relevance ranking", "recsys"],
        "recommendation systems": ["recommendation engine", "recommendation platform"],
        "hybrid search": ["retrieval systems", "vector retrieval"],
        "evaluation": ["offline evaluation", "model evaluation"],
        "evaluating": ["offline evaluation", "model evaluation"],
        "rag": ["retrieval augmented generation"],
        "qdrant": ["vector db", "milvus", "pinecone", "weaviate", "vector database"],
        "milvus": ["vector db", "qdrant", "pinecone", "weaviate", "vector database"],
        "machine learning": ["ml pipelines", "ml"],
        "ml pipelines": ["ml platform", "training pipeline", "production ml"]
    }
    
    action_verbs = ['built', 'implemented', 'deployed', 'designed', 'scaled', 'optimized', 'productionized']
    
    def process_row(combined_text, yoe, skills_list):
        skills_set = set([s.lower().strip() for s in skills_list])
        text_lower = combined_text.lower()
        
        matched_req = []
        missing_req = []
        
        for req in req_skills:
            req_lower = req.lower()
            syns = SYNONYMS.get(req_lower, [])
            
            # Check if req or any synonym is in text
            if req_lower in text_lower or any(syn in text_lower for syn in syns):
                matched_req.append(req)
            else:
                missing_req.append(req)
                
        matched_pref = []
        for pref in pref_skills:
            pref_lower = pref.lower()
            syns = SYNONYMS.get(pref_lower, [])
            if pref_lower in text_lower or any(syn in text_lower for syn in syns):
                matched_pref.append(pref)
                        
        req_score = len(matched_req) / len(req_skills) if req_skills else 1.0
        pref_score = len(matched_pref) / len(pref_skills) if pref_skills else 1.0
        
        # Evidence Validation
        claimed_advanced = [buzz for buzz in advanced_buzzwords if buzz in text_lower]
        skill_penalty = 0.0
        evidence_score = 1.0
        
        if len(claimed_advanced) > 0:
            valid_proximate_verbs = set()
            for claim in claimed_advanced:
                for verb in action_verbs:
                    pattern1 = r'\b{}\b.{{0,200}}\b{}\b'.format(verb, re.escape(claim.lower()))
                    pattern2 = r'\b{}\b.{{0,200}}\b{}\b'.format(re.escape(claim.lower()), verb)
                    if re.search(pattern1, text_lower) or re.search(pattern2, text_lower):
                        valid_proximate_verbs.add(verb)
            
            # Fix 3: Continuous Evidence Scoring
            # Scale based on how many distinct action verbs are used in proximity
            evidence_score = min(1.0, len(valid_proximate_verbs) / 3.0)
            
        # Skill-to-Role Consistency Penalty
        if yoe < 2.5 and len(claimed_advanced) >= 2:
            skill_penalty = 0.4  # Penalty for skill claims misaligned with seniority level
        elif yoe < 4.0 and len(claimed_advanced) >= 4:
            skill_penalty = 0.3
            
        return {
            "req_score": req_score,
            "pref_score": pref_score,
            "matched_req": ", ".join(matched_req),
            "missing_req": ", ".join(missing_req),
            "matched_pref": ", ".join(matched_pref),
            "skill_to_role_penalty": skill_penalty,
            "evidence_score": evidence_score
        }
        
    results = df.select(["combined_text", "years_of_experience", "skills_list"]).map_rows(
        lambda row: tuple(process_row(row[0], row[1], row[2]).values())
    )
    
    results = results.rename({
        "column_0": "req_score",
        "column_1": "pref_score",
        "column_2": "matched_req",
        "column_3": "missing_req",
        "column_4": "matched_pref",
        "column_5": "skill_to_role_penalty",
        "column_6": "evidence_score"
    })
    
    return pl.concat([df, results], how="horizontal")
