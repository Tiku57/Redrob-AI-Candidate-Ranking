import gzip
import json
import polars as pl

def parse_candidates(file_path: str, consulting_firms: set = None) -> pl.DataFrame:
    """
    Parses candidates from a JSONL.GZ file matching the Redrob Candidate Schema.
    Uses pure Python for initial flattening to safely handle nested structures and compute 
    complex boolean flags for JD disqualifications (pure consulting, pure research).
    """
    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'
    
    consulting_firms = consulting_firms or {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'mindtree', 'deloitte', 'pwc', 'kpmg', 'ey', 'ibm'}
    research_keywords = {'research assistant', 'academic', 'phd student', 'postdoc', 'research fellow'}
    
    flattened = []
    with open_func(file_path, mode) as f:
        # Check if it's a JSON array (like sample_candidates.json) or JSONL
        if file_path.endswith('.json'):
            # Load the entire array
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
        else:
            # Read line by line for JSONL / JSONL.GZ
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        for c in data:
            
            prof = c.get("profile", {})
            sig = c.get("redrob_signals", {})
            
            career = c.get("career_history", [])
            
            # Compute exact job durations and flags
            total_dur = 0
            companies = []
            titles = []
            
            for job in career:
                total_dur += job.get("duration_months", 0)
                companies.append(str(job.get("company", "")).lower())
                titles.append(str(job.get("title", "")).lower())
                
            avg_dur = total_dur / len(career) if career else 0
            
            # JD Disqualifier logic evaluated safely in Python
            is_pure_consulting = len(companies) > 0 and all(
                any(cf in comp for cf in consulting_firms) for comp in companies
            )
            
            is_pure_research = len(titles) > 0 and all(
                any(rk in t for rk in research_keywords) for t in titles
            )
            
            career_json = json.dumps(career).lower()
            skills_json = json.dumps(c.get("skills", [])).lower()
            
            raw_text = f"{prof.get('headline','')} {prof.get('summary','')} {career_json} {skills_json}".lower()
            combined_text = "".join([char if char.isalnum() or char.isspace() else " " for char in raw_text])
            
            # Compute advanced AI skills count for consistency checking
            skills = c.get("skills", [])
            advanced_ai_skills = len([s for s in skills if s.get("proficiency") == "advanced"])
            
            flat = {
                "candidate_id": c.get("candidate_id"),
                "profile_location": prof.get("location", ""),
                "profile_headline": prof.get("headline", ""),
                "profile_summary": prof.get("summary", ""),
                "years_of_experience": prof.get("years_of_experience", 0),
                "current_title": prof.get("current_title", "").lower(),
                "current_company": prof.get("current_company", "").lower(),
                
                "last_active_date": sig.get("last_active_date", "2020-01-01"),
                "recruiter_response_rate": sig.get("recruiter_response_rate", 1.0),
                "notice_period_days": sig.get("notice_period_days", 90),
                
                "interview_completion_rate": sig.get("interview_completion_rate", 1.0),
                "offer_acceptance_rate": sig.get("offer_acceptance_rate", 1.0),
                "github_activity_score": sig.get("github_activity_score", -1.0),
                "saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d", 0),
                
                "profile_completeness": sig.get("profile_completeness_score", 50) / 100.0,
                "search_appearances": sig.get("search_appearance_30d", 0),
                "endorsements": sig.get("endorsements_received", 0),
                "connection_count": sig.get("connection_count", 0),
                
                "avg_job_duration": avg_dur,
                "total_job_duration_months": total_dur,
                "is_pure_consulting": is_pure_consulting,
                "is_pure_research": is_pure_research,
                "advanced_ai_skills": advanced_ai_skills,
                
                "skills_list": [s.get("name", "").lower() for s in skills],
                "titles_list": titles,
                "companies_list": companies,
                "durations_list": [job.get("duration_months", 0) for job in career],
                "graduation_year": c.get("education", [{}])[0].get("graduation_year", 2020) if c.get("education") else 2020,
                
                "career_json": career_json,
                "skills_json": skills_json,
                "combined_text": combined_text
            }
            flattened.append(flat)
            
    df = pl.DataFrame(flattened)
    return df
