import re

def parse_jd(file_path: str) -> dict:
    """
    Parses the job description markdown to dynamically extract constraints, skills, and semantic queries.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    # 1. Title, Location, Experience
    title_match = re.search(r'#.*?(Job Description:)?\s*(.*)', content, re.IGNORECASE)
    role_title = title_match.group(2).strip() if title_match else "AI Engineer"
    
    loc_match = re.search(r'Location:\s*(.*)', content, re.IGNORECASE)
    location_preferences = loc_match.group(1).strip() if loc_match else ""
    
    exp_match = re.search(r'Experience Required:\s*(\d+)[^\d]*(\d+)?', content, re.IGNORECASE)
    min_yoe = int(exp_match.group(1)) if exp_match else 3
    max_yoe = int(exp_match.group(2)) if exp_match and exp_match.group(2) else 50
    
    # 2. Extract Consulting Firms Disqualifiers (Company Preferences)
    consulting_firms = []
    consult_match = re.search(r'consulting firms\s*\((.*?)\)', content, re.IGNORECASE)
    if consult_match:
        firms_str = consult_match.group(1)
        consulting_firms = {f.strip().lower() for f in firms_str.replace('etc.', '').split(',') if f.strip()}
    else:
        consulting_firms = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'mindtree'}
        
    # 3. Dynamic Skill Extraction
    # Try to find explicit lists, otherwise fallback to NLP/keyword sweeping on 'What you will do'
    required_skills = []
    preferred_skills = []
    
    # Simple explicit section parsers
    req_section = re.search(r'(Required Skills|Requirements|What we need)(.*?)(##|$)', content, re.IGNORECASE | re.DOTALL)
    if req_section:
        bullets = re.findall(r'[-*]\s*(.*)', req_section.group(2))
        required_skills.extend([b.strip().lower() for b in bullets])
        
    pref_section = re.search(r'(Preferred Skills|Nice to have)(.*?)(##|$)', content, re.IGNORECASE | re.DOTALL)
    if pref_section:
        bullets = re.findall(r'[-*]\s*(.*)', pref_section.group(2))
        preferred_skills.extend([b.strip().lower() for b in bullets])
        
    # Fallback to sweeping the text for standard ML terms if no explicit bullet lists
    if not required_skills:
        ai_vocabulary = ['recommendation systems', 'ranking', 'hybrid search', 'evaluating', 'ml pipelines', 'retrieval', 'rag', 'llm', 'nlp', 'python', 'pytorch', 'tensorflow', 'vector databases', 'qdrant', 'milvus']
        for word in ai_vocabulary:
            if word in content.lower():
                required_skills.append(word)
                
    if not preferred_skills:
        pref_vocabulary = ['distributed training', 'rlhf', 'agentic systems', 'langgraph', 'rust', 'c++']
        for word in pref_vocabulary:
            if word in content.lower():
                preferred_skills.append(word)

    # 4. Extract Core Semantic Query
    do_not_want_idx = content.lower().find("things we explicitly do not want")
    if do_not_want_idx != -1:
        positive_content = content[:do_not_want_idx]
    else:
        positive_content = content
        
    semantic_query = re.sub(r'#.*?\n', '', positive_content)
    semantic_query = re.sub(r'[^a-zA-Z0-9\s.,]', '', semantic_query)
    semantic_query = " ".join(semantic_query.split())
    
    if len(semantic_query.strip()) < 10:
        semantic_query = "Experienced AI Engineer building production recommendation systems, ranking algorithms, hybrid search, and evaluating ML pipelines."
        
    return {
        "role_title": role_title,
        "location_preferences": location_preferences,
        "min_yoe": min_yoe,
        "max_yoe": max_yoe,
        "consulting_firms": consulting_firms,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "semantic_query": semantic_query
    }
