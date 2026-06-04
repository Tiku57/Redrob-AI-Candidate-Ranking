from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl

def extract_lexical_features(df: pl.DataFrame, semantic_query: str):
    """
    Computes TF-IDF scores for the combined text against a highly targeted JD query.
    """
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    texts = df["combined_text"].to_list()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    query_vec = vectorizer.transform([semantic_query])
    
    return tfidf_matrix, query_vec
