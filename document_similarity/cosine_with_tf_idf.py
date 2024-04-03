from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess

# Vectorization - term frequency-inverse document frequency
vectorizer = TfidfVectorizer()

def cosine_similarity_with_tf_idf(docA, docB):
    """
    Usage:
    
    docA = "The sky is blue."
    docB = "The sun is bright."

    cosine_similarity_with_tf_idf(docA, docB)
    """
    
    docA_clean = preprocess(docA)
    docB_clean = preprocess(docB)

    tfidf_vectors = vectorizer.fit_transform([docA_clean, docB_clean])

    # Calculate Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors)
    
    return cosine_sim[0][1]

