from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess


# Sample documents
docA = "The sky is blue."
docB = "The sun is bright."

docA_clean = preprocess(docA)
docB_clean = preprocess(docB)


# Vectorization - term frequency-inverse document frequency
vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform([docA_clean, docB_clean])

from sklearn.metrics.pairwise import cosine_similarity

# Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors)
print(f"Cosine Similarity: {cosine_sim}")
