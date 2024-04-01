from preprocessing import preprocess

# Sample documents
docA = "The sky is blue."
docB = "The sun is bright."

docA_clean = preprocess(docA)
docB_clean = preprocess(docB)


# Vectorization - Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
# Fit and transform the documents
vectors = vectorizer.fit_transform([docA_clean, docB_clean]).toarray()

# Calculating Euclidean Distance
from scipy.spatial import distance

euclidean_dist = distance.euclidean(vectors[0], vectors[1])
print(f"Euclidean Distance between Document A and B: {euclidean_dist}")
