from scipy.spatial import distance
from preprocessing import preprocess

# Vectorization - Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


def euclidean_similarity(docA, docB):
    
    docA_clean = preprocess(docA)
    docB_clean = preprocess(docB)

    vectors = vectorizer.fit_transform([docA_clean, docB_clean]).toarray()

    euclidean_dist = distance.euclidean(vectors[0], vectors[1])
    print(f"Euclidean Distance between Document A and B: {euclidean_dist}")

    return euclidean_dist


# Sample documents
docA = "The sky is blue."
docB = "The sun is bright."

euclidean_similarity(docA, docB)

