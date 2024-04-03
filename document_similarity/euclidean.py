from scipy.spatial import distance
from preprocessing import preprocess

# Vectorization - Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


def euclidean_similarity(docA, docB):
    """
    Usage:
    
    docA = "The sky is blue."
    docB = "The sun is bright."

    euclidean_similarity(docA, docB)

    Disadvantage:

    Euclidean distance is sensitive to the magnitude of the vectors, 
    which can be a significant issue in document similarity tasks. 
    Documents of different lengths can lead to large Euclidean distances, 
    even if the documents are topically similar. 
    This is because longer documents may have higher word counts overall, 
    affecting the magnitude of their vector representations.
    """
    
    docA_clean = preprocess(docA)
    docB_clean = preprocess(docB)

    vectors = vectorizer.fit_transform([docA_clean, docB_clean]).toarray()

    euclidean_dist = distance.euclidean(vectors[0], vectors[1])

    return euclidean_dist
