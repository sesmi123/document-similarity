from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from abstract_similarity_startegy import SimilarityStrategy

class CosineSimilarity(SimilarityStrategy):

    # Vectorization - term frequency-inverse document frequency
    vectorizer = TfidfVectorizer()

    def calculate_similarity(self, docA, docB):
        """
        Usage:
        
        docA = "The sky is blue."
        docB = "The sun is bright."

        calculate_similarity(docA, docB)
        """

        tfidf_vectors = self.vectorizer.fit_transform([docA, docB])

        # Calculate Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors)
        
        return cosine_sim[0][1]

