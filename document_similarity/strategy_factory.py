from abstract_similarity_startegy import SimilarityStrategy
from jaccard import JaccardSimilarity
from euclidean import EuclideanSimilarity
from cosine_with_tf_idf import CosineSimilarity

class StrategyFactory:
    def __call__(self, choice: str) -> SimilarityStrategy:
        strategies = {
            '1': JaccardSimilarity(),
            '2': EuclideanSimilarity(),
            '3': CosineSimilarity(),
        }
        return strategies.get(choice)