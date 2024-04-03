from abc import ABC, abstractmethod

class SimilarityStrategy(ABC):
    @abstractmethod
    def calculate_similarity(self, text1, text2):
        pass
