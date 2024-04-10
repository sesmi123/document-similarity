import os
import glob
from abstract_similarity_startegy import SimilarityStrategy
from preprocessing import DocumentPreProcessor

class DocumentSimilarityChecker:

    def __init__(self, dir: str, doc_pre_processor: DocumentPreProcessor) -> None:
        self.doc_pre_processor = doc_pre_processor
        self.document_directory = dir
        self.preprocessed_documents = {}
        self._preprocess_and_load_docs_to_memory()

    def _list_txt_files(self, directory):
        return glob.glob(os.path.join(directory, '*.txt'))

    def _preprocess_and_load_docs_to_memory(self):
        documents = self._list_txt_files(self.document_directory)

        for file_path in documents:
            with open(file_path, "r") as f:
                self.preprocessed_documents[os.path.basename(file_path)] = self.doc_pre_processor.preprocess(f.read())

        return self.preprocessed_documents
    
    def get_most_matching_document(self, strategy: SimilarityStrategy, text):
        matching_document = (None, 0)
        text = self.doc_pre_processor.preprocess(text)

        for doc, content in self.preprocessed_documents.items():
            result = strategy.calculate_similarity(text, content)
            
            if matching_document[1] < result:
                matching_document = (doc, result)

        return matching_document