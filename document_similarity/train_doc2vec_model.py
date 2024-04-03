import os
import glob
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize

def read_and_preprocess(directory):
    documents = []
    for i, file_path in enumerate(glob.glob(os.path.join(directory, '*.txt'))):
        with open(file_path, 'r') as file:
            document = word_tokenize(file.read().lower())
            # Tag each document with an ID
            documents.append(TaggedDocument(document, [os.path.basename(file_path)]))
    return documents


def train_doc2vec(documents):
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("doc2vec_model")
    return model

if __name__ == "__main__":
    documents = read_and_preprocess("document_database_ml_training")
    doc2vec_model = train_doc2vec(documents)
