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

def infer_and_find_similar_using_doc2vec(model, new_document_text):
    # Tokenize the new document in the same way as your training documents
    new_document_words = word_tokenize(new_document_text.lower())
    inferred_vector = model.infer_vector(new_document_words)
    similar_documents = model.dv.most_similar([inferred_vector], topn=1)
    return similar_documents[0]

if __name__ == "__main__":
    documents = read_and_preprocess("document_database_ml_training")
    doc2vec_model = train_doc2vec(documents)
