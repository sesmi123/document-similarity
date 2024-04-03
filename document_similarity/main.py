import os
import glob

from preprocessing import preprocess
from jaccard import JaccardSimilarity
from euclidean import EuclideanSimilarity
from cosine_with_tf_idf import CosineSimilarity

def preprocess_and_store_documents():
    document_directory = "document_database"
    documents = list_txt_files(document_directory)
    preprocessed_documents = {}

    for file_path in documents:
        with open(file_path, "r") as f:
            preprocessed_documents[os.path.basename(file_path)] = preprocess(f.read())

    return preprocessed_documents

def list_txt_files(directory):
    return glob.glob(os.path.join(directory, '*.txt'))

def get_most_matching_document(strategy, text):
    matching_document = (None, 0)
    text = preprocess(text)

    for doc, content in doc_store.items():
        result = strategy.calculate_similarity(text, content)
        
        if matching_document[1] < result:
            matching_document = (doc, result)

    return matching_document

def get_user_input_text():
    print("Enter query or document content:")
    return input("> ")

def main():
    strategies = {
        '1': JaccardSimilarity(),
        '2': EuclideanSimilarity(),
        '3': CosineSimilarity(),
    }

    while True:
        print("\nChoose a similarity measure or type 'exit' to quit:")
        print("1. Jaccard's similarity")
        print("2. Euclidean similarity")
        print("3. Cosine similarity")
        choice = input("Enter your choice (1/2/3) or 'exit': ").strip().lower()

        if choice == 'exit':
            print("Exiting the application.")
            break
        elif choice in strategies:
            text = get_user_input_text()
            strategy = strategies[choice]
            result = get_most_matching_document(strategy, text)
            if result[0] is None:
                print("No matching documents found for the given query!!")
            else:
                print(f"Result: Found \"{result[0]}\" document with a match of {result[1]}")
                with open(f"document_database/{result[0]}", "r") as f:
                    print(f.read())
        else:
            print("Invalid choice. Please choose a valid option.")
            continue

if __name__ == "__main__":
    print("Preparing data store...")
    doc_store = preprocess_and_store_documents()
    main()
