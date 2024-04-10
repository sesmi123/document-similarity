from gensim.models import Doc2Vec
from document_similarity_checker import DocumentSimilarityChecker

from preprocessing import DocumentPreProcessor
from strategy_factory import StrategyFactory
from train_doc2vec_model import infer_and_find_similar_using_doc2vec


def display_results(result):
    if result is None or result[0] is None:
        print("No matching documents found for the given query!!")
    else:
        print(f"Result: Found \"{result[0]}\" document with a match of {result[1]}")
        with open(f"document_database/{result[0]}", "r") as f:
            print(f.read())

def get_user_input_text():
    print("Enter query or document content:")
    return input("> ")

def main():

    while True:

        print("\nChoose a similarity measure or type 'exit' to quit:")
        print("1. Jaccard's similarity")
        print("2. Euclidean similarity")
        print("3. Cosine similarity")
        print("4. Doc2Vec model")

        choice = input("Enter your choice (1/2/3/4) or 'exit': ").strip().lower()

        if choice == 'exit':
            print("Exiting the application.")
            break

        strategy = StrategyFactory()(choice)

        if strategy:
            text = get_user_input_text()
            result = doc_sim_checker.get_most_matching_document(strategy, text)
            display_results(result)
                    
        elif choice == '4':
            text = get_user_input_text()
            result = infer_and_find_similar_using_doc2vec(model, text)
            if result is None:
                print("No matching documents found for the given query!!")
            else:
                print(f"Result: Found \"{result[0]}\" document with a match of {result[1]}")
                with open(f"document_database_ml_training/{result[0]}", "r") as f:
                    print(f.read())
        else:
            print("Invalid choice. Please choose a valid option.")

if __name__ == "__main__":
    dir = "document_database"
    preprocessor = DocumentPreProcessor()
    doc_sim_checker = DocumentSimilarityChecker(dir, preprocessor)
    model = Doc2Vec.load("doc2vec_model")
    main()
