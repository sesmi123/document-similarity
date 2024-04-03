import os
import glob

from jaccard import jaccard_similarity
from euclidean import euclidean_similarity
from cosine_with_tf_idf import cosine_similarity_with_tf_idf

def list_txt_files(directory):
    return glob.glob(os.path.join(directory, '*.txt'))

def get_most_matching_document(choice, text):
    matching_document = (None, 0)
    document_directory = "document_database"
    documents = list_txt_files(document_directory)

    for file_path in documents:
        with open(file_path, "r") as f:
            doc_content = f.read()
            
            if choice == '1':
                result = jaccard_similarity(text, doc_content)
            elif choice == '2':
                result = euclidean_similarity(text, doc_content)
            elif choice == '3':
                result = cosine_similarity_with_tf_idf(text, doc_content)
            
            if matching_document[1] < result:
                matching_document = (os.path.basename(file_path), result)

    return matching_document

def get_user_input_text():
    print("Enter query or document content:")
    return input("> ")

def main():
    while True:
        print("\nChoose a similarity measure or type 'exit' to quit:")
        print("1. Jaccard's similarity")
        print("2. Euclidean similarity")
        print("3. Cosine similarity")
        choice = input("Enter your choice (1/2/3) or 'exit': ").strip().lower()

        if choice == 'exit':
            print("Exiting the application.")
            break
        elif choice in ['1', '2', '3']:
            text = get_user_input_text()
            result = get_most_matching_document(choice, text)
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
    main()