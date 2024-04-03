
def jaccard_similarity(doc1, doc2):
    """
    Example usage:
    doc1 = "the house had a tiny little mouse"
    doc2 = "the cat saw the mouse in the little house"
    similarity_score = jaccard_similarity(doc1, doc2)

    Disadvantage:
    As the size of the document increases, the number of common words will increase, 
    even though the two documents are semantically different.
    """

    words_doc1 = set(doc1.split())
    words_doc2 = set(doc2.split())
    
    # Find the intersection and union of the two sets
    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2)
    
    # Calculate Jaccard Similarity
    similarity = len(intersection) / len(union)
    return similarity

