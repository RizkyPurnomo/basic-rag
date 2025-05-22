import chromadb

chroma_client = chromadb.PersistentClient('../data/chromadb')
collection = chroma_client.get_collection(name='test_pdf_collection')


def retrieve_chunks(query: str, k: int = 10) -> list[str]:
    """
    Retrieve top-k most relevant document chunks from a ChromaDB collection.

    Parameters:
        collection: The ChromaDB collection to search.
        query (str): The user query string.
        k (int): Number of top results to retrieve.

    Returns:
        List of document chunks (strings).
    """
    result = collection.query(query_texts=[query], n_results=k)
    return result.get("documents", [[]])[0]  # Return empty list if nothing found


def remove_newlines(text_list: list[str]) -> list[str]:
    """
    Remove newline characters from each string in a list.

    Parameters:
        text_list (list[str]): List of strings with potential newline characters.

    Returns:
        List of strings without newline characters.
    """
    return [text.replace('\n', ' ') for text in text_list]