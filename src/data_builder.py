from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import uuid
import os
import time

# === CONFIGURATION ===
PDF_DIR = "../data/raw"
CHROMA_DB_PATH = "../data/chromadb"
COLLECTION_NAME = "test_pdf_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_pdf_paths(pdf_dir: str) -> list:
    """
    Get all PDF file paths from the specified directory.
    """
    return [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]


def load_and_split_pdfs(pdf_paths: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Load PDF files and split them into smaller text chunks.
    """
    print("1. LOAD AND SPLIT PDFS")
    start_time = time.time()
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    print(f"Loaded and split into {len(all_chunks)} chunks from {len(pdf_paths)} PDFs.")
    print("Time taken:", time.time() - start_time)
    return all_chunks


def prepare_chroma_inputs(chunks: list) -> tuple:
    """
    Prepare documents, metadata, and IDs for ChromaDB insertion.
    """
    print("2. PREPARE DATA FOR CHROMADB")
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    print("Sample Document:", documents[0][:100], "...")
    print("Sample Metadata:", metadatas[0])
    print("Sample ID:", ids[0])
    return documents, metadatas, ids


def insert_into_chromadb(documents: list, metadatas: list, ids: list, db_path: str, collection_name: str):
    """
    Create or get a ChromaDB collection and insert the documents.
    """
    print("3. INSERT INTO CHROMADB")
    start_time = time.time()

    print("Connecting to ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=db_path)

    print(f"Getting or creating collection: '{collection_name}'")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    print("Adding documents to collection...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Successfully added {len(documents)} documents.")
    print("Time taken:", time.time() - start_time)


def main():
    pdf_paths = get_pdf_paths(PDF_DIR)
    if not pdf_paths:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    chunks = load_and_split_pdfs(pdf_paths, CHUNK_SIZE, CHUNK_OVERLAP)
    documents, metadatas, ids = prepare_chroma_inputs(chunks)
    insert_into_chromadb(documents, metadatas, ids, CHROMA_DB_PATH, COLLECTION_NAME)


if __name__ == "__main__":
    main()
