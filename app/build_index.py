from typing import List

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from app.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


def build_index(chunked_documents: List[Document]) -> VectorStoreIndex:
    """
    Build and persist a Chroma-backed vector index from chunked documents.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    index = VectorStoreIndex.from_documents(
        chunked_documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index


def reset_collection() -> None:
    """
    Delete the existing Chroma collection if it exists.
    Useful when rebuilding the index from scratch.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    else:
        print(f"No existing collection found: {COLLECTION_NAME}")


if __name__ == "__main__":
    from app.load_documents import load_documents
    from app.chunk_documents import chunk_documents

    print("Loading raw documents...")
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} raw documents.")

    print("Chunking documents...")
    chunked_docs = chunk_documents(raw_docs)
    print(f"Created {len(chunked_docs)} chunks.")

    print("Resetting existing collection...")
    reset_collection()

    print("Building index...")
    build_index(chunked_docs)

    print("\nIndex build complete.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Chroma path: {CHROMA_DIR}")
