from typing import List

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from app.config import (
    COLLECTION_NAME,
    PROVIDER,
    get_chroma_dir,
    get_embed_model,
)


def get_embedding_model(provider: str | None = None):
    """
    Return the embedding model object for the selected provider.
    """
    selected_provider = provider or PROVIDER
    model_name = get_embed_model(selected_provider)

    if selected_provider == "local":
        return OllamaEmbedding(model_name=model_name)

    if selected_provider == "openai":
        return OpenAIEmbedding(model=model_name)

    raise ValueError(f"Unsupported provider: {selected_provider}")


def build_index(
    chunked_documents: List[Document],
    provider: str | None = None,
) -> VectorStoreIndex:
    """
    Build and persist a Chroma-backed vector index from chunked documents.
    """
    selected_provider = provider or PROVIDER
    chroma_dir = get_chroma_dir(selected_provider)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = get_embedding_model(selected_provider)

    index = VectorStoreIndex.from_documents(
        chunked_documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index


def reset_collection(provider: str | None = None) -> None:
    """
    Delete the existing Chroma collection for the selected provider if it exists.
    """
    selected_provider = provider or PROVIDER
    chroma_dir = get_chroma_dir(selected_provider)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME} ({selected_provider})")
    else:
        print(f"No existing collection found: {COLLECTION_NAME} ({selected_provider})")


if __name__ == "__main__":
    from app.load_documents import load_documents
    from app.chunk_documents import chunk_documents

    print(f"Provider: {PROVIDER}")
    print("Loading raw documents...")
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} raw documents.")

    print("Chunking documents...")
    chunked_docs = chunk_documents(raw_docs)
    print(f"Created {len(chunked_docs)} chunks.")

    print("Resetting existing collection...")
    reset_collection(PROVIDER)

    print("Building index...")
    build_index(chunked_docs, PROVIDER)

    print("\nIndex build complete.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Chroma path: {get_chroma_dir(PROVIDER)}")
