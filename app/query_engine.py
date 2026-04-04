from typing import Any, Dict, List

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from app.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, LLM_MODEL, TOP_K


SYSTEM_PROMPT = """
You are a legal research assistant for the fictional country Zephyria.

Rules:
1. Answer only from the provided retrieved context.
2. Do not rely on outside knowledge or Earth legal systems.
3. If the answer is not clearly supported by the retrieved context, say so.
4. Prefer precise references to document title, section, article, or case section when available.
5. Be concise but clear.
"""


def load_index() -> VectorStoreIndex:
    """
    Load the persisted Chroma-backed index.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


def get_query_engine():
    """
    Create a query engine with the local Ollama LLM.
    """
    index = load_index()

    llm = Ollama(
        model=LLM_MODEL,
        request_timeout=120.0,
        system_prompt=SYSTEM_PROMPT,
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=TOP_K,
        response_mode="compact",
    )
    return query_engine


def ask_question(question: str) -> Dict[str, Any]:
    """
    Ask a question against the index and return:
    - answer text
    - source metadata
    - retrieved chunk text
    """
    query_engine = get_query_engine()
    response = query_engine.query(question)

    sources: List[Dict[str, Any]] = []
    for node in response.source_nodes:
        metadata = dict(node.metadata) if node.metadata else {}
        sources.append(
            {
                "score": getattr(node, "score", None),
                "metadata": metadata,
                "text": node.text,
            }
        )

    return {
        "question": question,
        "answer": str(response),
        "sources": sources,
    }


if __name__ == "__main__":
    print("Zephyria RAG test mode")
    print("Type a question, or type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = ask_question(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for i, source in enumerate(result["sources"], start=1):
            meta = source["metadata"]
            print(f"\n[{i}] score={source['score']}")
            print(f"source_file: {meta.get('source_file', '')}")
            print(f"document_title: {meta.get('document_title', '')}")
            print(f"document_type: {meta.get('document_type', '')}")
            print(f"section: {meta.get('section', '')}")
            print(f"article: {meta.get('article', '')}")
            print(f"heading: {meta.get('heading', '')}")
            print("text preview:")
            print(source["text"][:500])
