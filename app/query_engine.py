from typing import Any, Dict, List

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from app.config import (
    APP_NAME,
    COLLECTION_NAME,
    PROVIDER,
    TOP_K,
    get_chroma_dir,
    get_embed_model,
    get_llm_model,
)


SYSTEM_PROMPT = """
You are a legal research assistant for the fictional country Wotukilandia.

You MUST follow these rules strictly:

1. Answer ONLY using the provided retrieved context.
2. Do NOT use outside knowledge.

3. PRIORITIZE EVIDENCE over summaries or claims:
   - Use evidence documents (logs, transcripts, reports) first
   - Do NOT treat case summaries or legal analysis as proof
   - Clearly distinguish between:
     - facts (from evidence)
     - claims (from case file)
     - legal reasoning

4. ALWAYS reference specific documents when making factual claims.
5. When citing, explicitly name:
   - the document title OR
   - the evidence type (e.g., "Synaptech Access Logs", "Diagnostic Session Transcript")

6. Be explicit and concrete:
   - Do NOT say "suggests" unless truly uncertain
   - Clearly explain WHY the evidence supports the conclusion

7. If evidence is missing or inconclusive, say so.
8. Combine multiple pieces of evidence when possible to support conclusions.
9. Retrieved context may contain extra material. Use only the information that is relevant to the user's question. Do not mention, summarize, or cite retrieved content unless it helps answer the question.

Preferred style:
- First: direct answer
- Then: explanation using specific evidence
- Clear, structured, and concise
"""


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


def get_llm(provider: str | None = None):
    """
    Return the LLM object for the selected provider.
    """
    selected_provider = provider or PROVIDER
    model_name = get_llm_model(selected_provider)

    if selected_provider == "local":
        return Ollama(
            model=model_name,
            request_timeout=120.0,
            system_prompt=SYSTEM_PROMPT,
        )

    if selected_provider == "openai":
        return OpenAI(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
        )

    raise ValueError(f"Unsupported provider: {selected_provider}")


def load_index(provider: str | None = None) -> VectorStoreIndex:
    """
    Load the persisted Chroma-backed index for the selected provider.
    """
    selected_provider = provider or PROVIDER
    chroma_dir = get_chroma_dir(selected_provider)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = get_embedding_model(selected_provider)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


def get_query_engine(provider: str | None = None):
    """
    Create a query engine for the selected provider.
    """
    selected_provider = provider or PROVIDER
    index = load_index(selected_provider)
    llm = get_llm(selected_provider)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=TOP_K,
        response_mode="compact",
    )
    return query_engine


def ask_question(question: str, provider: str | None = None) -> Dict[str, Any]:
    """
    Ask a question against the selected provider's index and return:
    - answer text
    - source metadata
    - retrieved chunk text
    """
    selected_provider = provider or PROVIDER
    query_engine = get_query_engine(selected_provider)
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
        "provider": selected_provider,
        "answer": str(response),
        "sources": sources,
    }


if __name__ == "__main__":
    print(f"{APP_NAME} test mode")
    print(f"Provider: {PROVIDER}")
    print("Type a question, or type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = ask_question(question, PROVIDER)
        print(result)

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
