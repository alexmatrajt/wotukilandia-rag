from typing import Any, Dict, List

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
)
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


RETRIEVAL_SCOPE_TO_DOCUMENT_TYPES = {
    "all_documents": None,
    "legal_authorities": ["constitution", "code", "statute", "regulation", "case_law"],
    "case_materials": ["case_file", "evidence", "legal_memo"],
    "evidence_only": ["evidence"],
    "case_law_only": ["case_law"],
}


def get_embedding_model(provider: str | None = None):
    selected_provider = provider or PROVIDER
    model_name = get_embed_model(selected_provider)

    if selected_provider == "local":
        return OllamaEmbedding(model_name=model_name)

    if selected_provider == "openai":
        return OpenAIEmbedding(model=model_name)

    raise ValueError(f"Unsupported provider: {selected_provider}")


def get_llm(provider: str | None = None):
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


def infer_auto_scope(question: str) -> str:
    """
    Lightweight rule-based router for retrieval scope.
    """
    q = question.lower()

    evidence_keywords = [
        "evidence",
        "log",
        "logs",
        "transcript",
        "report",
        "scan",
        "what happened",
        "show proof",
        "proof",
        "did synaptech",
        "did the client",
        "memory echo",
        "consent record",
    ]

    law_keywords = [
        "law",
        "statute",
        "regulation",
        "constitution",
        "case law",
        "precedent",
        "legal standard",
        "required",
        "allowed",
        "prohibited",
        "violate",
        "violation",
    ]

    strategy_keywords = [
        "argument",
        "arguments",
        "defense",
        "claim",
        "claims",
        "risk",
        "outcome",
        "likely",
        "support the client",
        "support the defendant",
        "strongest",
        "weakness",
        "strength",
    ]

    if any(k in q for k in strategy_keywords):
        return "all_documents"

    if any(k in q for k in evidence_keywords) and not any(k in q for k in law_keywords):
        return "case_materials"

    if any(k in q for k in law_keywords) and not any(k in q for k in evidence_keywords):
        return "legal_authorities"

    return "all_documents"


def build_metadata_filters(retrieval_scope: str, question: str) -> MetadataFilters | None:
    """
    Convert retrieval scope into LlamaIndex metadata filters.
    """
    selected_scope = retrieval_scope
    if selected_scope == "auto":
        selected_scope = infer_auto_scope(question)

    allowed_document_types = RETRIEVAL_SCOPE_TO_DOCUMENT_TYPES.get(selected_scope)

    if not allowed_document_types:
        return None

    return MetadataFilters(
        filters=[
            MetadataFilter(key="document_type", value=doc_type)
            for doc_type in allowed_document_types
        ],
        condition=FilterCondition.OR,
    )


def get_query_engine(
    provider: str | None = None,
    question: str | None = None,
    retrieval_scope: str = "all_documents",
):
    """
    Create a query engine for the selected provider and retrieval scope.
    """
    selected_provider = provider or PROVIDER
    index = load_index(selected_provider)
    llm = get_llm(selected_provider)

    filters = build_metadata_filters(retrieval_scope, question or "")

    retriever = index.as_retriever(
        similarity_top_k=TOP_K,
        filters=filters,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_mode="compact",
    )
    return query_engine


def ask_question(
    question: str,
    provider: str | None = None,
    retrieval_scope: str = "all_documents",
) -> Dict[str, Any]:
    """
    Ask a question against the selected provider's index and return:
    - answer text
    - source metadata
    - retrieved chunk text
    """
    selected_provider = provider or PROVIDER
    effective_scope = retrieval_scope
    if retrieval_scope == "auto":
        effective_scope = infer_auto_scope(question)

    query_engine = get_query_engine(
        provider=selected_provider,
        question=question,
        retrieval_scope=retrieval_scope,
    )
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
        "retrieval_scope": retrieval_scope,
        "effective_scope": effective_scope,
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

        scope = input(
            "Scope (auto/all_documents/legal_authorities/case_materials/evidence_only/case_law_only): "
        ).strip() or "auto"

        result = ask_question(question, PROVIDER, retrieval_scope=scope)

        print("\nANSWER:")
        print(result["answer"])

        print(f"\nRequested scope: {result['retrieval_scope']}")
        print(f"Effective scope: {result['effective_scope']}")

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
