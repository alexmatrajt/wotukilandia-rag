from app.load_documents import load_documents
from app.chunk_documents import chunk_documents
from app.build_index import build_index, reset_collection
from app.config import PROVIDER, get_chroma_dir


def main() -> None:
    print(f"Provider: {PROVIDER}")
    print(f"Target Chroma directory: {get_chroma_dir(PROVIDER)}\n")

    print("Step 1: Loading raw documents...")
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} raw documents.\n")

    print("Step 2: Chunking documents...")
    chunked_docs = chunk_documents(raw_docs)
    print(f"Created {len(chunked_docs)} chunks.\n")

    print("Step 3: Resetting existing index collection...")
    reset_collection(PROVIDER)
    print()

    print("Step 4: Building vector index...")
    build_index(chunked_docs, PROVIDER)
    print("Index build complete.\n")

    print(f"Done. Your Wotukilandia RAG knowledge base is ready for provider: {PROVIDER}")


if __name__ == "__main__":
    main()
