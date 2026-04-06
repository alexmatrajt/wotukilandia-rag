from pathlib import Path


# ---- Base paths ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_BASE_DIR = BASE_DIR / "chroma_db"


# ---- App identity ----
APP_NAME = "Wotukilandia Legal RAG"


# ---- Provider mode ----
# Options:
#   "local"  -> Ollama for embeddings + chat
#   "openai" -> OpenAI for embeddings + chat
PROVIDER = "local"
# PROVIDER = "openai"


# ---- Vector store ----
COLLECTION_NAME = "wotukilandia_legal_rag"

# Keep separate vector DB folders because embeddings from different
# providers/models should not share the same index.
CHROMA_DIRS = {
    "local": CHROMA_BASE_DIR / "local",
    "openai": CHROMA_BASE_DIR / "openai",
}


# ---- Local / Ollama models ----
LOCAL_LLM_MODEL = "llama3.1:8b"
LOCAL_EMBED_MODEL = "nomic-embed-text"


# ---- OpenAI models ----
OPENAI_LLM_MODEL = "gpt-4.1-nano"
OPENAI_EMBED_MODEL = "text-embedding-3-small"


# ---- Retrieval ----
TOP_K = 5


# ---- Chunking ----
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150


# ---- Supported folders and inferred document types ----
FOLDER_DOCUMENT_TYPES = {
    "constitution": "constitution",
    "codes": "code",
    "statutes": "statute",
    "regulations": "regulation",
    "cases": "case_law",
    "memos": "legal_memo",
    "case_files": "case_file",
    "evidence": "evidence",
    "glossary": "glossary",
}


# ---- File loading ----
SUPPORTED_EXTENSIONS = [".txt"]


def get_chroma_dir(provider: str | None = None) -> Path:
    """
    Return the Chroma directory for the selected provider.
    """
    selected_provider = provider or PROVIDER

    if selected_provider not in CHROMA_DIRS:
        raise ValueError(
            f"Unsupported provider: {selected_provider}. "
            f"Expected one of: {list(CHROMA_DIRS.keys())}"
        )

    return CHROMA_DIRS[selected_provider]


def get_llm_model(provider: str | None = None) -> str:
    """
    Return the chat model name for the selected provider.
    """
    selected_provider = provider or PROVIDER

    if selected_provider == "local":
        return LOCAL_LLM_MODEL
    if selected_provider == "openai":
        return OPENAI_LLM_MODEL

    raise ValueError(f"Unsupported provider: {selected_provider}")


def get_embed_model(provider: str | None = None) -> str:
    """
    Return the embedding model name for the selected provider.
    """
    selected_provider = provider or PROVIDER

    if selected_provider == "local":
        return LOCAL_EMBED_MODEL
    if selected_provider == "openai":
        return OPENAI_EMBED_MODEL

    raise ValueError(f"Unsupported provider: {selected_provider}")


def validate_paths() -> None:
    """
    Basic startup validation to catch missing folders early.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    if not CHROMA_BASE_DIR.exists():
        CHROMA_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for chroma_dir in CHROMA_DIRS.values():
        if not chroma_dir.exists():
            chroma_dir.mkdir(parents=True, exist_ok=True)
