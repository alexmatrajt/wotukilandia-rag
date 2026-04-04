from pathlib import Path


# ---- Base paths ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"


# ---- Vector store ----
COLLECTION_NAME = "zephyria_legal_rag"


# ---- Models ----
# Change these later if you choose different Ollama models.
LLM_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"


# ---- Retrieval ----
TOP_K = 5


# ---- Chunking ----
# For your project, we are mainly doing structure-aware chunking,
# but these are still useful fallback settings.
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


def validate_paths() -> None:
    """
    Basic startup validation to catch missing folders early.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    if not CHROMA_DIR.exists():
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
