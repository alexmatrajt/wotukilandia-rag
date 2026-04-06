from pathlib import Path
from typing import List, Dict, Any

from llama_index.core import Document

from app.config import DATA_DIR, FOLDER_DOCUMENT_TYPES, SUPPORTED_EXTENSIONS, validate_paths


def extract_title(text: str, fallback_name: str) -> str:
    """
    Try to extract a document title from the first few lines.
    Falls back to a cleaned filename if no title field is found.
    """
    for line in text.splitlines()[:10]:
        line = line.strip()
        if line.startswith("TITLE:") or line.startswith("DOCUMENT_TITLE:") or line.startswith("CASE_TITLE:"):
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()

    return fallback_name.replace("_", " ").replace(".txt", "").title()


def infer_document_type(top_level_folder_name: str) -> str:
    """
    Infer document type from the top-level folder name under data/.
    """
    return FOLDER_DOCUMENT_TYPES.get(top_level_folder_name, "unknown")


def load_single_file(file_path: Path) -> Document:
    """
    Load one text file and return it as a LlamaIndex Document
    with basic file-level metadata.

    Supports nested folders, such as:
    data/evidence/orin_tal_v_synaptech_dynamics/access_logs.txt
    """
    text = file_path.read_text(encoding="utf-8").strip()

    relative_path = file_path.relative_to(DATA_DIR)
    path_parts = relative_path.parts

    # Example:
    # ("evidence", "orin_tal_v_synaptech_dynamics", "access_logs.txt")
    top_level_folder = path_parts[0]
    source_file = file_path.name
    document_type = infer_document_type(top_level_folder)
    title = extract_title(text, file_path.stem)

    subfolder = ""
    if len(path_parts) > 2:
        subfolder = "/".join(path_parts[1:-1])

    metadata: Dict[str, Any] = {
        "source_file": source_file,
        "document_title": title,
        "document_type": document_type,
        "folder": top_level_folder,
        "subfolder": subfolder,
        "file_path": str(relative_path),
    }

    return Document(text=text, metadata=metadata)


def load_documents() -> List[Document]:
    """
    Load all supported files under the data directory, recursively.
    Returns a list of raw LlamaIndex Documents.
    """
    validate_paths()

    documents: List[Document] = []

    for file_path in sorted(DATA_DIR.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        doc = load_single_file(file_path)
        documents.append(doc)

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.\n")

    for doc in docs:
        print(f"- {doc.metadata['source_file']}")
        print(f"  title: {doc.metadata['document_title']}")
        print(f"  type: {doc.metadata['document_type']}")
        print(f"  folder: {doc.metadata['folder']}")
        print(f"  subfolder: {doc.metadata['subfolder']}")
        print(f"  file_path: {doc.metadata['file_path']}")
        print()
