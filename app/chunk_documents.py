import re
from typing import List, Dict, Any

from llama_index.core import Document

from app.config import MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS


SECTION_PATTERN = re.compile(r"^SECTION:\s*(.+)$", re.MULTILINE)
SUBSECTION_PATTERN = re.compile(r"^Subsection\s+([A-Z])\s*:\s*(.+)$", re.MULTILINE)
ARTICLE_PATTERN = re.compile(r"^(Article\s+\d+\.\d+)\s*:\s*(.+)$", re.MULTILINE)


def split_fallback_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Fallback splitter for documents that do not match our legal structure.
    Splits by character count with overlap.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, 0)

    return chunks


def extract_document_header_metadata(text: str) -> Dict[str, str]:
    """
    Extract top-level header fields like TITLE, DOCUMENT_TITLE, CASE_TITLE, VERSION, etc.
    """
    header = {}
    for line in text.splitlines()[:20]:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            if key and value:
                header[key] = value
    return header


def chunk_statute_like_document(doc: Document) -> List[Document]:
    """
    Chunk constitution, statutes, regulations, and codes by ARTICLE,
    while carrying SECTION metadata.
    """
    text = doc.text
    base_metadata = dict(doc.metadata)
    header_metadata = extract_document_header_metadata(text)

    matches = list(ARTICLE_PATTERN.finditer(text))
    if not matches:
        return [
            Document(
                text=chunk,
                metadata={**base_metadata, "section": "", "article": "", "heading": ""}
            )
            for chunk in split_fallback_text(text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
        ]

    chunks: List[Document] = []

    for i, match in enumerate(matches):
        article_label = match.group(1).strip()
        article_heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_block = text[start:end].strip()

        section_name = find_nearest_previous_section(text, start)

        chunk_text = build_chunk_text(
            document_title=base_metadata.get("document_title", ""),
            section=section_name,
            article=article_label,
            heading=article_heading,
            body=article_block,
        )

        chunk_metadata = {
            **base_metadata,
            "section": section_name,
            "article": article_label,
            "heading": article_heading,
            "version": header_metadata.get("VERSION", ""),
            "enacted": header_metadata.get("ENACTED", ""),
            "authority": header_metadata.get("AUTHORITY", header_metadata.get("ISSUING_BODY", "")),
        }

        chunks.append(Document(text=chunk_text, metadata=chunk_metadata))

    return chunks


def chunk_case_document(doc: Document) -> List[Document]:
    """
    Chunk case law by SECTION blocks such as FACTS, ISSUE, HOLDING, REASONING.
    """
    text = doc.text
    base_metadata = dict(doc.metadata)
    header_metadata = extract_document_header_metadata(text)

    sections = split_by_section_blocks(text)
    if not sections:
        return [
            Document(
                text=chunk,
                metadata={**base_metadata, "section": "", "case_title": base_metadata.get("document_title", "")}
            )
            for chunk in split_fallback_text(text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
        ]

    chunks: List[Document] = []

    for section_title, section_body in sections:
        chunk_text = build_chunk_text(
            document_title=base_metadata.get("document_title", ""),
            section=section_title,
            article="",
            heading="",
            body=section_body,
        )

        chunk_metadata = {
            **base_metadata,
            "case_title": header_metadata.get("CASE_TITLE", base_metadata.get("document_title", "")),
            "court": header_metadata.get("COURT", ""),
            "cycle": header_metadata.get("CYCLE", ""),
            "section": section_title,
            "article": "",
            "heading": "",
        }

        chunks.append(Document(text=chunk_text, metadata=chunk_metadata))

    return chunks


def chunk_memo_or_case_file(doc: Document) -> List[Document]:
    """
    Chunk memos and case files by SECTION blocks.
    """
    text = doc.text
    base_metadata = dict(doc.metadata)
    header_metadata = extract_document_header_metadata(text)

    sections = split_by_section_blocks(text)
    if not sections:
        return [
            Document(
                text=chunk,
                metadata={**base_metadata, "section": "", "article": "", "heading": ""}
            )
            for chunk in split_fallback_text(text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
        ]

    chunks: List[Document] = []

    for section_title, section_body in sections:
        subsection_matches = list(SUBSECTION_PATTERN.finditer(section_body))

        if subsection_matches:
            for i, match in enumerate(subsection_matches):
                sub_label = match.group(1).strip()
                sub_heading = match.group(2).strip()
                start = match.start()
                end = subsection_matches[i + 1].start() if i + 1 < len(subsection_matches) else len(section_body)
                subsection_body = section_body[start:end].strip()

                chunk_text = build_chunk_text(
                    document_title=base_metadata.get("document_title", ""),
                    section=section_title,
                    article=f"Subsection {sub_label}",
                    heading=sub_heading,
                    body=subsection_body,
                )

                chunk_metadata = {
                    **base_metadata,
                    "section": section_title,
                    "article": f"Subsection {sub_label}",
                    "heading": sub_heading,
                    "date": header_metadata.get("DATE", header_metadata.get("DATE_OPENED", "")),
                    "author": header_metadata.get("AUTHOR", ""),
                }

                chunks.append(Document(text=chunk_text, metadata=chunk_metadata))
        else:
            chunk_text = build_chunk_text(
                document_title=base_metadata.get("document_title", ""),
                section=section_title,
                article="",
                heading="",
                body=section_body,
            )

            chunk_metadata = {
                **base_metadata,
                "section": section_title,
                "article": "",
                "heading": "",
                "date": header_metadata.get("DATE", header_metadata.get("DATE_OPENED", "")),
                "author": header_metadata.get("AUTHOR", ""),
            }

            chunks.append(Document(text=chunk_text, metadata=chunk_metadata))

    return chunks


def chunk_generic_document(doc: Document) -> List[Document]:
    """
    Generic fallback chunking.
    """
    base_metadata = dict(doc.metadata)
    return [
        Document(
            text=chunk,
            metadata={**base_metadata, "section": "", "article": "", "heading": ""}
        )
        for chunk in split_fallback_text(doc.text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
    ]


def find_nearest_previous_section(text: str, position: int) -> str:
    """
    Find the closest SECTION heading before a given character position.
    """
    previous_section = ""
    for match in SECTION_PATTERN.finditer(text):
        if match.start() < position:
            previous_section = match.group(1).strip()
        else:
            break
    return previous_section


def split_by_section_blocks(text: str) -> List[tuple[str, str]]:
    """
    Split a document into (section_title, section_body) pairs based on SECTION: headings.
    """
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return []

    sections = []
    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_body = text[start:end].strip()
        sections.append((section_title, section_body))

    return sections


def build_chunk_text(
    document_title: str,
    section: str,
    article: str,
    heading: str,
    body: str,
) -> str:
    """
    Build chunk text with contextual headers included.
    This helps retrieval because the chunk carries its own context.
    """
    parts = []
    if document_title:
        parts.append(f"DOCUMENT_TITLE: {document_title}")
    if section:
        parts.append(f"SECTION: {section}")
    if article:
        parts.append(f"ARTICLE: {article}")
    if heading:
        parts.append(f"HEADING: {heading}")
    if body:
        parts.append(body.strip())

    return "\n".join(parts).strip()


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Main entry point. Routes documents to the correct chunking strategy
    based on document_type metadata.
    """
    chunked_docs: List[Document] = []

    for doc in documents:
        doc_type = doc.metadata.get("document_type", "unknown")

        if doc_type in {"constitution", "statute", "regulation", "code"}:
            chunked_docs.extend(chunk_statute_like_document(doc))
        elif doc_type == "case_law":
            chunked_docs.extend(chunk_case_document(doc))
        elif doc_type in {"legal_memo", "case_file", "evidence", "glossary"}:
            chunked_docs.extend(chunk_memo_or_case_file(doc))
        else:
            chunked_docs.extend(chunk_generic_document(doc))

    return chunked_docs


if __name__ == "__main__":
    from app.load_documents import load_documents

    raw_docs = load_documents()
    chunked = chunk_documents(raw_docs)

    print(f"Loaded {len(raw_docs)} raw documents.")
    print(f"Created {len(chunked)} chunks.\n")

    for i, chunk in enumerate(chunked[:10], start=1):
        print(f"CHUNK {i}")
        print("Metadata:", chunk.metadata)
        print("Text:")
        print(chunk.text[:700])
        print("-" * 80)

    # OUTPUT_PATH = "chunk_debug.txt"
    # with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    #     for i, chunk in enumerate(chunked, start=1):
    #         f.write(f"CHUNK {i}\n")
    #         f.write("METADATA:\n")
    #         for k, v in chunk.metadata.items():
    #             f.write(f"  {k}: {v}\n")

    #         f.write("\nTEXT:\n")
    #         f.write(chunk.text)
    #         f.write("\n" + "=" * 80 + "\n\n")

    # print(f"Saved chunks to {OUTPUT_PATH}")
