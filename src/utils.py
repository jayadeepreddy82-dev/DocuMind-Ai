"""
src/utils.py
Helper functions and presentation utilities for DocuMind AI.
Includes statistics computation, citation formatting for UI, and input validation.
"""

import re
from typing import List, Dict, Any, Set
from langchain_core.documents import Document
from src.rag_pipeline import SourceCitation


def calculate_document_stats(
    raw_documents: List[Document],
    chunks: List[Document]
) -> Dict[str, Any]:
    """
    Computes summary statistics for the processed documents and generated chunks.

    Args:
        raw_documents: List of page-level Document objects.
        chunks: List of split chunk Document objects.

    Returns:
        Dict[str, Any]: Dictionary with count metrics (num_files, num_pages, num_chunks, file_names).
    """
    unique_sources: Set[str] = set()
    total_pages = len(raw_documents)

    for doc in raw_documents:
        source_name = doc.metadata.get("source", "Unknown Document")
        unique_sources.add(source_name)

    total_characters = sum(len(c.page_content) for c in chunks)
    avg_chunk_length = round(total_characters / len(chunks), 1) if chunks else 0

    return {
        "num_files": len(unique_sources),
        "num_pages": total_pages,
        "num_chunks": len(chunks),
        "file_names": sorted(list(unique_sources)),
        "total_characters": total_characters,
        "avg_chunk_length": avg_chunk_length
    }


def format_citation_markdown(citation: SourceCitation, index: int) -> str:
    """
    Formats a single SourceCitation into a clean, modern GitHub-style Markdown card.

    Args:
        citation: SourceCitation instance.
        index: Citation display index (1-based).

    Returns:
        str: Formatted markdown string.
    """
    snippet_clean = citation.content_snippet.replace("\n", " ").strip()
    
    # Truncate overly long snippets if needed, maintaining readable sentence ends
    if len(snippet_clean) > 400:
        snippet_clean = snippet_clean[:397] + "..."

    return f"""**Source {index}**
- **📄 Document:** `{citation.source}`
- **📑 Page:** `{citation.page}`

> *"{snippet_clean}"*
"""


def validate_api_key_format(api_key: str) -> bool:
    """
    Validates whether an OpenAI API key conforms to standard naming patterns.

    Args:
        api_key: The API key string.

    Returns:
        bool: True if key appears valid, non-empty, and not a placeholder.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    stripped = api_key.strip()
    # Reject placeholders
    if "your_openai_api_key" in stripped or "placeholder" in stripped.lower():
        return False
    # OpenAI keys start with sk- or sk-proj- and have sufficient length
    if stripped.startswith("sk-") and len(stripped) >= 20:
        return True
    return False


def sanitize_filename(filename: str) -> str:
    """
    Cleans a filename to remove dangerous path traversal characters.

    Args:
        filename: Original filename.

    Returns:
        str: Safe filename.
    """
    return re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", filename)
