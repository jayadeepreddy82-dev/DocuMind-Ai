"""
src/text_splitter.py
Splits extracted PDF documents into semantically coherent, overlapping text chunks.
Preserves original document metadata (source filename, page number) on every chunk.
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Creates and configures a RecursiveCharacterTextSplitter instance.

    Why Chunk Overlap is Essential in RAG:
    --------------------------------------
    1. Context Preservation: Splitting at arbitrary character boundaries can cut 
       sentences, clauses, or key definitions in half.
    2. Boundary Continuity: An overlap of 200 characters ensures that the semantic meaning 
       spanning across adjacent chunks is not lost during embedding and retrieval.
    3. Retrieval Recall: If a query matches terms positioned near a chunk edge, the overlapping 
       window provides enough surrounding context for the vector embedding to capture the intent.

    Args:
        chunk_size: Maximum character length for each text chunk (default from config).
        chunk_overlap: Number of characters shared between adjacent chunks (default from config).

    Returns:
        RecursiveCharacterTextSplitter: Configured splitter instance.
    """
    effective_chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    # RecursiveCharacterTextSplitter splits by default across ["\n\n", "\n", " ", ""]
    # prioritizing paragraphs, sentences, and words before falling back to character slicing.
    return RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Splits a list of Document objects into smaller chunks while preserving all metadata.

    Args:
        documents: List of LangChain Document objects extracted from PDFs.
        chunk_size: Target character size for chunks.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List[Document]: List of split chunk Document objects with source and page metadata intact.
    """
    if not documents:
        logger.warning("split_documents called with an empty list of documents.")
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Attach chunk indexing metadata for enhanced traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    logger.info(
        f"Split {len(documents)} page documents into {len(chunks)} chunks "
        f"(chunk_size={splitter._chunk_size}, chunk_overlap={splitter._chunk_overlap})."
    )
    return chunks
