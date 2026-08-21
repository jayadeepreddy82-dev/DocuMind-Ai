"""
src/vector_store.py
Manages FAISS vector database indexing and similarity search.
Supports building vector indexes from document chunks and querying top-K relevant passages.
Designed with hooks for easy future addition of rerankers (e.g., Cohere, Cross-Encoders).
"""

import logging
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.config import TOP_K

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception raised when vector store creation or search fails."""
    pass


def create_vector_store(
    chunks: List[Document],
    embeddings: Embeddings
) -> FAISS:
    """
    Constructs an in-memory FAISS vector store from document chunks.

    Args:
        chunks: List of Document chunks to embed and index.
        embeddings: Initialized LangChain Embeddings model.

    Returns:
        FAISS: Indexed FAISS vector store instance.

    Raises:
        VectorStoreError: If chunks list is empty or indexing fails.
    """
    if not chunks:
        raise VectorStoreError("Cannot create vector store: No document chunks provided.")

    try:
        logger.info(f"Generating embeddings and indexing {len(chunks)} chunks in FAISS...")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        logger.info("FAISS vector store successfully created and indexed.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {str(e)}")
        raise VectorStoreError(f"Vector store indexing failed: {str(e)}")


def get_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: Optional[int] = None,
    with_scores: bool = False
) -> List[Document]:
    """
    Performs semantic similarity search against the FAISS vector database.

    Args:
        vector_store: Active FAISS vector store.
        query: User's input question or search query.
        top_k: Number of top relevant documents to retrieve (defaults to TOP_K).
        with_scores: If True, attaches L2 similarity distance score to metadata.

    Returns:
        List[Document]: Top-K most relevant document chunks.

    Raises:
        VectorStoreError: If retrieval fails.
    """
    if vector_store is None:
        raise VectorStoreError("Vector store is not initialized. Please process documents first.")

    k = top_k if top_k is not None else TOP_K
    cleaned_query = query.strip()
    if not cleaned_query:
        logger.warning("Empty query passed to vector store search.")
        return []

    try:
        logger.info(f"Executing semantic similarity search for query (Top-K={k})...")
        
        if with_scores:
            # similarity_search_with_score returns List[Tuple[Document, float]]
            docs_and_scores: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(
                query=cleaned_query,
                k=k
            )
            retrieved_docs: List[Document] = []
            for doc, score in docs_and_scores:
                doc.metadata["similarity_score"] = float(score)
                retrieved_docs.append(doc)
        else:
            retrieved_docs = vector_store.similarity_search(
                query=cleaned_query,
                k=k
            )

        # Hook for future reranking pipeline (e.g. FlashRank, Cohere Rerank, or Cross-Encoder)
        reranked_docs = apply_reranking_hook(retrieved_docs, cleaned_query)
        logger.info(f"Retrieved {len(reranked_docs)} relevant chunks from FAISS.")
        return reranked_docs

    except Exception as e:
        logger.error(f"Failed to query vector store: {str(e)}")
        raise VectorStoreError(f"Retrieval error: {str(e)}")


def apply_reranking_hook(
    documents: List[Document],
    query: str
) -> List[Document]:
    """
    Extensibility Hook: Allows integrating re-ranking models in the future.
    Currently acts as a pass-through returning the retrieved documents as-is.

    Args:
        documents: Documents retrieved from vector search.
        query: The user query.

    Returns:
        List[Document]: Reranked document list.
    """
    # Architecture hook: Cross-encoders or LLM-based rerankers can be plugged here
    return documents
