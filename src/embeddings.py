"""
src/embeddings.py
Factory module for initializing and managing vector embedding models.
Uses OpenAI text-embedding-3-small by default with robust API key validation.
"""

import logging
from typing import Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
import streamlit as st

from src.config import EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL, get_openai_api_key

logger = logging.getLogger(__name__)


class EmbeddingInitializationError(Exception):
    """Custom exception raised when embedding model initialization fails."""
    pass

@st.cache_resource
def get_embedding_model(
    provider: str = "local",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Embeddings:
    """
    Initializes and returns an Embeddings instance based on provider selection.

    Args:
        provider: 'local' (HuggingFace CPU - 100% Free, zero credits) or 'openai'.
        api_key: Optional OpenAI API key override when provider='openai'.
        model_name: Custom model name override.

    Returns:
        Embeddings: Configured embedding model instance.

    Raises:
        EmbeddingInitializationError: If model initialization or API key check fails.
    """
    cleaned_provider = (provider or "local").lower().strip()

    # ==========================================
    # 1. LOCAL FREE EMBEDDINGS (HuggingFace / CPU)
    # ==========================================
    if cleaned_provider in ["local", "huggingface", "free"]:
        effective_model = model_name or LOCAL_EMBEDDING_MODEL
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info(f"Loading local free HuggingFace embeddings model: {effective_model}...")
            embeddings = HuggingFaceEmbeddings(
                model_name=effective_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f"Successfully loaded local embeddings: {effective_model}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load local HuggingFace embeddings: {str(e)}")
            raise EmbeddingInitializationError(f"Failed to load local embedding model '{effective_model}': {str(e)}")

    # ==========================================
    # 2. OPENAI CLOUD EMBEDDINGS
    # ==========================================
    elif cleaned_provider == "openai":
        effective_api_key = (api_key or get_openai_api_key()).strip()
        if not effective_api_key or "your_openai_api_key" in effective_api_key or "placeholder" in effective_api_key.lower():
            raise EmbeddingInitializationError(
                "Valid OpenAI API key is missing. Please enter your real OpenAI key (sk-...) or switch to 'Local Free' mode in the sidebar."
            )

        effective_model = model_name or EMBEDDING_MODEL
        try:
            embeddings = OpenAIEmbeddings(
                model=effective_model,
                openai_api_key=effective_api_key,
                check_embedding_ctx_length=True
            )
            logger.info(f"Initialized OpenAIEmbeddings model: {effective_model}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIEmbeddings: {str(e)}")
            raise EmbeddingInitializationError(f"Failed to initialize OpenAI embedding model: {str(e)}")

    else:
        raise EmbeddingInitializationError(f"Unsupported embedding provider: '{provider}'. Choose 'local' or 'openai'.")
