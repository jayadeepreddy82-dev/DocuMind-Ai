"""
src/config.py
Centralized configuration management for DocuMind AI.
Loads environment variables and defines hyperparameters for embedding, chunking, retrieval, and LLM inference.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ==========================================
# MODEL & EMBEDDING CONFIGURATION
# ==========================================
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ==========================================
# CHUNKING & RETRIEVAL CONFIGURATION
# ==========================================
# Size of each chunk in characters
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))

# Overlap between consecutive chunks to preserve contextual continuity
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

# Number of top relevant chunks to retrieve from FAISS
TOP_K: int = int(os.getenv("TOP_K", 4))

# Temperature for LLM response generation (0.0 enforces deterministic, hallucination-free output)
TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.0))

# ==========================================
# RAG PROMPT CONFIGURATION
# ==========================================
RAG_SYSTEM_PROMPT: str = """You are a document question-answering assistant.

Answer the user's question ONLY using the provided context.

If the answer cannot be found in the provided context, respond:
'I could not find this information in the uploaded documents.'

Do not use outside knowledge.
Do not invent facts.
Be clear, concise, and accurate.

Context:
{context}"""

# ==========================================
# API KEY MANAGEMENT
# ==========================================
def get_openai_api_key() -> str:
    """Retrieve OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY", "").strip()

def get_groq_api_key() -> str:
    """Retrieve Groq API key from environment."""
    return os.getenv("GROQ_API_KEY", "").strip()

def get_google_api_key() -> str:
    """Retrieve Google Gemini API key from environment."""
    return os.getenv("GOOGLE_API_KEY", "").strip()
