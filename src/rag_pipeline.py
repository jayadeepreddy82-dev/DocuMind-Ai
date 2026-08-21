"""
src/rag_pipeline.py
Retrieval-Augmented Generation (RAG) execution pipeline.
Combines semantic retrieval from FAISS with deterministic LLM answer synthesis.
Ensures zero hallucinations with strict context grounding and generates source citations.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from src.config import (
    LLM_MODEL,
    TEMPERATURE,
    TOP_K,
    RAG_SYSTEM_PROMPT,
    get_openai_api_key,
)
from src.vector_store import get_relevant_chunks

logger = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    """Represents a single document source citation."""
    source: str
    page: int
    content_snippet: str
    chunk_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGResponse:
    """Represents the complete result of a RAG query execution."""
    answer: str
    sources: List[SourceCitation]
    retrieved_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "retrieved_count": self.retrieved_count,
        }


class RAGPipelineError(Exception):
    """Custom exception raised when RAG execution fails."""
    pass


def format_context_docs(docs: List[Document]) -> str:
    """
    Formats a list of retrieved Document objects into a clean contextual string for LLM injection.

    Args:
        docs: List of retrieved Document chunks.

    Returns:
        str: Cleanly structured context text block.
    """
    if not docs:
        return "No relevant context found in documents."

    formatted_blocks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        formatted_blocks.append(
            f"--- Document Chunk {idx} [File: {source}, Page: {page}] ---\n{content}"
        )

    return "\n\n".join(formatted_blocks)


def extract_source_citations(docs: List[Document]) -> List[SourceCitation]:
    """
    Extracts deduplicated and structured source citations from retrieved documents.

    Args:
        docs: List of retrieved Document chunks.

    Returns:
        List[SourceCitation]: List of citation objects.
    """
    citations: List[SourceCitation] = []
    
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown Document")
        page_num = doc.metadata.get("page", 1)
        chunk_idx = doc.metadata.get("chunk_index")
        
        # Clean and format snippet for citation display
        snippet = doc.page_content.strip()

        citations.append(
            SourceCitation(
                source=source_name,
                page=int(page_num) if isinstance(page_num, (int, str)) and str(page_num).isdigit() else 1,
                content_snippet=snippet,
                chunk_index=chunk_idx
            )
        )

    return citations


def get_llm(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None
):
    """
    Initializes a chat model instance based on provider selection.

    Args:
        provider: 'openai', 'groq', or 'google'.
        api_key: Optional API key override.
        model_name: Model identifier.
        temperature: Sampling temperature (defaults to TEMPERATURE: 0.0).

    Returns:
        BaseChatModel: Configured chat model instance.

    Raises:
        RAGPipelineError: If API key is missing or initialization fails.
    """
    cleaned_provider = (provider or "openai").lower().strip()
    temp = temperature if temperature is not None else TEMPERATURE

    # 1. OPENAI
    if cleaned_provider == "openai":
        effective_key = (api_key or get_openai_api_key()).strip()
        if not effective_key or "your_openai_api_key" in effective_key or "placeholder" in effective_key.lower():
            raise RAGPipelineError(
                "Valid OpenAI API key is missing. Please enter your key in the sidebar, or select a free LLM provider."
            )
        return ChatOpenAI(
            model=model_name or LLM_MODEL,
            temperature=temp,
            openai_api_key=effective_key,
            max_retries=2,
        )

    # 2. GROQ (FREE TIER)
    elif cleaned_provider in ["groq", "groq_free"]:
        from src.config import get_groq_api_key
        effective_key = (api_key or get_groq_api_key()).strip()
        if not effective_key:
            raise RAGPipelineError(
                "Groq API key is missing. Get a free key at https://console.groq.com and enter it in the sidebar."
            )
        from langchain_groq import ChatGroq
        effective_model = model_name or "openai/gpt-oss-20b"
        return ChatGroq(
            model=effective_model,
            temperature=temp,
            groq_api_key=effective_key,
            max_retries=2,
        )

    # 3. GOOGLE GEMINI (FREE TIER)
    elif cleaned_provider in ["google", "gemini", "google_genai"]:
        from src.config import get_google_api_key
        effective_key = (api_key or get_google_api_key()).strip()
        if not effective_key:
            raise RAGPipelineError(
                "Google Gemini API key is missing. Get a free key at https://aistudio.google.com and enter it in the sidebar."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            temperature=temp,
            google_api_key=effective_key,
            max_retries=2,
        )

    else:
        raise RAGPipelineError(f"Unsupported LLM provider: '{provider}'. Choose 'openai', 'groq', or 'google'.")


def execute_rag_pipeline(
    query: str,
    vector_store: FAISS,
    api_key: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    top_k: Optional[int] = None,
) -> RAGResponse:
    """
    Executes the complete end-to-end RAG pipeline for a given user query.

    Steps:
    1. Retrieve top-K relevant chunks using semantic search against FAISS.
    2. Format the retrieved chunks into a bounded context.
    3. Build a strict prompt instructing the LLM to answer ONLY from context.
    4. Call the LLM with temperature=0 for deterministic inference.
    5. Construct structured response with answer and source citations.

    Args:
        query: User's question.
        vector_store: Initialized FAISS vector store.
        api_key: Optional API key override.
        llm_provider: 'openai', 'groq', or 'google'.
        llm_model: Custom LLM model name.
        top_k: Number of chunks to retrieve.

    Returns:
        RAGResponse: Object containing generated answer and source citations.

    Raises:
        RAGPipelineError: If retrieval or LLM execution fails.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return RAGResponse(
            answer="Please enter a valid question.",
            sources=[],
            retrieved_count=0
        )

    if vector_store is None:
        raise RAGPipelineError("No documents have been processed. Please upload and process PDFs first.")

    try:
        # Step 1: Semantic Retrieval
        effective_k = top_k if top_k is not None else TOP_K
        retrieved_docs = get_relevant_chunks(
            vector_store=vector_store,
            query=cleaned_query,
            top_k=effective_k
        )

        if not retrieved_docs:
            logger.info("No matching document chunks found in vector store.")
            return RAGResponse(
                answer="I could not find this information in the uploaded documents.",
                sources=[],
                retrieved_count=0
            )

        # Step 2: Build Context & Citations
        context_text = format_context_docs(retrieved_docs)
        citations = extract_source_citations(retrieved_docs)

        # Step 3: Compose Prompt using modern LangChain ChatPromptTemplate
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        # Step 4: Initialize LLM and construct LCEL Chain
        llm = get_llm(provider=llm_provider, api_key=api_key, model_name=llm_model)
        chain = prompt_template | llm | StrOutputParser()

        # Step 5: Invoke Chain
        model_tag = getattr(llm, "model_name", getattr(llm, "model", llm_provider))
        logger.info(f"Invoking LLM ({model_tag}) with grounded RAG context...")
        answer = chain.invoke({
            "context": context_text,
            "question": cleaned_query
        })

        return RAGResponse(
            answer=answer.strip(),
            sources=citations,
            retrieved_count=len(retrieved_docs)
        )

    except Exception as e:
        logger.error(f"RAG Pipeline execution failed: {str(e)}")
        # If already a custom error, re-raise directly
        if isinstance(e, RAGPipelineError):
            raise e
        raise RAGPipelineError(f"Error processing your question: {str(e)}")
