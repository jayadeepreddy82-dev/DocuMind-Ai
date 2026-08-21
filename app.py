"""
app.py
DocuMind AI – RAG-Based PDF Document Search & Chatbot
Streamlit Web Application Entrypoint.

Provides a clean, intuitive, modern web interface for uploading multi-page PDF documents,
processing embeddings into a FAISS vector index, querying via RAG, and displaying
exact source citations (filename, page number, context snippet) with zero hallucinations.
"""

import logging
import streamlit as st

from src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    TEMPERATURE,
    get_openai_api_key,
)
from src.document_loader import load_pdf_documents, PDFProcessingError
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model, EmbeddingInitializationError
from src.vector_store import create_vector_store, VectorStoreError
from src.rag_pipeline import execute_rag_pipeline, RAGPipelineError, SourceCitation
from src.utils import calculate_document_stats, format_citation_markdown

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, production-grade styling
st.markdown("""
<style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header Card Styling */
    .hero-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FFFFFF;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    .badge-tag {
        display: inline-block;
        background: #0284C7;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        margin-left: 0.75rem;
        vertical-align: middle;
    }
    
    /* Metric & Stat Cards */
    .metric-card {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Welcome / Empty State Card */
    .welcome-card {
        background: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .welcome-card h3 {
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .welcome-card p {
        color: #64748B;
        font-size: 1rem;
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
    }
    .feature-grid {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .feature-item {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem 1rem;
        width: 230px;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
    .feature-title {
        font-weight: 700;
        color: #0F172A;
        font-size: 0.95rem;
    }
    .feature-desc {
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.4;
    }

    /* Citation Box */
    .citation-container {
        background: #F8FAFC;
        border-left: 3px solid #0284C7;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_session_state():
    """Initializes Streamlit session state variables to preserve context across reruns."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = {}
    if "processed_file_names" not in st.session_state:
        st.session_state.processed_file_names = []


init_session_state()


# ==========================================
# SIDEBAR UI & CONTROLS
# ==========================================
def render_sidebar():
    """Renders sidebar controls, provider selectors, PDF uploaders, and document statistics."""
    with st.sidebar:
        st.markdown("## 📄 DocuMind AI")
        st.markdown(
            "AI-powered PDF search using **Retrieval-Augmented Generation (RAG)** and semantic vector retrieval."
        )
        st.markdown("---")

        # 1. EMBEDDING ENGINE SELECTION
        st.markdown("### 🧠 Vector Embeddings")
        embedding_option = st.radio(
            "Embedding Provider:",
            options=["🖥️ Local Free (CPU / all-MiniLM-L6-v2)", "🌐 OpenAI (text-embedding-3-small)"],
            index=0,
            help="Local Free runs 100% on your machine with zero API keys and zero cost!"
        )
        is_local_embeddings = "Local Free" in embedding_option
        embedding_provider = "local" if is_local_embeddings else "openai"

        # 2. LLM GENERATION PROVIDER
        st.markdown("### 🤖 LLM Generation Provider")
        llm_option = st.selectbox(
            "Answer Generation Model:",
            options=[
                "⚡ Groq – LLaMA 3.1 8B (Fastest & Free)",
                "✨ Google Gemini 1.5 Flash (Free API Key)",
                "🌐 OpenAI – GPT-4o-mini (Paid Credits)"
            ],
            index=0,
            help="Select your preferred model provider. Groq and Google offer generous free tiers!"
        )

        llm_provider = "groq" if "Groq" in llm_option else ("google" if "Gemini" in llm_option else "openai")
        if llm_provider == "groq":
            selected_llm_model = "openai/gpt-oss-20b"
        elif llm_provider == "google":
            selected_llm_model = "gemini-1.5-flash"
        else:
            selected_llm_model = "gpt-4o-mini"

        # 3. DYNAMIC API KEY INPUT
        api_key_to_use = ""

        if llm_provider == "groq":
            from src.config import get_groq_api_key
            raw_groq = get_groq_api_key()
            groq_input = st.text_input(
                "Groq API Key (100% Free):",
                value=raw_groq if raw_groq and not "your_" in raw_groq else "",
                type="password",
                placeholder="gsk_...",
                help="Get a free instant API key at https://console.groq.com"
            )
            api_key_to_use = groq_input.strip() or raw_groq
            st.caption("👉 Get a free instant Groq key: [console.groq.com](https://console.groq.com)")

        elif llm_provider == "google":
            from src.config import get_google_api_key
            raw_gemini = get_google_api_key()
            gemini_input = st.text_input(
                "Google Gemini API Key (100% Free):",
                value=raw_gemini if raw_gemini and not "your_" in raw_gemini else "",
                type="password",
                placeholder="AIzaSy...",
                help="Get a free key at https://aistudio.google.com"
            )
            api_key_to_use = gemini_input.strip() or raw_gemini
            st.caption("👉 Get a free Gemini key: [aistudio.google.com](https://aistudio.google.com/app/apikey)")

        else: # OpenAI
            raw_openai = get_openai_api_key()
            openai_input = st.text_input(
                "OpenAI API Key (Paid Credits):",
                value=raw_openai if raw_openai and not "your_" in raw_openai else "",
                type="password",
                placeholder="sk-...",
                help="Enter your OpenAI API key."
            )
            api_key_to_use = openai_input.strip() or raw_openai

        # If OpenAI embeddings selected, require OpenAI key as well
        openai_embed_key = api_key_to_use if (not is_local_embeddings and llm_provider == "openai") else get_openai_api_key()
        if not is_local_embeddings and (not openai_embed_key or "your_" in openai_embed_key):
            st.warning("⚠️ OpenAI embeddings require an OpenAI API key.")

        st.markdown("---")
        st.markdown("### 📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "Select one or multiple PDF files:",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents to create the semantic vector database."
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            process_btn = st.button("🚀 Process", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

        # Handle Clear Action
        if clear_btn:
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.documents_processed = False
            st.session_state.processing_stats = {}
            st.session_state.processed_file_names = []
            st.success("Cleared all documents and chat history.")
            st.rerun()

        # Handle Document Processing Action
        if process_btn:
            if not is_local_embeddings and (not openai_embed_key or "your_" in openai_embed_key):
                st.error("Please enter a valid OpenAI API Key for OpenAI embeddings, or select 'Local Free'.")
            elif not uploaded_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs (extracting, chunking, embedding)..."):
                    try:
                        # 1. Extract Text & Metadata
                        raw_docs = load_pdf_documents(uploaded_files)
                        
                        # 2. Text Chunking
                        chunks = split_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

                        if not chunks:
                            st.error("No valid text chunks could be produced from the uploaded documents.")
                            return api_key_to_use, llm_provider, embedding_provider

                        # 3. Vector Embeddings (Local Free or OpenAI)
                        embeddings_model = get_embedding_model(
                            provider=embedding_provider,
                            api_key=openai_embed_key if not is_local_embeddings else None,
                        )

                        # 4. Build FAISS Vector Store
                        vector_store = create_vector_store(chunks=chunks, embeddings=embeddings_model)

                        # 5. Compute Statistics & Update Session State
                        stats = calculate_document_stats(raw_docs, chunks)
                        st.session_state.vector_store = vector_store
                        st.session_state.documents_processed = True
                        st.session_state.processing_stats = stats
                        st.session_state.processed_file_names = [f.name for f in uploaded_files]

                        st.success("✅ Documents processed successfully!")
                        st.rerun()

                    except PDFProcessingError as ppe:
                        st.error(f"❌ PDF Processing Error: {str(ppe)}")
                    except EmbeddingInitializationError as eie:
                        st.error(f"❌ Embedding Error: {str(eie)}")
                    except VectorStoreError as vse:
                        st.error(f"❌ Vector Store Error: {str(vse)}")
                    except Exception as e:
                        logger.exception("Unexpected error during document processing.")
                        st.error(f"❌ An error occurred during processing: {str(e)}")

        # Display Processing Statistics
        if st.session_state.documents_processed and st.session_state.processing_stats:
            st.markdown("---")
            st.markdown("### 📊 Document Statistics")
            stats = st.session_state.processing_stats

            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-value">{stats.get('num_files', 0)}</div><div class="metric-label">PDFs</div></div>""",
                    unsafe_allow_html=True
                )
            with col_stat2:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-value">{stats.get('num_pages', 0)}</div><div class="metric-label">Pages</div></div>""",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""<div class="metric-card"><div class="metric-value">{stats.get('num_chunks', 0)}</div><div class="metric-label">Vector Chunks</div></div>""",
                unsafe_allow_html=True
            )

            with st.expander("📄 Processed Files", expanded=False):
                for fname in stats.get("file_names", []):
                    st.markdown(f"- `{fname}`")

        # RAG Configuration Information Expander
        with st.expander("⚙️ Pipeline Configuration", expanded=False):
            st.markdown(f"**Embedding Engine:** `{'Local Free (all-MiniLM-L6-v2)' if is_local_embeddings else EMBEDDING_MODEL}`")
            st.markdown(f"**LLM Provider:** `{llm_provider.upper()}`")
            st.markdown(f"**Chunk Size:** `{CHUNK_SIZE}` chars")
            st.markdown(f"**Chunk Overlap:** `{CHUNK_OVERLAP}` chars")
            st.markdown(f"**Top-K Retrieval:** `{TOP_K}` chunks")
            st.markdown(f"**Temperature:** `{TEMPERATURE}` (Deterministic)")

        return (
            api_key_to_use,
            llm_provider,
            embedding_provider,
            selected_llm_model
        )


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
def render_main_content(
    active_api_key: str,
    llm_provider: str,
    embedding_provider: str,
    llm_model: str
):
    """Renders the main hero header, welcome message, chat history, and chat input."""
    # Hero Header Banner
    st.markdown(
        """<div class="hero-header">
<h1 class="hero-title">DocuMind AI <span class="badge-tag">RAG v1.0</span></h1>
<p class="hero-subtitle">Chat with your PDFs using Semantic Search & Hallucination-Free RAG</p>
</div>""",
        unsafe_allow_html=True
    )

    # Empty State / Welcome Screen if no documents are loaded
    if not st.session_state.documents_processed or st.session_state.vector_store is None:
        st.markdown(
            """<div class="welcome-card">
<h3>👋 Welcome to DocuMind AI!</h3>
<p>Upload your PDF documents from the sidebar to start asking questions grounded directly in your files.</p>
<div class="feature-grid">
<div class="feature-item">
<div class="feature-title">📁 Multi-PDF Support</div>
<div class="feature-desc">Upload and query across multiple multi-page PDF documents simultaneously.</div>
</div>
<div class="feature-item">
<div class="feature-title">⚡ FAISS Semantic Search</div>
<div class="feature-desc">Fast vector similarity retrieval powered by dense embeddings.</div>
</div>
<div class="feature-item">
<div class="feature-title">📚 Exact Citations</div>
<div class="feature-desc">Every answer links directly to the source filename and page number.</div>
</div>
<div class="feature-item">
<div class="feature-title">🛡️ Zero Hallucinations</div>
<div class="feature-desc">Strict system prompt ensures answers come strictly from document context.</div>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )
        return

    # Success notification after processing
    st.success("✅ Documents processed and indexed into FAISS. You can now ask questions below.")

    # Render Chat History
    for message in st.session_state.chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        sources = message.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)
            
            # Display source citations if available for assistant messages
            if role == "assistant" and sources:
                with st.expander("📚 Sources & Citations", expanded=False):
                    for idx, src in enumerate(sources, start=1):
                        citation_obj = src if isinstance(src, SourceCitation) else SourceCitation(**src)
                        st.markdown(format_citation_markdown(citation_obj, idx))

    # Chat Input Box
    user_query = st.chat_input("Ask a question about your uploaded documents...")

    if user_query:
        # 1. Validate Prerequisites
        if not active_api_key and llm_provider != "local":
            st.error(f"Please provide an API key for {llm_provider.upper()} in the sidebar.")
            return

        if st.session_state.vector_store is None:
            st.warning("Please upload and process your PDF documents first.")
            return

        # 2. Append User Message to State and Display
        st.session_state.chat_history.append({"role": "user", "content": user_query, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 3. Execute RAG Pipeline with Friendly Spinners
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Searching relevant document sections..."):
                    response = execute_rag_pipeline(
                        query=user_query,
                        vector_store=st.session_state.vector_store,
                        api_key=active_api_key,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        top_k=TOP_K
                    )

                # Display Generated Answer
                st.markdown(response.answer)

                # Display Citations
                if response.sources:
                    with st.expander("📚 Sources & Citations", expanded=False):
                        for idx, citation in enumerate(response.sources, start=1):
                            st.markdown(format_citation_markdown(citation, idx))

                # 4. Save Assistant Response in Chat History
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": [s.to_dict() for s in response.sources]
                })

            except RAGPipelineError as rpe:
                st.error(f"❌ RAG Error: {str(rpe)}")
            except Exception as e:
                logger.exception("Unexpected error in chat query handler.")
                st.error(f"❌ Failed to generate answer: {str(e)}")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    """Application entrypoint."""

    (
        active_api_key,
        llm_provider,
        embedding_provider,
        llm_model
    ) = render_sidebar()

    render_main_content(
        active_api_key,
        llm_provider,
        embedding_provider,
        llm_model
    )


if __name__ == "__main__":
    main()

