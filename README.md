# 📄 DocuMind AI

> An AI-powered RAG chatbot that allows users to upload PDF documents and ask questions using semantic search, FAISS, local embeddings, and LLMs.

## 🚀 Features

- 📁 Upload and chat with multiple PDF documents
- 📄 Multi-page PDF support
- ✂️ Text chunking with overlap
- 🧠 Local embeddings using Sentence Transformers
- 🔍 Semantic search with FAISS
- 🤖 LLM support via Groq, OpenAI, or Google Gemini
- 📚 Source citations with file names and page numbers
- 💬 Interactive Streamlit chat interface
- 🛡️ Context-grounded answers to reduce hallucinations

---

## 🏗️ Architecture

```text
PDF Documents
      │
      ▼
Text Extraction
      │
      ▼
Text Chunking
      │
      ▼
Local Embeddings
(MiniLM-L6-v2)
      │
      ▼
FAISS Vector Store
      │
      ▼
Semantic Search
(Retrieve Top-K Chunks)
      │
      ▼
Question + Context
      │
      ▼
LLM API
(Groq / OpenAI / Gemini)
      │
      ▼
Answer + Citations
```

---

## 🧠 How It Works

1. Upload one or more PDF documents.
2. Extract text from each PDF.
3. Split the text into overlapping chunks.
4. Convert chunks into vector embeddings.
5. Store embeddings in FAISS.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant chunks using semantic search.
8. Send the retrieved context and question to an LLM.
9. Display the generated answer with source citations.

---

## 💻 Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **FAISS**
- **Sentence Transformers**
- **Groq API**
- **OpenAI API**
- **Google Gemini**
- **PyPDF**
- **python-dotenv**

---

## 📂 Project Structure

```text
documind-ai/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── config.py
│   ├── document_processor.py
│   ├── embeddings.py
│   ├── rag_pipeline.py
│   └── vector_store.py
└── tests/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/documind-ai.git
cd documind-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key

# Optional
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

> ⚠️ Never upload your `.env` file or API keys to GitHub.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 🔒 RAG Prompting

The LLM is instructed to answer only using the retrieved document context.

If the answer is not available in the uploaded documents, it responds:

> "I could not find this information in the uploaded documents."

This helps reduce hallucination and keeps answers grounded in the provided PDFs.

---

## 🔐 Data Flow

The following operations run locally:

- PDF processing
- Text extraction
- Text chunking
- Local embedding generation
- FAISS vector search
- Semantic retrieval

The final answer is generated using a cloud LLM API such as **Groq, OpenAI, or Google Gemini**.

---

## 🔮 Future Improvements

- [ ] Persistent vector database
- [ ] Support for DOCX and TXT files
- [ ] Hybrid search
- [ ] Reranking
- [ ] Streaming responses
- [ ] Fully local LLM using Ollama
- [ ] Docker deployment

---

## 👨‍💻 Author

**Jayadeep**

Built to demonstrate practical knowledge of:

**RAG • LLMs • Semantic Search • Vector Databases • FAISS • LangChain • Streamlit**

---

⭐ If you found this project interesting, feel free to star the repository!
