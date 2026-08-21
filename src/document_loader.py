"""
src/document_loader.py
Handles loading, extraction, and validation of single and multiple PDF files.
Extracts text page-by-page while preserving essential metadata (source filename, 1-indexed page number).
Handles corrupted, empty, and password-protected PDFs safely without crashing.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import List, Union, Any
import pypdf
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Custom exception raised when PDF parsing or loading fails."""
    pass


def load_single_pdf_from_path(file_path: Union[str, Path]) -> List[Document]:
    """
    Extracts text and metadata from a local PDF file path.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        List[Document]: List of LangChain Document objects containing page text and metadata.

    Raises:
        PDFProcessingError: If the file is invalid, corrupted, or unreadable.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise PDFProcessingError(f"File not found: {file_path}")
    
    filename = path_obj.name
    try:
        with open(path_obj, "rb") as f:
            return extract_documents_from_bytes(f.read(), filename)
    except Exception as e:
        logger.error(f"Error reading PDF {filename}: {str(e)}")
        raise PDFProcessingError(f"Failed to read '{filename}': {str(e)}")


def extract_documents_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extracts pages and text from raw PDF bytes using PyPDF.

    Args:
        file_bytes: Raw binary content of the PDF file.
        filename: Name of the PDF file for metadata attribution.

    Returns:
        List[Document]: List of LangChain Document objects with 1-indexed page numbers.

    Raises:
        PDFProcessingError: If parsing fails or PDF contains no extractable text.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise PDFProcessingError(f"File '{filename}' is empty (0 bytes).")

    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        logger.error(f"Corrupted or invalid PDF header for '{filename}': {str(e)}")
        raise PDFProcessingError(f"Could not parse '{filename}'. File may be corrupted or not a valid PDF.")

    if pdf_reader.is_encrypted:
        try:
            # Attempt empty password decrypt
            pdf_reader.decrypt("")
        except Exception:
            raise PDFProcessingError(f"PDF '{filename}' is password protected and cannot be processed.")

    total_pages = len(pdf_reader.pages)
    if total_pages == 0:
        raise PDFProcessingError(f"PDF '{filename}' contains 0 pages.")

    documents: List[Document] = []
    extracted_text_count = 0

    for page_idx, page in enumerate(pdf_reader.pages):
        # 1-indexed page number for intuitive user citation
        user_page_num = page_idx + 1
        
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Warning: Failed to extract text from page {user_page_num} of '{filename}': {e}")
            page_text = ""

        page_text = page_text.strip()
        if page_text:
            extracted_text_count += 1
            doc = Document(
                page_content=page_text,
                metadata={
                    "source": filename,
                    "page": user_page_num,
                    "total_pages": total_pages,
                }
            )
            documents.append(doc)

    if not documents or extracted_text_count == 0:
        logger.warning(f"No extractable text found in '{filename}'. It may contain only scanned images without OCR.")
        raise PDFProcessingError(
            f"No extractable text found in '{filename}'. "
            "The document might be blank or consist purely of scanned images."
        )

    logger.info(f"Successfully loaded '{filename}': {len(documents)} text-containing pages extracted.")
    return documents


def load_pdf_documents(uploaded_files: List[Any]) -> List[Document]:
    """
    Processes a list of uploaded PDF files (Streamlit UploadedFile objects or file paths/bytes).

    Args:
        uploaded_files: List of Streamlit UploadedFile objects or file path strings.

    Returns:
        List[Document]: Aggregated list of LangChain Document objects from all valid PDFs.

    Raises:
        PDFProcessingError: If none of the files could be processed successfully.
    """
    if not uploaded_files:
        raise PDFProcessingError("No PDF files were provided for processing.")

    all_documents: List[Document] = []
    errors: List[str] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", None)
        
        try:
            # Case 1: Streamlit UploadedFile or file-like object with getvalue / read
            if hasattr(uploaded_file, "getvalue"):
                file_bytes = uploaded_file.getvalue()
                if not filename:
                    filename = "uploaded_document.pdf"
                docs = extract_documents_from_bytes(file_bytes, filename)
            elif hasattr(uploaded_file, "read"):
                file_bytes = uploaded_file.read()
                if not filename:
                    filename = "uploaded_document.pdf"
                docs = extract_documents_from_bytes(file_bytes, filename)
            # Case 2: File path string or Path object
            elif isinstance(uploaded_file, (str, Path)):
                filename = Path(uploaded_file).name
                docs = load_single_pdf_from_path(uploaded_file)
            else:
                raise ValueError(f"Unsupported file input type: {type(uploaded_file)}")

            all_documents.extend(docs)

        except Exception as e:
            error_msg = f"'{filename or 'Unknown'}': {str(e)}"
            logger.error(f"Failed to process file {error_msg}")
            errors.append(error_msg)

    if not all_documents and errors:
        raise PDFProcessingError(
            "Failed to extract text from all provided files:\n" + "\n".join(f"• {err}" for err in errors)
        )

    if errors:
        logger.warning(f"Some files had errors during processing: {errors}")

    return all_documents
