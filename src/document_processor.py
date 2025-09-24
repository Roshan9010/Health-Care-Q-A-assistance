"""
Document Processing Module for Healthcare Q&A Assistant
Handles PDF document ingestion, text extraction, and preprocessing.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import hashlib

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes PDF documents for the healthcare Q&A system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
                        
                return text
                
        except Exception as e:
            logger.error(f"Error reading PDF file {pdf_path}: {e}")
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text for better processing.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Preprocessed text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Healthcare-specific preprocessing
        # Normalize common medical abbreviations
        medical_replacements = {
            " mg ": " milligrams ",
            " ml ": " milliliters ",
            " cc ": " cubic centimeters ",
            " IV ": " intravenous ",
            " IM ": " intramuscular ",
            " PO ": " oral ",
            " PRN ": " as needed ",
            " BID ": " twice daily ",
            " TID ": " three times daily ",
            " QID ": " four times daily ",
            " QD ": " once daily ",
            " Q4H ": " every 4 hours ",
            " Q6H ": " every 6 hours ",
            " Q8H ": " every 8 hours ",
            " Q12H ": " every 12 hours ",
        }
        
        for abbrev, full_form in medical_replacements.items():
            text = text.replace(abbrev, full_form)
            
        return text
    
    def create_chunks(self, text: str, source: str) -> List[Document]:
        """
        Split text into chunks and create Document objects.
        
        Args:
            text (str): Text to be chunked
            source (str): Source file path
            
        Returns:
            List[Document]: List of document chunks
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(processed_text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "document_hash": self._get_document_hash(source)
                }
            )
            documents.append(doc)
            
        return documents
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete processing pipeline for a single PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Document]: Processed document chunks
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Validate file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError(f"No text could be extracted from PDF: {pdf_path}")
        
        # Create chunks
        documents = self.create_chunks(text, pdf_path)
        
        logger.info(f"Successfully processed {pdf_path}: {len(documents)} chunks created")
        return documents
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths (List[str]): List of PDF file paths
            
        Returns:
            List[Document]: Combined list of processed document chunks
        """
        all_documents = []
        
        for pdf_path in pdf_paths:
            try:
                documents = self.process_pdf(pdf_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                st.error(f"Error processing {pdf_path}: {e}")
                continue
                
        return all_documents
    
    def process_uploaded_files(self, uploaded_files) -> List[Document]:
        """
        Process files uploaded through Streamlit interface.
        
        Args:
            uploaded_files: Streamlit uploaded files
            
        Returns:
            List[Document]: Processed document chunks
        """
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process the file
                documents = self.process_pdf(temp_path)
                all_documents.extend(documents)
                
                # Clean up temporary file
                os.remove(temp_path)
                
                st.success(f"Successfully processed {uploaded_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process uploaded file {uploaded_file.name}: {e}")
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue
                
        return all_documents
    
    def _get_document_hash(self, file_path: str) -> str:
        """
        Generate a hash for the document for change detection.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: MD5 hash of the file
        """
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception:
            return ""
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Args:
            documents (List[Document]): List of processed documents
            
        Returns:
            Dict[str, Any]: Document statistics
        """
        if not documents:
            return {}
        
        total_chunks = len(documents)
        sources = set(doc.metadata.get("source", "") for doc in documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "unique_sources": len(sources),
            "total_characters": total_chars,
            "average_chunk_size": round(avg_chunk_size, 2),
            "sources": list(sources)
        }

# Utility functions
def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if a file is a valid PDF.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if valid PDF, False otherwise
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            return len(pdf_reader.pages) > 0
    except Exception:
        return False

def get_supported_file_types() -> List[str]:
    """
    Get list of supported file types.
    
    Returns:
        List[str]: List of supported file extensions
    """
    return [".pdf"]