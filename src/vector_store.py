"""
Vector Store Module for Healthcare Q&A Assistant
Manages FAISS vector database for document embeddings and similarity search.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareVectorStore:
    """
    Vector store implementation for healthcare documents using FAISS.
    Optimized for medical terminology and healthcare-specific content.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_path: str = "./vectorstore",
                 dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name (str): Name of the embedding model
            vector_store_path (str): Path to store the vector database
            dimension (int): Dimension of the embedding vectors
        """
        self.embedding_model_name = embedding_model_name
        self.vector_store_path = Path(vector_store_path)
        self.dimension = dimension
        
        # Create directory if it doesn't exist
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.document_metadata = []
        
        # Load existing index if available
        self._load_existing_index()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            
            # Use HuggingFace embeddings for LangChain compatibility
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Also keep a direct SentenceTransformer instance for flexibility
            self.sentence_transformer = SentenceTransformer(self.embedding_model_name)
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise Exception(f"Failed to initialize embeddings: {e}")
    
    def _load_existing_index(self):
        """Load existing FAISS index if available."""
        index_path = self.vector_store_path / "faiss_index"
        metadata_path = self.vector_store_path / "metadata.pkl"
        documents_path = self.vector_store_path / "documents.pkl"
        
        try:
            if (index_path.exists() and 
                metadata_path.exists() and 
                documents_path.exists()):
                
                logger.info("Loading existing vector store...")
                
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                
                # Load documents
                with open(documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
                
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            self._initialize_empty_index()
    
    def _initialize_empty_index(self):
        """Initialize an empty FAISS index."""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []
        self.document_metadata = []
        logger.info("Initialized empty vector store")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents (List[Document]): List of documents to add
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        try:
            # Extract text content for embedding
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self._generate_embeddings(texts)
            
            # Initialize index if empty
            if self.index is None:
                self._initialize_empty_index()
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store documents and metadata
            self.documents.extend(documents)
            
            # Extract metadata
            for doc in documents:
                metadata = doc.metadata.copy()
                metadata['text_length'] = len(doc.page_content)
                self.document_metadata.append(metadata)
            
            # Save the updated index
            self._save_index()
            
            logger.info(f"Successfully added {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise Exception(f"Failed to add documents to vector store: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        try:
            # Use sentence transformer for efficient batch processing
            embeddings = self.sentence_transformer.encode(
                texts,
                batch_size=32,
                show_progress_bar=len(texts) > 10
            )
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings.astype('float32')
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 2,
                         score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Perform similarity search for a query.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            score_threshold (float): Minimum similarity score threshold
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= score_threshold:  # Valid result
                    doc = self.documents[idx]
                    results.append((doc, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_healthcare_boost(self, 
                                              query: str, 
                                              k: int = 2,
                                              healthcare_terms: List[str] = None) -> List[Tuple[Document, float]]:
        """
        Enhanced similarity search with healthcare term boosting.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            healthcare_terms (List[str]): Healthcare terms to boost
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        # Default healthcare terms that get boosted
        if healthcare_terms is None:
            healthcare_terms = [
                'medication', 'dosage', 'treatment', 'diagnosis', 'symptom',
                'patient', 'procedure', 'therapy', 'prescription', 'clinical',
                'medical', 'drug', 'adverse', 'contraindication', 'indication'
            ]
        
        # Perform regular similarity search
        results = self.similarity_search(query, k * 2)  # Get more results for reranking
        
        # Boost scores for documents containing healthcare terms
        boosted_results = []
        for doc, score in results:
            boost_factor = 1.0
            doc_text_lower = doc.page_content.lower()
            
            # Calculate boost based on healthcare term presence
            for term in healthcare_terms:
                if term.lower() in doc_text_lower:
                    boost_factor += 0.1  # 10% boost per term
            
            # Apply boost
            boosted_score = score * boost_factor
            boosted_results.append((doc, boosted_score))
        
        # Sort by boosted scores and return top k
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:k]
    
    def get_relevant_context(self, 
                            query: str, 
                            max_context_length: int = 3000) -> str:
        """
        Get relevant context for a query, optimized for healthcare Q&A.
        
        Args:
            query (str): Search query
            max_context_length (int): Maximum length of context to return
            
        Returns:
            str: Relevant context for the query
        """
        # Search for relevant documents
        results = self.similarity_search_with_healthcare_boost(query, k=5)
        
        if not results:
            return ""
        
        # Combine relevant text chunks
        context_parts = []
        current_length = 0
        
        for doc, score in results:
            chunk_text = doc.page_content
            
            # Add source information
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 0)
            
            formatted_chunk = f"[Source: {Path(source).name}, Chunk {chunk_id}]\n{chunk_text}\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(formatted_chunk) > max_context_length:
                break
            
            context_parts.append(formatted_chunk)
            current_length += len(formatted_chunk)
        
        return "\n---\n".join(context_parts)
    
    def _save_index(self):
        """Save the FAISS index and metadata to disk."""
        try:
            if self.index is not None:
                # Save FAISS index
                index_path = self.vector_store_path / "faiss_index"
                faiss.write_index(self.index, str(index_path))
                
                # Save metadata
                metadata_path = self.vector_store_path / "metadata.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.document_metadata, f)
                
                # Save documents
                documents_path = self.vector_store_path / "documents.pkl"
                with open(documents_path, 'wb') as f:
                    pickle.dump(self.documents, f)
                
                logger.info("Vector store saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def clear_store(self):
        """Clear all documents from the vector store."""
        try:
            self._initialize_empty_index()
            
            # Remove saved files
            files_to_remove = [
                self.vector_store_path / "faiss_index",
                self.vector_store_path / "metadata.pkl",
                self.vector_store_path / "documents.pkl"
            ]
            
            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
            
            logger.info("Vector store cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    def get_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Store statistics
        """
        if not self.documents:
            return {"total_documents": 0}
        
        sources = set()
        total_chars = 0
        
        for doc, metadata in zip(self.documents, self.document_metadata):
            sources.add(metadata.get('source', 'Unknown'))
            total_chars += len(doc.page_content)
        
        return {
            "total_documents": len(self.documents),
            "unique_sources": len(sources),
            "total_characters": total_chars,
            "average_document_length": total_chars / len(self.documents),
            "embedding_dimension": self.dimension,
            "model_name": self.embedding_model_name
        }
    
    def search_by_source(self, source_name: str) -> List[Document]:
        """
        Get all documents from a specific source.
        
        Args:
            source_name (str): Name of the source file
            
        Returns:
            List[Document]: Documents from the specified source
        """
        matching_docs = []
        for doc in self.documents:
            doc_source = Path(doc.metadata.get('source', '')).name
            if source_name.lower() in doc_source.lower():
                matching_docs.append(doc)
        
        return matching_docs