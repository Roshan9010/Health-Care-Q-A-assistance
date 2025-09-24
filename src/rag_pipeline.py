"""
RAG Pipeline Module for Healthcare Q&A Assistant
Implements the Retrieval Augmented Generation pipeline using LangChain.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# LangChain imports - will be available at runtime
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.memory import ConversationBufferMemory
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
except ImportError:
    # Fallback imports for older versions
    try:
        from langchain.llms import HuggingFacePipeline
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate, ChatPromptTemplate
        from langchain.chains import RetrievalQA
        from langchain.memory import ConversationBufferMemory
        from langchain.schema import BaseRetriever, Document
    except ImportError as e:
        print(f"Warning: LangChain imports failed: {e}")
        # We'll handle this gracefully in the code
        BaseRetriever = object
        Document = dict

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Warning: transformers not available")
    
try:
    import streamlit as st
except ImportError:
    print("Warning: streamlit not available")

from .vector_store import HealthcareVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareRetriever:
    """Custom retriever for healthcare documents."""
    
    def __init__(self, vector_store: HealthcareVectorStore, k: int = 2):
        """
        Initialize the retriever.
        
        Args:
            vector_store (HealthcareVectorStore): Vector store instance
            k (int): Number of documents to retrieve (reduced to 2 for more focused answers)
        """
        self.vector_store = vector_store
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Document]: Relevant documents
        """
        results = self.vector_store.similarity_search_with_healthcare_boost(
            query, k=self.k
        )
        return [doc for doc, score in results]
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)
    
    def invoke(self, input_data):
        """Invoke method for LangChain compatibility."""
        if isinstance(input_data, str):
            return self.get_relevant_documents(input_data)
        elif isinstance(input_data, dict) and 'query' in input_data:
            return self.get_relevant_documents(input_data['query'])
        else:
            return []

class HealthcareRAGPipeline:
    """
    Complete RAG pipeline for healthcare Q&A system.
    Supports multiple LLM providers and optimized for medical content.
    """
    
    def __init__(self,
                 vector_store: HealthcareVectorStore,
                 llm_provider: str = "huggingface",
                 model_name: str = "microsoft/DialoGPT-medium",
                 temperature: float = 0.1,
                 max_tokens: int = 500):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store (HealthcareVectorStore): Vector store for document retrieval
            llm_provider (str): LLM provider ('openai', 'huggingface', 'local')
            model_name (str): Name of the model to use
            temperature (float): Temperature for text generation
            max_tokens (int): Maximum tokens to generate
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize retriever with reduced k for more focused answers
        self.retriever = HealthcareRetriever(vector_store, k=2)
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize chains
        self._initialize_chains()
    
    def _initialize_llm(self):
        """Initialize the language model based on provider."""
        try:
            if self.llm_provider.lower() == "openai":
                return self._initialize_openai_llm()
            elif self.llm_provider.lower() == "huggingface":
                return self._initialize_huggingface_llm()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Fallback to a simple model
            return self._initialize_fallback_llm()
    
    def _initialize_openai_llm(self):
        """Initialize OpenAI LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=api_key
        )
    
    def _initialize_huggingface_llm(self):
        """Initialize Hugging Face LLM."""
        try:
            # Use a lightweight model suitable for healthcare Q&A
            model_id = "microsoft/DialoGPT-medium"
            
            # Create text generation pipeline
            text_generator = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=model_id,
                max_length=512,
                temperature=self.temperature,
                do_sample=True,
                device=-1  # CPU
            )
            
            return HuggingFacePipeline(pipeline=text_generator)
            
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model: {e}")
            return self._initialize_fallback_llm()
    
    def _initialize_fallback_llm(self):
        """Initialize a simple fallback LLM."""
        # This is a simple mock LLM for demonstration
        class FallbackLLM:
            def __init__(self):
                self.temperature = 0.1
            
            def __call__(self, prompt):
                return "I'm a fallback response. Please configure a proper LLM."
            
            def invoke(self, prompt):
                return "I'm a fallback response. Please configure a proper LLM."
        
        logger.warning("Using fallback LLM. Please configure a proper language model.")
        return FallbackLLM()
    
    def _initialize_chains(self):
        """Initialize the RAG chains."""
        # Healthcare-specific prompt template
        self.qa_prompt = PromptTemplate(
            template="""You are a healthcare assistant helping medical professionals find information from medical documents. 
Use the following context to answer the question accurately and professionally.

Context:
{context}

Question: {question}

Guidelines:
- Provide accurate, evidence-based answers based on the context
- If the context doesn't contain enough information, say so clearly
- Include relevant details about dosages, contraindications, or warnings when applicable
- Always reference the source when possible
- Use professional medical terminology appropriately
- If discussing medications, include important safety information

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Conversational prompt for follow-up questions
        self.conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a healthcare assistant helping medical professionals. 
Use the following context and chat history to provide accurate, professional answers.
Always prioritize patient safety and evidence-based medicine."""),
            ("human", """Context: {context}
            
Question: {question}

Please provide a comprehensive answer based on the available information.""")
        ])
        
        # Initialize QA chain
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": self.qa_prompt},
                return_source_documents=True
            )
        except Exception as e:
            logger.error(f"Error initializing QA chain: {e}")
            self.qa_chain = None
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a healthcare-related question using RAG.
        
        Args:
            question (str): User's question
            
        Returns:
            Dict[str, Any]: Answer with metadata
        """
        try:
            start_time = datetime.now()
            
            # Retrieve relevant context
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the available documents to answer your question. Please try rephrasing your question or ensure that relevant documents have been uploaded.",
                    "sources": [],
                    "confidence": 0.0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Prepare context
            context = self._prepare_context(relevant_docs)
            
            # Generate answer using the LLM
            if self.qa_chain:
                try:
                    result = self.qa_chain({
                        "query": question,
                        "context": context
                    })
                    answer = result.get("result", "Unable to generate answer")
                    source_docs = result.get("source_documents", [])
                except Exception as e:
                    logger.error(f"Error with QA chain: {e}")
                    answer = self._generate_simple_answer(question, context)
                    source_docs = relevant_docs
            else:
                answer = self._generate_simple_answer(question, context)
                source_docs = relevant_docs
            
            # Calculate confidence based on retrieval scores
            confidence = self._calculate_confidence(relevant_docs)
            
            # Extract source information
            sources = self._extract_sources(source_docs)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "processing_time": processing_time,
                "context_used": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": 0.0
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        seen_sources = set()
        
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 0)
            
            # Create unique identifier for source + chunk combination
            source_key = f"{source}_{chunk_id}"
            
            # Skip if we've already seen this exact source + chunk combination
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            
            # Use simple numbering for context parts
            context_part = f"Section {len(context_parts) + 1} (Source: {source}):\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _generate_simple_answer(self, question: str, context: str) -> str:
        """Generate a simple answer when LLM chain fails."""
        # This is a fallback method for when the main LLM fails
        relevant_sentences = []
        
        # Simple keyword matching for basic responses
        question_lower = question.lower()
        context_sentences = context.split('. ')
        
        for sentence in context_sentences:
            if any(word in sentence.lower() for word in question_lower.split()):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return f"Based on the available documents: {' '.join(relevant_sentences[:3])}"
        else:
            return "I found relevant documents but couldn't generate a specific answer. Please review the source documents for detailed information."
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not documents:
            return 0.0
        
        # Simple confidence calculation based on number of relevant documents
        # and their metadata
        base_confidence = min(len(documents) / 5.0, 1.0)  # Max confidence with 5+ docs
        
        # Boost confidence if documents have medical terms
        medical_terms = ['medication', 'treatment', 'diagnosis', 'patient', 'clinical']
        medical_term_boost = 0.0
        
        for doc in documents:
            text_lower = doc.page_content.lower()
            for term in medical_terms:
                if term in text_lower:
                    medical_term_boost += 0.1
                    break
        
        confidence = min(base_confidence + (medical_term_boost / len(documents)), 1.0)
        return round(confidence, 2)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source_file = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 0)
            
            source_key = f"{source_file}_{chunk_id}"
            if source_key not in seen_sources:
                sources.append({
                    "file": os.path.basename(source_file) if source_file != 'Unknown' else 'Unknown',
                    "chunk_id": chunk_id,
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
                seen_sources.add(source_key)
        
        return sources
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        if hasattr(self.memory, 'chat_memory'):
            history = []
            for message in self.memory.chat_memory.messages:
                history.append({
                    "type": message.__class__.__name__,
                    "content": message.content
                })
            return history
        return []
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.memory.clear()
    
    def update_retrieval_settings(self, k: int = 5, score_threshold: float = 0.0):
        """
        Update retrieval settings.
        
        Args:
            k (int): Number of documents to retrieve
            score_threshold (float): Minimum similarity score
        """
        self.retriever.k = k
        if hasattr(self.retriever, 'score_threshold'):
            self.retriever.score_threshold = score_threshold

class HealthcareQueryProcessor:
    """Process and enhance healthcare queries."""
    
    @staticmethod
    def preprocess_query(query: str) -> str:
        """
        Preprocess a user query for better retrieval.
        
        Args:
            query (str): Raw user query
            
        Returns:
            str: Preprocessed query
        """
        # Expand common medical abbreviations
        abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'temp': 'temperature',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'pt': 'patient',
            'hx': 'history',
            'sx': 'symptoms',
            'fx': 'fracture'
        }
        
        processed_query = query.lower()
        for abbrev, full_form in abbreviations.items():
            processed_query = processed_query.replace(f' {abbrev} ', f' {full_form} ')
            processed_query = processed_query.replace(f' {abbrev}.', f' {full_form}')
        
        return processed_query
    
    @staticmethod
    def identify_query_type(query: str) -> str:
        """
        Identify the type of healthcare query.
        
        Args:
            query (str): User query
            
        Returns:
            str: Query type
        """
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['dose', 'dosage', 'mg', 'ml', 'how much']):
            return 'dosage'
        elif any(term in query_lower for term in ['contraindication', 'side effect', 'adverse']):
            return 'safety'
        elif any(term in query_lower for term in ['indication', 'used for', 'treat']):
            return 'indication'
        elif any(term in query_lower for term in ['procedure', 'protocol', 'guideline']):
            return 'procedure'
        else:
            return 'general'