"""
Healthcare Document Q&A Assistant - Main Streamlit Application
A RAG-powered chatbot for medical professionals to query healthcare documents.
"""

import os
import sys
import streamlit as st
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

# Import our modules
try:
    from src.document_processor import DocumentProcessor
    from src.vector_store import HealthcareVectorStore
    from src.rag_pipeline import HealthcareRAGPipeline
    from src.healthcare_utils import (
        HealthcareTerminologyProcessor, 
        DrugInteractionChecker, 
        HealthcareSafetyChecker
    )
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Healthcare Document Q&A Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = HealthcareVectorStore()
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    
    if 'terminology_processor' not in st.session_state:
        st.session_state.terminology_processor = HealthcareTerminologyProcessor()
    
    if 'drug_checker' not in st.session_state:
        st.session_state.drug_checker = DrugInteractionChecker()
    
    if 'safety_checker' not in st.session_state:
        st.session_state.safety_checker = HealthcareSafetyChecker()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_uploaded' not in st.session_state:
        st.session_state.documents_uploaded = False

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🏥 Healthcare Document Q&A Assistant</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to the Healthcare Document Q&A Assistant!</strong><br>
        This AI-powered system helps medical professionals find accurate information from 
        uploaded medical documents, guidelines, and policies using advanced RAG technology.
        Upload one document at a time for focused analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for document management and settings
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📁 Document Management</h2>', 
                   unsafe_allow_html=True)
        
        # Document upload section
        uploaded_file = st.file_uploader(
            "Upload Medical Document (PDF)",
            type=['pdf'],
            accept_multiple_files=False,
            help="Upload one medical guideline, policy, drug information, or other healthcare document"
        )
        
        if uploaded_file:
            if st.button("Process Document", type="primary"):
                process_document(uploaded_file)
        
        # Clear documents button
        if st.session_state.documents_uploaded:
            if st.button("Clear All Documents", type="secondary"):
                clear_documents()
        
        # Display document statistics
        display_document_stats()
        
        # Settings section
        st.markdown('<h2 class="sub-header">⚙️ Settings</h2>', 
                   unsafe_allow_html=True)
        
        # LLM provider selection
        llm_provider = st.selectbox(
            "Language Model Provider",
            ["huggingface", "openai"],
            help="Choose your preferred language model provider"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
            max_tokens = st.slider("Max Tokens", 100, 1000, 500, 50)
            retrieval_k = st.slider("Documents to Retrieve", 1, 10, 5, 1)
        
        # Initialize RAG pipeline with settings
        if st.button("Initialize/Update System"):
            initialize_rag_pipeline(llm_provider, temperature, max_tokens, retrieval_k)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.markdown('<h2 class="sub-header">💬 Ask Your Questions</h2>', 
                   unsafe_allow_html=True)
        
        # Display chat history
        display_chat_history()
        
        # Question input
        user_question = st.text_area(
            "Enter your healthcare question:",
            height=100,
            placeholder="e.g., What is the recommended dosage of metformin for type 2 diabetes?"
        )
        
        col_ask, col_clear = st.columns([3, 1])
        
        with col_ask:
            if st.button("Ask Question", type="primary", disabled=not st.session_state.documents_uploaded):
                if user_question.strip():
                    handle_question(user_question)
                else:
                    st.warning("Please enter a question.")
        
        with col_clear:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        # Healthcare utilities panel
        st.markdown('<h2 class="sub-header">🔧 Healthcare Tools</h2>', 
                   unsafe_allow_html=True)
        
        # Drug interaction checker
        with st.expander("Drug Interaction Checker"):
            drug_list_input = st.text_area(
                "Enter medications (one per line):",
                height=100,
                placeholder="metformin\nlisinopril\nwarfarin"
            )
            
            if st.button("Check Interactions"):
                if drug_list_input.strip():
                    check_drug_interactions(drug_list_input)
        
        # Medical terminology expander
        with st.expander("Medical Terminology Helper"):
            medical_text = st.text_area(
                "Enter text with medical abbreviations:",
                height=80,
                placeholder="Patient has HTN, DM, and takes BID dosing"
            )
            
            if st.button("Expand Abbreviations"):
                if medical_text.strip():
                    expand_medical_terms(medical_text)
        
        # System status
        display_system_status()

def process_document(uploaded_file):
    """Process uploaded PDF document."""
    try:
        with st.spinner("Processing document..."):
            # Clear existing documents first to avoid duplicates
            st.session_state.vector_store.clear_store()
            
            # Process document
            documents = st.session_state.document_processor.process_uploaded_files([uploaded_file])
            
            if documents:
                # Add to vector store
                st.session_state.vector_store.add_documents(documents)
                st.session_state.documents_uploaded = True
                
                # Show success message
                st.success(f"Successfully processed {uploaded_file.name} with {len(documents)} chunks!")
                
                # Show document statistics
                stats = st.session_state.document_processor.get_document_stats(documents)
                st.json(stats)
                
                st.rerun()
            else:
                st.error("No valid document could be processed.")
                
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.exception(e)

def clear_documents():
    """Clear all uploaded documents from the vector store."""
    try:
        with st.spinner("Clearing documents..."):
            # Clear the vector store
            st.session_state.vector_store.clear_store()
            st.session_state.documents_uploaded = False
            
            # Reset RAG pipeline
            st.session_state.rag_pipeline = None
            
            # Clear chat history
            st.session_state.chat_history = []
            
            st.success("All documents cleared successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error clearing documents: {str(e)}")
        st.exception(e)

def display_document_stats():
    """Display statistics about uploaded documents."""
    stats = st.session_state.vector_store.get_store_stats()
    
    if stats.get('total_documents', 0) > 0:
        st.markdown("**Document Statistics:**")
        st.metric("Total Documents", stats['total_documents'])
        st.metric("Unique Sources", stats['unique_sources'])
        st.metric("Total Characters", f"{stats['total_characters']:,}")
    else:
        st.info("No documents uploaded yet.")

def initialize_rag_pipeline(llm_provider, temperature, max_tokens, retrieval_k):
    """Initialize the RAG pipeline with given settings."""
    try:
        with st.spinner("Initializing RAG pipeline..."):
            st.session_state.rag_pipeline = HealthcareRAGPipeline(
                vector_store=st.session_state.vector_store,
                llm_provider=llm_provider,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Update retrieval settings
            if hasattr(st.session_state.rag_pipeline, 'update_retrieval_settings'):
                st.session_state.rag_pipeline.update_retrieval_settings(k=retrieval_k)
            
            st.success("RAG pipeline initialized successfully!")
            
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")

def handle_question(question):
    """Handle user question and generate response."""
    try:
        # Check if RAG pipeline is initialized
        if st.session_state.rag_pipeline is None:
            st.warning("Please initialize the system first using the sidebar settings.")
            return
        
        # Add user question to chat history
        st.session_state.chat_history.append({
            "type": "user",
            "content": question,
            "timestamp": datetime.now()
        })
        
        # Check for safety concerns
        safety_concerns = st.session_state.safety_checker.check_safety_concerns(question)
        safety_warning = st.session_state.safety_checker.generate_safety_warning(safety_concerns)
        
        with st.spinner("Searching documents and generating response..."):
            # Get answer from RAG pipeline
            result = st.session_state.rag_pipeline.answer_question(question)
            
            # Add assistant response to chat history
            assistant_response = {
                "type": "assistant",
                "content": result['answer'],
                "sources": result.get('sources', []),
                "confidence": result.get('confidence', 0.0),
                "processing_time": result.get('processing_time', 0.0),
                "safety_warning": safety_warning,
                "timestamp": datetime.now()
            }
            
            st.session_state.chat_history.append(assistant_response)
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.exception(e)

def display_chat_history():
    """Display the chat history."""
    if not st.session_state.chat_history:
        st.info("No questions asked yet. Upload some documents and start asking!")
        return
    
    for i, message in enumerate(st.session_state.chat_history):
        if message["type"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
        elif message["type"] == "assistant":
            # Display safety warning if present
            if message.get("safety_warning"):
                st.markdown(f"""
                <div class="warning-box">
                    {message["safety_warning"]}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{message.get('confidence', 0):.2f}")
            with col2:
                st.metric("Processing Time", f"{message.get('processing_time', 0):.2f}s")
            with col3:
                st.metric("Sources Used", len(message.get('sources', [])))
            
            # Display sources if available
            if message.get('sources'):
                with st.expander(f"View Sources ({len(message['sources'])})"):
                    for j, source in enumerate(message['sources']):
                        st.markdown(f"**Source {j+1}: {source['file']} (Chunk {source['chunk_id']})**")
                        st.text(source['preview'])
                        st.divider()

def check_drug_interactions(drug_list_input):
    """Check for drug interactions."""
    try:
        drugs = [drug.strip() for drug in drug_list_input.split('\n') if drug.strip()]
        
        if len(drugs) < 2:
            st.warning("Please enter at least 2 medications to check for interactions.")
            return
        
        interactions = st.session_state.drug_checker.check_interactions(drugs)
        
        if interactions:
            st.warning(f"Found {len(interactions)} potential interactions:")
            
            for interaction in interactions:
                severity_color = {
                    'major': '🔴',
                    'moderate': '🟡',
                    'minor': '🟢'
                }.get(interaction['severity'], '⚪')
                
                st.markdown(f"""
                **{severity_color} {interaction['drug1'].title()} ↔ {interaction['drug2'].title()}**
                - **Severity:** {interaction['severity'].title()}
                - **Description:** {interaction['description']}
                - **Management:** {interaction['management']}
                """)
        else:
            st.success("No known interactions found for the entered medications.")
            
    except Exception as e:
        st.error(f"Error checking drug interactions: {str(e)}")

def expand_medical_terms(medical_text):
    """Expand medical abbreviations in text."""
    try:
        expanded_text = st.session_state.terminology_processor.expand_abbreviations(medical_text)
        
        st.markdown("**Expanded Text:**")
        st.text_area("", value=expanded_text, height=100, disabled=True)
        
        # Identify drugs and specialties
        drugs = st.session_state.terminology_processor.identify_drug_mentions(medical_text)
        specialties = st.session_state.terminology_processor.identify_medical_specialty(medical_text)
        
        if drugs:
            st.markdown("**Medications Identified:**")
            for drug, category in drugs:
                st.write(f"- {drug.title()} ({category})")
        
        if specialties:
            st.markdown("**Medical Specialties:**")
            for specialty in specialties:
                st.write(f"- {specialty.title()}")
                
    except Exception as e:
        st.error(f"Error expanding medical terms: {str(e)}")

def display_system_status():
    """Display system status information."""
    st.markdown("**System Status:**")
    
    # Document status
    if st.session_state.documents_uploaded:
        st.success("📄 Documents loaded")
    else:
        st.warning("📄 No documents loaded")
    
    # RAG pipeline status
    if st.session_state.rag_pipeline is not None:
        st.success("🤖 RAG pipeline ready")
    else:
        st.warning("🤖 RAG pipeline not initialized")
    
    # Vector store status
    store_stats = st.session_state.vector_store.get_store_stats()
    if store_stats.get('total_documents', 0) > 0:
        st.success(f"🗃️ Vector store: {store_stats['total_documents']} docs")
    else:
        st.warning("🗃️ Vector store empty")

if __name__ == "__main__":
    main()