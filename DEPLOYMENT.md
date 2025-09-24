# Healthcare Document Q&A Assistant - Deployment Guide

## System Requirements

- Python 3.8 or higher
- Windows/Mac/Linux operating system
- Minimum 4GB RAM (8GB recommended)
- 2GB free disk space

## Installation Instructions

### 1. Clone or Download the Project

If you have the project files, ensure the following structure:

```
Healthcare/
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── healthcare_utils.py
├── data/
│   └── documents/         # Sample documents
├── vectorstore/           # FAISS storage (created automatically)
├── requirements.txt
├── .env.example
└── README.md
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv healthcare_env

# Activate virtual environment
# On Windows:
healthcare_env\Scripts\activate
# On Mac/Linux:
source healthcare_env/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# If you encounter issues, install key packages individually:
pip install streamlit langchain langchain-community langchain-openai
pip install faiss-cpu sentence-transformers transformers torch
pip install PyPDF2 python-dotenv numpy pandas
```

### 4. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings:
# - Add OpenAI API key if using OpenAI models
# - Configure other settings as needed
```

### 5. Run the Application

```bash
# Start the Streamlit application
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## Configuration Options

### Language Model Providers

1. **Hugging Face (Default - Free)**
   - No API key required
   - Uses local models
   - Good for testing and development

2. **OpenAI (Recommended for Production)**
   - Requires OpenAI API key
   - Better quality responses
   - Set `OPENAI_API_KEY` in .env file

### Environment Variables

Edit the `.env` file to configure:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
APP_TITLE=Healthcare Document Q&A Assistant
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PATH=./vectorstore

# LLM Settings
LLM_PROVIDER=huggingface  # or openai
TEMPERATURE=0.1
MAX_TOKENS=500
```

## Usage Instructions

### 1. Upload Documents

1. Open the application in your browser
2. Use the sidebar "Document Management" section
3. Upload PDF files containing medical documents
4. Click "Process Documents" to add them to the system

### 2. Initialize the System

1. In the sidebar "Settings" section
2. Choose your LLM provider (OpenAI or Hugging Face)
3. Adjust advanced settings if needed
4. Click "Initialize/Update System"

### 3. Ask Questions

1. In the main area, enter your healthcare question
2. Click "Ask Question"
3. Review the response and sources
4. Use the healthcare tools in the right panel for additional features

### 4. Healthcare Tools

- **Drug Interaction Checker**: Check for medication interactions
- **Medical Terminology Helper**: Expand medical abbreviations
- **System Status**: Monitor system health

## Sample Documents

The system includes sample medical documents in `data/documents/`:

- Diabetes management guidelines
- Hypertension protocols
- Medication safety guidelines

You can use these for testing before uploading your own documents.

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   # Reinstall packages if needed
   pip install --force-reinstall -r requirements.txt
   ```

2. **Memory Issues**
   ```bash
   # Reduce chunk size in .env
   CHUNK_SIZE=500
   # Use smaller embedding models
   ```

3. **Slow Performance**
   - Use OpenAI instead of local models
   - Reduce number of documents processed at once
   - Use more powerful hardware

4. **Model Loading Issues**
   ```bash
   # Clear cache and restart
   rm -rf ~/.cache/huggingface/
   # Or use different models in .env
   ```

### Performance Optimization

1. **For Better Accuracy**:
   - Use OpenAI GPT models
   - Increase chunk overlap
   - Upload high-quality, well-formatted documents

2. **For Better Speed**:
   - Use Hugging Face models
   - Reduce chunk size
   - Limit number of retrieved documents

3. **For Memory Efficiency**:
   - Use CPU-only versions of libraries
   - Process fewer documents at once
   - Use smaller embedding models

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Document Privacy**: Ensure uploaded documents comply with HIPAA/privacy regulations
3. **Access Control**: Implement proper authentication for production use
4. **Data Storage**: Consider encryption for sensitive medical documents

## Production Deployment

### Option 1: Local Server
```bash
# Run on specific port
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

### Option 2: Docker Deployment
```dockerfile
# Create Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Option 3: Cloud Deployment
- Deploy to Streamlit Cloud
- Use AWS/Azure/GCP with container services
- Configure proper security and scaling

## Data Management

### Document Processing
- Supports PDF files up to 50MB
- Automatically chunks documents for optimal retrieval
- Stores embeddings in FAISS vector database

### Vector Store Management
- Located in `./vectorstore/` directory
- Automatically persisted and loaded
- Can be cleared/reset from the interface

### Backup and Recovery
```bash
# Backup vector store
cp -r vectorstore vectorstore_backup

# Restore from backup
rm -rf vectorstore
cp -r vectorstore_backup vectorstore
```

## Support and Maintenance

### Regular Tasks
1. Update dependencies monthly
2. Monitor system performance
3. Backup vector store data
4. Review and update medical documents

### Logs and Monitoring
- Check browser console for errors
- Monitor memory usage
- Review response quality and accuracy

### Updates
```bash
# Update packages
pip install --upgrade -r requirements.txt

# Update LangChain
pip install --upgrade langchain langchain-community
```

## Legal and Compliance

⚠️ **Important**: This system is for educational and research purposes. For production use in healthcare:

1. Ensure compliance with HIPAA and other healthcare regulations
2. Implement proper access controls and audit logging
3. Validate all medical information with authoritative sources
4. Include appropriate disclaimers about AI-generated content
5. Consider liability and insurance requirements

Always consult with medical professionals for actual patient care decisions.

---

For technical support or questions, refer to the project documentation or contact your system administrator.