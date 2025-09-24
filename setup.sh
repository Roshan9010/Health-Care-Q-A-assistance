#!/bin/bash
# Healthcare Q&A Assistant Setup Script
# This script automates the setup process for the Healthcare Document Q&A Assistant

echo "🏥 Healthcare Document Q&A Assistant Setup Script"
echo "================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
if ! command_exists python3; then
    if ! command_exists python; then
        echo "❌ Error: Python is not installed. Please install Python 3.8 or higher."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python found: $PYTHON_CMD"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $PYTHON_VERSION found. Python 3.8 or higher is required."
    exit 1
fi

echo "✅ Python version $PYTHON_VERSION is compatible"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv healthcare_env

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created"

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source healthcare_env/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📥 Installing required packages..."
echo "This may take several minutes as it downloads large ML models..."

# Install packages in stages to handle potential issues
echo "Installing core packages..."
pip install streamlit python-dotenv PyPDF2 numpy pandas

echo "Installing AI/ML packages..."
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
pip install faiss-cpu

echo "Installing LangChain packages..."
pip install langchain langchain-community langchain-openai
pip install openai tiktoken

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install some packages. Trying alternative installation..."
    pip install -r requirements.txt --no-cache-dir
fi

echo "✅ Packages installed successfully"

# Create .env file from template
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    echo "📝 Creating .env configuration file..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it with your API keys if needed."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/documents
mkdir -p vectorstore

echo "✅ Directories created"

# Download sample model (optional)
echo "🤖 Downloading sample embedding model (this may take a few minutes)..."
python -c "
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('✅ Sample model downloaded successfully')
except Exception as e:
    print(f'⚠️ Warning: Could not download model: {e}')
    print('Model will be downloaded on first use')
"

# Test import
echo "🧪 Testing installation..."
python -c "
import streamlit
import langchain
import faiss
import sentence_transformers
import PyPDF2
print('✅ All core packages imported successfully')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Installation test passed"
else
    echo "⚠️ Warning: Some packages may not be properly installed"
fi

# Create startup script
echo "📝 Creating startup script..."
cat > start_app.sh << 'EOF'
#!/bin/bash
# Healthcare Q&A Assistant Startup Script

echo "🏥 Starting Healthcare Document Q&A Assistant..."

# Activate virtual environment
source healthcare_env/bin/activate

# Check if streamlit is available
if ! command -v streamlit >/dev/null 2>&1; then
    echo "❌ Error: Streamlit not found. Please run setup.sh first."
    exit 1
fi

# Start the application
echo "🚀 Launching Streamlit application..."
echo "The application will open in your browser at http://localhost:8501"
streamlit run app.py

EOF

chmod +x start_app.sh

# Create Windows startup script
cat > start_app.bat << 'EOF'
@echo off
echo 🏥 Starting Healthcare Document Q&A Assistant...

REM Activate virtual environment
call healthcare_env\Scripts\activate.bat

REM Check if streamlit is available
where streamlit >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Error: Streamlit not found. Please run setup first.
    pause
    exit /b 1
)

REM Start the application
echo 🚀 Launching Streamlit application...
echo The application will open in your browser at http://localhost:8501
streamlit run app.py

pause
EOF

echo "📄 Setup Summary:"
echo "=================="
echo "✅ Virtual environment: healthcare_env"
echo "✅ Python packages installed"
echo "✅ Configuration file created: .env"
echo "✅ Directories created: data/documents, vectorstore"
echo "✅ Startup scripts created: start_app.sh (Linux/Mac), start_app.bat (Windows)"
echo ""
echo "🚀 Next Steps:"
echo "1. Edit .env file with your API keys (optional)"
echo "2. Run the application:"
echo "   Linux/Mac: ./start_app.sh"
echo "   Windows: start_app.bat"
echo "   Manual: streamlit run app.py"
echo ""
echo "📚 For detailed instructions, see DEPLOYMENT.md"
echo ""
echo "🎉 Setup completed successfully!"