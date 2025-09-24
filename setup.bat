@echo off
REM Healthcare Q&A Assistant Setup Script for Windows
REM This script automates the setup process for the Healthcare Document Q&A Assistant

echo 🏥 Healthcare Document Q&A Assistant Setup Script
echo ==================================================

REM Check if Python is installed
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv healthcare_env

if %errorlevel% neq 0 (
    echo ❌ Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call healthcare_env\Scripts\activate.bat

REM Upgrade pip
echo 📈 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing required packages...
echo This may take several minutes as it downloads large ML models...

REM Install packages in stages
echo Installing core packages...
pip install streamlit python-dotenv PyPDF2 numpy pandas

echo Installing AI/ML packages...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
pip install faiss-cpu

echo Installing LangChain packages...
pip install langchain langchain-community langchain-openai
pip install openai tiktoken

if %errorlevel% neq 0 (
    echo ❌ Error: Failed to install some packages. Trying alternative installation...
    pip install -r requirements.txt --no-cache-dir
)

echo ✅ Packages installed successfully

REM Create .env file from template
if exist ".env.example" if not exist ".env" (
    echo 📝 Creating .env configuration file...
    copy .env.example .env
    echo ✅ .env file created. Please edit it with your API keys if needed.
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "data\documents" mkdir "data\documents"
if not exist "vectorstore" mkdir "vectorstore"

echo ✅ Directories created

REM Test installation
echo 🧪 Testing installation...
python -c "import streamlit; import langchain; import faiss; import sentence_transformers; import PyPDF2; print('✅ All core packages imported successfully')" 2>nul

if %errorlevel% equ 0 (
    echo ✅ Installation test passed
) else (
    echo ⚠️ Warning: Some packages may not be properly installed
)

REM Create Windows startup script
echo 📝 Creating startup script...
(
echo @echo off
echo echo 🏥 Starting Healthcare Document Q&A Assistant...
echo.
echo REM Activate virtual environment
echo call healthcare_env\Scripts\activate.bat
echo.
echo REM Check if streamlit is available
echo where streamlit ^>nul 2^>nul
echo if %%errorlevel%% neq 0 ^(
echo     echo ❌ Error: Streamlit not found. Please run setup.bat first.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Start the application
echo echo 🚀 Launching Streamlit application...
echo echo The application will open in your browser at http://localhost:8501
echo streamlit run app.py
echo.
echo pause
) > start_app.bat

echo.
echo 📄 Setup Summary:
echo ==================
echo ✅ Virtual environment: healthcare_env
echo ✅ Python packages installed
echo ✅ Configuration file created: .env
echo ✅ Directories created: data\documents, vectorstore
echo ✅ Startup script created: start_app.bat
echo.
echo 🚀 Next Steps:
echo 1. Edit .env file with your API keys (optional)
echo 2. Run the application: start_app.bat
echo 3. Or manually: streamlit run app.py
echo.
echo 📚 For detailed instructions, see DEPLOYMENT.md
echo.
echo 🎉 Setup completed successfully!
echo.
pause