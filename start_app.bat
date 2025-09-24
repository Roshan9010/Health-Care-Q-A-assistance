@echo off
echo 🏥 Starting Healthcare Document Q

REM Activate virtual environment
call healthcare_env\Scripts\activate.bat

REM Check if streamlit is available
where streamlit >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Error: Streamlit not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Start the application
echo 🚀 Launching Streamlit application...
echo The application will open in your browser at http://localhost:8501
streamlit run app.py

pause
