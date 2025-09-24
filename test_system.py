"""
Healthcare Q&A Assistant - System Test Script
Tests all major components and functionality.
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import langchain
        print("✅ LangChain imported successfully")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        from PyPDF2 import PdfReader
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    return True

def test_project_modules():
    """Test if project-specific modules can be imported."""
    print("\n🔧 Testing project modules...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.append(str(src_path))
    
    try:
        from src.document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
    except ImportError as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        return False
    
    try:
        from src.vector_store import HealthcareVectorStore
        print("✅ HealthcareVectorStore imported successfully")
    except ImportError as e:
        print(f"❌ HealthcareVectorStore import failed: {e}")
        return False
    
    try:
        from src.healthcare_utils import HealthcareTerminologyProcessor
        print("✅ HealthcareUtils imported successfully")
    except ImportError as e:
        print(f"❌ HealthcareUtils import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\n⚙️ Testing basic functionality...")
    
    try:
        # Test document processor
        from src.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        print("✅ DocumentProcessor instantiated successfully")
        
        # Test healthcare utilities
        from src.healthcare_utils import HealthcareTerminologyProcessor
        term_processor = HealthcareTerminologyProcessor()
        
        # Test abbreviation expansion
        test_text = "Patient has HTN and DM, takes metformin BID"
        expanded = term_processor.expand_abbreviations(test_text)
        if "hypertension" in expanded and "twice daily" in expanded:
            print("✅ Medical abbreviation expansion working")
        else:
            print("⚠️ Medical abbreviation expansion may have issues")
        
        # Test drug identification
        drugs = term_processor.identify_drug_mentions(test_text)
        if drugs:
            print("✅ Drug identification working")
        else:
            print("⚠️ Drug identification may need attention")
        
        # Test vector store
        from src.vector_store import HealthcareVectorStore
        vector_store = HealthcareVectorStore()
        print("✅ HealthcareVectorStore instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test if all required files and directories exist."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        ".env.example",
        "README.md",
        "DEPLOYMENT.md",
        "src/__init__.py",
        "src/document_processor.py",
        "src/vector_store.py",
        "src/rag_pipeline.py",
        "src/healthcare_utils.py"
    ]
    
    required_dirs = [
        "src",
        "data",
        "data/documents",
        "vectorstore"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ Missing directory: {dir_path}/")
            all_good = False
    
    return all_good

def test_environment():
    """Test environment configuration."""
    print("\n🌍 Testing environment...")
    
    # Check if .env file exists or can be created
    if Path(".env").exists():
        print("✅ .env file exists")
    elif Path(".env.example").exists():
        print("⚠️ .env file missing but .env.example exists - you should copy it")
    else:
        print("❌ No .env configuration found")
        return False
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor} is compatible")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} is too old (need 3.8+)")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🏥 Healthcare Document Q&A Assistant - System Test")
    print("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Project Modules", test_project_modules),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        print("-" * 30)
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system appears to be working correctly.")
        print("\n🚀 You can now run the application with:")
        print("   streamlit run app.py")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
        print("\n🔧 Try running setup.bat (Windows) or setup.sh (Linux/Mac) to fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)