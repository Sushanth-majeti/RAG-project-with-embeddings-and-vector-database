"""
Setup and validation script for RAG evaluation project.
Run this before executing main.py to ensure all dependencies are installed.
"""
import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.8."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")


def install_requirements():
    """Install requirements from requirements.txt."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt", "-q"
        ])
        print("✓ All dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Run manually: pip install -r requirements.txt")
        return False
    return True


def check_imports():
    """Check if key packages can be imported."""
    print("\n📚 Checking imports...")
    required = [
        'numpy',
        'pandas',
        'qdrant_client',
        'sentence_transformers',
    ]
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} not found")
            return False
    return True


def create_directories():
    """Create necessary project directories."""
    print("\n📁 Creating directories...")
    dirs = ['projects', 'results', 'data', 'qdrant_storage']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"  ✓ {d}/")


def verify_files():
    """Verify key files exist."""
    print("\n🔍 Verifying files...")
    files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'data/queries.json',
        'src/__init__.py',
        'src/chunking.py',
        'src/embeddings.py',
        'src/evaluation.py',
        'src/document_loader.py',
        'src/vector_db.py',
        'src/utils.py',
    ]
    
    for f in files:
        if Path(f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ⚠️  Missing: {f}")
    

def main():
    """Run all setup checks."""
    print("=" * 60)
    print("RAG EVALUATION PROJECT - SETUP")
    print("=" * 60)
    
    check_python_version()
    
    if not install_requirements():
        sys.exit(1)
    
    if not check_imports():
        print("\n❌ Some imports failed. Install dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    create_directories()
    verify_files()
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\n▶️  Next step: python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
