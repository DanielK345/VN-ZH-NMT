#!/bin/bash

# Vietnamese-Chinese Machine Translation - Utility Script
# ========================================================
# This script provides various utility functions

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show project info
show_info() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Vietnamese-Chinese Machine Translation System               ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "📊 Project Statistics:"
    echo ""
    
    # Count files
    PYTHON_FILES=$(find . -name "*.py" -type f | grep -v __pycache__ | wc -l)
    NOTEBOOKS=$(find . -name "*.ipynb" -type f | wc -l)
    DOCS=$(find . -name "*.md" -type f | wc -l)
    
    echo "  Python files:    $PYTHON_FILES"
    echo "  Notebooks:       $NOTEBOOKS"
    echo "  Documentation:   $DOCS"
    echo ""
    
    # Check directories
    echo "📁 Directory Structure:"
    echo "  training/        $([ -d training ] && echo '✓' || echo '✗')"
    echo "  inference/       $([ -d inference ] && echo '✓' || echo '✗')"
    echo "  model/           $([ -d model ] && echo '✓' || echo '✗')"
    echo "  notebooks/       $([ -d notebooks ] && echo '✓' || echo '✗')"
    echo "  scripts/         $([ -d scripts ] && echo '✓' || echo '✗')"
    echo "  dataset/         $([ -d dataset ] && echo '✓' || echo '✗')"
    echo ""
    
    # Check key files
    echo "📄 Key Files:"
    echo "  README.md        $([ -f README.md ] && echo '✓' || echo '✗')"
    echo "  requirements.txt $([ -f requirements.txt ] && echo '✓' || echo '✗')"
    echo "  QUICKSTART.py    $([ -f QUICKSTART.py ] && echo '✓' || echo '✗')"
    echo ""
}

# Function to clean up cache and temporary files
clean_cache() {
    echo -e "${YELLOW}Cleaning cache and temporary files...${NC}"
    echo ""
    
    # Remove Python cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove pytest cache
    rm -rf .pytest_cache/ 2>/dev/null || true
    
    # Remove mypy cache
    rm -rf .mypy_cache/ 2>/dev/null || true
    
    # Remove coverage files
    rm -rf .coverage 2>/dev/null || true
    rm -rf htmlcov/ 2>/dev/null || true
    
    # Remove temporary files
    find . -type f -name "*.tmp" -delete 2>/dev/null || true
    find . -type f -name "*.bak" -delete 2>/dev/null || true
    
    echo -e "${GREEN}✓ Cache cleaned${NC}"
    echo ""
}

# Function to verify project structure
verify_project() {
    echo -e "${YELLOW}Verifying project structure...${NC}"
    echo ""
    
    python3 << 'PYTHON_END'
import os
import sys
from pathlib import Path

def verify():
    errors = []
    warnings = []
    
    # Required directories
    required_dirs = ['training', 'inference', 'model', 'notebooks', 'dataset']
    for d in required_dirs:
        if not os.path.isdir(d):
            errors.append(f"Missing directory: {d}/")
    
    # Required files
    required_files = ['README.md', 'requirements.txt', '.gitignore']
    for f in required_files:
        if not os.path.isfile(f):
            errors.append(f"Missing file: {f}")
    
    # Check Python files
    python_files = list(Path('.').rglob('*.py'))
    if len(python_files) < 10:
        warnings.append(f"Few Python files found: {len(python_files)}")
    
    # Print results
    if errors:
        print("❌ Errors found:")
        for e in errors:
            print(f"   {e}")
        return False
    else:
        print("✅ All checks passed!")
        if warnings:
            print("⚠️  Warnings:")
            for w in warnings:
                print(f"   {w}")
        return True

if __name__ == "__main__":
    verify()
PYTHON_END
    echo ""
}

# Function to show usage statistics
show_stats() {
    echo -e "${YELLOW}Computing statistics...${NC}"
    echo ""
    
    python3 << 'PYTHON_END'
import os
from pathlib import Path
from collections import defaultdict

def analyze_project():
    stats = defaultdict(int)
    
    # Count lines and files
    for py_file in Path('.').rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                stats['python_lines'] += lines
                stats['python_files'] += 1
        except:
            pass
    
    # Count markdown
    for md_file in Path('.').rglob('*.md'):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                stats['doc_lines'] += lines
                stats['doc_files'] += 1
        except:
            pass
    
    # Count notebooks
    stats['notebooks'] = len(list(Path('./notebooks').glob('*.ipynb')))
    
    # Print stats
    print("📊 Code Statistics:")
    print(f"  Python files:     {stats['python_files']}")
    print(f"  Python lines:     {stats['python_lines']}")
    print(f"  Documentation:    {stats['doc_files']} files, {stats['doc_lines']} lines")
    print(f"  Notebooks:        {stats['notebooks']}")
    print("")
    
    total = stats['python_lines'] + stats['doc_lines']
    print(f"  Total lines:      {total}")

if __name__ == "__main__":
    analyze_project()
PYTHON_END
    echo ""
}

# Function to generate requirements
generate_requirements() {
    echo -e "${YELLOW}Checking Python environment...${NC}"
    echo ""
    
    python3 << 'PYTHON_END'
import sys

packages = {
    'torch': 'PyTorch',
    'sentencepiece': 'SentencePiece',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'tqdm': 'tqdm',
    'sacrebleu': 'SacreBLEU'
}

print("📦 Installed Packages:")
print("")

missing = []
for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {name:15} {version}")
    except ImportError:
        print(f"  ✗ {name:15} NOT INSTALLED")
        missing.append(module)

print("")
if missing:
    print(f"❌ Missing packages: {', '.join(missing)}")
    print("Run: pip install -r requirements.txt")
else:
    print("✅ All required packages installed")
PYTHON_END
    echo ""
}

# Function to show help
show_help() {
    echo "Usage: bash utils.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  info              Show project information"
    echo "  verify            Verify project structure"
    echo "  stats             Show code statistics"
    echo "  clean             Clean cache and temporary files"
    echo "  check-env         Check Python environment and packages"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash scripts/utils.sh info"
    echo "  bash scripts/utils.sh verify"
    echo "  bash scripts/utils.sh clean"
    echo ""
}

# Main
COMMAND="${1:-help}"

case $COMMAND in
    info)
        show_info
        ;;
    verify)
        verify_project
        ;;
    stats)
        show_stats
        ;;
    clean)
        clean_cache
        ;;
    check-env)
        generate_requirements
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Use 'bash scripts/utils.sh help' for usage information"
        exit 1
        ;;
esac
