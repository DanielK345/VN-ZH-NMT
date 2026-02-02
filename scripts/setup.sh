#!/bin/bash

# Vietnamese-Chinese Machine Translation - Setup Script
# ======================================================
# This script sets up the project environment and installs dependencies

set -e  # Exit on error

echo "🚀 Setting up Vietnamese-Chinese Machine Translation System..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}Virtual environment activated${NC}"
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip3 install --upgrade pip setuptools wheel

# Install dependencies
echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import sentencepiece; print(f'SentencePiece installed')"
python3 -c "import pandas; print(f'Pandas installed')"

echo ""
echo -e "${GREEN}✅ Setup completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review README.md for configuration options"
echo "  2. Prepare your data in dataset/ folder"
echo "  3. Run: bash scripts/train.sh (to train model)"
echo "  4. Run: bash scripts/inference.sh (to run inference)"
echo ""
