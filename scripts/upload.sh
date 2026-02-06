#!/bin/bash

# Vietnamese-Chinese Machine Translation - Upload to Hugging Face Hub
# =====================================================================
# This script uploads trained model checkpoints to Hugging Face Hub

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PRIVATE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --private)
            PRIVATE=true
            shift
            ;;
        --description)
            DESCRIPTION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash upload.sh [OPTIONS]"
            echo ""
            echo "Upload trained model to Hugging Face Hub"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH      Path to model checkpoint (.pt file) [REQUIRED]"
            echo "  --repo-id ID           Hugging Face repo ID: username/repo-name [REQUIRED]"
            echo "  --token TOKEN          HF API token (uses HF_TOKEN env var if not set)"
            echo "  --private              Make repository private (default: public)"
            echo "  --description TEXT     Custom model description"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Upload best model to public repo"
            echo "  bash upload.sh \\"
            echo "    --checkpoint checkpoints_bidirectional/best_model.pt \\"
            echo "    --repo-id username/vn-zh-translation"
            echo ""
            echo "  # Upload with private repo and custom description"
            echo "  bash upload.sh \\"
            echo "    --checkpoint checkpoints_bidirectional/best_model.pt \\"
            echo "    --repo-id username/vn-zh-translation \\"
            echo "    --private \\"
            echo "    --description 'v2 with improved curriculum learning'"
            echo ""
            echo "Setup:"
            echo "  1. Create a Hugging Face account: https://huggingface.co"
            echo "  2. Create an access token: https://huggingface.co/settings/tokens"
            echo "  3. Set HF_TOKEN environment variable:"
            echo "     export HF_TOKEN='your_token_here'"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: --checkpoint is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$REPO_ID" ]; then
    echo -e "${RED}Error: --repo-id is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: Checkpoint file not found: $CHECKPOINT_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Upload to Hugging Face Hub                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found in PATH${NC}"
    exit 1
fi

# Build Python command
PYTHON_CMD="python upload_to_hub.py"
PYTHON_CMD="$PYTHON_CMD '$CHECKPOINT_PATH'"
PYTHON_CMD="$PYTHON_CMD --repo-id '$REPO_ID'"

if [ -n "$HF_TOKEN" ]; then
    PYTHON_CMD="$PYTHON_CMD --token '$HF_TOKEN'"
fi

if [ "$PRIVATE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --private"
fi

if [ -n "$DESCRIPTION" ]; then
    PYTHON_CMD="$PYTHON_CMD --description '$DESCRIPTION'"
fi

# Run upload
echo -e "${YELLOW}Starting upload...${NC}"
echo ""
eval $PYTHON_CMD

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Upload completed successfully!${NC}"
    echo ""
    echo "Model is now available on Hugging Face Hub:"
    echo "  https://huggingface.co/$REPO_ID"
else
    echo -e "${RED}❌ Upload failed with exit code: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi
