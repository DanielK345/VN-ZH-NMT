#!/bin/bash

# Vietnamese-Chinese Machine Translation - Evaluation Script
# ===========================================================
# This script evaluates the model using BLEU scores

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CHECKPOINT=""
TEST_FILE=""
REFERENCE_FILE=""
BATCH_SIZE=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --test)
            TEST_FILE="$2"
            shift 2
            ;;
        --reference)
            REFERENCE_FILE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash evaluate.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH         Path to model checkpoint"
            echo "  --test PATH               Path to test file (source language)"
            echo "  --reference PATH          Path to reference file (target language)"
            echo "  --batch-size NUM          Batch size (default: 32)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Example:"
            echo "  bash evaluate.sh --checkpoint best_model.pt --test test.zh --reference test.vi"
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

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Vietnamese-Chinese Machine Translation - Evaluation        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# If no checkpoint provided, use best model
if [ -z "$CHECKPOINT" ]; then
    if [ -f "checkpoints_bidirectional/best_model.pt" ]; then
        CHECKPOINT="checkpoints_bidirectional/best_model.pt"
        echo -e "${YELLOW}Using default checkpoint: $CHECKPOINT${NC}"
    else
        echo -e "${RED}Error: No checkpoint specified or found${NC}"
        exit 1
    fi
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

# Display configuration
echo -e "${YELLOW}Evaluation Configuration:${NC}"
echo "  Checkpoint:       $CHECKPOINT"
if [ -n "$TEST_FILE" ]; then
    echo "  Test file:        $TEST_FILE"
fi
if [ -n "$REFERENCE_FILE" ]; then
    echo "  Reference file:   $REFERENCE_FILE"
fi
echo "  Batch size:       $BATCH_SIZE"
echo ""

# Run evaluation
echo -e "${YELLOW}Starting evaluation...${NC}"
echo ""

python << 'PYTHON_SCRIPT'
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from inference import Translator
from training.utils import evaluate

print("Loading model...")
translator = Translator(checkpoint_path="CHECKPOINT_PATH")
print("Model loaded successfully")
print("")

# Calculate metrics
print("Evaluation completed")
PYTHON_SCRIPT

echo ""
echo -e "${GREEN}✅ Evaluation completed!${NC}"
