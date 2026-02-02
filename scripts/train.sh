#!/bin/bash

# Vietnamese-Chinese Machine Translation - Training Script
# =========================================================
# This script trains the machine translation model

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SKIP_TRAINING=false
FORCE_REPROCESS=false
RESUME_FROM=""
NUM_EPOCHS=40
BATCH_SIZE=128
LR=0.0002

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --force-reprocess)
            FORCE_REPROCESS=true
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-training           Skip model training (data processing only)"
            echo "  --force-reprocess         Force reprocessing of all data"
            echo "  --resume CHECKPOINT       Resume training from checkpoint"
            echo "  --epochs NUM              Number of epochs (default: 40)"
            echo "  --batch-size SIZE         Batch size (default: 128)"
            echo "  --lr LEARNING_RATE        Learning rate (default: 0.0002)"
            echo "  --help                    Show this help message"
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
echo -e "${BLUE}║     Vietnamese-Chinese Machine Translation - Training          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if required files exist
echo -e "${YELLOW}Checking prerequisites...${NC}"

if [ ! -d "dataset/train" ]; then
    echo -e "${RED}Error: dataset/train directory not found${NC}"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

# Display configuration
echo -e "${YELLOW}Training Configuration:${NC}"
echo "  Epochs:           $NUM_EPOCHS"
echo "  Batch size:       $BATCH_SIZE"
echo "  Learning rate:    $LR"
echo "  Skip training:    $SKIP_TRAINING"
echo "  Force reprocess:  $FORCE_REPROCESS"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resume from:      $RESUME_FROM"
fi
echo ""

# Build Python command
PYTHON_CMD="python -m training.main"
PYTHON_CMD="$PYTHON_CMD --num_epochs $NUM_EPOCHS"
PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --lr $LR"

if [ "$SKIP_TRAINING" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip_training"
fi

if [ "$FORCE_REPROCESS" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --force_reprocess"
fi

if [ -n "$RESUME_FROM" ]; then
    PYTHON_CMD="$PYTHON_CMD --resume_from $RESUME_FROM"
fi

# Run training
echo -e "${YELLOW}Starting training pipeline...${NC}"
echo -e "${BLUE}Command: $PYTHON_CMD${NC}"
echo ""

$PYTHON_CMD

EXITCODE=$?

echo ""
if [ $EXITCODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  - Check checkpoints in: checkpoints_bidirectional/"
    echo "  - Run inference: bash scripts/inference.sh"
    echo "  - Evaluate model: bash scripts/evaluate.sh"
else
    echo -e "${RED}❌ Training failed with exit code: $EXITCODE${NC}"
    exit $EXITCODE
fi
