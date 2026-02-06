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
OVERRIDE_EPOCHS=""
OVERRIDE_LR=""

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
            OVERRIDE_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            OVERRIDE_LR="$2"
            shift 2
            ;;
        --batch-size)
            echo -e "${YELLOW}⚠️  Note: --batch-size is no longer supported.${NC}"
            echo "Batch size is configured in training/config.py instead."
            shift 2
            ;;
        --help)
            echo "Usage: bash train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-training           Skip model training (data processing only)"
            echo "  --force-reprocess         Force reprocessing of all data"
            echo "  --resume CHECKPOINT       Resume training from checkpoint"
            echo "  --epochs N                Override number of epochs (only with --resume)"
            echo "  --lr RATE                 Override learning rate (only with --resume)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash train.sh"
            echo "  bash train.sh --resume checkpoints_bidirectional/best_model.pt"
            echo "  bash train.sh --resume checkpoints_bidirectional/best_model.pt --epochs 50"
            echo "  bash train.sh --resume checkpoints_bidirectional/best_model.pt --epochs 50 --lr 0.0001"
            echo ""
            echo "Customizing Base Hyperparameters:"
            echo "  Edit training/config.py to change base epochs, batch_size, and learning rate"
            echo "  Example: RopeConfig(num_epochs=40, batch_size=128, lr=0.0002)"
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
echo "  Skip training:    $SKIP_TRAINING"
echo "  Force reprocess:  $FORCE_REPROCESS"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resume from:      $RESUME_FROM"
fi
if [ -n "$OVERRIDE_EPOCHS" ]; then
    echo "  Override epochs:  $OVERRIDE_EPOCHS"
fi
if [ -n "$OVERRIDE_LR" ]; then
    echo "  Override lr:      $OVERRIDE_LR"
fi
echo ""
echo -e "${YELLOW}Note:${NC} Base hyperparameters (epochs, batch_size, lr) are configured in training/config.py"
echo ""

# Build Python command
# Hyperparameters (epochs, batch_size, lr) can be overridden for any training mode
# The CLI supports: --base_dir, --force_reprocess, --skip_training, --resume_checkpoint, --epochs, --lr
PYTHON_CMD="python -m training.main --base_dir ."

if [ "$SKIP_TRAINING" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip_training"
fi

if [ "$FORCE_REPROCESS" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --force_reprocess"
fi

if [ -n "$RESUME_FROM" ]; then
    PYTHON_CMD="$PYTHON_CMD --resume_checkpoint $RESUME_FROM"
fi

# Add optional override parameters (valid for any training mode)
if [ -n "$OVERRIDE_EPOCHS" ]; then
    PYTHON_CMD="$PYTHON_CMD --epochs $OVERRIDE_EPOCHS"
fi

if [ -n "$OVERRIDE_LR" ]; then
    PYTHON_CMD="$PYTHON_CMD --lr $OVERRIDE_LR"
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
