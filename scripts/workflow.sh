#!/bin/bash

# Vietnamese-Chinese Machine Translation - Full Workflow Script
# ==============================================================
# This script runs the complete workflow: setup, training, and inference

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SETUP=false
TRAIN=false
INFERENCE=false
DATA_ONLY=false
TEST_INFERENCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            SETUP=true
            TRAIN=true
            TEST_INFERENCE=true
            shift
            ;;
        --setup)
            SETUP=true
            shift
            ;;
        --train)
            TRAIN=true
            shift
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --inference)
            INFERENCE=true
            shift
            ;;
        --test-inference)
            TEST_INFERENCE=true
            shift
            ;;
        --help)
            echo "Usage: bash workflow.sh [OPTIONS]"
            echo ""
            echo "Workflow Options:"
            echo "  --all                     Run complete workflow (setup + train + test inference)"
            echo "  --setup                   Run setup script only"
            echo "  --train                   Run training script only"
            echo "  --data-only               Run training data processing only (no model training)"
            echo "  --inference               Run inference (requires checkpoint)"
            echo "  --test-inference          Run test inference with sample data"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash workflow.sh --all                  # Complete workflow"
            echo "  bash workflow.sh --setup --train        # Setup and train"
            echo "  bash workflow.sh --data-only            # Prepare data only"
            echo "  bash workflow.sh --test-inference       # Test with sample data"
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

# If no options provided, show help
if [ "$SETUP" = false ] && [ "$TRAIN" = false ] && [ "$INFERENCE" = false ] && \
   [ "$DATA_ONLY" = false ] && [ "$TEST_INFERENCE" = false ]; then
    echo "Use --help for usage information or specify an action (--all, --setup, --train, etc.)"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Vietnamese-Chinese MT - Workflow Orchestration          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Setup
if [ "$SETUP" = true ]; then
    echo -e "${YELLOW}[1/5] Running setup...${NC}"
    echo ""
    bash scripts/setup.sh
    echo ""
    echo -e "${GREEN}✓ Setup completed${NC}"
    echo ""
fi

# Data processing only
if [ "$DATA_ONLY" = true ]; then
    echo -e "${YELLOW}[2/5] Running data processing...${NC}"
    echo ""
    bash scripts/train.sh --skip-training --force-reprocess
    echo ""
    echo -e "${GREEN}✓ Data processing completed${NC}"
    echo ""
    exit 0
fi

# Training
if [ "$TRAIN" = true ]; then
    echo -e "${YELLOW}[2/5] Running training...${NC}"
    echo ""
    bash scripts/train.sh
    echo ""
    echo -e "${GREEN}✓ Training completed${NC}"
    echo ""
fi

# Test inference
if [ "$TEST_INFERENCE" = true ]; then
    echo -e "${YELLOW}[3/5] Running test inference...${NC}"
    echo ""
    
    # Check if checkpoint exists
    CHECKPOINT=""
    if [ -f "checkpoints_bidirectional/best_model.pt" ]; then
        CHECKPOINT="checkpoints_bidirectional/best_model.pt"
    elif [ -f "checkpoints_bidirectional/last_model.pt" ]; then
        CHECKPOINT="checkpoints_bidirectional/last_model.pt"
    fi
    
    if [ -z "$CHECKPOINT" ]; then
        echo -e "${RED}Error: No checkpoint found in checkpoints_bidirectional/${NC}"
        echo "Please run training first: bash scripts/train.sh"
        exit 1
    fi
    
    echo "Using checkpoint: $CHECKPOINT"
    echo ""
    
    # Create test file if it doesn't exist
    TEST_INPUT="test_input.txt"
    if [ ! -f "$TEST_INPUT" ]; then
        echo "Creating test input file..."
        cat > "$TEST_INPUT" << 'EOF'
这是一个测试句子。
你好，世界！
今天天气很好。
EOF
    fi
    
    # Run inference
    bash scripts/inference.sh \
        --checkpoint "$CHECKPOINT" \
        --input "$TEST_INPUT" \
        --output test_output.csv \
        --beam-size 3
    
    echo ""
    echo -e "${GREEN}✓ Test inference completed${NC}"
    echo ""
    echo "Results saved to: test_output.csv"
    echo ""
fi

# Regular inference
if [ "$INFERENCE" = true ]; then
    echo -e "${YELLOW}[4/5] Running inference...${NC}"
    echo ""
    bash scripts/inference.sh --help
    echo ""
fi

echo -e "${GREEN}✅ Workflow completed!${NC}"
echo ""
