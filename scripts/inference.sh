#!/bin/bash

# Vietnamese-Chinese Machine Translation - Inference Script
# ===========================================================
# This script runs inference on input data

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CHECKPOINT=""
INPUT_FILE=""
OUTPUT_FILE=""
BEAM_SIZE=3
TOP_K=5
LENGTH_PENALTY=0.6
BATCH_SIZE=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --beam-size)
            BEAM_SIZE="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --length-penalty)
            LENGTH_PENALTY="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --greedy)
            BEAM_SIZE=1
            shift
            ;;
        --help)
            echo "Usage: bash inference.sh [OPTIONS]"
            echo ""
            echo "Required Options:"
            echo "  --checkpoint PATH         Path to model checkpoint"
            echo "  --input PATH              Path to input file (one sentence per line)"
            echo "  --output PATH             Path to output CSV file"
            echo ""
            echo "Optional Decoding Options:"
            echo "  --beam-size NUM           Beam search size (default: 3)"
            echo "  --top-k NUM               Top-k filtering (default: 5)"
            echo "  --length-penalty FLOAT    Length penalty (default: 0.6)"
            echo "  --greedy                  Use greedy decoding (sets beam-size=1)"
            echo ""
            echo "Other Options:"
            echo "  --batch-size NUM          Batch size (default: 32)"
            echo "  --device DEVICE           cuda or cpu (default: cuda)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash inference.sh --checkpoint best_model.pt --input test.txt --output results.csv"
            echo "  bash inference.sh --checkpoint best_model.pt --input test.txt --output results.csv --greedy"
            echo "  bash inference.sh --checkpoint best_model.pt --input test.txt --output results.csv --beam-size 5 --device cpu"
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
if [ -z "$CHECKPOINT" ] || [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Vietnamese-Chinese Machine Translation - Inference         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Display configuration
echo -e "${YELLOW}Inference Configuration:${NC}"
echo "  Checkpoint:       $CHECKPOINT"
echo "  Input:            $INPUT_FILE"
echo "  Output:           $OUTPUT_FILE"
echo "  Beam size:        $BEAM_SIZE"
echo "  Top-k:            $TOP_K"
echo "  Length penalty:   $LENGTH_PENALTY"
echo "  Batch size:       $BATCH_SIZE"
echo ""

# Count input lines
INPUT_COUNT=$(wc -l < "$INPUT_FILE")
echo -e "${YELLOW}Input Summary:${NC}"
echo "  Total lines:      $INPUT_COUNT"
echo ""

# Run inference
echo -e "${YELLOW}Starting inference...${NC}"
python -m inference.main \
    --checkpoint "$CHECKPOINT" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --beam_size "$BEAM_SIZE" \
    --top_k "$TOP_K" \
    --length_penalty "$LENGTH_PENALTY" \
    --batch_size "$BATCH_SIZE"

EXITCODE=$?

echo ""
if [ $EXITCODE -eq 0 ]; then
    # Count output lines
    OUTPUT_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo -e "${GREEN}✅ Inference completed successfully!${NC}"
    echo ""
    echo "Output Summary:"
    echo "  Output file:      $OUTPUT_FILE"
    echo "  Output lines:     $OUTPUT_COUNT"
    echo ""
    echo "Command to view results:"
    echo "  head -10 $OUTPUT_FILE"
else
    echo -e "${RED}❌ Inference failed with exit code: $EXITCODE${NC}"
    exit $EXITCODE
fi
