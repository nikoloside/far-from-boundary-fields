#!/bin/bash
#
# Exp 4: MFCD Definition Validation Runner
#
# This script should be run AFTER exp1-3 complete
# It uses the reconstruction results from exp1-3 to validate Symmetric MFCD
#
# Prerequisites:
#   - Exp 1-3 must have completed and generated meshes
#   - Original ground truth meshes should be in data/original_meshes/
#
# Usage:
#   bash scripts/run_exp4_mfcd_validation.sh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Print header
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}         EXP 4: MFCD DEFINITION VALIDATION${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
echo ""

MISSING=0

# Check Exp 1 results
if [ ! -d "experiments/exp1_udf_baseline/results/meshes" ]; then
    echo -e "${RED}✗ Exp 1 results not found${NC}"
    echo "  Please run: bash scripts/run_core_experiments.sh"
    MISSING=1
else
    EXP1_MESHES=$(ls experiments/exp1_udf_baseline/results/meshes/*.ply 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Exp 1 results found (${EXP1_MESHES} meshes)${NC}"
fi

# Check Exp 2 results (optional but recommended)
if [ ! -d "experiments/exp2_training_trick_ablation/results/meshes" ]; then
    echo -e "${YELLOW}⚠ Exp 2 results not found (optional)${NC}"
else
    EXP2_MESHES=$(ls experiments/exp2_training_trick_ablation/results/meshes/*.ply 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Exp 2 results found (${EXP2_MESHES} meshes)${NC}"
fi

# Check Exp 3 results (optional but recommended)
if [ ! -d "experiments/exp3_activation_ablation/results/meshes" ]; then
    echo -e "${YELLOW}⚠ Exp 3 results not found (optional)${NC}"
else
    EXP3_MESHES=$(ls experiments/exp3_activation_ablation/results/meshes/*.ply 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Exp 3 results found (${EXP3_MESHES} meshes)${NC}"
fi

# Check ground truth
if [ ! -d "data/original_meshes" ]; then
    echo -e "${RED}✗ Ground truth meshes not found at data/original_meshes/${NC}"
    MISSING=1
else
    GT_MESHES=$(ls data/original_meshes/*.ply 2>/dev/null | wc -l)
    if [ "$GT_MESHES" -eq 0 ]; then
        echo -e "${RED}✗ No ground truth meshes found in data/original_meshes/${NC}"
        MISSING=1
    else
        echo -e "${GREEN}✓ Ground truth meshes found (${GT_MESHES} meshes)${NC}"
    fi
fi

echo ""

if [ "$MISSING" -eq 1 ]; then
    echo -e "${RED}Prerequisites not met. Cannot run Exp 4.${NC}"
    exit 1
fi

# Run Exp 4
START_TIME=$(date +%s)

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Running MFCD validation with results from Exp 1-3...${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

python experiments/exp4_mfcd_definition/run.py

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}         EXP 4: MFCD VALIDATION COMPLETED${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Runtime: ${DURATION}s (${DURATION_MIN} min)"
echo ""
echo "Results location:"
echo "  - Exp 4: experiments/exp4_mfcd_definition/results/"
echo "    ├── toy_examples/          # Toy validation results"
echo "    ├── metrics/               # Symmetric MFCD metrics"
echo "    └── figures/               # Comparison visualizations"
echo ""
echo "Key findings:"
echo "  - Symmetric MFCD vs Traditional MFCD correlation analysis"
echo "  - Fragment precision/recall correlation"
echo "  - Method ranking comparison"
echo ""
echo "Logs: docs/experiments_log.md"
echo ""
echo -e "${GREEN}✅ MFCD validation completed!${NC}"
