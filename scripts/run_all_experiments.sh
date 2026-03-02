#!/bin/bash
#
# Master Experiment Runner
#
# Runs experiments in the correct order:
# 1. Exp 1: FFB vs UDF vs NeuralUDF + Flooding vs MIND
# 2. Exp 2: FFB+Flooding vs Training Tricks
# 3. Exp 3: FFB+Flooding vs Activations
# 4. Exp 4: MFCD Definition Validation
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --minimal
#   bash scripts/run_all_experiments.sh --quick

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
MODE=""
if [[ "$*" == *"--minimal"* ]]; then
    MODE="--minimal"
    echo "Mode: MINIMAL"
elif [[ "$*" == *"--quick"* ]]; then
    MODE="--quick"
    echo "Mode: QUICK"
else
    echo "Mode: FULL"
fi

cd "$ROOT_DIR"

# Print header
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}             RUNNING ALL EXPERIMENTS (1-4)${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Complete verification sequence:"
echo "  Phase 1: Exp 1-3 (Core experiments - generate reconstruction results)"
echo "    1. Exp 1: FFB vs UDF vs NeuralUDF + Flooding vs MIND"
echo "    2. Exp 2: FFB+Flooding vs Training Tricks"
echo "    3. Exp 3: FFB+Flooding vs Activations"
echo "  Phase 2: Exp 4 (MFCD validation using Exp 1-3 results)"
echo "    4. Exp 4: Symmetric MFCD Definition Validation"
echo ""
echo "NOTE: This runs ALL experiments end-to-end"
echo "      For faster testing, use: bash scripts/run_core_experiments.sh"
echo ""

START_TIME=$(date +%s)

# ========== EXP 1: UDF BASELINE ==========
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}EXP 1: UDF Baseline (编码+架构+后处理对比)${NC}"
echo -e "${GREEN}======================================================================${NC}"
python experiments/exp1_udf_baseline/run.py $MODE

# ========== EXP 2: TRAINING TRICK ABLATION ==========
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}EXP 2: Training Trick Ablation (训练技巧消融)${NC}"
echo -e "${GREEN}======================================================================${NC}"
python experiments/exp2_training_trick_ablation/run.py $MODE

# ========== EXP 3: ACTIVATION ABLATION ==========
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}EXP 3: Activation Ablation (激活函数消融)${NC}"
echo -e "${GREEN}======================================================================${NC}"
python experiments/exp3_activation_ablation/run.py $MODE

# ========== EXP 4: MFCD DEFINITION (使用Exp 1-3结果) ==========
echo ""
echo -e "${YELLOW}======================================================================${NC}"
echo -e "${YELLOW}EXP 4: MFCD Definition Validation (使用Exp 1-3结果)${NC}"
echo -e "${YELLOW}======================================================================${NC}"
echo "This experiment validates Symmetric MFCD using results from Exp 1-3"
echo ""
python experiments/exp4_mfcd_definition/run.py $MODE

# ========== SUMMARY ==========
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}              ALL EXPERIMENTS (1-4) COMPLETED${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Total runtime: ${DURATION}s (${DURATION_MIN} min)"
echo ""
echo "Results locations:"
echo "  Core experiments (reconstruction):"
echo "    - Exp 1: experiments/exp1_udf_baseline/results/"
echo "    - Exp 2: experiments/exp2_training_trick_ablation/results/"
echo "    - Exp 3: experiments/exp3_activation_ablation/results/"
echo ""
echo "  MFCD validation:"
echo "    - Exp 4: experiments/exp4_mfcd_definition/results/"
echo ""
echo "Logs: docs/experiments_log.md"
echo ""
echo "Summary:"
echo "  - Phase 1 (Exp 1-3): Generated reconstruction meshes"
echo "  - Phase 2 (Exp 4): Validated Symmetric MFCD definition"
echo ""
echo -e "${GREEN}✅ All experiments (1-4) completed!${NC}"
echo ""
echo "Note: Exp 5 (voxel ablation) is optional and not included in this run"
