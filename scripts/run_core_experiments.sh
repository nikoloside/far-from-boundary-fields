#!/bin/bash
#
# Core Experiments Runner (Exp 1-3 Only)
#
# Runs ONLY the core verification experiments:
# 1. Exp 1: FFB vs UDF vs NeuralUDF + Flooding vs MIND
# 2. Exp 2: FFB+Flooding vs Training Tricks
# 3. Exp 3: FFB+Flooding vs Activations
#
# NOTE: Exp 4 (MFCD validation) should be run AFTER these complete
#
# Usage:
#   bash scripts/run_core_experiments.sh
#   bash scripts/run_core_experiments.sh --minimal
#   bash scripts/run_core_experiments.sh --quick

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
echo -e "${BLUE}            RUNNING CORE EXPERIMENTS (EXP 1-3)${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Core verification sequence:"
echo "  Phase 1: Exp 1 - Encoding & Architecture & Post-processing"
echo "  Phase 2: Exp 2 - Training strategies ablation"
echo "  Phase 3: Exp 3 - Activation functions ablation"
echo ""
echo "NOTE: Exp 4 (MFCD validation) runs separately after these complete"
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

# ========== SUMMARY ==========
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}              CORE EXPERIMENTS (1-3) COMPLETED${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Total runtime: ${DURATION}s (${DURATION_MIN} min)"
echo ""
echo "Results locations:"
echo "  - Exp 1: experiments/exp1_udf_baseline/results/"
echo "  - Exp 2: experiments/exp2_training_trick_ablation/results/"
echo "  - Exp 3: experiments/exp3_activation_ablation/results/"
echo ""
echo "Logs: docs/experiments_log.md"
echo ""
echo -e "${YELLOW}Next step: Run Exp 4 to validate MFCD definition using above results${NC}"
echo "  bash scripts/run_exp4_mfcd_validation.sh"
echo ""
echo -e "${GREEN}✅ Core experiments (1-3) completed!${NC}"
