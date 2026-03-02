#!/bin/bash
#
# Complete Pipeline for UDF Baseline Experiments
#
# This script runs the entire pipeline for all 5 data objects:
# 1. Data encoding (FFB-DF and UDF)
# 2. Model training (FFB-MLP, UDF-MLP, NeuralUDF-MLP)
# 3. Mesh extraction (with and without MIND)
# 4. Metric computation
# 5. Visualization
#
# Usage:
#   bash scripts/run_complete_pipeline.sh              # Full pipeline
#   bash scripts/run_complete_pipeline.sh --quick      # Quick test
#   bash scripts/run_complete_pipeline.sh --help       # Show help

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters
EPOCHS=30
RESOLUTION=256
MIND_ITER=200
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            EPOCHS=1
            RESOLUTION=64
            MIND_ITER=10
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick         Quick test mode (1 epoch, low resolution)"
            echo "  --epochs N      Number of training epochs (default: 30)"
            echo "  --resolution N  Grid resolution for MIND (default: 256)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}      Complete Pipeline for UDF Baseline Experiments${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Resolution: $RESOLUTION"
echo "  MIND iterations: $MIND_ITER"
echo "  Quick mode: $QUICK_MODE"
echo ""

# Change to root directory
cd "$ROOT_DIR"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="data/results/complete_pipeline"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_log_${TIMESTAMP}.txt"

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Phase header
phase_header() {
    log ""
    log "${BLUE}======================================================================${NC}"
    log "${BLUE}$1${NC}"
    log "${BLUE}======================================================================${NC}"
    log ""
}

# Start time
START_TIME=$(date +%s)

# ========== PHASE 1: DATA ENCODING ==========
phase_header "PHASE 1: DATA ENCODING"

# Check if data already exists
if [ -d "data/npz-resample" ] && [ -d "data/npz-udf" ]; then
    log "${YELLOW}⏭️  Data already exists, skipping encoding${NC}"
else
    log "${GREEN}▶️  1.1 FFB-DF Encoding${NC}"
    python src/encoder_ffb-df_mlp.py 2>&1 | tee -a "$LOG_FILE"

    log "${GREEN}▶️  1.2 UDF Encoding${NC}"
    python src/encoder_udf_mesh.py 2>&1 | tee -a "$LOG_FILE"
fi

# ========== PHASE 2: MODEL TRAINING ==========
phase_header "PHASE 2: MODEL TRAINING"

log "${GREEN}▶️  2.1 Training FFB-MLP${NC}"
python src/train_ffb_mlp.py \
    --npz_dir data/npz-resample \
    --epochs $EPOCHS \
    --output_dir data/ckpts/ffb_mlp \
    2>&1 | tee -a "$LOG_FILE"

log "${GREEN}▶️  2.2 Training UDF-MLP${NC}"
python src/train_udf_mlp.py \
    --npz_dir data/npz-udf \
    --epochs $EPOCHS \
    --output_dir data/ckpts/udf_mlp \
    2>&1 | tee -a "$LOG_FILE"

log "${GREEN}▶️  2.3 Training NeuralUDF-MLP${NC}"
python src/train_neuraludf_mlp.py \
    --npz_dir data/npz-udf \
    --epochs $EPOCHS \
    --d_hidden 256 \
    --n_layers 6 \
    --skip_in 4 \
    --multires 6 \
    --output_dir data/ckpts/neuraludf_mlp \
    2>&1 | tee -a "$LOG_FILE"

# ========== PHASE 3: MESH EXTRACTION ==========
phase_header "PHASE 3: MESH EXTRACTION WITH MIND"

mkdir -p data/results/meshes

log "${GREEN}▶️  3.1 Extracting UDF-MLP + MIND${NC}"
python src/extract_mesh_with_mind.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_mlp_mind.ply \
    --resolution $RESOLUTION \
    --max_iter $MIND_ITER \
    2>&1 | tee -a "$LOG_FILE" || log "${RED}⚠️  UDF-MLP + MIND failed${NC}"

log "${GREEN}▶️  3.2 Extracting FFB-MLP + MIND${NC}"
python src/extract_mesh_with_mind.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mlp_mind.ply \
    --resolution $RESOLUTION \
    --max_iter $MIND_ITER \
    2>&1 | tee -a "$LOG_FILE" || log "${RED}⚠️  FFB-MLP + MIND failed${NC}"

log "${GREEN}▶️  3.3 Extracting NeuralUDF-MLP + MIND${NC}"
python src/extract_mesh_with_mind.py \
    --model_type neuraludf_mlp \
    --ckpt data/ckpts/neuraludf_mlp/neuraludf_mlp.pth \
    --output data/results/meshes/neuraludf_mlp_mind.ply \
    --resolution $RESOLUTION \
    --max_iter $MIND_ITER \
    2>&1 | tee -a "$LOG_FILE" || log "${RED}⚠️  NeuralUDF-MLP + MIND failed${NC}"

# ========== PHASE 4: EVALUATION ==========
phase_header "PHASE 4: METRIC COMPUTATION"

if [ -d "data/original_meshes" ]; then
    log "${GREEN}▶️  4.1 Computing Symmetric MFCD${NC}"
    python experiments/mfcd_definition/symmetric_mfcd.py \
        --batch \
        --orig-dir data/original_meshes \
        --recon-dirs \
            udf_mlp:data/results/meshes \
            ffb_mlp:data/results/meshes \
            neuraludf:data/results/meshes \
        --output-dir data/results/complete_pipeline/mfcd \
        --num-samples 5000 \
        2>&1 | tee -a "$LOG_FILE" || log "${RED}⚠️  MFCD computation failed${NC}"
else
    log "${YELLOW}⚠️  Original meshes not found, skipping evaluation${NC}"
fi

# ========== PHASE 5: VISUALIZATION ==========
phase_header "PHASE 5: VISUALIZATION"

log "${YELLOW}📊 Visualization - TODO: Create comparison plots${NC}"

# ========== SUMMARY ==========
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

log ""
log "${GREEN}======================================================================${NC}"
log "${GREEN}                      PIPELINE COMPLETED${NC}"
log "${GREEN}======================================================================${NC}"
log ""
log "Total runtime: ${DURATION}s (${DURATION_MIN} min)"
log ""
log "Results saved to:"
log "  - Checkpoints: data/ckpts/"
log "  - Meshes: data/results/meshes/"
log "  - Metrics: data/results/complete_pipeline/mfcd/"
log "  - Log: $LOG_FILE"
log ""
log "${GREEN}✅ Pipeline completed successfully!${NC}"
log ""
