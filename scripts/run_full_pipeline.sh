#!/bin/bash
# Full pipeline: encode -> train -> visualize
# Usage: ./scripts/run_full_pipeline.sh [--minimal]
set -e
cd "$(dirname "$0")/.."

MODE=""
[ "$1" = "--minimal" ] && MODE="--minimal"
[ "$1" = "--fast" ] && MODE="--fast"

echo "=== 1. FFB-DF encoder ==="
python src/encoder_ffb-df_mlp.py $MODE || true

echo "=== 2. UDF mesh encoder ==="
python src/encoder_udf_mesh.py $MODE || true

# Use dummy data if encoders produced nothing (e.g. slow on large meshes)
if [ ! -f data/npz-resample/1.npz ] 2>/dev/null; then
  echo "No npz-resample; creating minimal dummy data for pipeline test..."
  python -c "
import numpy as np, os
for i in range(1,4):
  os.makedirs('data/npz-resample', exist_ok=True)
  pts = np.random.rand(1500,3).astype(np.float32)*2-1
  vals = np.random.randn(1500).astype(np.float32)*0.1
  np.savez(f'data/npz-resample/{i}.npz', poisson_grid_points=pts, sdf_values=vals)
  os.makedirs('data/npz-udf', exist_ok=True)
  np.savez(f'data/npz-udf/{i}.npz', poisson_grid_points=pts, udf_values=np.abs(vals))
"
fi

echo "=== 3. Train FFB-MLP ==="
python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 30

echo "=== 4. Train UDF-MLP (MIND/NeuralUDF style) ==="
python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 30

echo "=== 5. Generate & render PNG ==="
python scripts/generate_and_render.py

echo "=== Done. PNGs in data/results/udf_baseline/qualitative/ ==="
ls -la data/results/udf_baseline/qualitative/
