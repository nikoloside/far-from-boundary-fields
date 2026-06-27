# FFB-MLP Experiments

VQ-MLP framework for fractured solid reconstruction. Compares UDF encoding strategies, training techniques, activation functions, and external SOTA methods.

## Architecture

**VQ-MLP** (from `VQ-mlp-origin-siren.py`):
- **MultiLatentEncoder**: Siren(7D -> 120) — encodes collision conditions (pos, dir, impulse) from CSV
- **PosEncoder**: Siren(3D -> 128) — encodes spatial coordinates
- **ImplicitFunction**: 8-layer MLP (512 hidden, skip at layer 5) — predicts distance field
- **DeepSDFLoss**: L1/L2/Weighted-L2 with clamping + latent norm regularization

**Training**: Point-level batching. All shapes' points flattened (5 shapes x 130k pts = 650k items). `batch_size=8` = 8 points per batch = ~81k batches/epoch.

## Quick Start

```bash
# Train with FFB encoding (default: NB sampling, L1 loss, SiLU activation)
python src/train_vq_mlp.py --encoding_type ffb --data_dir data/ --epochs 80000

# Extract mesh
python src/infer_vq_mlp.py --encoding_type ffb --model_dir data/ckpts/vq_mlp_ffb \
    --shape_id 1 --output mesh_1.ply --resolution 256

# Run experiments (add --minimal for quick smoke test, --quick for medium run)
python experiments/exp1_encoding/run.py --minimal
python experiments/exp2_training_tricks/run.py --minimal
python experiments/exp3_external_methods/run.py --minimal
python experiments/exp4_activation/run.py --minimal
```

## Repository Structure

```
src/
  train_vq_mlp.py              # Core training (point-level batching, all encodings)
  infer_vq_mlp.py              # Inference: grid query + marching cubes + voxel vis
  encoder_ffb-df_mlp.py        # FFB encoding: per-fragment normalized SDF
  encoder_udf_mesh.py          # UDF encoding: abs(SDF)
  encoder_truncated_udf.py     # Truncated UDF: min(abs(SDF), 0.1)
  encoder_flip_truncated_udf.py # Flip-Truncated UDF: sign(SDF)*min(abs(SDF), 0.1)
  FFB-MLP_VQ-MLP/              # Original reference implementations

experiments/
  run_experiment.py             # Shared utilities (log, run_cmd)
  eval_utils.py                 # SymMFCD, loss curves, bar charts, mesh rendering
  exp1_encoding/run.py          # Exp1: Encoding comparison (4 types)
  exp2_training_tricks/run.py   # Exp2: Sampling x Loss ablation (8 conditions)
  exp3_external_methods/run.py  # Exp3: NDC / MeshUDF / CAP-UDF vs ours
  exp4_activation/run.py        # Exp4: SiLU / Siren / SoftPlus ablation

data/
  obj/                          # Ground truth OBJ meshes (1.obj - 5.obj)
  csv/                          # Collision conditions per shape (1.csv - 5.csv)
  npz-resample/                 # FFB encoded points (NB sampling)
  npz-resample-uniform/         # FFB encoded points (uniform sampling)
  npz-udf/                      # UDF encoded points (NB sampling)
  npz-truncated-udf/            # Truncated UDF encoded points
  npz-flip-truncated-udf/       # Flip-Truncated UDF encoded points

docs/
  ARCHITECTURE.md               # VQ-MLP architecture details
  EXPERIMENTS.md                # 4-layer experiment design
  ENCODINGS.md                  # Encoding type specifications
```

## Evaluation

**SymMFCD** (Symmetric Mean Fragment Chamfer Distance): Primary metric comparing reconstructed mesh to ground truth OBJ. Lower = better.

All experiments produce:
- `output/figures/loss_curves.png` — training loss over epochs
- `output/figures/symmfcd_comparison.png` — SymMFCD bar chart
- `output/meshes/*.ply` — extracted meshes
- `output/voxels/*.gif` + `*.nii` — voxel visualizations
