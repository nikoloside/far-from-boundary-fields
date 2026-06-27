# Experiment Design

4-layer experiment system comparing UDF encoding strategies for fractured solid reconstruction.

All experiments use the same VQ-MLP framework (`src/train_vq_mlp.py`), training 5 shapes together with CSV-based collision conditioning and point-level batching.

## Experiment Overview

| Exp | Purpose | Variable | Fixed |
|-----|---------|----------|-------|
| **Exp1** | Encoding comparison | FFB / UDF / TUDF / Flip-TUDF | SiLU + NB + L1 |
| **Exp2** | Training technique ablation | NB vs Uniform x L1/L2/WL2 | SiLU, on FFB and UDF |
| **Exp3** | External method comparison | NDC / MeshUDF / CAP-UDF vs ours | Same OBJ inputs |
| **Exp4** | Activation ablation | SiLU / Siren / SoftPlus | FFB + NB + L1 |

## Running Modes

All experiments support three modes:

| Mode | Flag | Epochs | Batch Size | Resolution | Shapes | Purpose |
|------|------|--------|------------|------------|--------|---------|
| MINIMAL | `--minimal` | 2 | 2500 | 64 | 1 | Smoke test (seconds) |
| QUICK | `--quick` | 15 | 128 | 128 | all | Sanity check (minutes) |
| FULL | (none) | 3,000 | 128 | 256 | all | Publication results (hours) |

## Exp1: Encoding Comparison

**Question**: Is FFB encoding necessary? Can UDF variants achieve equivalent quality?

**Runner**: `experiments/exp1_encoding/run.py`

**4 conditions** (same architecture, only encoding differs):

| Condition | Encoding | NPZ Dir |
|-----------|----------|---------|
| ffb | FFB (per-fragment normalized SDF) | `npz-resample/` |
| udf | Unsigned distance field | `npz-udf/` |
| truncated_udf | Clamped UDF (max 0.1) | `npz-truncated-udf/` |
| flip_truncated_udf | Signed truncated ([-0.1, 0.1]) | `npz-flip-truncated-udf/` |

**Pipeline**:
1. Ensure NPZ encodings exist (run encoders if missing)
2. Train VQ-MLP per encoding
3. Extract meshes + voxel GIF/NII
4. Compute SymMFCD
5. Visualize: loss curves, SymMFCD bars, mesh renders

**Output**: `experiments/exp1_encoding/output/`

## Exp2: Training Technique Ablation

**Question**: How do sampling strategy and loss function affect reconstruction?

**Runner**: `experiments/exp2_training_tricks/run.py`

**8 conditions** (2 encodings x 4 technique combos):

| Encoding | Sampling | Loss | Tag |
|----------|----------|------|-----|
| FFB | NB | L1 (DeepSDF) | `ffb_NB_L1` |
| FFB | NB | L2 (MSE) | `ffb_NB_L2` |
| FFB | NB | Weighted L2 | `ffb_NB_WL2` |
| FFB | Uniform | L1 | `ffb_Uniform_L1` |
| UDF | NB | L1 | `udf_NB_L1` |
| UDF | NB | L2 | `udf_NB_L2` |
| UDF | NB | Weighted L2 | `udf_NB_WL2` |
| UDF | Uniform | L1 | `udf_Uniform_L1` |

**Prerequisites**: Uniform NPZ data (`npz-resample-uniform/`, `npz-udf-uniform/`) must exist for uniform sampling conditions.

**Output**: `experiments/exp2_training_tricks/output/`

## Exp3: External Methods Comparison

**Question**: Is our FFB encoding + Marching Cubes better than UDF + specialized extraction (MeshUDF/NDC)?

**Motivation**: Reviewer challenge — "UDF + MeshUDF extraction can also reconstruct, why need FFB?"

**Runner**: `experiments/exp3_external_methods/run.py`

**4 methods** compared on the same 5 shapes:

| Method | Field Learning | Mesh Extraction | Input |
|--------|---------------|-----------------|-------|
| **Ours** | FFB + VQ-MLP | Marching Cubes | CSV conditions |
| **UDF + MeshUDF** | UDF + VQ-MLP | MeshUDF | CSV conditions |
| **UDF + NDC** | UDF + VQ-MLP | NDC | CSV conditions |
| **CAP-UDF** | CAP-UDF (end-to-end) | CAP-UDF built-in | Surface points |

- MeshUDF and NDC use the **UDF-trained model** (not FFB), isolating: "does FFB encoding matter, even when UDF gets better extraction?"
- CAP-UDF is an independent end-to-end baseline

**Prerequisites**: Run `bash experiments/exp3_external_methods/setup_repos.sh` to clone repos. Missing repos are skipped gracefully.

**Output**: `experiments/exp3_external_methods/output/`

## Exp4: Activation Function Ablation

**Question**: Is SiLU optimal for the ImplicitFunction MLP?

**Runner**: `experiments/exp4_activation/run.py`

**3 conditions** (FFB encoding only):

| Activation | Description |
|-----------|-------------|
| SiLU | `nn.SiLU()` — current default |
| Siren | Siren layers (sinusoidal activation) |
| SoftPlus | `nn.Softplus()` |

**Output**: `experiments/exp4_activation/output/`

## Evaluation Metric: SymMFCD

**Symmetric Mean Fragment Chamfer Distance**: Samples points from both ground truth and reconstructed mesh, computes bidirectional Chamfer distance. Lower = better.

Implementation: `experiments/eval_utils.py :: compute_symmfcd()`

## Output Structure (per experiment)

```
output/
  ckpts/{condition}/          # Trained models + loss_history.json
  meshes/{condition}_{sid}.ply  # Extracted meshes
  metrics/symmfcd_results.json  # SymMFCD per condition per shape
  figures/
    loss_curves.png            # Training loss comparison
    symmfcd_comparison.png     # SymMFCD bar chart
    mesh_comparison_{sid}.png  # Side-by-side mesh renders
  voxels/{condition}/          # GIF + NII voxel visualizations
```

## CLI Reference

```bash
# Core training
python src/train_vq_mlp.py \
  --encoding_type {ffb|udf|truncated_udf|flip_truncated_udf} \
  --sampling {nb|uniform} \
  --loss_type {l1|l2|weighted_l2} \
  --activation {silu|siren|softplus} \
  --batch_size 128 \
  --epochs 1000 \
  --sample_limit 130000 \
  --max_shapes 5

# Inference + mesh extraction
python src/infer_vq_mlp.py \
  --encoding_type ffb \
  --model_dir path/to/ckpt \
  --shape_id 1 \
  --output mesh.ply \
  --resolution 256 \
  --voxel_gif --voxel_nii \
  --render_mesh
```
