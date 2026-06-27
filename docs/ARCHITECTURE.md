# VQ-MLP Architecture

Reference implementation: `src/FFB-MLP_VQ-MLP/VQ-mlp-origin-siren.py`

## Network Components

### MultiLatentEncoder (Siren 7D -> 120)
Encodes collision conditions from CSV files into a latent feature vector.

- **Input**: `concat(pos(3), direction(3), impulse(1))` = 7D
- **Output**: `z_feature` (120D)
- **Network**: Single Siren layer

CSV parsing (`parse_csv_condition`):
- Row 2 (sphere1): position (fields 2-4), direction (fields 8-10)
- Impulse = `norm(direction) / 304527.0` (max impulse normalization)

### PosEncoder (Siren 3D -> 128)
Encodes 3D spatial coordinates.

- **Input**: xyz coordinates (3D)
- **Output**: `feature_coords` (128D)
- **Network**: Single Siren layer

### ImplicitFunction (8-layer MLP)
Predicts distance field values from concatenated features.

- **Input**: `concat(z_feature(120), latent_vec(8), feature_coords(128))` = 256D
- **Hidden**: 512D, 8 layers
- **Skip connection**: At layer 5 (concat input with layer 4 output -> layer 5)
- **Output**: tanh-activated scalar
- **Activation**: SiLU (default), Siren, or SoftPlus (configurable via `--activation`)

Layer structure:
```
input(256) -> L1(512) -> L2(512) -> L3(512) -> L4(512-256=256)
-> [skip: cat(256, 256)=512] -> L5(512) -> L6(512) -> L7(512) -> L8(1) -> tanh
```

### Latent Vectors (VQ Codebook)
Per-shape learnable vectors, optimized jointly with network parameters.

- **Shape**: `(num_shapes, noisy_dim=8)`
- **Init**: `N(0, 0.01)`
- **Regularization**: L2 norm penalty in DeepSDFLoss

## Loss Function: DeepSDFLoss

```
L = loss_fn(clamp(pred, -delta, delta), clamp(gt, -delta, delta)) + latent_norm_reg
```

- **L1**: `nn.L1Loss()` — default
- **L2**: `nn.MSELoss().mean()`
- **Weighted L2**: MSE with 10x weight for points where `|gt| < 0.1` (near-surface emphasis)
- **Latent norm**: `sum(latent^2) / sd^2`, `sd=0.01`
- **Delta** (clamp range): 1.0 for FFB, 0.1 for UDF variants

## Training: Point-Level Batching

**PointLevelDataset** (`train_vq_mlp.py`):

1. Pre-loads ALL shapes' NPZ data and CSV conditions at init
2. Flattens all points into single arrays:
   - `self.points`: `(N_total, 3)` — coordinates
   - `self.values`: `(N_total,)` — distance field values
   - `self.shape_idx`: `(N_total,)` — which shape each point belongs to
3. Per-shape metadata stored separately:
   - `self.shape_pos[si]`, `self.shape_dir[si]`, `self.shape_imp[si]` — CSV conditions
   - `self.latent_vectors[si]` — learnable latent vector

**`__getitem__(idx)`** returns ONE point:
```python
(shape_pos, shape_dir, shape_imp, point_xyz, value, latent_vec, shape_index)
```

**Scale**: 5 shapes x 130,000 pts = 650,000 items. `batch_size=8` -> 81,250 batches/epoch.

**Optimizer**: Adam(lr=0.0001) over `[latent_vectors] + decoder.params + encoder.params + featureEncoder.params`

## Inference

**`infer_vq_mlp.py`**:

1. Load saved model (full objects via `torch.load`, NOT state_dict)
2. Create dense grid `[-1,1]^3` at target resolution
3. Query decoder in chunks (shape-level: coords dim=3, broadcast z_feature/latent)
4. Marching cubes on predicted volume -> PLY mesh

Optional outputs: voxel GIF animation, NIfTI volume, multi-view mesh renders.

## Model Saving Format

```
{save_path}/
  vqmlp-decoder.pt          # torch.save(decoder_object)
  vqmlp-encoder.pt          # torch.save(encoder_object)
  vqmlp-featureEncoder.pt   # torch.save(featureEncoder_object)
  vqmlp-codes.npz           # np.savez(latent_vectors)
  loss_history.json          # {"epochs": [...], "losses": [...]}
  config.json                # argparse config
  networks/                  # Per-epoch checkpoints
    vqmlp{epoch}-decoder.pt
    ...
```

Note: Models are saved as full objects (not state_dict), requiring `weights_only=False` and the defining classes to be importable when loading.
