# Encoding Types

Four ways to represent the distance field from fractured solid fragments.

## Overview

| Encoding | Formula | NPZ Directory | value_key | Delta | Signed |
|----------|---------|---------------|-----------|-------|--------|
| FFB | Per-fragment normalized SDF | `npz-resample/` | `sdf_values` | 1.0 | Yes |
| UDF | `abs(SDF)` | `npz-udf/` | `udf_values` | 0.1 | No |
| Truncated UDF | `min(abs(SDF), 0.1)` | `npz-truncated-udf/` | `udf_values` | 0.1 | No |
| Flip-Truncated UDF | `sign(SDF) * min(abs(SDF), 0.1)` | `npz-flip-truncated-udf/` | `sdf_values` | 0.1 | Yes |

## FFB (Fragment-Based Field)

**NOT standard SDF.** Each fragment's internal points are normalized by that fragment's `max_dist` to `[-1, 0]`. External points keep their original positive distance. This preserves per-fragment structure.

- Encoder: `src/encoder_ffb-df_mlp.py`
- NB sampling: `npz-resample/`
- Uniform sampling: `npz-resample-uniform/`
- Historical note: "npz-resample" is legacy naming — it means FFB encoding with near-boundary sampling.

## UDF (Unsigned Distance Field)

Standard unsigned distance: `abs(SDF)`. No sign information — all values >= 0. Surface is at UDF = 0.

- Encoder: `src/encoder_udf_mesh.py`
- NB sampling: `npz-udf/`
- Uniform sampling: `npz-udf-uniform/` (may need generation)

## Truncated UDF

Clamped unsigned distance: `min(abs(SDF), 0.1)`. Far-field points all map to 0.1, focusing network capacity on near-surface region.

- Encoder: `src/encoder_truncated_udf.py`
- NB sampling only: `npz-truncated-udf/`
- Uniform sampling: Not supported

## Flip-Truncated UDF

Signed truncated field: `sign(SDF) * min(abs(SDF), 0.1)`. Combines sign information with truncation. Values in `[-0.1, 0.1]`.

- Encoder: `src/encoder_flip_truncated_udf.py`
- NB sampling only: `npz-flip-truncated-udf/`
- Uniform sampling: Not supported

## NPZ File Format

Each `{shape_id}.npz` contains:
- `poisson_grid_points`: `(N, 3)` float32 — 3D coordinates
- `sdf_values` or `udf_values`: `(N,)` float32 — distance field values

Typical size: ~130,000 points per shape with NB (near-boundary) sampling.

## Sampling Strategies

### Near-Boundary (NB) — default
Points are concentrated near the surface (where distance field changes rapidly). This is the sampling used in the original VQ-MLP.

### Uniform
Points are uniformly distributed in `[-1,1]^3`. Tests whether NB sampling is important for reconstruction quality.

Only FFB and UDF have uniform sampling variants. Controlled by `--sampling {nb|uniform}` in `train_vq_mlp.py`.
