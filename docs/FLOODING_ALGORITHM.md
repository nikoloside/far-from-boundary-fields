# Flooding (Watershed) Mesh Extraction Algorithm

## Overview

The flooding algorithm extracts individual fracture fragment meshes from a learned distance field using ImageJ's 3D watershed segmentation. The core idea: convert the predicted distance field to a "topographic landscape" where each fragment forms a basin, flood from surface minima, and let watershed dams separate the fragments.

**Script**: `src/extract_mesh_flooding.py`

---

## Pipeline

```
Model Inference (128³)
    ↓
FFB: negate → +0.03 offset on negatives → abs() → UDF
UDF: use as-is
    ↓
Normalization to Integer Range (0–250)
    ↓
ImageJ 3D Watershed Flooding
    ↓
Label Map + GIF/NII Saved (intermediate)
    ↓
Per-Label Marching Cubes → Individual Fragments
    ↓
Mesh Boolean ∩ GT Object
    ↓
Final Fragment Meshes
```

---

## Step-by-Step Algorithm

### Step 1: Volume Inference

- Query the trained model on a uniform **128³** grid in `[-1, 1]³`
- Output: `volume[i,j,k]` = predicted distance value at each voxel
- Supported model types: **FFB-MLP** (signed), **UDF-MLP** (unsigned), **NeuralUDF** (unsigned)
- Saves: `raw_volume.nii` + `raw_volume.gif` (slice animation)

### Step 2: FFB → Cover Cage → Unified UDF

Both FFB and UDF paths converge to a unified UDF representation before watershed.

#### FFB Path (Signed Distance Field)

FFB-MLP outputs a **signed** distance field (igl convention: inside = negative, outside = positive).

1. **Negate**: `volume = -volume`
   - Now: inside = positive, outside = negative
   - Surface is at zero-crossing

2. **Add cover cage offset**: Add `+0.03` to the outside (negative) values
   - Shifts the zero-crossing **outward** by 0.03
   - Creates a thin **cover cage** shell around each fragment
   - Purpose: the cage ensures that watershed treats each fragment as a connected basin, preventing fragment surfaces from being split by noise

3. **Convert to UDF**: `volume = abs(volume)`
   - Inside fragment: positive values (distance from surface inward)
   - Cover cage (0 ~ 0.03 shell): small positive values near zero
   - Far outside: `abs(negative value)` = large positive values
   - Result: UDF where 0 = fragment surface + cage, increasing with distance

#### UDF Path (Unsigned Distance Field)

UDF-MLP outputs **unsigned** distance (all >= 0, surface at 0).

- Already in the right form: 0 = surface, increasing with distance
- No sign flip or cage needed

### Step 3: Normalization to Integer Range (0–250)

ImageJ watershed operates on integer-like voxel intensities. Float decimals (0.01, 0.02, ...) are too fine for its minimum unit of 1.

**Scaling logic**:
- Original UDF range: `[0, ~2.5]` (max distance in `[-1,1]³` space)
- Scale factor: `~100×`
- After scaling: `[0, ~250]`
- Precision: `0.01` in original space ≈ `1` in scaled space
- Surface (UDF = 0) maps to `0`
- Farthest point maps to `~250`

```python
scale_factor = 250.0 / udf_volume.max()
discrete = (udf_volume * scale_factor).astype(np.float32)
# discrete: 0 = surface, ~250 = farthest from any surface
# original 0.01 ≈ 1 in normalized space
```

Saves: `udf_volume.nii/.gif` (after cage conversion) + `discrete_volume.nii/.gif` (normalized)

### Step 4: ImageJ 3D Watershed Flooding

The watershed treats the normalized volume as a 3D topographic landscape:

- **Basins** (low values near 0): fragment surfaces → each fragment is a valley
- **Peaks** (high values near 250): far from any surface → ridges between fragments
- **Flooding**: water rises from each basin (fragment surface) simultaneously
- **Dams**: where two flood fronts meet, a dam (boundary) is built

**ImageJ BeanShell script** (MorphoLibJ plugin):
```java
// Parameters
radius = 2;          // not used directly
tolerance = 10;      // extended minima tolerance (in scaled units, ≈ 0.1 original)
conn = 6;            // 6-connectivity (face-adjacent voxels)
dams = true;         // build watershed dams (boundaries)

// Algorithm
regionalMinima = MinimaAndMaxima3D.extendedMinima(image, tolerance, conn);
imposedMinima = MinimaAndMaxima3D.imposeMinima(image, regionalMinima, conn);
labeledMinima = BinaryImages.componentsLabeling(regionalMinima, conn, 32);
resultStack = Watershed.computeWatershed(imposedMinima, labeledMinima, conn, dams);
```

**Key parameters**:
- `tolerance = 10`: In scaled space, this means basins must differ by at least 10 units (≈ 0.1 in original UDF space) to be considered separate fragments. This filters noise-induced false minima.
- `conn = 6`: 6-connectivity ensures conservative segmentation (only face-adjacent voxels connect, not edge or corner neighbors).
- `dams = true`: Explicit boundaries between labels.

**Output**: 3D label map where each voxel has an integer label (fragment ID), with dam voxels = 0 and background = -1.

### Step 5: Label Map Post-Processing

1. **Mask exterior**: Set voxels outside the object (original UDF > threshold) to `-1`
2. **Clear boundaries**: Set volume borders (first/last slice in x, y, z) to `-1`
3. **Filter noise**: Remove labels with fewer than `filtNoisy` voxels (default: 50)
4. **Save as NIfTI**: Save label map using nibabel for visualization and debugging

```python
# Save label map for inspection
ni_img = nib.Nifti1Image(label_map, affine=np.eye(4))
nib.save(ni_img, os.path.join(work_path, "watershed_labels.nii"))
```

The saved `.nii` file can be viewed in:
- **3D Slicer**: Load as label map, inspect individual fragments
- **ITK-SNAP**: Overlay labels on volume
- **Fiji/ImageJ**: Native format support

A **direct marching cubes** at `isolevel=0.03` is also extracted as an intermediate result for visual comparison (saved as `direct_mc.ply`).

### Step 6: Per-Label Marching Cubes

For each unique label in the watershed result:
1. Create a binary volume: `label_volume = (label_map == label_id)`
2. Run marching cubes (via vedo) at `isolevel=0.5` on the binary volume
3. Apply coordinate transforms:
   ```python
   mesh.shift(-res/2, -res/2, -res/2)   # Center at origin (voxel → normalized)
   mesh.scale(2/res, 2/res, 2/res)        # Scale to [-1, 1]³
   mesh.rotate_x(180).rotate_y(-90).rotate_z(90)  # Align to data coordinate system
   ```
4. Save individual fragments to `fragments/fragment_{label_id}.ply`

### Step 7: Mesh Boolean with GT Object

The flooding result may include the **cover cage** (for FFB) or have imprecise outer boundaries. Mesh boolean intersection with the ground-truth object clips each fragment to the correct outer boundary.

```
Final Fragment = Watershed Fragment ∩ GT Object
```

This ensures:
- The cover cage is removed (it extends beyond the GT surface)
- Fragment boundaries align precisely with the original solid's exterior
- Only interior fracture surfaces come from the learned field

Uses `trimesh.intersection()` per fragment. If boolean fails for a fragment, the raw fragment is kept.

---

## FFB Cover Cage: Why It's Needed

Without the cage, FFB fragments may have:
- Surface noise causing false zero-crossings
- Thin gaps between close fragments that watershed merges
- Open surfaces where the signed field is ambiguous

The +0.03 offset creates a thin **closed shell** around each fragment:
```
Original surface (SDF = 0)
    ↕ 0.03 gap
Cover cage surface (shifted zero-crossing)
```

After `abs()`, both the original surface and cage surface have UDF ≈ 0, forming a connected valley for watershed. The cage acts as a "bucket" that holds each fragment's basin together.

The cage is removed in Step 7 by mesh boolean with the GT object.

---

## Data Flow Summary

```
FFB-MLP Model                          UDF-MLP Model
     ↓                                      ↓
Signed field                           Unsigned field
(inside<0, outside>0)                  (surface=0, all>=0)
     ↓ negate                               ↓
(inside>0, outside<0)                       │
     ↓ +0.03 offset on negatives            │
(cage: outside[-0.03,0]→[0,0.03])           │
     ↓ abs()                                │
     └──────────── UDF ─────────────────────┘
                    ↓
            Scale to 0–250
                    ↓
          ImageJ 3D Watershed
                    ↓
             Label Map (.nii)
                    ↓
          Per-Label Marching Cubes
                    ↓
       Mesh Boolean ∩ GT Object
                    ↓
          Final Fragment Meshes
```

---

## Intermediate Outputs (for Debugging)

All saved in `flooding_work_{output_stem}/`:

| File | Description |
|------|-------------|
| `raw_volume.nii` / `.gif` | Model output before any processing |
| `udf_volume.nii` / `.gif` | Unified UDF (after cage for FFB) |
| `discrete_volume.nii` / `.gif` | Normalized 0–250 volume fed to ImageJ |
| `input_discrete.nii` | Actual input to watershed (same as discrete) |
| `watershed_labels.nii` / `.gif` | Label map from watershed (each fragment = unique int) |
| `direct_mc.ply` | Direct marching cubes at isolevel=0.03 (intermediate) |
| `fragments/fragment_{id}.ply` | Individual fragment meshes from watershed |
| `boolean_fragments/fragment_{id}.ply` | Fragments after ∩ GT object |

All `.nii` files viewable in 3D Slicer, ITK-SNAP, or Fiji.
All `.gif` files show axial slice animations of the 3D volume.

---

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--resolution` | 128 | Volume grid resolution (128³) |
| `--cage_offset` | 0.03 | Cover cage offset for FFB; also isolevel for direct MC |
| `--target_max` | 250 | Target max for ImageJ normalization |
| `--tolerance` | 10 | Watershed: minimum basin depth in normalized units (≈0.1 original) |
| `--conn` | 6 | Watershed connectivity (6=face, 26=full) |
| `--filt_noisy` | 50 | Minimum voxel count per label (smaller → removed) |
| `--gt_obj` | None | GT object for mesh boolean (optional) |
| `--model_type` | required | `ffb_mlp`, `udf_mlp`, or `neuraludf_mlp` |

---

## External Method Normalization (Exp3)

Each external mesh extraction method expects UDF values in a different format.
Our VQ-MLP UDF model may output values with a global offset (e.g., range [0.13, 0.34]
instead of starting at 0). A common preprocessing step is `vol_shifted = vol - vol.min()`
to bring the surface to UDF=0.

### MeshUDF Normalization

Source: `MeshUDF/optimize_chamfer_A_to_B.py`

| Property | Value |
|----------|-------|
| Coordinate space | [-1, 1]³ |
| `voxel_size` | `2.0 / (N-1)` |
| `spacing` | `[voxel_size] * 3` |
| Gradient threshold | `UDF < 2 * voxel_size` (~0.031 for N=128) |
| Vertex offset | `verts -= 1` (for voxel_origin=[-1,-1,-1]) |
| Face filter | `max_UDF_at_verts < voxel_size / 6` (~0.0026 for N=128) |

### NDC Normalization

Source: `NDC/dataset.py`

| Property | Value |
|----------|-------|
| File format | `.sdf` binary |
| Denormalization | `stored_value * grid_size` |
| Expected units | Voxel-unit distances (1 = one voxel spacing) |
| Near-surface mask | `abs(denormalized) < 1` |
| Clip range | `[-2, 2]` after denormalization |
| Conversion | `stored = coord_distance / 2` (for [-1,1]³ grid) |
| Vertex rescale | `v = v / grid_size * 2 - 1` (grid coords → [-1,1]³) |
