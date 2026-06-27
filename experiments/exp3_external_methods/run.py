#!/usr/bin/env python3
"""
Exp3: External Methods Comparison

Core argument: Our FFB encoding + simple Marching Cubes is better than
UDF + specialized UDF extraction methods (MeshUDF, NDC).

Comparison:
  1. Ours:        FFB + VQ-MLP → Marching Cubes
  2. UDF+MeshUDF: UDF + VQ-MLP → MeshUDF extraction (custom MC with gradients)
  3. UDF+NDC:     UDF + VQ-MLP → NDC extraction (learned dual contouring)
  4. CAP-UDF:     End-to-end (independent baseline, needs GPU)

All VQ-MLP methods use the same training setup (CSV conditioning, same shapes).
MeshUDF and NDC use the UDF-trained model's volume, NOT the FFB model.

Prerequisites:
    - Repos cloned in experiments/exp3_external_methods/repos/
    - NDC: Cython built + pretrained weights downloaded
    - MeshUDF: Cython built
    - CAP-UDF: CUDA chamfer extension built (needs GPU)

Usage:
    python experiments/exp3_external_methods/run.py
    python experiments/exp3_external_methods/run.py --minimal
    python experiments/exp3_external_methods/run.py --quick
"""
import os
import sys
import json
import glob
import struct
import shutil
import numpy as np

_exp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_exp_dir))
import run_experiment
from eval_utils import (compute_symmfcd, plot_loss_curves, plot_mfcd_bar_chart,
                        render_mesh_comparison)

log = run_experiment.log
run_cmd = run_experiment.run_cmd
ROOT = run_experiment.ROOT

EXP_ID = "exp3_external_methods"
OUT_DIR = os.path.join(_exp_dir, "output")
REPOS_DIR = os.path.join(_exp_dir, "repos")

MINIMAL = "--minimal" in sys.argv
QUICK = "--quick" in sys.argv

ALL_SHAPE_IDS = ['1', '2', '3', '4', '5']
GT_OBJ_DIR = os.path.join(ROOT, "data", "obj")
SQUIRREL_OBJ = os.path.join(ROOT, "data", "squirrel.obj")

METHODS = ['ours_ffb', 'udf_meshudf', 'udf_ndc', 'capudf']


def _get_available_shapes():
    return [sid for sid in ALL_SHAPE_IDS
            if os.path.exists(os.path.join(GT_OBJ_DIR, f"{sid}.obj"))]


def _check_repos():
    """Check which external repos are set up and ready."""
    available = {}

    # MeshUDF: needs Cython extension
    meshudf_dir = os.path.join(REPOS_DIR, "MeshUDF")
    meshudf_so = glob.glob(os.path.join(meshudf_dir, "custom_mc", "_marching_cubes_lewiner_cy*.so"))
    available['meshudf'] = os.path.isdir(meshudf_dir) and len(meshudf_so) > 0
    if not available['meshudf']:
        if os.path.isdir(meshudf_dir):
            log("  WARNING: MeshUDF Cython not built. Run: cd repos/MeshUDF/custom_mc && python setup.py build_ext --inplace")
        else:
            log("  WARNING: MeshUDF repo not found")

    # NDC: needs Cython extension + pretrained weights
    ndc_dir = os.path.join(REPOS_DIR, "NDC")
    ndc_so = glob.glob(os.path.join(ndc_dir, "cutils*.so"))
    ndc_weights_bool = os.path.join(ndc_dir, "weights", "weights_undc_udf_bool.pth")
    ndc_weights_float = os.path.join(ndc_dir, "weights", "weights_undc_udf_float.pth")
    ndc_has_weights = os.path.exists(ndc_weights_bool) and os.path.exists(ndc_weights_float)
    available['ndc'] = os.path.isdir(ndc_dir) and len(ndc_so) > 0 and ndc_has_weights
    if not available['ndc']:
        if not os.path.isdir(ndc_dir):
            log("  WARNING: NDC repo not found")
        elif len(ndc_so) == 0:
            log("  WARNING: NDC Cython not built. Run: cd repos/NDC && python setup.py build_ext --inplace")
        elif not ndc_has_weights:
            log("  WARNING: NDC pretrained weights missing. Download weights_undc_udf_bool.pth and weights_undc_udf_float.pth to repos/NDC/weights/")

    # CAP-UDF: needs CUDA chamfer extension
    capudf_dir = os.path.join(REPOS_DIR, "CAP-UDF")
    available['capudf'] = os.path.isdir(capudf_dir)
    if not available['capudf']:
        log("  WARNING: CAP-UDF repo not found")

    return available


def _get_training_params():
    epochs = 2 if MINIMAL else (15 if QUICK else 1500)
    batch_size = 2500 if MINIMAL else (128 if QUICK else 128)
    save_interval = 1 if MINIMAL else (5 if QUICK else 100)
    sample_limit = 5000 if MINIMAL else (30000 if QUICK else 130000)
    return epochs, batch_size, save_interval, sample_limit


# ========== Our Method (FFB + Marching Cubes) ==========

def run_ours(shape_ids, resolution=128):
    """Train FFB VQ-MLP and extract meshes via Marching Cubes."""
    ckpt_dir = os.path.join(OUT_DIR, "ckpts", "ours_ffb")
    epochs, batch_size, save_interval, sample_limit = _get_training_params()

    epochs = _get_training_params()[0]
    if run_experiment.is_training_complete(ckpt_dir, epochs):
        log(f"  SKIP ours_ffb: already trained to {epochs} epochs")
    else:
        resume_flag = ""
        if os.path.exists(os.path.join(ckpt_dir, "vqmlp-decoder.pt")):
            resume_flag = "--resume "
            log("  Resuming: FFB + VQ-MLP...")
        else:
            log("  Training: FFB + VQ-MLP...")
        run_cmd(
            f"python src/train_vq_mlp.py "
            f"--encoding_type ffb "
            f"--data_dir data/ "
            f"--save_path {ckpt_dir} "
            f"--epochs {epochs} "
            f"--batch_size {batch_size} "
            f"--save_interval {save_interval} "
            f"--sample_limit {sample_limit} "
            f"--max_shapes {len(shape_ids)} "
            f"{resume_flag}"
            f"--proj_name vqmlp",
            EXP_ID, "Train ours_ffb")

    if not os.path.exists(os.path.join(ckpt_dir, "vqmlp-decoder.pt")):
        log("  Our method: training failed, skipping extraction")
        return ckpt_dir

    for sid in shape_ids:
        mesh_path = os.path.join(OUT_DIR, "meshes", f"ours_ffb_{sid}.obj")
        run_cmd(
            f"python src/infer_vq_mlp.py "
            f"--encoding_type ffb "
            f"--model_dir {ckpt_dir} "
            f"--data_dir data/ "
            f"--shape_id {sid} "
            f"--output {mesh_path} "
            f"--resolution {resolution} "
            f"--proj_name vqmlp "
            f"--batch_chunks {16 if MINIMAL else 64} "
            f"--voxel_gif --voxel_nii "
            f"--voxel_resolution 64 "
            f"--output_dir {os.path.join(OUT_DIR, 'voxels', 'ours_ffb')} "
            f"--render_mesh",
            EXP_ID, f"Infer ours shape {sid}")

    return ckpt_dir


# ========== UDF Model (shared by MeshUDF and NDC) ==========

def train_udf_model(shape_ids):
    """Train UDF VQ-MLP (used as input for MeshUDF and NDC extraction)."""
    ckpt_dir = os.path.join(OUT_DIR, "ckpts", "udf_model")
    epochs, batch_size, save_interval, sample_limit = _get_training_params()

    epochs = _get_training_params()[0]
    if run_experiment.is_training_complete(ckpt_dir, epochs):
        log(f"  SKIP udf_model: already trained to {epochs} epochs")
    else:
        resume_flag = ""
        if os.path.exists(os.path.join(ckpt_dir, "vqmlp-decoder.pt")):
            resume_flag = "--resume "
            log("  Resuming: UDF + VQ-MLP (for MeshUDF/NDC extraction)...")
        else:
            log("  Training: UDF + VQ-MLP (for MeshUDF/NDC extraction)...")
        run_cmd(
            f"python src/train_vq_mlp.py "
            f"--encoding_type udf "
            f"--data_dir data/ "
            f"--save_path {ckpt_dir} "
            f"--epochs {epochs} "
            f"--batch_size {batch_size} "
            f"--save_interval {save_interval} "
            f"--sample_limit {sample_limit} "
            f"--max_shapes {len(shape_ids)} "
            f"{resume_flag}"
            f"--proj_name vqmlp",
            EXP_ID, "Train udf_model")

    return ckpt_dir


def generate_udf_volume_and_gradients(udf_model_dir, shape_ids, resolution=128):
    """Generate UDF volumes AND gradient fields from trained UDF model.

    Returns vol_dir containing:
      - {sid}_udf_vol.npy: (resolution, resolution, resolution) float32
      - {sid}_udf_grad.npy: (resolution, resolution, resolution, 3) float32
    """
    vol_dir = os.path.join(OUT_DIR, "udf_volumes")
    os.makedirs(vol_dir, exist_ok=True)

    if not os.path.exists(os.path.join(udf_model_dir, "vqmlp-decoder.pt")):
        log("  UDF model not trained, skipping volume generation")
        return vol_dir

    for sid in shape_ids:
        vol_path = os.path.join(vol_dir, f"{sid}_udf_vol.npy")
        grad_path = os.path.join(vol_dir, f"{sid}_udf_grad.npy")
        if os.path.exists(vol_path) and os.path.exists(grad_path):
            log(f"  UDF volume+gradients shape {sid}: already exists")
            continue

        log(f"  Generating UDF volume + gradients: shape {sid} ({resolution}^3)...")
        # Use a separate script to avoid inline quoting issues
        script = f'''
import sys, os
sys.path.insert(0, 'src')
import torch
import torch.nn.functional as F
import numpy as np
import train_vq_mlp
import __main__
for _c in ['ImplicitFunction','PosEncoder','MultiLatentEncoder']:
    setattr(__main__, _c, getattr(train_vq_mlp, _c))
from infer_vq_mlp import parse_csv_condition, infer_volume

device = 'cuda' if torch.cuda.is_available() else 'cpu'
decoder = torch.load('{udf_model_dir}/vqmlp-decoder.pt', map_location=device, weights_only=False)
encoder = torch.load('{udf_model_dir}/vqmlp-encoder.pt', map_location=device, weights_only=False)
fe = torch.load('{udf_model_dir}/vqmlp-featureEncoder.pt', map_location=device, weights_only=False)
codes = np.load('{udf_model_dir}/vqmlp-codes.npz')
lvec = torch.from_numpy(codes[codes.files[0]]).float().to(device)

csv_p = 'data/csv/{sid}.csv'
pos, d, imp = parse_csv_condition(csv_p)
pos_t = torch.from_numpy(pos).float().to(device)
dir_t = torch.from_numpy(d).float().to(device)
imp_t = torch.from_numpy(imp).float().to(device)

decoder.eval(); encoder.eval(); fe.eval()
fz = fe(pos_t, dir_t, imp_t)

# Generate UDF volume
vol = infer_volume(decoder, encoder, fe, fz, lvec[{int(sid)-1}], resolution={resolution}, device=device)
vol = np.maximum(vol, 0).astype(np.float32)
np.save('{vol_path}', vol)
print(f'UDF volume saved: shape={sid}, range=[{{vol.min():.4f}}, {{vol.max():.4f}}]')

# Save NIfTI for debugging
import nibabel as nib
nii_path = '{vol_path}'.replace('.npy', '.nii')
nib.save(nib.Nifti1Image(vol, np.eye(4)), nii_path)
print(f'UDF NIfTI saved: {{nii_path}}')

# Save GIF (axial slices)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
frames = []
vmin, vmax = vol.min(), vol.max()
for z in range(vol.shape[2]):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(vol[:, :, z], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'UDF z={{z}}/{{vol.shape[2]}}')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    buf.close()
gif_path = '{vol_path}'.replace('.npy', '.gif')
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=80, loop=0)
print(f'UDF GIF saved: {{gif_path}} ({{len(frames)}} frames)')

# Generate gradient field using finite differences
# grad[i,j,k] = normalized gradient of UDF at voxel (i,j,k)
grad = np.zeros((*vol.shape, 3), dtype=np.float32)
# Central differences (edges use forward/backward)
for axis in range(3):
    grad_1d = np.gradient(vol, axis=axis)
    grad[..., axis] = grad_1d

# Normalize gradients (direction towards surface = negative gradient for UDF)
norms = np.linalg.norm(grad, axis=-1, keepdims=True)
norms = np.clip(norms, 1e-8, None)
grad = -grad / norms  # negative = pointing towards surface

np.save('{grad_path}', grad)
print(f'Gradient field saved: shape={sid}, grad shape={{grad.shape}}')
'''
        script_path = os.path.join(vol_dir, f"gen_vol_{sid}.py")
        with open(script_path, 'w') as f:
            f.write(script)
        run_cmd(f"python {script_path}", EXP_ID, f"UDF vol+grad shape {sid}")
        os.remove(script_path)

    return vol_dir


# ========== UDF + MeshUDF Extraction ==========

def run_meshudf(shape_ids, udf_vol_dir, resolution=128):
    """Extract meshes from UDF volumes using MeshUDF's custom marching cubes.

    MeshUDF requires BOTH the UDF volume AND gradient directions.
    Core function: udf_mc_lewiner(volume, grads, spacing) from custom_mc/

    Normalization for MeshUDF (from optimize_chamfer_A_to_B.py):
      - Grid in [-1,1]³, voxel_size = 2.0/(N-1)
      - spacing = [voxel_size]*3
      - Gradients computed only where UDF < 2*voxel_size (near-surface)
      - After MC: verts offset by -1 (voxel_origin=[-1,-1,-1])
      - Face filtering: keep faces where max UDF at vertices < voxel_size/6

    Key: Our UDF values may have a global offset (min >> 0).
    We shift UDF so min → 0, bringing the "surface" to UDF=0.
    Gradients are recomputed on the shifted volume.
    """
    meshudf_dir = os.path.join(REPOS_DIR, "MeshUDF")

    for sid in shape_ids:
        vol_path = os.path.join(udf_vol_dir, f"{sid}_udf_vol.npy")
        grad_path = os.path.join(udf_vol_dir, f"{sid}_udf_grad.npy")
        mesh_out = os.path.join(OUT_DIR, "meshes", f"udf_meshudf_{sid}.obj")

        if not os.path.exists(vol_path) or not os.path.exists(grad_path):
            log(f"  MeshUDF: UDF volume/gradients missing for shape {sid}, skipping")
            continue

        log(f"  MeshUDF: Extracting mesh for shape {sid}...")
        script = f'''
import sys, os
import numpy as np
custom_mc_dir = os.path.join('{meshudf_dir}', 'custom_mc')
os.chdir(custom_mc_dir)
sys.path.insert(0, custom_mc_dir)
from _marching_cubes_lewiner import udf_mc_lewiner
import trimesh

vol_raw = np.load('{vol_path}').astype(np.float32)
grads = np.load('{grad_path}').astype(np.float32)
N = vol_raw.shape[0]

print(f'UDF raw: shape={{vol_raw.shape}}, range=[{{vol_raw.min():.6f}}, {{vol_raw.max():.6f}}]')

surface_offset = 0.013
vol = (vol_raw - surface_offset).astype(np.float32)
print(f'UDF after shift(-{{surface_offset}}): range=[{{vol.min():.6f}}, {{vol.max():.6f}}]')

# Recompute gradients on shifted volume
grad_new = np.zeros((*vol.shape, 3), dtype=np.float32)
for axis in range(3):
    grad_new[..., axis] = np.gradient(vol, axis=axis)
norms = np.linalg.norm(grad_new, axis=-1, keepdims=True)
norms = np.clip(norms, 1e-8, None)
grads = (-grad_new / norms).astype(np.float32)

# MeshUDF spacing: voxel size in coordinate space [-1,1]³
voxel_size = 2.0 / (N - 1)
print(f'voxel_size={{voxel_size:.6f}}, gradient threshold=2*vs={{2*voxel_size:.6f}}')

near_surface = np.sum(vol < 2 * voxel_size)
print(f'Near-surface voxels (UDF < 2*voxel_size): {{near_surface}}')

vol = np.ascontiguousarray(vol)
grads = np.ascontiguousarray(grads)

try:
    vertices, faces, normals, values = udf_mc_lewiner(
        vol, grads, spacing=(voxel_size, voxel_size, voxel_size))

    if len(vertices) == 0 or len(faces) == 0:
        print('MeshUDF: No mesh extracted (empty result)')
        sys.exit(0)

    # Vertex offset for voxel_origin=[-1,-1,-1] (MeshUDF convention)
    vertices = vertices - 1.0
    print(f'After MC: {{len(vertices)}} verts, {{len(faces)}} faces')

    # Face filtering: remove faces far from surface (MeshUDF post-processing)
    # Interpolate UDF at vertex positions to get per-vertex distance
    from scipy.interpolate import RegularGridInterpolator
    grid_1d = np.linspace(-1, 1, N)
    interp = RegularGridInterpolator((grid_1d, grid_1d, grid_1d), vol,
                                      bounds_error=False, fill_value=vol.max())
    vert_udf = interp(vertices)
    # Keep faces where ALL vertices have small UDF (near surface)
    max_udf_per_face = np.max(vert_udf[faces], axis=1)
    face_mask = max_udf_per_face < voxel_size / 6.0
    filtered_faces = faces[face_mask]
    print(f'Face filtering (UDF < vs/6={{voxel_size/6:.6f}}): {{len(faces)}} → {{len(filtered_faces)}} faces')

    if len(filtered_faces) == 0:
        # Fallback: less aggressive filtering
        face_mask = max_udf_per_face < voxel_size
        filtered_faces = faces[face_mask]
        print(f'Relaxed filter (UDF < vs={{voxel_size:.6f}}): {{len(filtered_faces)}} faces')

    if len(filtered_faces) > 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=filtered_faces)
    else:
        # Last fallback: save unfiltered
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(f'WARNING: face filter removed all faces, using unfiltered')

    # --- Mesh post-processing: fix normals + clean ---
    # Remove degenerate faces (API varies by trimesh version)
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    # Remove duplicate faces
    unique_mask = mesh.unique_faces()
    mesh.update_faces(unique_mask)
    # Fix face winding / normals (make consistent outward-facing)
    mesh.fix_normals()
    mesh.fill_holes()
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    # Keep all components — MeshUDF produces fragmented surfaces by design
    mesh.export('{mesh_out}')
    print(f'MeshUDF mesh saved: {{len(mesh.vertices)}} verts, {{len(mesh.faces)}} faces')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'MeshUDF extraction failed: {{e}}')
'''
        script_path = os.path.join(OUT_DIR, f"meshudf_{sid}.py")
        with open(script_path, 'w') as f:
            f.write(script)
        run_cmd(f"python {script_path}", EXP_ID, f"MeshUDF extract {sid}")
        os.remove(script_path)


# ========== UDF + NDC Extraction ==========

def _write_sdf_file(volume, filepath):
    """Write a 3D numpy array to SDFGen .sdf format.

    Format:
        #sdf\\n
        dims X Y Z\\n
        <empty line>\\n
        <float32 binary data>
    """
    with open(filepath, 'wb') as f:
        f.write(b'#sdf\n')
        dims = volume.shape
        f.write(f'dims {dims[0]} {dims[1]} {dims[2]}\n'.encode())
        f.write(b'\n')
        f.write(volume.astype(np.float32).tobytes())


def run_ndc(shape_ids, udf_vol_dir, resolution=128):
    """Extract meshes from UDF volumes using NDC (Neural Dual Contouring).

    NDC uses pretrained networks (bool + float) to predict edge crossings
    and vertex positions from a UDF grid.

    NDC normalization (from dataset.py):
      - Reads .sdf file as LOD_input
      - Denormalizes: gt_input_ = LOD_input * grid_size  (grid_size = N-1)
      - For UDF: gt_input_ = abs(gt_input_)
      - Creates edge mask: (-1 < gt_input_ < 1) → near-surface voxels
      - Clips: gt_input = clip(gt_input, -2, 2)
      - Comment: "each cell in the input is a unit cube"

    So denormalized values must be in VOXEL UNITS (1 = one voxel spacing).
    For near-surface detection: denormalized UDF < 1 at the surface.

    Conversion from our coordinate space [-1,1]³:
      - voxel_spacing = 2 / (N-1) in coordinate space
      - voxel_distance = coord_distance / voxel_spacing
      - stored_value = voxel_distance / grid_size = coord_distance / 2

    Key: We also shift UDF min → 0 to bring surface to UDF=0.
    """
    ndc_dir = os.path.join(REPOS_DIR, "NDC")

    for sid in shape_ids:
        vol_path = os.path.join(udf_vol_dir, f"{sid}_udf_vol.npy")
        mesh_out = os.path.join(OUT_DIR, "meshes", f"udf_ndc_{sid}.obj")

        if not os.path.exists(vol_path):
            log(f"  NDC: UDF volume missing for shape {sid}, skipping")
            continue

        log(f"  NDC: Extracting mesh for shape {sid}...")

        # Load and prepare volume for NDC
        vol_raw = np.load(vol_path).astype(np.float32)
        vol_raw = np.maximum(vol_raw, 0)  # Clamp negatives to 0 (on surface)

        log(f"  NDC: UDF raw range=[{vol_raw.min():.6f}, {vol_raw.max():.6f}]")

        surface_offset = 0.013
        vol = (vol_raw - surface_offset).astype(np.float32)
        log(f"  NDC: UDF after shift(-{surface_offset}): range=[{vol.min():.6f}, {vol.max():.6f}]")

        # Convert coordinate distances to NDC's expected format
        # NDC denormalizes: stored * grid_size → expects voxel-unit distances
        # stored = coord_distance / 2
        grid_size = vol.shape[0] - 1
        vol_normalized = vol / 2.0

        # Verify: after NDC denormalization (stored * grid_size):
        denorm_min = vol_normalized.min() * grid_size
        denorm_max = vol_normalized.max() * grid_size
        denorm_surface = 0 * grid_size  # surface → 0
        near_surface_count = np.sum((vol_normalized * grid_size) < 1)
        log(f"  NDC: After denorm: range=[{denorm_min:.4f}, {denorm_max:.4f}], "
            f"near-surface voxels (denorm<1): {near_surface_count}")

        # Write to .sdf format
        sdf_path = os.path.join(udf_vol_dir, f"{sid}_ndc_input.sdf")
        _write_sdf_file(vol_normalized, sdf_path)

        # NDC output goes to samples/quicktest_undc_udf.obj
        ndc_sample_dir = os.path.join(OUT_DIR, "ndc_samples", sid)
        os.makedirs(ndc_sample_dir, exist_ok=True)

        run_cmd(
            f"cd {ndc_dir} && PYTHONPATH={ndc_dir}:$PYTHONPATH python main.py "
            f"--test_input {os.path.abspath(sdf_path)} "
            f"--input_type udf "
            f"--method undc "
            f"--postprocessing "
            f"--sample_dir {os.path.abspath(ndc_sample_dir)} "
            f"--gpu 0",
            EXP_ID, f"NDC extract {sid}")

        # Copy output OBJ → PLY, rescale vertices, fix normals
        ndc_obj = os.path.join(ndc_sample_dir, "quicktest_undc_udf.obj")
        if os.path.exists(ndc_obj):
            try:
                import trimesh
                mesh = trimesh.load(ndc_obj, force='mesh')
                # NDC outputs vertices in grid coordinates [0, grid_size]
                # Rescale to our [-1,1]³ space: v = v / grid_size * 2 - 1
                mesh.vertices = mesh.vertices / grid_size * 2.0 - 1.0
                # Post-processing: fix normals + clean
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.update_faces(mesh.unique_faces())
                mesh.fix_normals()
                mesh.remove_unreferenced_vertices()
                # Keep all components — NDC produces fragmented surfaces by design
                mesh.export(mesh_out)
                log(f"  NDC: shape {sid} mesh saved ({len(mesh.vertices)} verts, "
                    f"bounds=[{mesh.vertices.min():.3f}, {mesh.vertices.max():.3f}])")
            except Exception as e:
                log(f"  NDC: Failed to convert OBJ→PLY for shape {sid}: {e}")
        else:
            log(f"  NDC: No output for shape {sid} (expected {ndc_obj})")


# ========== CAP-UDF (End-to-End) ==========

def run_capudf(shape_ids, resolution=128):
    """CAP-UDF: OBJ -> surface point cloud -> end-to-end UDF learning + mesh.

    CAP-UDF is an independent end-to-end baseline that:
    1. Takes raw surface point cloud as input
    2. Learns UDF from scratch using consistency-aware progressive training
    3. Extracts mesh using gradient-based sign flipping + marching cubes

    Requires: GPU with CUDA, compiled chamfer_dist extension.
    Input: .npy point cloud in data/owndata/input/
    Output: mesh at outs/{dir}/mesh/60000_mesh.obj
    """
    capudf_dir = os.path.join(REPOS_DIR, "CAP-UDF")
    import torch
    if not torch.cuda.is_available():
        log("  CAP-UDF: SKIPPED (no GPU available)")
        return

    # Check chamfer extension
    try:
        sys.path.insert(0, capudf_dir)
        from extensions.chamfer_dist import ChamferDistanceL1
        sys.path.pop(0)
    except ImportError:
        log("  CAP-UDF: SKIPPED (chamfer_dist not compiled)")
        log("  Build: cd repos/CAP-UDF/extensions/chamfer_dist && python setup.py install")
        return

    # Prepare input directory
    input_dir = os.path.join(capudf_dir, "data", "owndata", "input")
    os.makedirs(input_dir, exist_ok=True)

    # Training iterations based on mode
    if MINIMAL:
        step1, step2 = 100, 200
        mcube_res = 64
    elif QUICK:
        step1, step2 = 2000, 4000
        mcube_res = 128
    else:
        step1, step2 = 40000, 60000
        mcube_res = resolution

    # Create a custom config for our settings
    config_path = os.path.join(OUT_DIR, "capudf_config.conf")
    with open(config_path, 'w') as f:
        f.write(f"""general {{
    base_exp_dir = {os.path.join(OUT_DIR, 'capudf_outs')}/
    recording = [
        ./,
        ./models
    ]
}}

dataset {{
    data_dir = {os.path.join(capudf_dir, 'data', 'owndata')}/
}}

train {{
    learning_rate = 0.001
    step1_maxiter = {step1}
    step2_maxiter = {step2}
    warm_up_end = {min(1000, step1 // 4)}
    eval_num_points = {100000 if MINIMAL else 1000000}
    df_filter = 0.01
    far = -1
    outlier = 0.002
    extra_points_rate = 1
    low_range = 1.1

    batch_size = 5000
    batch_size_step2 = 20000

    save_freq = {step2}
    val_freq = {step2}
    val_mesh_freq = {step2}
    report_freq = {max(step1 // 4, 1)}

    igr_weight = 0.1
    mask_weight = 0.0
    load_ckpt = none
}}

model {{
    udf_network {{
        d_out = 1
        d_in = 3
        d_hidden = 256
        n_layers = 8
        skip_in = [4]
        multires = 0
        bias = 0.5
        scale = 1.0
        geometric_init = True
        weight_norm = True
    }}
}}
""")

    num_surface_points = 3000 if MINIMAL else (10000 if QUICK else 50000)

    for sid in shape_ids:
        obj_path = os.path.join(GT_OBJ_DIR, f"{sid}.obj")
        mesh_out = os.path.join(OUT_DIR, "meshes", f"capudf_{sid}.obj")

        if os.path.exists(mesh_out):
            log(f"  CAP-UDF: shape {sid} already done")
            continue

        # Prepare input point cloud
        pc_name = f"shape_{sid}"
        pc_path = os.path.join(input_dir, f"{pc_name}.npy")
        if not os.path.exists(pc_path):
            log(f"  CAP-UDF: Sampling {num_surface_points} points from shape {sid}...")
            run_cmd(
                f"python -c \""
                f"import trimesh; import numpy as np; "
                f"mesh = trimesh.load('{obj_path}', force='mesh'); "
                f"pts, _ = trimesh.sample.sample_surface(mesh, {num_surface_points}); "
                f"np.save('{pc_path}', pts.astype(np.float32)); "
                f"print(f'Saved point cloud: {{pts.shape}}')\"",
                EXP_ID, f"CAP-UDF sample {sid}")

        # Remove cached query data to force regeneration
        query_data = os.path.join(capudf_dir, "data", "owndata", "query_data", f"{pc_name}.npz")
        if os.path.exists(query_data):
            os.remove(query_data)

        # Run CAP-UDF training + mesh extraction
        capudf_out_dir = f"shape_{sid}"
        log(f"  CAP-UDF: Training for shape {sid} ({step2} iterations)...")
        run_cmd(
            f"cd {capudf_dir} && PYTHONPATH={capudf_dir}:$PYTHONPATH python run.py "
            f"--conf {os.path.abspath(config_path)} "
            f"--dataname {pc_name} "
            f"--dir {capudf_out_dir} "
            f"--mcube_resolution {mcube_res} "
            f"--gpu 0",
            EXP_ID, f"CAP-UDF train {sid}")

        # Find and copy output mesh
        capudf_mesh = os.path.join(OUT_DIR, "capudf_outs", capudf_out_dir, "mesh",
                                   f"{step2}_mesh.obj")
        if os.path.exists(capudf_mesh):
            try:
                import trimesh
                mesh = trimesh.load(capudf_mesh, force='mesh')
                mesh.export(mesh_out)
                log(f"  CAP-UDF: shape {sid} mesh saved ({len(mesh.vertices)} verts)")
            except Exception as e:
                log(f"  CAP-UDF: Failed to convert mesh for shape {sid}: {e}")
        else:
            log(f"  CAP-UDF: No output mesh for shape {sid} (expected {capudf_mesh})")


# ========== Main ==========

def main():
    log(f"=== {EXP_ID} START ===")
    log(f"Mode: {'MINIMAL' if MINIMAL else 'QUICK' if QUICK else 'FULL'}")
    run_experiment.check_dependencies()

    shape_ids = _get_available_shapes()
    if MINIMAL:
        shape_ids = shape_ids[:1]
    log(f"Shapes: {shape_ids}")

    os.makedirs(os.path.join(OUT_DIR, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "voxels"), exist_ok=True)

    resolution = 64 if MINIMAL else (128 if QUICK else 128)
    repo_status = _check_repos()
    log(f"Repo status: {repo_status}")

    # === Step 1: Train our method (FFB + MC) ===
    log("\n--- Step 1: Our Method (FFB + Marching Cubes) ---")
    run_ours(shape_ids, resolution=resolution)

    # === Step 1b: Flooding (Fragment Extraction for FFB) ===
    log("\n--- Step 1b: Flooding (FFB Fragment Extraction) ---")
    os.makedirs(os.path.join(OUT_DIR, "flooding"), exist_ok=True)
    for sid in shape_ids:
        nii_path = os.path.join(OUT_DIR, "voxels", "ours_ffb",
                                f"ffb_{sid}_voxel_{resolution}.nii")
        if not os.path.exists(nii_path):
            log(f"  Skipping flooding shape {sid}: no volume NII")
            continue
        flood_output = os.path.join(OUT_DIR, "flooding", f"ours_ffb_{sid}.obj")
        gt_flag = f"--gt_obj {SQUIRREL_OBJ}" if os.path.exists(SQUIRREL_OBJ) else ""
        run_cmd(
            f"python src/extract_mesh_flooding.py "
            f"--volume {nii_path} "
            f"--field_type signed "
            f"--output {flood_output} "
            f"--tolerance 25 "
            f"--conn 6 "
            f"{gt_flag}",
            EXP_ID, f"Flooding ours shape {sid}")

    # === Step 2: Train UDF model (shared by MeshUDF and NDC) ===
    need_udf = repo_status.get('meshudf') or repo_status.get('ndc')
    if need_udf:
        log("\n--- Step 2: Train UDF Model (for MeshUDF/NDC) ---")
        udf_model_dir = train_udf_model(shape_ids)

        # === Step 3: Generate UDF volumes + gradients ===
        log("\n--- Step 3: Generate UDF Volumes + Gradients ---")
        udf_vol_dir = generate_udf_volume_and_gradients(udf_model_dir, shape_ids, resolution)

        # Also extract UDF + standard Marching Cubes for reference
        if os.path.exists(os.path.join(udf_model_dir, "vqmlp-decoder.pt")):
            for sid in shape_ids:
                mesh_path = os.path.join(OUT_DIR, "meshes", f"udf_mc_{sid}.obj")
                run_cmd(
                    f"python src/infer_vq_mlp.py "
                    f"--encoding_type udf "
                    f"--model_dir {udf_model_dir} "
                    f"--data_dir data/ "
                    f"--shape_id {sid} "
                    f"--output {mesh_path} "
                    f"--resolution {resolution} "
                    f"--proj_name vqmlp "
                    f"--batch_chunks {16 if MINIMAL else 64} "
                    f"--voxel_gif --voxel_nii "
                    f"--voxel_resolution 64 "
                    f"--output_dir {os.path.join(OUT_DIR, 'voxels', 'udf_mc')} "
                    f"--render_mesh",
                    EXP_ID, f"Infer UDF+MC shape {sid}")
    else:
        log("\n--- Steps 2-3: SKIPPED (no MeshUDF/NDC repos ready) ---")
        udf_vol_dir = None

    # === Step 4: MeshUDF extraction ===
    if repo_status.get('meshudf') and udf_vol_dir:
        log("\n--- Step 4: UDF + MeshUDF Extraction ---")
        run_meshudf(shape_ids, udf_vol_dir, resolution=resolution)
    else:
        log("\n--- Step 4: MeshUDF SKIPPED ---")

    # === Step 5: NDC extraction ===
    if repo_status.get('ndc') and udf_vol_dir:
        log("\n--- Step 5: UDF + NDC Extraction ---")
        run_ndc(shape_ids, udf_vol_dir, resolution=resolution)
    else:
        log("\n--- Step 5: NDC SKIPPED ---")

    # === Step 6: CAP-UDF (end-to-end) ===
    if repo_status.get('capudf'):
        log("\n--- Step 6: CAP-UDF (End-to-End) ---")
        run_capudf(shape_ids, resolution=resolution)
    else:
        log("\n--- Step 6: CAP-UDF SKIPPED ---")

    # === Step 7: Mesh clipping ∩ squirrel.obj for all methods ===
    log("\n--- Step 7: Mesh Clipping ∩ squirrel.obj ---")
    if os.path.exists(SQUIRREL_OBJ):
        import trimesh
        import vedo as vd
        squirrel_vedo = vd.Mesh(SQUIRREL_OBJ)
        log(f"  Squirrel mesh: {squirrel_vedo.npoints} verts")
        bool_dir = os.path.join(OUT_DIR, "boolean")
        os.makedirs(bool_dir, exist_ok=True)

        # Clip non-flooding methods to squirrel interior
        # These meshes are non-manifold so boolean won't work.
        # Use vedo inside_points (VTK vtkSelectEnclosedPoints) for robust containment.
        for method in METHODS:
            if method == 'ours_ffb':
                continue  # ours already does boolean inside flooding pipeline
            for sid in shape_ids:
                mesh_path = os.path.join(OUT_DIR, "meshes", f"{method}_{sid}.obj")
                bool_path = os.path.join(bool_dir, f"{method}_{sid}.obj")
                if not os.path.exists(mesh_path):
                    continue
                try:
                    recon = trimesh.load(mesh_path, force='mesh', process=False)
                    # Use vedo inside_points for robust vertex containment
                    inside_ids = squirrel_vedo.inside_points(
                        recon.vertices.tolist(), return_ids=True)
                    inside = np.zeros(len(recon.vertices), dtype=bool)
                    inside[inside_ids] = True
                    # Keep faces where ALL 3 vertices are inside
                    face_inside = inside[recon.faces].all(axis=1)
                    clipped = recon.submesh([np.where(face_inside)[0]], append=True)
                    clipped.remove_unreferenced_vertices()
                    if len(clipped.faces) > 0:
                        clipped.export(bool_path)
                        log(f"  {method} {sid}: {len(recon.faces)} → {len(clipped.faces)} faces after clipping")
                    else:
                        log(f"  {method} {sid}: empty after clipping, using raw mesh")
                        recon.export(bool_path)
                except Exception as e:
                    log(f"  {method} {sid}: clipping failed ({e}), using raw mesh")
                    try:
                        import shutil as _shutil
                        _shutil.copy2(mesh_path, bool_path)
                    except Exception:
                        pass
    else:
        log("  WARNING: squirrel.obj not found, skipping boolean")

    # === EVALUATION (ours uses flooding, others use boolean results) ===
    log("\n--- Evaluation ---")
    all_results = {}
    for method in METHODS:
        all_results[method] = {}
        for sid in shape_ids:
            gt_obj = os.path.join(GT_OBJ_DIR, f"{sid}.obj")
            if method == 'ours_ffb':
                mesh_path = os.path.join(OUT_DIR, "flooding", f"ours_ffb_{sid}.obj")
            else:
                # Use boolean result if available, otherwise raw mesh
                bool_path = os.path.join(OUT_DIR, "boolean", f"{method}_{sid}.obj")
                mesh_path = bool_path if os.path.exists(bool_path) else \
                    os.path.join(OUT_DIR, "meshes", f"{method}_{sid}.obj")
            if not os.path.exists(mesh_path):
                log(f"  SKIP {method} shape {sid}: mesh not found: {mesh_path}")
                continue
            if not os.path.exists(gt_obj):
                log(f"  SKIP {method} shape {sid}: GT obj not found: {gt_obj}")
                continue
            try:
                result = compute_symmfcd(gt_obj, mesh_path)
                all_results[method][sid] = result
                log(f"  {method} shape {sid}: SymMFCD = {result['symmetric_mfcd']:.6f}")
            except Exception as e:
                log(f"  {method} shape {sid}: Failed: {e}")

    metrics_path = os.path.join(OUT_DIR, "metrics", "symmfcd_results.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # === VISUALIZATION ===
    log("\n--- Visualization ---")

    # Loss curves (ours vs UDF)
    loss_files, loss_labels = [], []
    for tag, label in [("ours_ffb", "Ours (FFB+MC)"), ("udf_model", "UDF model")]:
        lf = os.path.join(OUT_DIR, "ckpts", tag, "loss_history.json")
        if os.path.exists(lf):
            loss_files.append(lf)
            loss_labels.append(label)
    if loss_files:
        plot_loss_curves(loss_files, loss_labels,
                         os.path.join(OUT_DIR, "figures", "loss_curves.png"),
                         title="Training Loss: FFB vs UDF")

    # SymMFCD bar chart
    avg_mfcd = {}
    for method in METHODS:
        vals = [r['symmetric_mfcd'] for r in all_results.get(method, {}).values()
                if 'symmetric_mfcd' in r and not np.isnan(r['symmetric_mfcd'])]
        if vals:
            avg_mfcd[method] = float(np.mean(vals))
    if avg_mfcd:
        plot_mfcd_bar_chart(
            list(avg_mfcd.keys()), list(avg_mfcd.values()),
            os.path.join(OUT_DIR, "figures", "symmfcd_comparison.png"),
            title="SymMFCD: FFB+MC vs UDF+MeshUDF vs UDF+NDC vs CAP-UDF")

    # Mesh comparison per shape (ours uses flooding)
    for sid in shape_ids:
        available_methods = []
        available_paths = []
        for method in METHODS:
            if method == 'ours_ffb':
                mp = os.path.join(OUT_DIR, "flooding", f"ours_ffb_{sid}.obj")
            else:
                mp = os.path.join(OUT_DIR, "meshes", f"{method}_{sid}.obj")
            available_methods.append(method)
            available_paths.append(mp if os.path.exists(mp) else None)
        render_mesh_comparison(
            available_paths, available_methods,
            os.path.join(OUT_DIR, "figures", f"mesh_comparison_{sid}.png"),
            title=f"Methods Comparison: Shape {sid}")

    # === SUMMARY ===
    log(f"\n=== {EXP_ID} SUMMARY ===")
    log(f"Argument: FFB+MC vs UDF+specialized_extraction vs CAP-UDF")
    log(f"Repos available: {repo_status}")
    mesh_files = glob.glob(os.path.join(OUT_DIR, "meshes", "*.obj"))
    log(f"Meshes: {len(mesh_files)}")
    if avg_mfcd:
        for m, v in avg_mfcd.items():
            log(f"  {m}: {v:.6f}")
    log(f"Results: {OUT_DIR}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()
