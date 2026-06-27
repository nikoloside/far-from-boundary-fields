"""
Signed-UDF encoder from mesh.
Computes global signed distance (inside < 0, outside > 0) without per-fragment
normalization. Unlike FFB, the raw signed distance is preserved.
Output: poisson_grid_points (Nx3), sdf_values (N)
"""
import glob
import numpy as np
import os
import scipy
import igl
from tqdm import tqdm
import vedo
import argparse


def _as_array(x):
    # vedo exposes .points/.cells as a property in some versions, a method in others
    return np.array(x() if callable(x) else x)


ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="Use fewer samples (4000) for quick test")
ap.add_argument("--minimal", action="store_true", help="Minimal: 1500 uniform samples only")
ap.add_argument("--uniform_only", action="store_true",
                help="Use only uniform sampling (no near-surface enrichment), full sample count")
ap.add_argument("--save_path", default=None,
                help="Override output directory (relative to workspacePath)")
ap.add_argument("--shape_id", default=None,
                help="Only encode a specific shape ID (e.g., '1')")
args = ap.parse_args()
mlp_sample_num = 1500 if args.minimal else (4000 if args.fast else 64000)
use_minimal = args.minimal
use_uniform_only = args.uniform_only
bool_multi_fragment = True

workspacePath = "data/"
targetPath = "obj/"
savePath = args.save_path if args.save_path else "npz-signed-udf/"

os.makedirs(os.path.join(workspacePath, savePath), exist_ok=True)

AB_paths = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join(workspacePath, targetPath, "*.obj"))]
AB_paths = sorted(AB_paths)

if args.shape_id:
    AB_paths = [p for p in AB_paths if p == args.shape_id]

print("Signed-UDF encoder, objects:", AB_paths)

for path in AB_paths:
    expTime = path
    print("start load:", path)
    objName = os.path.join(workspacePath, targetPath, f"{expTime}.obj")

    if bool_multi_fragment:
        mesh = vedo.load(objName)
        objs = mesh.split()
        all_vertices = []
        all_faces = []

        for obj in objs:
            vertices = _as_array(obj.points).astype(np.float64)
            faces = _as_array(obj.cells)
            if len(faces.shape) == 1:
                faces = faces.reshape(-1, 3)
            elif len(faces.shape) == 2 and faces.shape[1] > 3:
                faces = faces[:, :3]
            faces = faces.astype(np.int64)  # igl requires int64 faces
            all_vertices.append(vertices)
            all_faces.append(faces)
    else:
        vertices, faces = igl.read_triangle_mesh(objName)
        all_vertices = [vertices]
        all_faces = [faces]

    # Compute signed-UDF value for a batch of points (raw SDF, no normalization)
    def compute_signed_udf(pts):
        sdf = np.full(len(pts), np.inf)
        for obj_v, obj_f in zip(all_vertices, all_faces):
            s = igl.signed_distance(pts, obj_v, obj_f)[0]
            closer = np.abs(s) < np.abs(sdf)
            sdf[closer] = s[closer]
        return sdf

    # Total target count
    total_enriched = int(mlp_sample_num * 0.5) * 10 + (mlp_sample_num - int(mlp_sample_num * 0.5))

    if use_minimal:
        poisson_grid_points = np.random.rand(mlp_sample_num, 3).astype(np.float64) * 2 - 1
        sdf_values = compute_signed_udf(poisson_grid_points)
    elif use_uniform_only:
        poisson_grid_points = np.random.rand(total_enriched, 3).astype(np.float64) * 2 - 1
        sdf_values = compute_signed_udf(poisson_grid_points)
    else:
        # Stratified value-based sampling (same approach as FFB encoder):
        #   Phase 1: collect points with |value| <= 0.05 → 70% of total
        #   Phase 2: collect points with |value| > 0.05 → 30% of total
        # Signed-UDF has shallow interior (~-0.05 to -0.37), so use ±0.05 boundary.
        near_target = int(total_enriched * 0.7)
        far_target = total_enriched - near_target

        near_pts = np.array([]).reshape(0, 3)
        near_vals = np.array([])
        far_pts = np.array([]).reshape(0, 3)
        far_vals = np.array([])

        batch_size_sample = 100000
        max_iterations = 200

        for iteration in tqdm(range(max_iterations), desc=f"Stratified sampling for {path}"):
            sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
            pts = sampler.random(batch_size_sample)
            pts = 2 * pts - 1

            vals = compute_signed_udf(pts)

            near_mask = (vals >= -0.05) & (vals <= 0.05)
            far_mask = ~near_mask

            if near_mask.any():
                near_pts = np.vstack([near_pts, pts[near_mask]]) if len(near_pts) > 0 else pts[near_mask]
                near_vals = np.concatenate([near_vals, vals[near_mask]])
            if far_mask.any():
                far_pts = np.vstack([far_pts, pts[far_mask]]) if len(far_pts) > 0 else pts[far_mask]
                far_vals = np.concatenate([far_vals, vals[far_mask]])

            if len(near_pts) >= near_target and len(far_pts) >= far_target:
                break

        if len(near_pts) > near_target:
            idx = np.random.choice(len(near_pts), near_target, replace=False)
            near_pts, near_vals = near_pts[idx], near_vals[idx]
        if len(far_pts) > far_target:
            idx = np.random.choice(len(far_pts), far_target, replace=False)
            far_pts, far_vals = far_pts[idx], far_vals[idx]

        poisson_grid_points = np.vstack([near_pts, far_pts])
        sdf_values = np.concatenate([near_vals, far_vals])

        print(f"Stratified: near[-0.05,0.05]={len(near_pts)}, far={len(far_pts)}, total={len(poisson_grid_points)}")
        print(f"  Value range: [{sdf_values.min():.4f}, {sdf_values.max():.4f}]")

    npyName = os.path.join(workspacePath, savePath, f"{expTime}.npz")
    print("start save:", npyName)
    np.savez(npyName, poisson_grid_points=poisson_grid_points, sdf_values=sdf_values)

    data = np.load(npyName)
    print(f"Saved: {len(data['poisson_grid_points'])} points, SDF range [{data['sdf_values'].min():.4f}, {data['sdf_values'].max():.4f}]")
