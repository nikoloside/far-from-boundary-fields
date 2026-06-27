"""
Truncated-UDF encoder from mesh.
Computes min(|SDF(p)|, threshold) for each query point.
Output: poisson_grid_points (Nx3), udf_values (N)
"""
import glob
import numpy as np
import os
import scipy
import igl
from tqdm import tqdm
import vedo
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="Use fewer samples (4000) for quick test")
ap.add_argument("--minimal", action="store_true", help="Minimal: 1500 uniform samples only")
ap.add_argument("--uniform_only", action="store_true",
                help="Use only uniform sampling (no near-surface enrichment), full sample count")
ap.add_argument("--threshold", type=float, default=0.1,
                help="Truncation threshold (default: 0.1)")
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
savePath = args.save_path if args.save_path else "npz-truncated-udf/"

os.makedirs(os.path.join(workspacePath, savePath), exist_ok=True)

AB_paths = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join(workspacePath, targetPath, "*.obj"))]
AB_paths = sorted(AB_paths)

if args.shape_id:
    AB_paths = [p for p in AB_paths if p == args.shape_id]

print("Truncated-UDF encoder, objects:", AB_paths)

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
            vertices = np.array(obj.points)
            faces = np.array(obj.cells)
            if len(faces.shape) == 1:
                faces = faces.reshape(-1, 3)
            elif len(faces.shape) == 2 and faces.shape[1] > 3:
                faces = faces[:, :3]
            faces = faces.astype(np.int32)
            all_vertices.append(vertices)
            all_faces.append(faces)
    else:
        vertices, faces = igl.read_triangle_mesh(objName)
        all_vertices = [vertices]
        all_faces = [faces]

    # Same sample count logic as UDF encoder
    near_surface_num = int(mlp_sample_num * 0.5)
    uniform_num = mlp_sample_num - near_surface_num
    if use_minimal:
        uniform_num = mlp_sample_num
        near_surface_num = 0
    elif use_uniform_only:
        total_enriched = int(mlp_sample_num * 0.5) * 10 + (mlp_sample_num - int(mlp_sample_num * 0.5))
        uniform_num = total_enriched
        near_surface_num = 0
    else:
        near_surface_num *= 10

    near_surface_points = np.array([]).reshape(0, 3)
    max_iterations = 20 if use_minimal else 100

    for iteration in tqdm(range(max_iterations), desc=f"Sampling for {path}", disable=(use_minimal or use_uniform_only)):
        if use_minimal or use_uniform_only:
            coarse_points = np.random.rand(uniform_num, 3).astype(np.float64) * 2 - 1
            near_surface_selected = coarse_points
            break
        coarse_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
        coarse_points = coarse_sampler.random(near_surface_num * 2)
        coarse_points = 2 * coarse_points - 1

        # Compute UDF (unsigned) for filtering
        coarse_udf = np.full(len(coarse_points), np.inf)
        for obj_vertices, obj_faces in zip(all_vertices, all_faces):
            result = igl.signed_distance(
                coarse_points, obj_vertices, obj_faces,
                sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED
            )
            s = result[0]
            coarse_udf = np.minimum(coarse_udf, s)

        near_surface_mask = coarse_udf <= args.threshold
        new_near_surface_points = coarse_points[near_surface_mask]

        near_surface_points = (
            np.vstack([near_surface_points, new_near_surface_points])
            if len(near_surface_points) > 0
            else new_near_surface_points
        )

        if len(near_surface_points) >= near_surface_num:
            break

    if not use_minimal and not use_uniform_only:
        if len(near_surface_points) >= near_surface_num:
            indices = np.random.choice(len(near_surface_points), near_surface_num, replace=False)
            near_surface_selected = near_surface_points[indices]
        else:
            near_surface_selected = near_surface_points
            print(f"Warning: Only {len(near_surface_points)} near-surface points (requested {near_surface_num})")

    if use_minimal or use_uniform_only:
        poisson_grid_points = near_surface_selected
    else:
        uniform_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
        uniform_points = uniform_sampler.random(uniform_num)
        uniform_points = 2 * uniform_points - 1
        poisson_grid_points = np.vstack([near_surface_selected, uniform_points])

    # Compute UDF (unsigned distance)
    udf_values = np.full(len(poisson_grid_points), np.inf)
    for obj_vertices, obj_faces in zip(all_vertices, all_faces):
        result = igl.signed_distance(
            poisson_grid_points, obj_vertices, obj_faces,
            sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED
        )
        s = result[0]
        udf_values = np.minimum(udf_values, s)

    # Truncate: min(udf, threshold)
    udf_values = np.minimum(udf_values, args.threshold)

    npyName = os.path.join(workspacePath, savePath, f"{expTime}.npz")
    print("start save:", npyName)
    np.savez(npyName, poisson_grid_points=poisson_grid_points, udf_values=udf_values)

    data = np.load(npyName)
    print(f"Saved: {len(data['poisson_grid_points'])} points, UDF range [{data['udf_values'].min():.4f}, {data['udf_values'].max():.4f}]")
