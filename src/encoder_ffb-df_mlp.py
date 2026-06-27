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
ap.add_argument("--minimal", action="store_true", help="Minimal run: 1500 uniform samples only (no near-surface)")
ap.add_argument("--uniform_only", action="store_true",
                help="Use only uniform sampling (no near-surface enrichment), full sample count")
ap.add_argument("--threshold", type=float, default=0.3,
                help="Near-surface threshold for FFBDF mode (default: 0.3)")
ap.add_argument("--save_path", default=None,
                help="Override output directory (relative to workspacePath)")
ap.add_argument("--shape_id", default=None,
                help="Only encode a specific shape ID (e.g., '1')")
args = ap.parse_args()
mlp_sample_num = 1500 if args.minimal else (4000 if args.fast else 64000)
use_minimal = args.minimal
use_uniform_only = args.uniform_only
bool_ffbdf = True  # FFBDF flag

# projName = "squirrel"
# projectPath = "_out_" + projName + "/"
# workspacePath = "/data/data-mlp/" + projectPath
workspacePath = "data/"
targetPath = "obj/"
savePath = args.save_path if args.save_path else ("npz-resample-uniform/" if use_uniform_only else "npz-resample/")

# 创建保存目录
os.makedirs(os.path.join(workspacePath, savePath), exist_ok=True)

AB_paths = [x.split('.')[0] for x in glob.glob(workspacePath + 'obj/*.obj')]
AB_paths = sorted(AB_paths)

if args.shape_id:
    AB_paths = [p for p in AB_paths if os.path.basename(p) == args.shape_id]

print(AB_paths)

for path in AB_paths:
    expTime = os.path.basename(path).split(".")[0]

    print("start load:", path)
    objName = workspacePath + "obj/" + str(expTime) + ".obj"
    
    if bool_ffbdf:
        # FFBDF mode: load single mesh and split into multiple objects
        mesh = vedo.load(objName)
        objs = mesh.split()
        all_vertices = []
        all_faces = []
        max_distances = []
        
        # Process each object from the split
        for obj in objs:
            vertices = np.array(obj.points)
            
            # Use cells instead of faces as suggested by vedo
            faces = np.array(obj.cells)
            
            # Ensure faces are in the correct format for igl (triangles)
            if len(faces.shape) == 1:
                # If faces is a flat array, reshape it to triangles
                faces = faces.reshape(-1, 3)
            elif len(faces.shape) == 2 and faces.shape[1] > 3:
                # If faces have more than 3 vertices, convert to triangles
                # This is a simple triangulation - for complex cases you might need more sophisticated triangulation
                faces = faces[:, :3]
            
            # Ensure faces are in the correct data type for igl
            faces = faces.astype(np.int32)
            
            all_vertices.append(vertices)
            all_faces.append(faces)
            
            # Compute farthest distance for this object using SDF
            # Sample points in a grid around the object
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            max_size = np.max(bbox_size)
            
            # Create a grid of points around the object
            grid_size = 128
            x = np.linspace(bbox_center[0] - max_size, bbox_center[0] + max_size, grid_size)
            y = np.linspace(bbox_center[1] - max_size, bbox_center[1] + max_size, grid_size)
            z = np.linspace(bbox_center[2] - max_size, bbox_center[2] + max_size, grid_size)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            # Compute SDF for grid points
            sdf_values = igl.signed_distance(grid_points, vertices, faces)[0]
            
            # Compute farthest distance as max(-sdf) for points inside the object
            inside_mask = sdf_values < 0
            if np.any(inside_mask):
                farthest_dist = np.max(-sdf_values[inside_mask])
            else:
                # Alert if no points are inside the object
                print(f"Warning: No points found inside object {objName}")
                farthest_dist = np.linalg.norm(bbox_max - bbox_min)
            
            max_distances.append(farthest_dist)
    else:
        # Original mode: single object
        vertices, faces = igl.read_triangle_mesh(objName)
        all_vertices = [vertices]
        all_faces = [faces]
        max_distances = [1]  # Not used in original mode
    
    # Compute FFB value for a batch of points (multi-fragment normalized SDF)
    def compute_ffb_values(pts):
        sdf = np.full(len(pts), np.inf)
        for obj_v, obj_f, md in zip(all_vertices, all_faces, max_distances):
            obj_sdf = igl.signed_distance(pts, obj_v, obj_f)[0]
            inside = obj_sdf < 0
            obj_sdf[inside] = obj_sdf[inside] / md
            sdf = np.minimum(sdf, obj_sdf)
        return sdf

    # Total target count
    total_enriched = int(mlp_sample_num * 0.5) * 10 + (mlp_sample_num - int(mlp_sample_num * 0.5))
    # = 352000 for mlp_sample_num=64000

    if use_minimal:
        # Minimal mode: just uniform random
        poisson_grid_points = np.random.rand(mlp_sample_num, 3).astype(np.float64) * 2 - 1
        sdf_values = compute_ffb_values(poisson_grid_points)
    elif use_uniform_only:
        # Uniform-only mode: all uniform, same total count
        poisson_grid_points = np.random.rand(total_enriched, 3).astype(np.float64) * 2 - 1
        sdf_values = compute_ffb_values(poisson_grid_points)
    else:
        # Stratified value-based sampling:
        #   Phase 1: collect points with value in [-0.2, 0.2] → 70% of total
        #   Phase 2: collect points with value outside [-0.2, 0.2] → 30% of total
        # Points rejected in phase 1 (outside [-0.2, 0.2]) are kept for phase 2.
        near_target = int(total_enriched * 0.7)   # 70% near-surface
        far_target = total_enriched - near_target  # 30% deep + far

        near_pts = np.array([]).reshape(0, 3)
        near_vals = np.array([])
        far_pts = np.array([]).reshape(0, 3)
        far_vals = np.array([])

        batch_size_sample = 100000  # points per sampling iteration
        max_iterations = 200

        for iteration in tqdm(range(max_iterations), desc=f"Stratified sampling for {path}"):
            # Sample batch
            sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
            pts = sampler.random(batch_size_sample)
            pts = 2 * pts - 1

            # Compute FFB values
            vals = compute_ffb_values(pts)

            # Bin by value range
            near_mask = (vals >= -0.2) & (vals <= 0.2)
            far_mask = ~near_mask

            if near_mask.any():
                near_pts = np.vstack([near_pts, pts[near_mask]]) if len(near_pts) > 0 else pts[near_mask]
                near_vals = np.concatenate([near_vals, vals[near_mask]])
            if far_mask.any():
                far_pts = np.vstack([far_pts, pts[far_mask]]) if len(far_pts) > 0 else pts[far_mask]
                far_vals = np.concatenate([far_vals, vals[far_mask]])

            # Check if both bins are full
            if len(near_pts) >= near_target and len(far_pts) >= far_target:
                break

        # Subsample each bin to target
        if len(near_pts) > near_target:
            idx = np.random.choice(len(near_pts), near_target, replace=False)
            near_pts, near_vals = near_pts[idx], near_vals[idx]
        if len(far_pts) > far_target:
            idx = np.random.choice(len(far_pts), far_target, replace=False)
            far_pts, far_vals = far_pts[idx], far_vals[idx]

        poisson_grid_points = np.vstack([near_pts, far_pts])
        sdf_values = np.concatenate([near_vals, far_vals])

        print(f"Stratified: near[-0.2,0.2]={len(near_pts)}, far={len(far_pts)}, total={len(poisson_grid_points)}")
        print(f"  Value range: [{sdf_values.min():.4f}, {sdf_values.max():.4f}]")

    npyName = os.path.join(workspacePath, savePath + str(expTime) + ".npz")

    print("start save:", npyName)

    np.savez(npyName, poisson_grid_points=poisson_grid_points, sdf_values=sdf_values)

    data = np.load(npyName)
    print(data["poisson_grid_points"], data["sdf_values"])

    # break