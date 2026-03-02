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
args = ap.parse_args()
mlp_sample_num = 1500 if args.minimal else (4000 if args.fast else 64000)
use_minimal = args.minimal
bool_ffbdf = True  # FFBDF flag

# projName = "squirrel"
# projectPath = "_out_" + projName + "/"
# workspacePath = "/data/data-mlp/" + projectPath
workspacePath = "data/"
targetPath = "obj/"
savePath = "npz-resample/" # "npz-ffbdf"

# 创建保存目录
os.makedirs(os.path.join(workspacePath, savePath), exist_ok=True)

AB_paths = [x.split('.')[0] for x in glob.glob(workspacePath + 'obj/*.obj')]
AB_paths = sorted(AB_paths)

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
    
    pd_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius = 0.025)
    
    # 计算需要的点数
    near_surface_num = int(mlp_sample_num * 0.5)  # 50% 在表面附近
    uniform_num = mlp_sample_num - near_surface_num
    if use_minimal:
        # Minimal: pure uniform, no near-surface
        uniform_num = mlp_sample_num
        near_surface_num = 0
    else:
        near_surface_num *= 10
    
    # 重复采样直到获得足够的接近表面的点
    near_surface_points = np.array([]).reshape(0, 3)
    max_iterations = 20 if use_minimal else 100
    
    for iteration in tqdm(range(max_iterations), desc=f"Sampling for {path}", disable=use_minimal):
        if use_minimal and iteration > 0:
            break
        # 在 [-1, 1] 范围内进行采样
        if use_minimal:
            coarse_points = np.random.rand(uniform_num, 3).astype(np.float64) * 2 - 1
        else:
            coarse_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
            coarse_points = coarse_sampler.random(near_surface_num * 2)  # 采样更多点
            coarse_points = 2 * coarse_points - 1
        
        if bool_ffbdf:
            # FFBDF: compute SDF using multiple objects
            coarse_sdf = np.full(len(coarse_points), np.inf)
            for i, (obj_vertices, obj_faces, max_dist) in enumerate(zip(all_vertices, all_faces, max_distances)):
                # Compute signed distance to this object
                obj_sdf = igl.signed_distance(coarse_points, obj_vertices, obj_faces)[0]
                
                # If point is inside this object, normalize by max_dist
                inside_mask = obj_sdf < 0
                obj_sdf[inside_mask] = obj_sdf[inside_mask] / max_dist
                
                # Take minimum distance across all objects
                coarse_sdf = np.minimum(coarse_sdf, obj_sdf)
        else:
            # Original: single object SDF
            coarse_sdf = igl.signed_distance(coarse_points, vertices, faces)[0]
        
        # 检查 bool_ffbdf 是否为 True
        if bool_ffbdf:
            # 筛选出 SDF 值在 [-0.1, 0.1] 附近的点
            near_surface_mask = np.abs(coarse_sdf) <= 0.3
        else:
            # 如果 bool_ffbdf 为 False，使用不同的阈值
            near_surface_mask = np.abs(coarse_sdf) <= 0.1
        new_near_surface_points = coarse_points[near_surface_mask]
        
        # 添加到已有的点中
        near_surface_points = np.vstack([near_surface_points, new_near_surface_points]) if len(near_surface_points) > 0 else new_near_surface_points
        
        # 如果已经获得足够的点，跳出循环
        if use_minimal:
            near_surface_selected = coarse_points  # use all uniform points
            break
        if len(near_surface_points) >= near_surface_num:
            break
    
    # 从接近表面的点中随机选择需要的数量
    if not use_minimal:
        if len(near_surface_points) >= near_surface_num:
            indices = np.random.choice(len(near_surface_points), near_surface_num, replace=False)
            near_surface_selected = near_surface_points[indices]
        else:
            near_surface_selected = near_surface_points
            print(f"Warning: Only found {len(near_surface_points)} points near surface, less than requested {near_surface_num}")
    
    # 重新采样均匀分布的点
    if use_minimal:
        uniform_points = np.array([]).reshape(0, 3)
    else:
        uniform_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
        uniform_points = uniform_sampler.random(uniform_num)
    if len(uniform_points) > 0:
        uniform_points = 2 * uniform_points - 1
    
    # 合并两种采样点
    poisson_grid_points = np.vstack([near_surface_selected, uniform_points]) if len(uniform_points) > 0 else near_surface_selected
    
    if bool_ffbdf:
        # FFBDF: compute final SDF using multiple objects
        sdf_values = np.full(len(poisson_grid_points), np.inf)
        for i, (obj_vertices, obj_faces, max_dist) in enumerate(zip(all_vertices, all_faces, max_distances)):
            # Compute signed distance to this object
            obj_sdf = igl.signed_distance(poisson_grid_points, obj_vertices, obj_faces)[0]
            
            # If point is inside this object, normalize by max_dist
            inside_mask = obj_sdf < 0
            obj_sdf[inside_mask] = obj_sdf[inside_mask] / max_dist
            
            # Take minimum distance across all objects
            sdf_values = np.minimum(sdf_values, obj_sdf)
    else:
        # Original: single object SDF
        sdf_values = igl.signed_distance(poisson_grid_points, vertices, faces)[0]

    npyName = os.path.join(workspacePath, savePath + str(expTime) + ".npz")

    print("start save:", npyName)

    np.savez(npyName, poisson_grid_points=poisson_grid_points, sdf_values=sdf_values)

    data = np.load(npyName)
    print(data["poisson_grid_points"], data["sdf_values"])

    # break