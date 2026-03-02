import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import argparse

def sample_points(mesh, num_points=10000):
    """Uniformly sample points from mesh surface."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def load_obj_shapes(obj_path):
    """Load and split OBJ file into individual shapes."""
    mesh = trimesh.load(obj_path)
    # Split mesh into connected components
    components = mesh.split(only_watertight=True)
    return components

def chamfer_distance(points1, points2):
    """Calculate Chamfer Distance between two point sets."""
    dist_matrix = cdist(points1, points2)
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)
    chamfer_dist = np.mean(min_dist_1_to_2) + np.mean(min_dist_2_to_1)
    return chamfer_dist

def save_intermediate_results(individual_errors, method_avg_errors, method, shape_index, output_dir='.'):
    """Save intermediate results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'individual_errors': individual_errors,
        'method_avg_errors': method_avg_errors,
        'last_method': method,
        'last_shape_index': shape_index,
        'timestamp': time.time()
    }
    json_path = os.path.join(output_dir, 'chamfer_distance_intermediate.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)

def load_intermediate_results(output_dir='.'):
    """Load intermediate results from JSON file if it exists."""
    json_path = os.path.join(output_dir, 'chamfer_distance_intermediate.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
        return results
    return None

def main(output_dir='.'):
    """Main function to calculate Chamfer Distance errors.
    
    Args:
        output_dir: Directory to save results (JSON and PNG files). Defaults to current directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original OBJ files (1.obj to 10.obj)
    original_objs = []
    for i in range(1, 11):
        obj_path = f"./objs/squirrel-feature-origin/{i}.obj"
        if os.path.exists(obj_path):
            original_objs.append(obj_path)
        else:
            print(f"Warning: {obj_path} not found")
    
    original_objs = sorted(original_objs)
    
    # Method is "origin" (comparing origin with itself)
    methods = ["origin"]
    
    # Try to load intermediate results
    intermediate = load_intermediate_results(output_dir)
    if intermediate:
        print("Found intermediate results. Loading...")
        individual_errors = intermediate['individual_errors']
        method_avg_errors = intermediate['method_avg_errors']
        last_method = intermediate['last_method']
        shape_index = intermediate['last_shape_index']
        
        # Find the index of the last processed method
        start_method_idx = methods.index(last_method) + 1
        print(f"Resuming from method: {methods[start_method_idx]}")
    else:
        print("No intermediate results found. Starting from scratch...")
        individual_errors = []
        method_avg_errors = {}
        start_method_idx = 0
        shape_index = 0
    
    num_sample_points = 5000
    
    # For each method
    for method in tqdm(methods[start_method_idx:], desc="Processing methods"):
        print(f"\nProcessing method: {method}")
        method_errors = []
        
        # For each original obj (compare with itself)
        for orig_obj in tqdm(original_objs, desc=f"Processing objects for {method}", leave=False):
            orig_name = os.path.basename(orig_obj).replace('.obj', '')
            print("current obj: ", orig_name, "\n")
                
            # Compare origin with itself
            filtered_obj = orig_obj
            
            if not os.path.exists(filtered_obj):
                print(f"Warning: {filtered_obj} not found")
                method_errors.append(np.nan)
                individual_errors.append([method, shape_index, np.nan])
                shape_index += 1
                continue
            
            print(f"Comparing {orig_obj} with itself")
            
            # Load both original and filtered shapes
            orig_shapes = load_obj_shapes(orig_obj)
            filtered_shapes = load_obj_shapes(filtered_obj)
            
            # For each shape in original obj
            for orig_shape in tqdm(orig_shapes, desc=f"Processing shapes in {orig_name}", leave=False):
                orig_points = sample_points(orig_shape, num_sample_points)
                min_error = float('inf')
                
                # Compare with each shape in filtered obj
                for filtered_shape in tqdm(filtered_shapes, desc="Comparing with filtered shapes", leave=False):
                    filtered_points = sample_points(filtered_shape, num_sample_points)
                    error = chamfer_distance(orig_points, filtered_points)
                    min_error = min(min_error, error)
                
                if min_error != float('inf'):
                    method_errors.append(min_error)
                    individual_errors.append([method, shape_index, min_error])
                else:
                    method_errors.append(np.nan)
                    individual_errors.append([method, shape_index, np.nan])
                shape_index += 1
            
            # Save intermediate results after each object
            save_intermediate_results(individual_errors, method_avg_errors, method, shape_index, output_dir)
        
        # Calculate average error for this method
        method_avg_errors[method] = np.nanmean(method_errors)
        
        # Save intermediate results after each method
        save_intermediate_results(individual_errors, method_avg_errors, method, shape_index, output_dir)
    
    # Print results
    print("\nIndividual Errors [method, index, error]:")
    for method, idx, error in individual_errors:
        print(f"{method}, {idx}, {error:.6f}")
    
    print("\nAverage Errors by Method:")
    for method, avg_error in method_avg_errors.items():
        print(f"{method}: {avg_error:.6f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Group errors by method for plotting
    method_errors = {}
    for method in methods:
        method_errors[method] = [e for m, i, e in individual_errors if m == method]
    
    shape_indices = range(len(next(iter(method_errors.values()))))
    
    for method, errors in method_errors.items():
        plt.plot(shape_indices, errors, label=method, marker='o', markersize=4)
    
    plt.xlabel('Shape Index')
    plt.ylabel('Chamfer Distance Error')
    plt.title('Chamfer Distance Error by Method')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'chamfer_distance_errors.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Chamfer Distance errors between original and filtered OBJ files')
    parser.add_argument('--output-dir', '-o', type=str, default='.', 
                        help='Directory to save results (JSON and PNG files). Defaults to current directory.')
    args = parser.parse_args()
    main(output_dir=args.output_dir)
