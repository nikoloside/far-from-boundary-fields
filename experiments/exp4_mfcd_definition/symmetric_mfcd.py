"""
Symmetric Multi-Fragment Chamfer Distance (SymMFCD)

This implements the bidirectional/symmetric version of MFCD.

Key difference from one-directional MFCD:
- One-directional: For each fragment in A, find closest in B → CD
- Symmetric: Above + For each fragment in B, find closest in A → CD

Mathematical definition:
    SymMFCD(A, B) = (1/|A|) * Σ_i min_j CD(a_i, b_j)  [A→B direction]
                  + (1/|B|) * Σ_j min_i CD(b_j, a_i)  [B→A direction]

Where:
- A, B are sets of fragments (meshes)
- CD is bidirectional Chamfer Distance between point clouds
- a_i is i-th fragment in A
- b_j is j-th fragment in B
"""

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


def chamfer_distance(points1, points2):
    """
    Calculate bidirectional Chamfer Distance between two point sets.

    CD(P1, P2) = mean(min_dist(P1→P2)) + mean(min_dist(P2→P1))

    Args:
        points1: (N, 3) array of points
        points2: (M, 3) array of points

    Returns:
        Bidirectional Chamfer Distance (scalar)
    """
    dist_matrix = cdist(points1, points2)

    # P1 → P2: for each point in P1, find min distance to P2
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)

    # P2 → P1: for each point in P2, find min distance to P1
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)

    # Bidirectional CD
    chamfer_dist = np.mean(min_dist_1_to_2) + np.mean(min_dist_2_to_1)

    return chamfer_dist


def symmetric_mfcd(fragments_A, fragments_B, num_sample_points=10000):
    """
    Calculate Symmetric Multi-Fragment Chamfer Distance.

    SymMFCD(A, B) = MFCD(A→B) + MFCD(B→A)

    Where:
        MFCD(A→B) = (1/|A|) * Σ_i min_j CD(a_i, b_j)
        MFCD(B→A) = (1/|B|) * Σ_j min_i CD(b_j, a_i)

    Args:
        fragments_A: List of trimesh objects (fragments in set A)
        fragments_B: List of trimesh objects (fragments in set B)
        num_sample_points: Number of points to sample from each fragment

    Returns:
        sym_mfcd: Symmetric MFCD (scalar)
        mfcd_a_to_b: One-directional MFCD from A to B
        mfcd_b_to_a: One-directional MFCD from B to A
        fragment_errors_a_to_b: List of per-fragment errors (A→B)
        fragment_errors_b_to_a: List of per-fragment errors (B→A)
    """

    # Direction 1: A → B
    # For each fragment in A, find closest fragment in B
    fragment_errors_a_to_b = []

    for frag_a in tqdm(fragments_A, desc="Computing MFCD (A→B)", leave=False):
        points_a = sample_points(frag_a, num_sample_points)
        min_error = float('inf')

        # Find closest fragment in B
        for frag_b in fragments_B:
            points_b = sample_points(frag_b, num_sample_points)
            error = chamfer_distance(points_a, points_b)
            min_error = min(min_error, error)

        if min_error != float('inf'):
            fragment_errors_a_to_b.append(min_error)
        else:
            fragment_errors_a_to_b.append(np.nan)

    mfcd_a_to_b = np.nanmean(fragment_errors_a_to_b)


    # Direction 2: B → A
    # For each fragment in B, find closest fragment in A
    fragment_errors_b_to_a = []

    for frag_b in tqdm(fragments_B, desc="Computing MFCD (B→A)", leave=False):
        points_b = sample_points(frag_b, num_sample_points)
        min_error = float('inf')

        # Find closest fragment in A
        for frag_a in fragments_A:
            points_a = sample_points(frag_a, num_sample_points)
            error = chamfer_distance(points_b, points_a)
            min_error = min(min_error, error)

        if min_error != float('inf'):
            fragment_errors_b_to_a.append(min_error)
        else:
            fragment_errors_b_to_a.append(np.nan)

    mfcd_b_to_a = np.nanmean(fragment_errors_b_to_a)


    # Symmetric MFCD
    sym_mfcd = mfcd_a_to_b + mfcd_b_to_a

    return sym_mfcd, mfcd_a_to_b, mfcd_b_to_a, fragment_errors_a_to_b, fragment_errors_b_to_a


def load_obj_shapes(obj_path):
    """Load and split OBJ file into individual fragments."""
    mesh = trimesh.load(obj_path)
    # Split mesh into connected components
    components = mesh.split(only_watertight=False)
    return components


def compare_meshes(orig_obj_path, recon_obj_path, num_sample_points=5000):
    """
    Compare two meshes using Symmetric MFCD.

    Args:
        orig_obj_path: Path to original (ground truth) mesh
        recon_obj_path: Path to reconstructed mesh
        num_sample_points: Number of points to sample per fragment

    Returns:
        Dictionary with comparison results
    """
    print(f"\nComparing:")
    print(f"  Original: {orig_obj_path}")
    print(f"  Reconstructed: {recon_obj_path}")

    # Load fragments
    orig_fragments = load_obj_shapes(orig_obj_path)
    recon_fragments = load_obj_shapes(recon_obj_path)

    print(f"  Original fragments: {len(orig_fragments)}")
    print(f"  Reconstructed fragments: {len(recon_fragments)}")

    # Compute Symmetric MFCD
    sym_mfcd, mfcd_o2r, mfcd_r2o, errors_o2r, errors_r2o = symmetric_mfcd(
        orig_fragments, recon_fragments, num_sample_points
    )

    results = {
        'symmetric_mfcd': sym_mfcd,
        'mfcd_orig_to_recon': mfcd_o2r,
        'mfcd_recon_to_orig': mfcd_r2o,
        'num_orig_fragments': len(orig_fragments),
        'num_recon_fragments': len(recon_fragments),
        'fragment_errors_orig_to_recon': errors_o2r,
        'fragment_errors_recon_to_orig': errors_r2o,
    }

    print(f"\nResults:")
    print(f"  SymMFCD: {sym_mfcd:.6f}")
    print(f"  MFCD (Orig→Recon): {mfcd_o2r:.6f}")
    print(f"  MFCD (Recon→Orig): {mfcd_r2o:.6f}")

    return results


def batch_comparison(orig_dir, recon_dirs, output_dir='.', num_sample_points=5000):
    """
    Compare multiple reconstruction methods against original meshes.

    Args:
        orig_dir: Directory containing original OBJ files
        recon_dirs: Dictionary of {method_name: directory_path} for reconstructed meshes
        output_dir: Directory to save results
        num_sample_points: Number of points to sample per fragment

    Returns:
        Dictionary of results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all original OBJ files
    orig_objs = sorted(glob.glob(os.path.join(orig_dir, "*.obj")))

    if not orig_objs:
        print(f"Warning: No OBJ files found in {orig_dir}")
        return {}

    print(f"Found {len(orig_objs)} original meshes")

    all_results = {}

    # For each method
    for method_name, recon_dir in recon_dirs.items():
        print(f"\n{'='*60}")
        print(f"Processing method: {method_name}")
        print(f"{'='*60}")

        method_results = {
            'per_mesh': [],
            'symmetric_mfcd_mean': None,
            'mfcd_orig_to_recon_mean': None,
            'mfcd_recon_to_orig_mean': None,
        }

        # For each original mesh
        for orig_obj_path in tqdm(orig_objs, desc=f"Processing {method_name}"):
            obj_name = os.path.basename(orig_obj_path)
            recon_obj_path = os.path.join(recon_dir, obj_name)

            if not os.path.exists(recon_obj_path):
                print(f"Warning: {recon_obj_path} not found, skipping")
                continue

            # Compare this pair
            results = compare_meshes(orig_obj_path, recon_obj_path, num_sample_points)
            results['obj_name'] = obj_name
            method_results['per_mesh'].append(results)

        # Compute mean metrics
        if method_results['per_mesh']:
            method_results['symmetric_mfcd_mean'] = np.mean([
                r['symmetric_mfcd'] for r in method_results['per_mesh']
            ])
            method_results['mfcd_orig_to_recon_mean'] = np.mean([
                r['mfcd_orig_to_recon'] for r in method_results['per_mesh']
            ])
            method_results['mfcd_recon_to_orig_mean'] = np.mean([
                r['mfcd_recon_to_orig'] for r in method_results['per_mesh']
            ])

        all_results[method_name] = method_results

    # Save results to JSON
    json_path = os.path.join(output_dir, 'symmetric_mfcd_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method_name, results in all_results.items():
        print(f"\n{method_name}:")
        print(f"  SymMFCD: {results['symmetric_mfcd_mean']:.6f}")
        print(f"  MFCD (Orig→Recon): {results['mfcd_orig_to_recon_mean']:.6f}")
        print(f"  MFCD (Recon→Orig): {results['mfcd_recon_to_orig_mean']:.6f}")

    # Plot comparison
    plot_comparison(all_results, output_dir)

    return all_results


def plot_comparison(all_results, output_dir='.'):
    """Plot comparison of different methods."""

    methods = list(all_results.keys())

    if not methods:
        print("No results to plot")
        return

    # Extract metrics
    sym_mfcd_means = [all_results[m]['symmetric_mfcd_mean'] for m in methods]
    mfcd_o2r_means = [all_results[m]['mfcd_orig_to_recon_mean'] for m in methods]
    mfcd_r2o_means = [all_results[m]['mfcd_recon_to_orig_mean'] for m in methods]

    # Create bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # SymMFCD
    axes[0].bar(methods, sym_mfcd_means, color='steelblue')
    axes[0].set_ylabel('Symmetric MFCD')
    axes[0].set_title('Symmetric MFCD (lower is better)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # MFCD Orig→Recon
    axes[1].bar(methods, mfcd_o2r_means, color='coral')
    axes[1].set_ylabel('MFCD (Orig→Recon)')
    axes[1].set_title('MFCD: Original → Reconstruction')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    # MFCD Recon→Orig
    axes[2].bar(methods, mfcd_r2o_means, color='mediumseagreen')
    axes[2].set_ylabel('MFCD (Recon→Orig)')
    axes[2].set_title('MFCD: Reconstruction → Original')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'symmetric_mfcd_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")


def main():
    """Main function with example usage."""
    parser = argparse.ArgumentParser(
        description='Calculate Symmetric Multi-Fragment Chamfer Distance (SymMFCD)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

1. Compare two meshes:
    python symmetric_mfcd.py \\
        --orig path/to/original.obj \\
        --recon path/to/reconstructed.obj

2. Batch comparison of multiple methods:
    python symmetric_mfcd.py \\
        --batch \\
        --orig-dir path/to/original_meshes/ \\
        --recon-dirs udf_mlp:path/to/udf_results/ \\
                     ffb_mlp:path/to/ffb_results/ \\
                     neuraludf:path/to/neuraludf_results/ \\
        --output-dir results/mfcd_comparison/
        """
    )

    parser.add_argument('--orig', type=str, help='Path to original OBJ file')
    parser.add_argument('--recon', type=str, help='Path to reconstructed OBJ file')
    parser.add_argument('--batch', action='store_true', help='Batch comparison mode')
    parser.add_argument('--orig-dir', type=str, help='Directory with original OBJ files')
    parser.add_argument('--recon-dirs', nargs='+', help='Method:directory pairs (e.g., udf:path/to/udf/)')
    parser.add_argument('--output-dir', '-o', type=str, default='.', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of sample points per fragment')

    args = parser.parse_args()

    if args.batch:
        # Batch comparison mode
        if not args.orig_dir or not args.recon_dirs:
            print("Error: --orig-dir and --recon-dirs required for batch mode")
            return

        # Parse recon_dirs
        recon_dirs = {}
        for pair in args.recon_dirs:
            if ':' not in pair:
                print(f"Warning: Invalid format '{pair}', expected 'method:directory'")
                continue
            method, directory = pair.split(':', 1)
            recon_dirs[method] = directory

        batch_comparison(args.orig_dir, recon_dirs, args.output_dir, args.num_samples)

    elif args.orig and args.recon:
        # Single comparison mode
        results = compare_meshes(args.orig, args.recon, args.num_samples)

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, 'symmetric_mfcd_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    else:
        print("Error: Either provide --orig and --recon, or use --batch mode")
        parser.print_help()


if __name__ == "__main__":
    main()
