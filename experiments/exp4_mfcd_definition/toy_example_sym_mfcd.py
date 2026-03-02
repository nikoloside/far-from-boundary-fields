"""
Toy Example: Demonstrating the difference between
One-directional MFCD and Symmetric MFCD

This script creates simple toy scenarios to show when
one-directional MFCD fails and symmetric MFCD succeeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def chamfer_distance(points1, points2):
    """Bidirectional Chamfer Distance between two point clouds."""
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points1, points2)
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)
    return np.mean(min_dist_1_to_2) + np.mean(min_dist_2_to_1)


def one_directional_mfcd(fragments_A, fragments_B):
    """One-directional MFCD: A → B only."""
    errors = []
    for frag_a in fragments_A:
        min_error = min([chamfer_distance(frag_a, frag_b) for frag_b in fragments_B])
        errors.append(min_error)
    return np.mean(errors), errors


def symmetric_mfcd(fragments_A, fragments_B):
    """Symmetric MFCD: A → B + B → A."""
    # A → B
    errors_a2b = []
    for frag_a in fragments_A:
        min_error = min([chamfer_distance(frag_a, frag_b) for frag_b in fragments_B])
        errors_a2b.append(min_error)
    mfcd_a2b = np.mean(errors_a2b)

    # B → A
    errors_b2a = []
    for frag_b in fragments_B:
        min_error = min([chamfer_distance(frag_b, frag_a) for frag_a in fragments_A])
        errors_b2a.append(min_error)
    mfcd_b2a = np.mean(errors_b2a)

    return mfcd_a2b + mfcd_b2a, mfcd_a2b, mfcd_b2a


def create_sphere_fragment(center, radius, num_points=100):
    """Create a spherical fragment as point cloud."""
    phi = np.random.uniform(0, 2*np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)
    return np.column_stack([x, y, z])


def scenario_1_missing_fragments():
    """
    Scenario 1: Missing Fragments
    Original: 5 fragments
    Reconstructed: 3 fragments (missing 2)
    """
    print("\n" + "="*60)
    print("Scenario 1: Missing Fragments")
    print("="*60)

    # Original: 5 fragments in a row
    orig_fragments = [
        create_sphere_fragment([i*2, 0, 0], 0.5, 200)
        for i in range(5)
    ]

    # Reconstructed: only first 3 fragments
    recon_fragments = [
        create_sphere_fragment([i*2, 0, 0], 0.5, 200)
        for i in range(3)
    ]

    print(f"Original fragments: {len(orig_fragments)}")
    print(f"Reconstructed fragments: {len(recon_fragments)}")

    # One-directional MFCD
    one_dir_mfcd, _ = one_directional_mfcd(orig_fragments, recon_fragments)
    print(f"\nOne-directional MFCD (Orig→Recon): {one_dir_mfcd:.6f}")

    # Symmetric MFCD
    sym_mfcd, mfcd_o2r, mfcd_r2o = symmetric_mfcd(orig_fragments, recon_fragments)
    print(f"\nSymmetric MFCD: {sym_mfcd:.6f}")
    print(f"  MFCD (Orig→Recon): {mfcd_o2r:.6f}  ← High (missing fragments)")
    print(f"  MFCD (Recon→Orig): {mfcd_r2o:.6f}  ← Low (all recon have match)")

    print("\n⚠️  Analysis:")
    print("  - One-directional MFCD is HIGH due to missing fragments 4 and 5")
    print("  - MFCD(Recon→Orig) is LOW because all 3 reconstructed fragments match")
    print("  - Symmetric MFCD captures both directions")

    return orig_fragments, recon_fragments


def scenario_2_extra_fragments():
    """
    Scenario 2: Extra Fragments
    Original: 3 fragments
    Reconstructed: 5 fragments (2 extra noise fragments)
    """
    print("\n" + "="*60)
    print("Scenario 2: Extra Fragments")
    print("="*60)

    # Original: 3 fragments
    orig_fragments = [
        create_sphere_fragment([i*2, 0, 0], 0.5, 200)
        for i in range(3)
    ]

    # Reconstructed: 3 correct + 2 noise fragments
    recon_fragments = [
        create_sphere_fragment([i*2, 0, 0], 0.5, 200)
        for i in range(3)
    ] + [
        create_sphere_fragment([3*2, 2, 0], 0.3, 200),  # Noise fragment 1
        create_sphere_fragment([4*2, -2, 0], 0.3, 200), # Noise fragment 2
    ]

    print(f"Original fragments: {len(orig_fragments)}")
    print(f"Reconstructed fragments: {len(recon_fragments)} (2 extra)")

    # One-directional MFCD
    one_dir_mfcd, _ = one_directional_mfcd(orig_fragments, recon_fragments)
    print(f"\nOne-directional MFCD (Orig→Recon): {one_dir_mfcd:.6f}")

    # Symmetric MFCD
    sym_mfcd, mfcd_o2r, mfcd_r2o = symmetric_mfcd(orig_fragments, recon_fragments)
    print(f"\nSymmetric MFCD: {sym_mfcd:.6f}")
    print(f"  MFCD (Orig→Recon): {mfcd_o2r:.6f}  ← Low (all orig have match)")
    print(f"  MFCD (Recon→Orig): {mfcd_r2o:.6f}  ← High (2 noise fragments)")

    print("\n⚠️  Analysis:")
    print("  - One-directional MFCD is LOW (misses the problem!)")
    print("  - MFCD(Recon→Orig) is HIGH due to 2 noise fragments")
    print("  - Only symmetric MFCD detects the extra fragments")

    return orig_fragments, recon_fragments


def scenario_3_position_shift():
    """
    Scenario 3: Position Shift
    Original: 3 fragments
    Reconstructed: 3 fragments (same count, but shifted positions)
    """
    print("\n" + "="*60)
    print("Scenario 3: Position Shift")
    print("="*60)

    # Original: 3 fragments
    orig_fragments = [
        create_sphere_fragment([i*2, 0, 0], 0.5, 200)
        for i in range(3)
    ]

    # Reconstructed: 3 fragments with position shift
    shift = 0.3
    recon_fragments = [
        create_sphere_fragment([i*2 + shift, shift, 0], 0.5, 200)
        for i in range(3)
    ]

    print(f"Original fragments: {len(orig_fragments)}")
    print(f"Reconstructed fragments: {len(recon_fragments)} (position shifted)")

    # One-directional MFCD
    one_dir_mfcd, _ = one_directional_mfcd(orig_fragments, recon_fragments)
    print(f"\nOne-directional MFCD (Orig→Recon): {one_dir_mfcd:.6f}")

    # Symmetric MFCD
    sym_mfcd, mfcd_o2r, mfcd_r2o = symmetric_mfcd(orig_fragments, recon_fragments)
    print(f"\nSymmetric MFCD: {sym_mfcd:.6f}")
    print(f"  MFCD (Orig→Recon): {mfcd_o2r:.6f}")
    print(f"  MFCD (Recon→Orig): {mfcd_r2o:.6f}")

    print("\n⚠️  Analysis:")
    print("  - Both directions show similar error (symmetric shift)")
    print("  - Symmetric MFCD properly captures the bidirectional shift")
    print("  - One-directional MFCD only sees half of the problem")

    return orig_fragments, recon_fragments


def scenario_4_perfect_reconstruction():
    """
    Scenario 4: Perfect Reconstruction
    Original: 3 fragments
    Reconstructed: 3 fragments (perfect match)
    """
    print("\n" + "="*60)
    print("Scenario 4: Perfect Reconstruction")
    print("="*60)

    # Original: 3 fragments
    orig_fragments = [
        create_sphere_fragment([i*2, 0, 0], 0.5, 200)
        for i in range(3)
    ]

    # Reconstructed: perfect copy
    recon_fragments = [
        create_sphere_fragment([i*2, 0, 0], 0.5, 200)
        for i in range(3)
    ]

    print(f"Original fragments: {len(orig_fragments)}")
    print(f"Reconstructed fragments: {len(recon_fragments)} (perfect)")

    # One-directional MFCD
    one_dir_mfcd, _ = one_directional_mfcd(orig_fragments, recon_fragments)
    print(f"\nOne-directional MFCD (Orig→Recon): {one_dir_mfcd:.6f}")

    # Symmetric MFCD
    sym_mfcd, mfcd_o2r, mfcd_r2o = symmetric_mfcd(orig_fragments, recon_fragments)
    print(f"\nSymmetric MFCD: {sym_mfcd:.6f}")
    print(f"  MFCD (Orig→Recon): {mfcd_o2r:.6f}")
    print(f"  MFCD (Recon→Orig): {mfcd_r2o:.6f}")

    print("\n✅ Analysis:")
    print("  - Both metrics are near-zero (perfect reconstruction)")
    print("  - Small non-zero values due to random sampling")

    return orig_fragments, recon_fragments


def visualize_scenario(orig_fragments, recon_fragments, title, filename):
    """Visualize original and reconstructed fragments."""
    fig = plt.figure(figsize=(14, 6))

    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    for i, frag in enumerate(orig_fragments):
        ax1.scatter(frag[:, 0], frag[:, 1], frag[:, 2],
                   s=1, alpha=0.6, label=f'Orig {i+1}')
    ax1.set_title(f'{title}\nOriginal ({len(orig_fragments)} fragments)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend(markerscale=5)

    # Reconstructed
    ax2 = fig.add_subplot(122, projection='3d')
    for i, frag in enumerate(recon_fragments):
        ax2.scatter(frag[:, 0], frag[:, 1], frag[:, 2],
                   s=1, alpha=0.6, label=f'Recon {i+1}')
    ax2.set_title(f'{title}\nReconstructed ({len(recon_fragments)} fragments)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend(markerscale=5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {filename}")


def main():
    """Run all toy scenarios."""
    print("\n" + "="*60)
    print("TOY EXAMPLES: One-directional vs Symmetric MFCD")
    print("="*60)

    output_dir = "experiments/mfcd_definition/toy_examples/"
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Scenario 1: Missing fragments
    orig1, recon1 = scenario_1_missing_fragments()
    visualize_scenario(orig1, recon1, "Scenario 1: Missing Fragments",
                      f"{output_dir}/scenario_1_missing.png")

    # Scenario 2: Extra fragments
    orig2, recon2 = scenario_2_extra_fragments()
    visualize_scenario(orig2, recon2, "Scenario 2: Extra Fragments",
                      f"{output_dir}/scenario_2_extra.png")

    # Scenario 3: Position shift
    orig3, recon3 = scenario_3_position_shift()
    visualize_scenario(orig3, recon3, "Scenario 3: Position Shift",
                      f"{output_dir}/scenario_3_shift.png")

    # Scenario 4: Perfect reconstruction
    orig4, recon4 = scenario_4_perfect_reconstruction()
    visualize_scenario(orig4, recon4, "Scenario 4: Perfect Reconstruction",
                      f"{output_dir}/scenario_4_perfect.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n✅ Symmetric MFCD advantages:")
    print("  1. Detects missing fragments (Scenario 1)")
    print("  2. Detects extra/noise fragments (Scenario 2)")
    print("  3. Captures bidirectional shift (Scenario 3)")
    print("  4. Symmetric: SymMFCD(A,B) = SymMFCD(B,A)")
    print("\n❌ One-directional MFCD limitations:")
    print("  1. Misses extra fragments (Scenario 2)")
    print("  2. Only captures one direction of error")
    print("  3. Not symmetric")
    print("\n💡 Recommendation:")
    print("  Use Symmetric MFCD for comprehensive evaluation!")
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
