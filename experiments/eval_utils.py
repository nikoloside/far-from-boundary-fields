"""
Shared evaluation and visualization utilities for experiments.

Functions:
- compute_symmfcd: Compare a reconstructed mesh against ground truth OBJ fragments
- plot_loss_curves: Overlay loss curves from multiple loss_history.json files
- plot_mfcd_bar_chart: Create SymMFCD comparison bar chart
- visualize_sampling: Visualize NPZ point cloud sampling density
"""

import os
import gc
import json
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_symmfcd(gt_obj_path, recon_mesh_path, num_samples=5000,
                    volume_trim=1.0):
    """
    Compute SymMFCD v2: volume-weighted, nearest-neighbor, no penalty.

    Pipeline:
      1. Load GT and Recon fragments
      2. Trim: keep top 95% cumulative volume (discard smallest fragments)
      3. Sample: volume sampling for manifold, surface sampling for non-manifold
      4. Directed CD nearest-neighbor matching (no Hungarian, no penalty)
      5. Volume-weighted average in both directions

    gt_obj_path: directory of OBJ files or single OBJ (split into components)
    recon_mesh_path: single OBJ/PLY whose connected components are fragments

    Returns dict with: symmetric_mfcd, mfcd_gt_to_recon, mfcd_recon_to_gt,
                       num_gt_fragments, num_recon_fragments
    """
    import trimesh
    from scipy.spatial import cKDTree

    # === Load GT fragments ===
    if os.path.isdir(gt_obj_path):
        gt_objs = sorted(glob.glob(os.path.join(gt_obj_path, "*.obj")))
    elif os.path.isfile(gt_obj_path):
        gt_objs = [gt_obj_path]
    else:
        raise FileNotFoundError(f"GT path not found: {gt_obj_path}")

    if not gt_objs:
        raise FileNotFoundError(f"No OBJ files found at {gt_obj_path}")

    gt_fragments = []
    for obj_path in gt_objs:
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            for geom in mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh) and len(geom.faces) >= 250:
                    gt_fragments.append(geom)
        elif isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
            parts = mesh.split(only_watertight=False)
            for part in parts:
                if isinstance(part, trimesh.Trimesh) and len(part.faces) >= 250:
                    gt_fragments.append(part)

    # === Load Recon fragments ===
    recon_mesh = trimesh.load(recon_mesh_path, force='mesh', process=False)
    if isinstance(recon_mesh, trimesh.Scene):
        recon_fragments = [g for g in recon_mesh.geometry.values()
                           if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    elif isinstance(recon_mesh, trimesh.Trimesh):
        recon_fragments = recon_mesh.split(only_watertight=False)
    else:
        recon_fragments = [recon_mesh]

    recon_fragments = [f for f in recon_fragments
                       if isinstance(f, trimesh.Trimesh) and len(f.faces) >= 250]

    if not gt_fragments or not recon_fragments:
        return {
            'symmetric_mfcd': float('nan'),
            'mfcd_gt_to_recon': float('nan'),
            'mfcd_recon_to_gt': float('nan'),
            'num_gt_fragments': len(gt_fragments),
            'num_recon_fragments': len(recon_fragments),
        }

    # === Fragment size (volume for manifold, area for non-manifold) ===
    def _frag_size(f):
        if f.is_watertight and abs(f.volume) > 1e-10:
            return abs(f.volume)
        return f.area

    # === Trim: keep top 90% cumulative volume ===
    def _trim_fragments(fragments, trim_ratio):
        sizes = np.array([_frag_size(f) for f in fragments])
        order = np.argsort(-sizes)  # descending
        cumsum = np.cumsum(sizes[order])
        total = cumsum[-1]
        cutoff = np.searchsorted(cumsum, total * trim_ratio) + 1
        keep_idx = order[:cutoff]
        return [fragments[i] for i in keep_idx]

    n_gt_raw, n_recon_raw = len(gt_fragments), len(recon_fragments)
    gt_fragments = _trim_fragments(gt_fragments, volume_trim)
    recon_fragments = _trim_fragments(recon_fragments, volume_trim)

    if not gt_fragments or not recon_fragments:
        return {
            'symmetric_mfcd': float('nan'),
            'mfcd_gt_to_recon': float('nan'),
            'mfcd_recon_to_gt': float('nan'),
            'num_gt_fragments': 0,
            'num_recon_fragments': 0,
        }

    n_gt = len(gt_fragments)
    n_recon = len(recon_fragments)

    # === Compute weights (after trimming, re-normalize) ===
    gt_sizes = np.array([_frag_size(f) for f in gt_fragments])
    recon_sizes = np.array([_frag_size(f) for f in recon_fragments])
    gt_weights = gt_sizes / gt_sizes.sum() if gt_sizes.sum() > 0 else np.ones(n_gt) / n_gt
    recon_weights = recon_sizes / recon_sizes.sum() if recon_sizes.sum() > 0 else np.ones(n_recon) / n_recon

    gt_metric = "vol" if all(f.is_watertight for f in gt_fragments) else "area"
    recon_metric = "vol" if all(f.is_watertight for f in recon_fragments) else "area"

    # === Sampling: volume for manifold, surface for non-manifold ===
    def _sample(mesh, n):
        if len(mesh.faces) == 0:
            return mesh.vertices[:n]
        if mesh.is_watertight:
            try:
                points = trimesh.sample.volume_mesh(mesh, count=n)
                if len(points) >= n // 2:
                    return points
            except Exception:
                pass
        # Fallback: surface sampling
        points, _ = trimesh.sample.sample_surface(mesh, n)
        return points

    gt_pts = [_sample(f, num_samples) for f in gt_fragments]
    recon_pts = [_sample(f, num_samples) for f in recon_fragments]

    # === Directed CD: d(A→B) = mean min dist from A points to B points ===
    def _directed_cd(pts_a, pts_b):
        tree_b = cKDTree(pts_b)
        dists, _ = tree_b.query(pts_a)
        return float(np.mean(dists))

    # === Compute directed CD matrices ===
    # cd_g2r[i,j] = directed CD from GT_i → Recon_j
    # cd_r2g[j,i] = directed CD from Recon_j → GT_i
    cd_g2r = np.zeros((n_gt, n_recon))
    cd_r2g = np.zeros((n_recon, n_gt))
    for i in range(n_gt):
        for j in range(n_recon):
            cd_g2r[i, j] = _directed_cd(gt_pts[i], recon_pts[j])
            cd_r2g[j, i] = _directed_cd(recon_pts[j], gt_pts[i])

    # === Weighted nearest-neighbor matching (no Hungarian, no penalty) ===
    # GT→Recon: each GT fragment finds its best Recon match
    gt_best_j = np.argmin(cd_g2r, axis=1)
    gt_to_recon = float(np.sum(gt_weights * cd_g2r[np.arange(n_gt), gt_best_j]))

    # Recon→GT: each Recon fragment finds its best GT match
    recon_best_i = np.argmin(cd_r2g, axis=1)
    recon_to_gt = float(np.sum(recon_weights * cd_r2g[np.arange(n_recon), recon_best_i]))

    sym_mfcd = (gt_to_recon + recon_to_gt) / 2.0

    print(f"  SymMFCD: {n_gt_raw}→{n_gt} GT({gt_metric}) × {n_recon_raw}→{n_recon} Recon({recon_metric}) [top {volume_trim*100:.0f}% vol]")
    print(f"  GT→Recon: {gt_to_recon:.6f}, Recon→GT: {recon_to_gt:.6f}, SymMFCD: {sym_mfcd:.6f}")

    result = {
        'symmetric_mfcd': sym_mfcd,
        'mfcd_gt_to_recon': gt_to_recon,
        'mfcd_recon_to_gt': recon_to_gt,
        'num_gt_fragments': n_gt,
        'num_recon_fragments': n_recon,
        'num_gt_raw': n_gt_raw,
        'num_recon_raw': n_recon_raw,
    }
    gc.collect()
    return result


def plot_loss_curves(loss_files, labels, output_path, title="Training Loss Curves",
                     log_scale=True):
    """
    Plot overlaid loss curves from multiple loss_history.json files.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(loss_files), 9)))

    for i, (fpath, label) in enumerate(zip(loss_files, labels)):
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found, skipping")
            continue
        with open(fpath, 'r') as f:
            data = json.load(f)
        ax.plot(data['epochs'], data['losses'], label=label,
                color=colors[i % len(colors)], linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    if log_scale:
        ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {output_path}")


def plot_mfcd_bar_chart(method_names, mfcd_values, output_path,
                        title="SymMFCD Comparison (lower is better)"):
    """
    Create a bar chart comparing SymMFCD across methods.
    mfcd_values can be floats or dicts with 'symmetric_mfcd' key.
    """
    values = []
    for v in mfcd_values:
        if isinstance(v, dict):
            values.append(v.get('symmetric_mfcd', float('nan')))
        else:
            values.append(v)

    fig, ax = plt.subplots(figsize=(max(8, len(method_names) * 1.5), 6))
    bars = ax.bar(range(len(method_names)), values, color='steelblue',
                  edgecolor='black', linewidth=0.5)

    for bar_obj, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2., bar_obj.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Symmetric MFCD', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SymMFCD bar chart saved to {output_path}")


def visualize_sampling(npz_path, output_path, obj_path=None, value_key=None):
    """
    Visualize NPZ point cloud sampling: 2x2 subplot with XY/XZ/YZ slices + histogram.

    Args:
        npz_path: Path to NPZ file with poisson_grid_points and sdf_values/udf_values
        output_path: Path to save PNG
        obj_path: Optional OBJ path to overlay mesh cross-section
        value_key: Key for values in NPZ ('sdf_values' or 'udf_values'). Auto-detected if None.
    """
    data = np.load(npz_path)
    pts = data['poisson_grid_points']

    if value_key is None:
        value_key = 'sdf_values' if 'sdf_values' in data else 'udf_values'
    vals = data[value_key].ravel()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Sampling: {os.path.basename(npz_path)} ({len(pts)} pts)", fontsize=14)

    # XY slice (z near 0)
    z_mask = np.abs(pts[:, 2]) < 0.05
    ax = axes[0, 0]
    sc = ax.scatter(pts[z_mask, 0], pts[z_mask, 1], c=vals[z_mask], s=1, cmap='RdBu_r',
                    vmin=-0.5, vmax=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title(f'XY slice (|z|<0.05, {z_mask.sum()} pts)')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label=value_key)

    # XZ slice (y near 0)
    y_mask = np.abs(pts[:, 1]) < 0.05
    ax = axes[0, 1]
    sc = ax.scatter(pts[y_mask, 0], pts[y_mask, 2], c=vals[y_mask], s=1, cmap='RdBu_r',
                    vmin=-0.5, vmax=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Z')
    ax.set_title(f'XZ slice (|y|<0.05, {y_mask.sum()} pts)')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label=value_key)

    # YZ slice (x near 0)
    x_mask = np.abs(pts[:, 0]) < 0.05
    ax = axes[1, 0]
    sc = ax.scatter(pts[x_mask, 1], pts[x_mask, 2], c=vals[x_mask], s=1, cmap='RdBu_r',
                    vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Y'); ax.set_ylabel('Z')
    ax.set_title(f'YZ slice (|x|<0.05, {x_mask.sum()} pts)')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label=value_key)

    # Histogram of values
    ax = axes[1, 1]
    ax.hist(vals, bins=100, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='zero level')
    ax.set_xlabel(value_key); ax.set_ylabel('Count')
    ax.set_title(f'Value distribution (mean={vals.mean():.4f}, std={vals.std():.4f})')
    near_surface = np.sum(np.abs(vals) < 0.1)
    ax.legend([f'zero level', f'|val|<0.1: {near_surface} ({100*near_surface/len(vals):.1f}%)'])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sampling visualization saved to {output_path}")


def render_mesh_comparison(mesh_paths, labels, output_path, title="Mesh Comparison"):
    """
    Render multiple meshes side by side for visual comparison.

    Args:
        mesh_paths: list of PLY/OBJ file paths
        labels: list of labels for each mesh
        output_path: path to save PNG
        title: figure title
    """
    import trimesh
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(mesh_paths)
    fig = plt.figure(figsize=(5 * n, 5))
    fig.suptitle(title, fontsize=14)

    # First pass: load meshes and determine global axis limits
    meshes = []
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for mesh_path in mesh_paths:
        if mesh_path is None or not os.path.exists(mesh_path):
            meshes.append(None)
            continue
        try:
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [g for g in mesh.geometry.values()
                     if isinstance(g, trimesh.Trimesh)]
                )
            # Subsample large meshes for faster rendering
            if len(mesh.faces) > 20000:
                indices = np.random.default_rng(42).choice(
                    len(mesh.faces), 20000, replace=False
                )
                mesh = mesh.submesh([indices], append=True)
            meshes.append(mesh)
            global_min = np.minimum(global_min, mesh.vertices.min(axis=0))
            global_max = np.maximum(global_max, mesh.vertices.max(axis=0))
        except Exception as e:
            print(f"Warning: failed to load {mesh_path}: {e}")
            meshes.append(None)

    # Compute consistent axis range
    if np.all(np.isfinite(global_min)) and np.all(np.isfinite(global_max)):
        center = (global_min + global_max) / 2.0
        half_range = (global_max - global_min).max() / 2.0 * 1.1
    else:
        center = np.zeros(3)
        half_range = 1.0

    # Second pass: render each mesh
    for i, (mesh, label) in enumerate(zip(meshes, labels)):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        ax.set_title(label, fontsize=11)

        if mesh is None:
            ax.text2D(0.5, 0.5, "N/A", transform=ax.transAxes,
                      fontsize=20, ha='center', va='center', color='gray')
        else:
            verts = mesh.vertices
            faces = mesh.faces
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], faces, verts[:, 2],
                color='steelblue', edgecolor='none', alpha=0.8, linewidth=0
            )

        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=135)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mesh comparison saved to {output_path}")


def render_mesh_multiview(mesh_path, output_path, title=None):
    """Render a single mesh from 4 viewpoints (front, right, top, perspective)."""
    import trimesh
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if not os.path.exists(mesh_path):
        print(f"Warning: mesh not found: {mesh_path}")
        return

    try:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values()
                 if isinstance(g, trimesh.Trimesh)]
            )
    except Exception as e:
        print(f"Warning: failed to load {mesh_path}: {e}")
        return

    # Subsample large meshes for faster rendering
    if len(mesh.faces) > 20000:
        indices = np.random.default_rng(42).choice(
            len(mesh.faces), 20000, replace=False
        )
        mesh = mesh.submesh([indices], append=True)

    verts = mesh.vertices
    faces = mesh.faces
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    half_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2.0 * 1.1

    viewpoints = [
        ("Front", 0, 0),
        ("Right", 0, 90),
        ("Top", 90, 0),
        ("Perspective", 25, 135),
    ]

    fig = plt.figure(figsize=(20, 5))
    if title is None:
        title = os.path.basename(mesh_path)
    fig.suptitle(title, fontsize=14)

    for i, (view_name, elev, azim) in enumerate(viewpoints):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        ax.set_title(view_name, fontsize=11)
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2],
            color='steelblue', edgecolor='none', alpha=0.8, linewidth=0
        )
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Multi-view render saved to {output_path}")
