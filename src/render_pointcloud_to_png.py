"""
Render point cloud from NPZ to PNG for validation.
Supports sdf_values and udf_values. Uses matplotlib (reliable headless) with optional vedo 3D.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def render_npz_to_png_matplotlib(
    npz_path,
    out_path,
    value_key=None,
    near_surface_range=0.2,
    point_size=2,
    show_all=False,
):
    """Matplotlib 3D scatter - reliable for headless."""
    data = np.load(npz_path)
    pts = data["poisson_grid_points"]
    if value_key is None:
        value_key = "udf_values" if "udf_values" in data else "sdf_values"
    vals = data[value_key]
    if vals.ndim > 1:
        vals = vals.ravel()

    if show_all:
        sel_pts, sel_vals = pts, vals
    else:
        if "udf" in value_key.lower():
            mask = vals <= near_surface_range
        else:
            mask = np.abs(vals) <= near_surface_range
        sel_pts = pts[mask]
        sel_vals = vals[mask]
        if len(sel_pts) == 0:
            sel_pts, sel_vals = pts, vals
        if len(sel_pts) > 8000:
            idx = np.random.choice(len(sel_pts), 8000, replace=False)
            sel_pts, sel_vals = sel_pts[idx], sel_vals[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    vmin, vmax = float(np.percentile(sel_vals, 2)), float(np.percentile(sel_vals, 98))
    sc = ax.scatter(
        sel_pts[:, 0], sel_pts[:, 1], sel_pts[:, 2],
        c=sel_vals, cmap="coolwarm", s=point_size, vmin=vmin, vmax=vmax
    )
    plt.colorbar(sc, shrink=0.6, label=value_key)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=45)
    plt.title(os.path.basename(npz_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def render_npz_to_png(
    npz_path,
    out_path,
    value_key=None,
    near_surface_range=0.15,
    point_size=3,
    show_all=False,
):
    """Use matplotlib (always works headless)."""
    render_npz_to_png_matplotlib(
        npz_path, out_path, value_key, near_surface_range, point_size, show_all
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="Input npz file")
    parser.add_argument("-o", "--out", default=None, help="Output PNG path")
    parser.add_argument("--value_key", default=None, choices=["sdf_values", "udf_values"])
    parser.add_argument("--range", type=float, default=0.15)
    parser.add_argument("--all", action="store_true", help="Show all points")
    args = parser.parse_args()

    out = args.out
    if not out:
        base = os.path.splitext(os.path.basename(args.npz_path))[0]
        out = f"data/results/udf_baseline/qualitative/{base}_pc.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    render_npz_to_png(
        args.npz_path, out,
        value_key=args.value_key,
        near_surface_range=args.range,
        show_all=args.all,
    )


if __name__ == "__main__":
    main()
