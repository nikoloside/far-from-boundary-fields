#!/usr/bin/env python3
"""
Generate npz from trained FFB-MLP and UDF-MLP, then render all to PNG.
"""
import os
import sys
import glob
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Add NeuralUDF
sys.path.insert(0, os.path.join(ROOT, "experiments", "udf_baseline", "NeuralUDF"))

from src.train_ffb_mlp import SimpleSDFMLP
from src.train_udf_mlp import SimpleUDFMLP
from src.encoder_udf_model import encode_from_query_func, generate_same_sampling


def load_ffb_mlp(ckpt_path="data/ckpts/ffb_mlp/ffb_mlp.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSDFMLP(d_hidden=128, n_layers=4, multires=4).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return lambda x: model.sdf(x)


def load_udf_mlp(ckpt_path="data/ckpts/udf_mlp/udf_mlp.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUDFMLP(d_hidden=128, n_layers=4, multires=4).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return lambda x: model.udf(x)


def main():
    out_dir = "data/results/udf_baseline/qualitative"
    os.makedirs(out_dir, exist_ok=True)

    # Get sample points from first npz
    npz_udf = sorted(glob.glob("data/npz-udf/*.npz"))
    npz_ffb = sorted(glob.glob("data/npz-resample/*.npz"))
    if not npz_udf:
        print("No npz-udf; run encoder_udf_mesh.py first")
        return
    pts = np.load(npz_udf[0])["poisson_grid_points"]
    # Subsample for faster render
    if len(pts) > 10000:
        idx = np.random.choice(len(pts), 10000, replace=False)
        pts = pts[idx]

    ffb_ckpt = "data/ckpts/ffb_mlp/ffb_mlp.pth"
    udf_ckpt = "data/ckpts/udf_mlp/udf_mlp.pth"

    if os.path.exists(ffb_ckpt):
        print("Generating FFB-MLP encoding...")
        pts_f, sdf_f = encode_from_query_func(load_ffb_mlp(ffb_ckpt), pts.copy())
        np.savez(os.path.join(out_dir, "ffb_mlp_pred.npz"), poisson_grid_points=pts_f, sdf_values=sdf_f)

    if os.path.exists(udf_ckpt):
        print("Generating UDF-MLP (MIND/NeuralUDF style) encoding...")
        pts_u, udf_u = encode_from_query_func(load_udf_mlp(udf_ckpt), pts.copy())
        np.savez(os.path.join(out_dir, "udf_mlp_pred.npz"), poisson_grid_points=pts_u, udf_values=udf_u)

    # Render all to PNG
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from render_pointcloud_to_png import render_npz_to_png

    to_render = []
    for f in npz_ffb:
        to_render.append((f, f"ffb_{os.path.basename(f)}", "sdf_values"))
    for f in npz_udf:
        to_render.append((f, f"udf_mesh_{os.path.basename(f)}", "udf_values"))
    if os.path.exists(os.path.join(out_dir, "ffb_mlp_pred.npz")):
        to_render.append((os.path.join(out_dir, "ffb_mlp_pred.npz"), "ffb_mlp_pred", "sdf_values"))
    if os.path.exists(os.path.join(out_dir, "udf_mlp_pred.npz")):
        to_render.append((os.path.join(out_dir, "udf_mlp_pred.npz"), "udf_mlp_pred", "udf_values"))

    for npz_path, name, vk in to_render:
        if not os.path.exists(npz_path):
            continue
        png_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}.png")
        render_npz_to_png(npz_path, png_path, value_key=vk)
        print(f"Rendered {png_path}")


if __name__ == "__main__":
    main()
