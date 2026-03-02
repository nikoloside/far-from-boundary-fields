"""
Encode UDF using trained NeuralUDF model.
Run from project root: python experiments/udf_baseline/encode_neuraludf.py --exp_dir <path> --npz <points.npz> --out <out.npz>
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "NeuralUDF"))
from pyhocon import ConfigFactory
from models.fields import UDFNetwork
from tqdm import tqdm

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SAVE_DIR = "npz-udf-neuraludf"


def load_neuraludf(exp_dir: str, conf_path: str | None = None):
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"No .pth in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])

    conf_path = conf_path or os.path.join(os.path.dirname(__file__), "NeuralUDF", "confs", "udf_dtu_blending.conf")
    conf = ConfigFactory.parse_file(conf_path)
    geo = conf["model.udf_network"]

    net = UDFNetwork(
        d_in=geo.get_int("d_in", 3),
        d_out=geo.get_int("d_out", 257),
        d_hidden=geo.get_int("d_hidden", 256),
        n_layers=geo.get_int("n_layers", 8),
        skip_in=tuple(geo.get_list("skip_in", [4])),
        multires=geo.get_int("multires", 6),
        scale=geo.get_float("scale", 1.0),
        bias=geo.get_float("bias", 0.5),
    ).cuda()
    ckpt = torch.load(ckpt_path, map_location="cuda")
    net.load_state_dict(ckpt["udf_network_fine"], strict=True)
    net.eval()
    return net


def encode(npz_path: str, exp_dir: str, out_path: str, batch_size: int = 8192):
    net = load_neuraludf(exp_dir)
    data = np.load(npz_path)
    points = data["poisson_grid_points"]
    pts_t = torch.from_numpy(points.astype(np.float32)).cuda()

    udf_list = []
    for i in tqdm(range(0, len(pts_t), batch_size)):
        batch = pts_t[i : i + batch_size]
        with torch.no_grad():
            udf = net.udf(batch).cpu().numpy().ravel()
        udf_list.append(udf)
    udf_values = np.concatenate(udf_list)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, poisson_grid_points=points, udf_values=udf_values)
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="NeuralUDF exp dir with checkpoints/")
    parser.add_argument("--npz", required=True, help="Input npz with poisson_grid_points")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch_size", type=int, default=8192)
    args = parser.parse_args()

    out = args.out
    if not out:
        name = os.path.basename(args.npz)
        out = os.path.join(WORKSPACE, SAVE_DIR, name)
    encode(args.npz, args.exp_dir, out, args.batch_size)
