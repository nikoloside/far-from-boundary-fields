"""
Encode UDF using MIND query_func (trained UDF network).
MIND expects a query_func(pts_tensor) -> udf_tensor. Provide your trained model wrapper here.
Run: python experiments/udf_baseline/encode_mind.py --npz <points.npz> --out <out.npz>
(Requires implementing load_mind_query_func for your MIND checkpoint)
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MIND", "src"))

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SAVE_DIR = "npz-udf-mind"


def load_mind_query_func(ckpt_path: str | None = None):
    """
    Return a query_func(pts) -> udf for MIND.
    Override this or pass your trained UDF network.
    Example: lambda pts: model(pts).squeeze(-1)
    """
    if ckpt_path is None:
        raise NotImplementedError(
            "Provide your MIND model loader. "
            "MIND uses query_func passed to MIND() - load your trained UDF net and wrap as query_func(pts)->udf"
        )
    # TODO: load MIND checkpoint, return callable
    # from mind import MIND
    # model = ... load from ckpt_path
    # return lambda pts: model.query_func(pts)
    raise NotImplementedError("Implement load_mind_query_func with your MIND checkpoint path")


def encode(npz_path: str, query_func, out_path: str, batch_size: int = 8192):
    data = np.load(npz_path)
    points = data["poisson_grid_points"]
    pts_t = torch.from_numpy(points.astype(np.float32)).cuda()

    udf_list = []
    for i in tqdm(range(0, len(pts_t), batch_size)):
        batch = pts_t[i : i + batch_size]
        with torch.no_grad():
            udf = query_func(batch)
            if isinstance(udf, torch.Tensor):
                udf = udf.cpu().numpy().ravel()
        udf_list.append(udf)
    udf_values = np.concatenate(udf_list)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, poisson_grid_points=points, udf_values=udf_values)
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Input npz with poisson_grid_points")
    parser.add_argument("--ckpt", help="MIND checkpoint path (implement load_mind_query_func)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch_size", type=int, default=8192)
    args = parser.parse_args()

    query_func = load_mind_query_func(args.ckpt)
    out = args.out or os.path.join(WORKSPACE, SAVE_DIR, os.path.basename(args.npz))
    encode(args.npz, query_func, out, args.batch_size)
