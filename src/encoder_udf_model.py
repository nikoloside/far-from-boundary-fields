"""
UDF encoder from a trained model (NeuralUDF, MIND, or any query_func).
Same output format as encoder_ffb-df_mlp: poisson_grid_points, udf_values.
Supports: (1) load points from existing npz, (2) generate same sampling pattern.
"""
import os
import numpy as np
import torch
from tqdm import tqdm

# Default paths
WORKSPACE = "data/"
SAVE_PATH_UDF_MESH = "npz-udf/"
SAVE_PATH_NEURALUDF = "npz-udf-neuraludf/"
SAVE_PATH_MIND = "npz-udf-mind/"


def generate_same_sampling(num_points: int = 64000) -> np.ndarray:
    """Generate same near-surface + uniform sampling in [-1,1]^3."""
    import scipy.stats.qmc
    near_num = int(num_points * 0.5)
    uniform_num = num_points - near_num
    near_num *= 10

    near_surface_points = []
    for _ in range(100):
        sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
        pts = sampler.random(near_num * 2)
        pts = 2 * pts - 1
        if len(near_surface_points) + len(pts) >= near_num:
            remain = near_num - len(near_surface_points)
            near_surface_points.extend(pts[:remain])
            break
        near_surface_points.extend(pts)

    near_surface_points = np.array(near_surface_points[:near_num])
    uniform_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
    uniform_points = uniform_sampler.random(uniform_num)
    uniform_points = 2 * uniform_points - 1
    return np.vstack([near_surface_points, uniform_points])


def encode_from_query_func(
    query_func,
    points=None,
    batch_size: int = 8192,
    device: str = None,
) -> tuple:
    """
    Get UDF values at points using query_func(pt_tensor) -> udf_tensor.
    query_func: callable(torch.Tensor) -> torch.Tensor, shape (N,1)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if points is None:
        points = generate_same_sampling()
    points_t = torch.from_numpy(points.astype(np.float32)).to(device)

    udf_list = []
    for i in tqdm(range(0, len(points_t), batch_size), desc="Query UDF"):
        batch = points_t[i : i + batch_size]
        with torch.no_grad():
            udf = query_func(batch)
        if isinstance(udf, torch.Tensor):
            udf = udf.cpu().numpy().ravel()
        udf_list.append(udf)

    udf_values = np.concatenate(udf_list)
    return points, udf_values


def encode_from_npz_and_model(
    npz_path: str,
    query_func,
    batch_size: int = 8192,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Load points from FFB-DF or UDF npz, query model for values."""
    data = np.load(npz_path)
    pts_key = "poisson_grid_points" if "poisson_grid_points" in data else "points"
    points = data[pts_key]
    return encode_from_query_func(query_func, points, batch_size, device)


def encode_neuraludf(
    exp_dir: str,
    npz_path: str | None = None,
    points: np.ndarray | None = None,
    out_path: str | None = None,
) -> str:
    """
    Use NeuralUDF checkpoint to encode. exp_dir = base_exp_dir (e.g. exp/udf/dtu/case/)
    """
    import sys
    neuraludf_root = os.path.join(os.path.dirname(__file__), "..", "experiments", "udf_baseline", "NeuralUDF")
    sys.path.insert(0, neuraludf_root)
    from pyhocon import ConfigFactory
    from models.fields import UDFNetwork

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"NeuralUDF checkpoints dir not found: {ckpt_dir}")
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoint in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])

    conf_path = os.path.join(exp_dir, "..", "..", "confs", "udf_dtu_blending.conf")
    if not os.path.exists(conf_path):
        conf_path = os.path.join(neuraludf_root, "confs", "udf_dtu_blending.conf")
    conf = ConfigFactory.parse_file(conf_path)
    geo_conf = conf["model.udf_network"]

    net = UDFNetwork(
        d_in=geo_conf.get_int("d_in", 3),
        d_out=geo_conf.get_int("d_out", 257),
        d_hidden=geo_conf.get_int("d_hidden", 256),
        n_layers=geo_conf.get_int("n_layers", 8),
        skip_in=tuple(geo_conf.get_list("skip_in", [4])),
        multires=geo_conf.get_int("multires", 6),
        scale=geo_conf.get_float("scale", 1.0),
        bias=geo_conf.get_float("bias", 0.5),
    ).cuda()

    ckpt = torch.load(ckpt_path, map_location="cuda")
    net.load_state_dict(ckpt["udf_network_fine"], strict=True)
    net.eval()

    def query(x):
        return net.udf(x)

    if points is None and npz_path:
        data = np.load(npz_path)
        points = data["poisson_grid_points"]
    elif points is None:
        points = generate_same_sampling()

    pts, udf_values = encode_from_query_func(query, points)

    if out_path is None:
        out_path = os.path.join(WORKSPACE, SAVE_PATH_NEURALUDF, os.path.basename(npz_path or "pred.npz"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, poisson_grid_points=pts, udf_values=udf_values)
    return out_path


def encode_mind(
    query_func,
    npz_path: str | None = None,
    points: np.ndarray | None = None,
    out_path: str | None = None,
) -> str:
    """
    Use MIND query_func to encode. query_func(pts_tensor) -> udf_tensor.
    """
    if points is None and npz_path:
        data = np.load(npz_path)
        points = data["poisson_grid_points"]
    elif points is None:
        points = generate_same_sampling()

    pts, udf_values = encode_from_query_func(query_func, points)

    if out_path is None:
        name = os.path.basename(npz_path) if npz_path else "mind_pred.npz"
        out_path = os.path.join(WORKSPACE, SAVE_PATH_MIND, name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, poisson_grid_points=pts, udf_values=udf_values)
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "neuraludf", "mind"], default="sample")
    parser.add_argument("--npz", type=str, help="Input npz with points")
    parser.add_argument("--exp_dir", type=str, help="NeuralUDF exp dir with ckpts")
    parser.add_argument("--out", type=str, help="Output npz path")
    args = parser.parse_args()

    if args.mode == "sample":
        pts = generate_same_sampling()
        np.savez(args.out or "data/npz-udf/sample_points.npz", poisson_grid_points=pts)
        print("Saved sample points to", args.out or "data/npz-udf/sample_points.npz")
    elif args.mode == "neuraludf":
        encode_neuraludf(args.exp_dir, args.npz, out_path=args.out)
    elif args.mode == "mind":
        print("MIND mode requires query_func - use encode_mind() from Python")
