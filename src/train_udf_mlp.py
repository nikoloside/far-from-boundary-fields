"""
Train a simple UDF MLP on npz-udf encodings.
Architecture compatible with NeuralUDF / MIND query_func.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

NEURALUDF_ROOT = os.path.join(os.path.dirname(__file__), "..", "experiments", "udf_baseline", "NeuralUDF")
sys.path.insert(0, NEURALUDF_ROOT)


class NpzDataset(Dataset):
    def __init__(self, npz_paths, value_key="udf_values"):
        self.data = []
        for p in npz_paths:
            d = np.load(p)
            pts = d["poisson_grid_points"].astype(np.float32)
            vals = d[value_key].astype(np.float32)
            if vals.ndim > 1:
                vals = vals.squeeze()
            self.data.append((pts, vals))
        self.lengths = [len(d[0]) for d in self.data]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, (pts, vals) in enumerate(self.data):
            if idx < self.lengths[i]:
                return pts[idx], vals[idx]
            idx -= self.lengths[i]
        return self.data[-1][0][-1], self.data[-1][1][-1]


class SimpleUDFMLP(nn.Module):
    """Lightweight UDF MLP, compatible with MIND/NeuralUDF query_func."""
    def __init__(self, d_in=3, d_hidden=256, n_layers=6, multires=4):
        super().__init__()
        from models.embedder import get_embedder
        self.embed_fn, embed_dim = get_embedder(multires, input_dims=d_in)
        dims = [embed_dim] + [d_hidden] * (n_layers - 1) + [1]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.activation = nn.Softplus(beta=100)
        self.scale = 1.0

    def forward(self, x):
        x = x * self.scale
        x = self.embed_fn(x)
        for i, lin in enumerate(self.layers[:-1]):
            x = self.activation(lin(x))
        x = self.layers[-1](x)
        return torch.abs(x) / self.scale

    def udf(self, x):
        return self.forward(x)


def train(npz_dir, ckpt_dir, epochs=100, batch_size=4096, lr=1e-4):
    from glob import glob
    npz_files = sorted(glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No npz in {npz_dir}")

    all_pts, all_vals = [], []
    for f in npz_files:
        d = np.load(f)
        all_pts.append(d["poisson_grid_points"].astype(np.float32))
        v = d["udf_values"] if "udf_values" in d else d["sdf_values"]
        all_vals.append(np.abs(v).astype(np.float32).ravel())
    pts = np.vstack(all_pts)
    vals = np.concatenate(all_vals)
    print(f"Training on {len(pts)} samples from {len(npz_files)} files")

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(pts),
        torch.from_numpy(vals).unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUDFMLP(d_hidden=128, n_layers=4, multires=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(ckpt_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        for pts_b, vals_b in loader:
            pts_b = pts_b.to(device)
            vals_b = vals_b.to(device)
            pred = model.udf(pts_b)
            loss = nn.functional.mse_loss(pred, vals_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={loss_sum/len(loader):.6f}")
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "udf_mlp.pth"))
    print(f"Saved to {ckpt_dir}/udf_mlp.pth")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", default="data/npz-udf")
    parser.add_argument("--ckpt_dir", default="data/ckpts/udf_mlp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    train(args.npz_dir, args.ckpt_dir, args.epochs, args.batch_size)
