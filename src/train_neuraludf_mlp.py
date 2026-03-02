"""
Train NeuralUDF MLP

This script trains a complete NeuralUDF architecture on UDF data.

Key differences from simple UDF-MLP:
- Full NeuralUDF architecture (6-8 layers, 256 hidden dim)
- Skip connections at layer 4
- Geometric initialization
- Weight normalization
- Higher multires encoding (6)

Usage:
    python src/train_neuraludf_mlp.py --npz_dir data/npz-udf --epochs 100
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ========== Position Encoding ==========

class Embedder:
    """Positional encoding using Fourier features."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    """Create position encoder."""
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# ========== NeuralUDF Network ==========

class NeuralUDFNetwork(nn.Module):
    """
    Complete NeuralUDF architecture with:
    - Skip connections
    - Geometric initialization
    - Weight normalization
    - Softplus activation
    """
    def __init__(self,
                 d_in=3,
                 d_out=1,
                 d_hidden=256,
                 n_layers=6,
                 skip_in=(4,),
                 multires=6,
                 scale=1.0,
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True,
                 udf_type='abs'):
        super(NeuralUDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        # Build layers
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            # Geometric initialization
            if geometric_init:
                if l == self.num_layers - 2:
                    # Last layer
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    # First layer with position encoding
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    # Skip connection layers
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    # Other layers
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            # Weight normalization
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.udf_type = udf_type

    def udf_out(self, x):
        """Output activation for UDF."""
        if self.udf_type == 'abs':
            return torch.abs(x)
        elif self.udf_type == 'square':
            return x ** 2
        elif self.udf_type == 'sdf':
            return x
        return x

    def forward(self, inputs):
        """Forward pass."""
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            # Skip connection
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            # Activation (except last layer)
            if l < self.num_layers - 2:
                x = self.activation(x)

        return self.udf_out(x) / self.scale

    def udf(self, x):
        """Query UDF value."""
        return self.forward(x)


# ========== Dataset ==========

class UDFDataset(Dataset):
    """Dataset for UDF training."""
    def __init__(self, points, udf_values):
        self.points = torch.from_numpy(points).float()
        self.udf_values = torch.from_numpy(udf_values).float().reshape(-1, 1)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.udf_values[idx]


# ========== Training ==========

def load_udf_data(npz_dir):
    """Load all UDF .npz files."""
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    print(f"Loading {len(npz_files)} NPZ files from {npz_dir}")

    all_pts, all_vals = [], []
    for f in tqdm(npz_files, desc="Loading data"):
        d = np.load(f)
        pts = d["poisson_grid_points"].astype(np.float32)

        # Load UDF values
        if "udf_values" in d:
            vals = d["udf_values"].astype(np.float32)
        elif "sdf_values" in d:
            # Fallback: convert SDF to UDF
            vals = np.abs(d["sdf_values"].astype(np.float32))
        else:
            raise KeyError(f"Neither 'udf_values' nor 'sdf_values' found in {f}")

        all_pts.append(pts)
        all_vals.append(vals.ravel())

    all_pts = np.concatenate(all_pts, axis=0)
    all_vals = np.concatenate(all_vals, axis=0)

    print(f"Total points: {len(all_pts):,}")
    print(f"UDF range: [{all_vals.min():.4f}, {all_vals.max():.4f}]")

    return all_pts, all_vals


def train_neuraludf(args):
    """Train NeuralUDF model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    all_pts, all_vals = load_udf_data(args.npz_dir)

    # Create dataset and dataloader
    dataset = UDFDataset(all_pts, all_vals)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Create model
    print("\n" + "="*60)
    print("NeuralUDF Architecture:")
    print("="*60)
    print(f"  Layers: {args.n_layers}")
    print(f"  Hidden dim: {args.d_hidden}")
    print(f"  Skip connections: {args.skip_in}")
    print(f"  Multires: {args.multires}")
    print(f"  Geometric init: {args.geometric_init}")
    print(f"  Weight norm: {args.weight_norm}")
    print("="*60 + "\n")

    model = NeuralUDFNetwork(
        d_in=3,
        d_out=1,
        d_hidden=args.d_hidden,
        n_layers=args.n_layers,
        skip_in=tuple(args.skip_in),
        multires=args.multires,
        scale=args.scale,
        bias=args.bias,
        geometric_init=args.geometric_init,
        weight_norm=args.weight_norm,
        udf_type=args.udf_type
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    print("\nStarting training...")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for points, udf_gt in pbar:
            points = points.to(device)
            udf_gt = udf_gt.to(device)

            # Forward
            udf_pred = model(points)

            # Loss
            loss = criterion(udf_pred, udf_gt)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"neuraludf_mlp_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "neuraludf_mlp.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NeuralUDF MLP")

    # Data
    parser.add_argument('--npz_dir', type=str, default='data/npz-udf',
                        help='Directory containing UDF .npz files')

    # Model architecture
    parser.add_argument('--d_hidden', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of layers (default: 6)')
    parser.add_argument('--skip_in', type=int, nargs='+', default=[4],
                        help='Skip connection layers (default: [4])')
    parser.add_argument('--multires', type=int, default=6,
                        help='Positional encoding frequency (default: 6)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Input scaling factor (default: 1.0)')
    parser.add_argument('--bias', type=float, default=0.5,
                        help='Output bias for geometric init (default: 0.5)')
    parser.add_argument('--geometric_init', action='store_true', default=True,
                        help='Use geometric initialization')
    parser.add_argument('--no_geometric_init', dest='geometric_init', action='store_false',
                        help='Disable geometric initialization')
    parser.add_argument('--weight_norm', action='store_true', default=True,
                        help='Use weight normalization')
    parser.add_argument('--no_weight_norm', dest='weight_norm', action='store_false',
                        help='Disable weight normalization')
    parser.add_argument('--udf_type', type=str, default='abs', choices=['abs', 'square', 'sdf'],
                        help='UDF output type (default: abs)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='Batch size (default: 8192)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs (default: 20)')

    # Output
    parser.add_argument('--output_dir', type=str, default='data/ckpts/neuraludf_mlp',
                        help='Output directory for checkpoints')

    args = parser.parse_args()

    train_neuraludf(args)


if __name__ == "__main__":
    main()
