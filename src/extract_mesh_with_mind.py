"""
Extract mesh using MIND optimization

This script:
1. Loads a trained UDF/FFB model
2. Creates a UDF query function
3. Uses MIND to extract and optimize mesh
4. Saves the result

Usage:
    python src/extract_mesh_with_mind.py \
        --model_type udf_mlp \
        --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
        --output data/results/meshes/udf_mlp_mind.ply \
        --resolution 256
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import trimesh

# Add MIND to path (try both old and new experiment directory names)
_mind_paths = [
    os.path.join(os.path.dirname(__file__), '../experiments/exp1_udf_baseline/MIND/src'),
    os.path.join(os.path.dirname(__file__), '../experiments/udf_baseline/MIND/src'),
]
for _p in _mind_paths:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
try:
    from mind import MIND
except ImportError:
    MIND = None
    print("WARNING: MIND module not available. MIND extraction will be skipped.")


# ========== Model Definitions ==========

class Embedder:
    """Positional encoding."""
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


class SimpleMLPSDF(nn.Module):
    """SDF MLP for FFB-DF (matches SimpleSDFMLP in train_ffb_mlp.py)."""
    def __init__(self, d_in=3, d_hidden=128, n_layers=4, multires=4):
        super().__init__()
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
        return self.layers[-1](x) / self.scale


class SimpleMLPUDF(nn.Module):
    """UDF MLP (matches SimpleUDFMLP in train_udf_mlp.py)."""
    def __init__(self, d_in=3, d_hidden=128, n_layers=4, multires=4):
        super().__init__()
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


class NeuralUDFNetwork(nn.Module):
    """Complete NeuralUDF architecture."""
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

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.udf_type = udf_type

    def udf_out(self, x):
        if self.udf_type == 'abs':
            return torch.abs(x)
        elif self.udf_type == 'square':
            return x ** 2
        elif self.udf_type == 'sdf':
            return x
        return x

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return self.udf_out(x) / self.scale


# ========== Model Loading ==========

def load_model(model_type, ckpt_path, device):
    """Load trained model."""
    print(f"Loading {model_type} from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if model_type == 'udf_mlp':
        model = SimpleMLPUDF(d_hidden=128, n_layers=4, multires=4)
    elif model_type == 'ffb_mlp':
        model = SimpleMLPSDF(d_hidden=128, n_layers=4, multires=4)
    elif model_type == 'neuraludf_mlp':
        # NeuralUDF (6 layers, 256 dim, skip connections)
        if 'args' in ckpt:
            args = ckpt['args']
            model = NeuralUDFNetwork(
                d_hidden=args.get('d_hidden', 256),
                n_layers=args.get('n_layers', 6),
                skip_in=tuple(args.get('skip_in', [4])),
                multires=args.get('multires', 6),
                scale=args.get('scale', 1.0),
                bias=args.get('bias', 0.5),
                geometric_init=args.get('geometric_init', True),
                weight_norm=args.get('weight_norm', True),
                udf_type=args.get('udf_type', 'abs')
            )
        else:
            # Default NeuralUDF config
            model = NeuralUDFNetwork(
                d_hidden=256,
                n_layers=6,
                skip_in=(4,),
                multires=6,
                scale=1.0,
                bias=0.5,
                geometric_init=True,
                weight_norm=True,
                udf_type='abs'
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    return model


def create_query_function(model, device, use_abs=True):
    """Create UDF query function for MIND."""
    def query_func(xyz):
        """
        Query UDF values at given points.

        Args:
            xyz: (N, 3) tensor of query points

        Returns:
            (N, 1) tensor of UDF values
        """
        with torch.no_grad():
            udf = model(xyz)

            # Ensure non-negative (UDF)
            if use_abs:
                udf = torch.abs(udf)

            return udf

    return query_func


# ========== Mesh Extraction ==========

def extract_mesh_with_mind(args):
    """Extract mesh using MIND optimization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.model_type, args.ckpt, device)

    # Create query function
    use_abs = args.model_type != 'ffb_mlp'  # FFB already has sign info
    query_func = create_query_function(model, device, use_abs=use_abs)

    # Setup MIND
    print("\n" + "="*60)
    print("MIND Configuration:")
    print("="*60)
    print(f"  Resolution: {args.resolution}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Laplacian weight: {args.laplacian_weight}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Bounds: [{args.bound_min}, {args.bound_max}]")
    print("="*60 + "\n")

    if MIND is None:
        print("ERROR: MIND module not available. Cannot proceed.")
        print("Make sure the NeuralUDF submodule is properly initialized:")
        print("  git submodule update --init --recursive")
        sys.exit(1)

    mind = MIND(
        query_func=query_func,
        resolution=args.resolution,
        r1=args.r1,
        r2=args.r2,
        max_iter=args.max_iter,
        sample_pc_iter=args.sample_pc_iter,
        laplacian_weight=args.laplacian_weight,
        bound_min=args.bound_min,
        bound_max=args.bound_max,
        max_batch=args.max_batch,
        learning_rate=args.learning_rate,
        warm_up_end=args.warm_up_end,
        report_freq=args.report_freq
    )

    # Run MIND
    print("Running MIND mesh extraction and optimization...")
    final_mesh = mind.run()

    # Save result
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_mesh.export(args.output)
    print(f"\nMesh saved to: {args.output}")

    # Print statistics
    print(f"\nMesh statistics:")
    print(f"  Vertices: {len(final_mesh.vertices):,}")
    print(f"  Faces: {len(final_mesh.faces):,}")
    print(f"  Bounds: {final_mesh.bounds}")

    # Check if mesh is watertight
    if hasattr(final_mesh, 'is_watertight'):
        print(f"  Watertight: {final_mesh.is_watertight}")

    return final_mesh


def main():
    parser = argparse.ArgumentParser(description="Extract mesh using MIND optimization")

    # Model
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['udf_mlp', 'ffb_mlp', 'neuraludf_mlp'],
                        help='Type of model to load')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')

    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output mesh path (.ply)')

    # MIND parameters
    parser.add_argument('--resolution', type=int, default=256,
                        help='Grid resolution for MIND (default: 256)')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='Maximum optimization iterations (default: 200)')
    parser.add_argument('--laplacian_weight', type=float, default=1000.0,
                        help='Laplacian regularization weight (default: 1000.0)')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate for optimization (default: 0.0005)')
    parser.add_argument('--r1', type=float, default=0.04,
                        help='MIND parameter r1 (default: 0.04)')
    parser.add_argument('--r2', type=float, default=0.01,
                        help='MIND parameter r2 (default: 0.01)')
    parser.add_argument('--sample_pc_iter', type=int, default=100,
                        help='Point cloud sampling iterations (default: 100)')
    parser.add_argument('--max_batch', type=int, default=100000,
                        help='Maximum batch size for UDF queries (default: 100000)')
    parser.add_argument('--warm_up_end', type=int, default=25,
                        help='Warm-up ending iteration (default: 25)')
    parser.add_argument('--report_freq', type=int, default=1,
                        help='Reporting frequency (default: 1)')

    # Bounds
    parser.add_argument('--bound_min', type=float, nargs=3, default=[-1.0, -1.0, -1.0],
                        help='Minimum bounds (default: -1 -1 -1)')
    parser.add_argument('--bound_max', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Maximum bounds (default: 1 1 1)')

    args = parser.parse_args()

    extract_mesh_with_mind(args)


if __name__ == "__main__":
    main()
