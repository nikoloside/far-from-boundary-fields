"""
Extract mesh using Flooding (Watershed) Algorithm

This script:
1. Loads a trained model (FFB/UDF/NeuralUDF)
2. Infers volume field (256³)
3. Applies watershed flooding algorithm
4. Extracts mesh using marching cubes

Core method: FFB-MLP + Flooding (Our Method)

Usage:
    # Our method
    python src/extract_mesh_flooding.py \
        --model_type ffb_mlp \
        --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
        --output data/results/meshes/ffb_mlp_flooding.ply

    # Comparison
    python src/extract_mesh_flooding.py \
        --model_type udf_mlp \
        --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
        --output data/results/meshes/udf_mlp_flooding.ply
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import vedo as vd
import nibabel as nib
from tqdm import tqdm


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


class SimpleMLPUDF(nn.Module):
    """Simple MLP for UDF/FFB (4 layers, 128 dim)."""
    def __init__(self, d_hidden=128, n_layers=4, multires=4):
        super().__init__()
        self.multires = multires

        if multires > 0:
            self.embed_fn, d_in = get_embedder(multires, input_dims=3)
        else:
            self.embed_fn = None
            d_in = 3

        layers = []
        layers.append(nn.Linear(d_in, d_hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(d_hidden, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.embed_fn is not None:
            x = self.embed_fn(x)
        return self.net(x)


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
        model = SimpleMLPUDF(d_hidden=128, n_layers=4, multires=4)
    elif model_type == 'neuraludf_mlp':
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

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    return model


# ========== Volume Inference ==========

def infer_volume(model, resolution=256, batch_size=100000, device='cuda'):
    """
    Infer volume from model.

    Args:
        model: Trained model
        resolution: Volume resolution (default: 256)
        batch_size: Batch size for inference
        device: Device

    Returns:
        volume: (resolution, resolution, resolution) array
    """
    print(f"Inferring volume at resolution {resolution}³...")

    # Create coordinate grid
    coords = np.mgrid[-1:1:(resolution * 1j),
                      -1:1:(resolution * 1j),
                      -1:1:(resolution * 1j)]
    coords = np.ascontiguousarray(coords.reshape(3, -1).transpose())

    # Split into batches
    n_points = len(coords)
    n_batches = (n_points + batch_size - 1) // batch_size

    volume_flat = []

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Inferring"):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_points)

            batch_coords = torch.from_numpy(coords[start:end]).float().to(device)
            batch_output = model(batch_coords)

            volume_flat.append(batch_output.cpu().numpy())

    volume_flat = np.concatenate(volume_flat, axis=0).ravel()
    volume = volume_flat.reshape(resolution, resolution, resolution)

    print(f"Volume inferred: shape={volume.shape}, range=[{volume.min():.4f}, {volume.max():.4f}]")

    return volume


# ========== Flooding Algorithm ==========

def getMaskForSdf(data):
    """Get mask for SDF (inside=1, outside=0)."""
    mask = data.copy()
    mask[data >= 0] = 1
    mask[data < 0] = 0
    return mask


def flooding_with_imagej(volume, work_path, isolevel=0.03, use_imagej=True):
    """
    Apply flooding (watershed) algorithm and extract mesh.

    Args:
        volume: Input volume (SDF/UDF)
        work_path: Working directory
        isolevel: Iso-level for marching cubes
        use_imagej: Use ImageJ watershed (requires ImageJ setup)

    Returns:
        mesh: Extracted mesh
    """
    print("\n" + "="*60)
    print("Flooding Algorithm")
    print("="*60)

    os.makedirs(work_path, exist_ok=True)

    resolution = volume.shape[0]
    shift = resolution / 2
    scale = 1 / resolution * 2

    # Apply isolevel offset
    data = volume.copy()
    data = data + isolevel

    # Direct reconstruction (without watershed)
    print("Performing direct marching cubes...")
    vol_direct = vd.Volume(volume).isosurface(isolevel).smooth()
    vol_direct = vol_direct.shift(-shift, -shift, -shift).scale(scale, scale, scale)
    vol_direct = vol_direct.rotate_x(180).rotate_y(-90).rotate_z(90)
    vol_direct.write(os.path.join(work_path, "direct_marching_cubes.obj"))
    print("Direct reconstruction completed.")

    if not use_imagej:
        print("Skipping ImageJ watershed (use_imagej=False)")
        return vol_direct

    # ImageJ watershed (optional)
    try:
        import imagej

        print("Initializing ImageJ...")
        # Try to load ImageJ (requires proper setup)
        try:
            ij = imagej.init('sc.fiji:fiji', mode="headless", add_legacy=True)
            print(f"ImageJ version: {ij.getVersion()}")
        except Exception as e:
            print(f"ImageJ initialization failed: {e}")
            print("Falling back to direct marching cubes")
            return vol_direct

        # Get mask
        mask = getMaskForSdf(data)

        # Normalize for ImageJ
        discrete = data.copy()
        discrete[discrete < 0] *= -1
        discrete = 255 - (discrete + 1) / 2 * 255

        # Save intermediate
        ni_img = nib.Nifti1Image(discrete, affine=np.eye(4))
        nib.save(ni_img, os.path.join(work_path, "discrete.nii"))

        # Convert to ImageJ format
        dataset = ij.py.to_java(discrete.astype(np.float32))
        imp = ij.py.to_imageplus(dataset)

        # Watershed script
        print("Running watershed segmentation...")
        script = """
        // @ImagePlus(label="Input image") imp
        // @OUTPUT ImagePlus resultImage

        import ij.IJ;
        import ij.ImagePlus;
        import inra.ijpb.binary.BinaryImages;
        import inra.ijpb.morphology.MinimaAndMaxima3D;
        import inra.ijpb.watershed.Watershed;

        radius = 2;
        tolerance = 10;
        strConn = "6";
        dams = true;
        conn = Integer.parseInt(strConn);

        image = imp.getImageStack().duplicate();
        regionalMinima = MinimaAndMaxima3D.extendedMinima(image, tolerance, conn);
        imposedMinima = MinimaAndMaxima3D.imposeMinima(image, regionalMinima, conn);
        labeledMinima = BinaryImages.componentsLabeling(regionalMinima, conn, 32);
        resultStack = Watershed.computeWatershed(imposedMinima, labeledMinima, conn, dams);

        resultImage = new ImagePlus("watershed", resultStack);
        resultImage.setCalibration(imp.getCalibration());
        """

        args = {"imp": imp}
        result = ij.py.run_script("BeanShell", script, args)
        resultImp = result.getOutput("resultImage")

        # Convert back to numpy
        xr = ij.py.from_java(resultImp)
        xr = np.array(xr)
        xr[mask == 0] = -1

        # Set boundaries to -1
        xr[0, :, :] = -1
        xr[-1, :, :] = -1
        xr[:, 0, :] = -1
        xr[:, -1, :] = -1
        xr[:, :, 0] = -1
        xr[:, :, -1] = -1

        # Filter small regions
        filtNoisy = 50
        unique, counts = np.unique(xr, return_counts=True)
        for value in unique[counts < filtNoisy]:
            xr[xr == int(value)] = -1

        # Save segmentation result
        ni_img = nib.Nifti1Image(xr, affine=np.eye(4))
        nib.save(ni_img, os.path.join(work_path, "watershed_labels.nii"))

        # Marching cubes on watershed result
        print("Performing marching cubes on watershed result...")
        vol_watershed = vd.Volume(xr).isosurface(isolevel).smooth()
        vol_watershed = vol_watershed.shift(-shift, -shift, -shift).scale(scale, scale, scale)
        vol_watershed = vol_watershed.rotate_x(180).rotate_y(-90).rotate_z(90)
        vol_watershed.write(os.path.join(work_path, "watershed_marching_cubes.obj"))
        print("Watershed-based reconstruction completed.")

        return vol_watershed

    except ImportError:
        print("ImageJ not available (imagej package not installed)")
        print("Falling back to direct marching cubes")
        return vol_direct

    except Exception as e:
        print(f"Error during watershed: {e}")
        print("Falling back to direct marching cubes")
        return vol_direct


# ========== Main Pipeline ==========

def extract_mesh_flooding(args):
    """Extract mesh using flooding algorithm."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.model_type, args.ckpt, device)

    # Infer volume
    volume = infer_volume(model, args.resolution, args.batch_size, device)

    # Save volume (optional)
    if args.save_volume:
        volume_path = args.output.replace('.ply', '.nii').replace('.obj', '.nii')
        ni_img = nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(ni_img, volume_path)
        print(f"Volume saved to: {volume_path}")

    # Create working directory
    work_dir = os.path.join(os.path.dirname(args.output), "flooding_work")

    # Apply flooding and extract mesh
    mesh = flooding_with_imagej(
        volume,
        work_dir,
        isolevel=args.isolevel,
        use_imagej=args.use_imagej
    )

    # Save final mesh
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mesh.write(args.output)
    print(f"\nFinal mesh saved to: {args.output}")

    # Print statistics
    print(f"\nMesh statistics:")
    print(f"  Vertices: {mesh.N()}")
    print(f"  Faces: {mesh.NCells()}")
    print(f"  Bounds: {mesh.bounds()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract mesh using Flooding (Watershed) Algorithm",
        epilog="""
Examples:

  # Our method (FFB-MLP + Flooding)
  python src/extract_mesh_flooding.py \\
      --model_type ffb_mlp \\
      --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \\
      --output data/results/meshes/ffb_mlp_flooding.ply

  # Comparison (UDF-MLP + Flooding)
  python src/extract_mesh_flooding.py \\
      --model_type udf_mlp \\
      --ckpt data/ckpts/udf_mlp/udf_mlp.pth \\
      --output data/results/meshes/udf_mlp_flooding.ply
        """
    )

    # Model
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['udf_mlp', 'ffb_mlp', 'neuraludf_mlp'],
                        help='Type of model to load')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')

    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output mesh path (.ply or .obj)')

    # Inference
    parser.add_argument('--resolution', type=int, default=256,
                        help='Volume resolution (default: 256)')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Batch size for inference (default: 100000)')

    # Flooding
    parser.add_argument('--isolevel', type=float, default=0.03,
                        help='Iso-level for marching cubes (default: 0.03)')
    parser.add_argument('--use_imagej', action='store_true', default=False,
                        help='Use ImageJ watershed (requires ImageJ setup)')
    parser.add_argument('--no_imagej', dest='use_imagej', action='store_false',
                        help='Skip ImageJ watershed, use direct marching cubes only')

    # Save intermediate
    parser.add_argument('--save_volume', action='store_true',
                        help='Save inferred volume as .nii file')

    args = parser.parse_args()

    extract_mesh_flooding(args)


if __name__ == "__main__":
    main()
