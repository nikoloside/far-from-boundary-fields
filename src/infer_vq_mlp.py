"""
VQ-MLP inference and mesh extraction.
Loads trained VQ-MLP model, queries a dense grid, extracts mesh via marching cubes.

Usage:
    python src/infer_vq_mlp.py --encoding_type ffb --model_dir data/ckpts/vq_mlp_ffb \
        --shape_id 1 --output meshes/ffb_1.obj --resolution 256
"""
import argparse
import os
import numpy as np
import torch

# Import encoding config from training script
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)


def parse_csv_condition(csv_path):
    """Parse collision condition from CSV (same as train_vq_mlp.py)."""
    maxImpulse = 304527.0
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    row = lines[1].strip().split(';')
    pos = np.array([float(row[2]), float(row[3]), float(row[4])], dtype=np.float32)
    direction = np.array([float(row[8]), float(row[9]), float(row[10])], dtype=np.float32)
    impulse = np.array([np.linalg.norm(direction) / maxImpulse], dtype=np.float32)
    return pos, direction, impulse


def infer_volume(decoder, encoder, featureEncoder, feature_z, latent_vec,
                 resolution=256, device='cuda', batch_chunks=64):
    """Query the model on a dense grid and return the volume."""
    # Create grid [-1, 1]^3
    coords = np.mgrid[-1:1:(resolution * 1j), -1:1:(resolution * 1j), -1:1:(resolution * 1j)]
    coords = np.ascontiguousarray(coords.reshape(3, -1).transpose())

    # Split into chunks to avoid OOM
    chunks = np.array_split(coords, batch_chunks)
    predicted_sdfs = []

    with torch.no_grad():
        for chunk in chunks:
            torch_grid = torch.from_numpy(chunk).float().to(device).unsqueeze(0)
            feature_coords = encoder(torch_grid)
            pred = decoder(feature_z.unsqueeze(0), latent_vec.unsqueeze(0), feature_coords)
            predicted_sdfs.append(pred.cpu().numpy())

    volume = np.concatenate(predicted_sdfs, axis=1).reshape(resolution, resolution, resolution)
    return volume


def extract_mesh_mc(volume, resolution=256, isolevel=0.0, signed=True):
    """Extract mesh from volume using vedo isosurface (VTK marching cubes)."""
    import vedo as vd

    if signed:
        # For signed fields (FFB, flip-truncated): negate so inside=positive
        volume_mc = -volume
        level = isolevel
    else:
        # For unsigned fields (UDF, truncated-UDF): extract isosurface at small positive value
        volume_mc = volume
        level = 0.03 if isolevel == 0.0 else isolevel

    spacing = (2.0 / (resolution - 1),) * 3
    origin = (-1.0, -1.0, -1.0)

    try:
        vol = vd.Volume(volume_mc, spacing=spacing, origin=origin)
        mesh = vol.isosurface(value=level)
        verts = mesh.vertices
        verts = np.asarray(verts() if callable(verts) else verts)
        faces = mesh.cells
        faces = np.asarray(faces() if callable(faces) else faces)
        if len(verts) == 0:
            print("Marching cubes: empty result")
            return None, None
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return None, None

    return verts, faces


def save_ply(verts, faces, output_path):
    """Save mesh as PLY."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(output_path)
    print(f"Saved mesh: {output_path} ({len(verts)} verts, {len(faces)} faces)")


def save_voxel_gif(volume, output_path):
    """Save axial slices of a 3D volume as an animated GIF."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    frames = []
    vmin, vmax = volume.min(), volume.max()
    resolution = volume.shape[2]

    for z in range(resolution):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        im = ax.imshow(volume[:, :, z], cmap='viridis', vmin=vmin, vmax=vmax,
                        origin='lower')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'z={z}/{resolution}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=72)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0)
    print(f"Saved voxel GIF: {output_path} ({len(frames)} frames)")


def save_voxel_nii(volume, output_path):
    """Save a 3D volume as a NIfTI .nii file."""
    import nibabel as nib

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(img, output_path)
    print(f"Saved NIfTI: {output_path} (shape {volume.shape})")


def render_mesh_views(verts, faces, output_path):
    """Render mesh from 4 viewpoints (front, side, top, perspective) and save as 2x2 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Subsample faces for faster rendering if mesh is large
    faces = np.asarray(faces)
    render_faces = faces
    if len(faces) > 50000:
        indices = np.random.choice(len(faces), 50000, replace=False)
        render_faces = faces[indices]
        print(f"Subsampled mesh for rendering: {len(faces)} -> {len(render_faces)} faces")

    # 4 viewpoints: (elevation, azimuth)
    views = [
        (20, 45, 'Perspective'),
        (0, 0, 'Front'),
        (0, 90, 'Side'),
        (90, 0, 'Top'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10),
                              subplot_kw={'projection': '3d'})

    for ax, (elev, azim, title) in zip(axes.flat, views):
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=render_faces, cmap='coolwarm',
                        edgecolor='none', alpha=0.8, linewidth=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved mesh rendering: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VQ-MLP inference and mesh extraction")
    parser.add_argument('--encoding_type', type=str, required=True,
                        choices=['ffb', 'udf', 'truncated_udf', 'signed_udf'])
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory with saved model files')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Data directory for CSV files')
    parser.add_argument('--shape_id', type=str, required=True,
                        help='Shape ID to reconstruct')
    parser.add_argument('--output', type=str, required=True,
                        help='Output PLY path')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--isolevel', type=float, default=0.05)
    parser.add_argument('--proj_name', type=str, default='vqmlp')
    parser.add_argument('--batch_chunks', type=int, default=64,
                        help='Number of chunks for grid inference')
    parser.add_argument('--activation', type=str, default='silu',
                        choices=['silu', 'siren', 'softplus'],
                        help='Activation used during training (must match)')
    # Voxel visualization
    parser.add_argument('--voxel_gif', action='store_true',
                        help='Save voxel slices as animated GIF')
    parser.add_argument('--voxel_nii', action='store_true',
                        help='Save volume as NIfTI .nii file')
    parser.add_argument('--voxel_resolution', type=int, default=64,
                        help='Resolution for voxel visualization (separate from mesh resolution)')
    # Mesh rendering
    parser.add_argument('--render_mesh', action='store_true',
                        help='Save multi-angle mesh renderings as PNG')
    # Output directory
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for all output files (auto-names based on encoding_type and shape_id)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj = args.proj_name

    # Make model classes available in __main__ namespace for torch.load unpickling
    # (torch.save saves full model objects, which pickle references __main__.ClassName)
    import train_vq_mlp as _tvq
    import __main__
    for _cls_name in ['ImplicitFunction', 'PosEncoder', 'MultiLatentEncoder']:
        if not hasattr(__main__, _cls_name):
            setattr(__main__, _cls_name, getattr(_tvq, _cls_name))

    # Load models (same format as VQ-mlp-origin-siren.py)
    # weights_only=False needed because torch.save saves full model objects
    decoder = torch.load(os.path.join(args.model_dir, f"{proj}-decoder.pt"),
                         map_location=device, weights_only=False)
    encoder = torch.load(os.path.join(args.model_dir, f"{proj}-encoder.pt"),
                         map_location=device, weights_only=False)
    featureEncoder = torch.load(os.path.join(args.model_dir, f"{proj}-featureEncoder.pt"),
                                map_location=device, weights_only=False)

    # Load latent codes
    codes_data = np.load(os.path.join(args.model_dir, f"{proj}-codes.npz"))
    latent_vectors = torch.from_numpy(codes_data[codes_data.files[0]]).float().to(device)

    # Determine shape index from config
    config_path = os.path.join(args.model_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
        npz_dir = os.path.join(config.get('data_dir', 'data/'),
                               {'ffb': 'npz-resample', 'udf': 'npz-udf',
                                'truncated_udf': 'npz-truncated-udf',
                                'signed_udf': 'npz-signed-udf'}[args.encoding_type])
        shape_ids = sorted([os.path.basename(f).split('.')[0]
                           for f in os.listdir(npz_dir) if f.endswith('.npz')])
    else:
        # Fallback: assume shapes are 1-5
        shape_ids = [str(i) for i in range(1, 6)]

    if args.shape_id not in shape_ids:
        print(f"Warning: shape_id {args.shape_id} not found in {shape_ids}, using index 0")
        shape_idx = 0
    else:
        shape_idx = shape_ids.index(args.shape_id)

    latent_vec = latent_vectors[shape_idx].to(device)

    # Load collision condition from CSV
    csv_path = os.path.join(args.data_dir, 'csv', f'{args.shape_id}.csv')
    if os.path.exists(csv_path):
        pos, direction, impulse = parse_csv_condition(csv_path)
    else:
        print(f"Warning: CSV not found at {csv_path}, using zero condition")
        pos = np.zeros(3, dtype=np.float32)
        direction = np.zeros(3, dtype=np.float32)
        impulse = np.zeros(1, dtype=np.float32)

    pos_t = torch.from_numpy(pos).float().to(device)
    dir_t = torch.from_numpy(direction).float().to(device)
    imp_t = torch.from_numpy(impulse).float().to(device)

    decoder.eval()
    encoder.eval()
    featureEncoder.eval()

    with torch.no_grad():
        feature_z = featureEncoder(pos_t, dir_t, imp_t)

    # Encoding type determines sign convention
    from train_vq_mlp import ENCODING_CONFIG
    cfg = ENCODING_CONFIG[args.encoding_type]

    # Auto-generate output filenames when --output_dir is set
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = f"{args.encoding_type}_{args.shape_id}"
    else:
        base_name = None

    # --- Voxel visualization (at voxel_resolution) ---
    if args.voxel_gif or args.voxel_nii:
        voxel_res = args.voxel_resolution
        print(f"Inferring voxel volume: shape={args.shape_id}, encoding={args.encoding_type}, "
              f"resolution={voxel_res}")
        voxel_volume = infer_volume(decoder, encoder, featureEncoder, feature_z, latent_vec,
                                    resolution=voxel_res, device=device,
                                    batch_chunks=args.batch_chunks)
        print(f"Voxel volume range: [{voxel_volume.min():.4f}, {voxel_volume.max():.4f}]")

        if args.voxel_gif:
            gif_path = (os.path.join(args.output_dir, f"{base_name}_voxel_{voxel_res}.gif")
                        if args.output_dir else args.output.replace('.obj', f'_voxel_{voxel_res}.gif'))
            save_voxel_gif(voxel_volume, gif_path)

        if args.voxel_nii:
            nii_path = (os.path.join(args.output_dir, f"{base_name}_voxel_{voxel_res}.nii")
                        if args.output_dir else args.output.replace('.obj', f'_voxel_{voxel_res}.nii'))
            save_voxel_nii(voxel_volume, nii_path)

    # --- Mesh extraction (at full resolution) ---
    print(f"Inferring volume: shape={args.shape_id}, encoding={args.encoding_type}, "
          f"resolution={args.resolution}")

    volume = infer_volume(decoder, encoder, featureEncoder, feature_z, latent_vec,
                          resolution=args.resolution, device=device,
                          batch_chunks=args.batch_chunks)

    print(f"Volume range: [{volume.min():.4f}, {volume.max():.4f}]")

    # Save voxel GIF/NII at mesh resolution too if --voxel_gif and resolution differs
    if args.voxel_gif and args.resolution != args.voxel_resolution:
        gif_path_hires = (os.path.join(args.output_dir, f"{base_name}_voxel_{args.resolution}.gif")
                          if args.output_dir else args.output.replace('.obj', f'_voxel_{args.resolution}.gif'))
        save_voxel_gif(volume, gif_path_hires)

    if args.voxel_nii and args.resolution != args.voxel_resolution:
        nii_path_hires = (os.path.join(args.output_dir, f"{base_name}_voxel_{args.resolution}.nii")
                          if args.output_dir else args.output.replace('.obj', f'_voxel_{args.resolution}.nii'))
        save_voxel_nii(volume, nii_path_hires)

    verts, faces = extract_mesh_mc(volume, resolution=args.resolution,
                                   isolevel=args.isolevel, signed=cfg['signed'])

    if verts is not None and len(verts) > 0:
        # Always save PLY to --output path (used by eval phase to find meshes)
        save_ply(verts, faces, args.output)
        # Also save a copy in --output_dir if specified
        if args.output_dir:
            ply_copy = os.path.join(args.output_dir, f"{base_name}.obj")
            save_ply(verts, faces, ply_copy)

        # --- Mesh rendering ---
        if args.render_mesh:
            render_path = (os.path.join(args.output_dir, f"{base_name}_render.png")
                           if args.output_dir else args.output.replace('.obj', '_render.png'))
            render_mesh_views(verts, faces, render_path)
    else:
        print("No mesh extracted (empty volume or marching cubes failed)")


if __name__ == "__main__":
    main()
