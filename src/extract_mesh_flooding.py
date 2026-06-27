"""
Extract mesh using Flooding (Watershed) Algorithm

Pipeline:
  1. Load volume (from NII/NPZ or legacy model inference)
  2. Convert to watershed-ready volume:
     - Signed fields (FFB): +cage_offset → abs() → UDF → invert (max-vol)
       Fragment centers (far from surface) → low values (basin seeds)
       Surface → high values (ridges between fragments)
     - Unsigned fields (UDF): invert (max-vol), same logic
  3. Normalize to 0–250 integer range for ImageJ
  4. ImageJ 3D watershed flooding → label map
  5. Mask out exterior voxels using interior mask:
     - Signed fields: raw_volume < 0 (inside surface)
     - Unsigned fields: squirrel.obj contains test
  6. Per-label marching cubes on interior-only labels → fragment meshes
  7. Mesh boolean ∩ GT object → smooth outer shell from squirrel.obj

Usage:
    python src/extract_mesh_flooding.py \
        --model_type ffb_mlp \
        --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
        --gt_obj data/obj/1.obj \
        --output data/results/meshes/ffb_mlp_flooding.obj

    python src/extract_mesh_flooding.py \
        --model_type udf_mlp \
        --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
        --gt_obj data/obj/1.obj \
        --output data/results/meshes/udf_mlp_flooding.obj
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

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    return model


# ========== Volume Inference ==========

def infer_volume(model, resolution=128, batch_size=100000, device='cuda'):
    """
    Infer volume from model on a uniform grid in [-1, 1]³.

    Returns:
        volume: (resolution, resolution, resolution) array
    """
    print(f"Inferring volume at resolution {resolution}³...")

    coords = np.mgrid[-1:1:(resolution * 1j),
                      -1:1:(resolution * 1j),
                      -1:1:(resolution * 1j)]
    coords = np.ascontiguousarray(coords.reshape(3, -1).transpose())

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


# ========== Visualization ==========

def save_voxel_gif(volume, output_path, title_prefix=""):
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
        ax.set_title(f'{title_prefix}z={z}/{resolution}')
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


def save_nii(volume, output_path, name="volume"):
    """Save a 3D volume as NIfTI .nii file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    ni_img = nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4))
    nib.save(ni_img, output_path)
    print(f"Saved {name}: {output_path} (shape {volume.shape}, "
          f"range [{volume.min():.4f}, {volume.max():.4f}])")


# ========== Volume Processing ==========

def ffb_to_udf_with_cage(volume, cage_offset=0.03):
    """
    Convert FFB signed field to UDF with cover cage.

    FFB outputs signed distance (igl convention: inside=negative, outside=positive).

    Steps:
        1. Negate: inside=positive, outside=negative
        2. Add +cage_offset to negative (outside) values → shifts zero-crossing outward
        3. abs() → unified UDF with thin cover cage shell

    The cage ensures each fragment forms a closed basin for watershed.
    It is removed later by mesh boolean with the GT object.
    """
    print(f"FFB → UDF with cover cage (offset={cage_offset})")

    # Step 1: Negate (inside=positive, outside=negative)
    vol = -volume.copy()
    print(f"  After negate: range [{vol.min():.4f}, {vol.max():.4f}]")

    # Step 2: Add offset to negative (outside) values
    # This shifts the zero-crossing outward by cage_offset
    neg_mask = vol < 0
    vol[neg_mask] += cage_offset
    print(f"  After +{cage_offset} offset on negatives: range [{vol.min():.4f}, {vol.max():.4f}]")
    print(f"  Cage shell voxels (was [-{cage_offset}, 0], now [0, {cage_offset}]): "
          f"{np.sum((vol >= 0) & (vol <= cage_offset) & neg_mask)}")

    # Step 3: abs() → UDF
    vol = np.abs(vol)
    print(f"  After abs(): range [{vol.min():.4f}, {vol.max():.4f}]")

    return vol


def normalize_udf_for_imagej(udf_volume, target_max=250.0):
    """
    Normalize UDF to integer range [0, target_max] for ImageJ watershed.

    ImageJ cannot handle float decimals — minimum resolution is 1.
    Scaling so that 0.01 in original UDF ≈ 1 in normalized space.

    Args:
        udf_volume: UDF volume (0 = surface, increasing with distance)
        target_max: target maximum value (default: 250)

    Returns:
        discrete: normalized volume in [0, target_max]
        scale_factor: the scaling factor used
    """
    vmax = udf_volume.max()
    if vmax < 1e-8:
        print("WARNING: UDF volume is nearly zero everywhere!")
        return udf_volume.copy(), 1.0

    scale_factor = target_max / vmax
    discrete = (udf_volume * scale_factor).astype(np.float32)

    print(f"Normalized UDF for ImageJ: scale_factor={scale_factor:.2f}, "
          f"range [{discrete.min():.1f}, {discrete.max():.1f}]")
    print(f"  Original 0.01 → {0.01 * scale_factor:.1f} in normalized space")

    return discrete, scale_factor


# ========== Direct Marching Cubes (Intermediate Result) ==========

def direct_marching_cubes(volume, resolution, isolevel=0.03, field_type='signed'):
    """
    Direct marching cubes for intermediate visualization.
    Uses vedo Volume.isosurface() with proper origin/spacing for [-1,1]³.
    """
    print(f"\n--- Direct Marching Cubes (intermediate, isolevel={isolevel}) ---")

    if field_type == 'signed':
        mc_vol = -volume  # negate so inside > 0
    else:
        mc_vol = volume

    spacing = (2.0 / (resolution - 1),) * 3
    origin = (-1.0, -1.0, -1.0)
    vol = vd.Volume(mc_vol, spacing=spacing, origin=origin)
    mesh = vol.isosurface(value=isolevel)
    mesh.smooth(niter=50)

    npts = mesh.npoints if hasattr(mesh, 'npoints') else mesh.N()
    ncells = mesh.ncells if hasattr(mesh, 'ncells') else mesh.NCells()
    print(f"  Direct MC mesh: {npts} verts, {ncells} faces (smoothed)")

    return mesh


# ========== Flooding Algorithm ==========

def flooding_with_imagej(discrete_volume, work_path, tolerance=10, conn=6,
                         filt_noisy=50):
    """
    Apply ImageJ 3D watershed on the normalized UDF volume.

    Args:
        discrete_volume: inverted UDF normalized to 0–250 (0=fragment center/basin, 250=surface/ridge)
        work_path: working directory for intermediate files
        tolerance: extended minima tolerance (in normalized units)
        conn: connectivity (6=face-adjacent)
        filt_noisy: minimum voxel count per label

    Returns:
        label_map: 3D array of integer labels (fragment IDs), -1=background/dam
    """
    print("\n" + "=" * 60)
    print("ImageJ 3D Watershed Flooding")
    print("=" * 60)

    os.makedirs(work_path, exist_ok=True)

    # Save input volume for debugging
    save_nii(discrete_volume, os.path.join(work_path, "input_discrete.nii"),
             "input to watershed")

    try:
        import imagej

        print("Initializing ImageJ (Fiji + MorphoLibJ, headless)...")
        ij = imagej.init([
            "sc.fiji:fiji",
            "fr.inra.ijpb:MorphoLibJ_:1.6.5",
        ], mode="headless", add_legacy=True)
        print(f"ImageJ version: {ij.getVersion()}")

    except ImportError:
        print("ERROR: imagej package not installed. pip install pyimagej")
        return None
    except Exception as e:
        print(f"ERROR: ImageJ initialization failed: {e}")
        return None

    # Convert to ImageJ format
    dataset = ij.py.to_java(discrete_volume.astype(np.float32))
    imp = ij.py.to_imageplus(dataset)

    # Watershed BeanShell script (MorphoLibJ)
    print(f"Running watershed (tolerance={tolerance}, conn={conn})...")
    script = f"""
    // @ImagePlus(label="Input image") imp
    // @OUTPUT ImagePlus resultImage

    import ij.IJ;
    import ij.ImagePlus;
    import inra.ijpb.binary.BinaryImages;
    import inra.ijpb.morphology.MinimaAndMaxima3D;
    import inra.ijpb.watershed.Watershed;

    tolerance = {tolerance};
    conn = {conn};
    dams = false;

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
    label_map = np.array(ij.py.from_java(resultImp)).astype(np.int32)

    # Clear volume boundaries
    label_map[0, :, :] = -1
    label_map[-1, :, :] = -1
    label_map[:, 0, :] = -1
    label_map[:, -1, :] = -1
    label_map[:, :, 0] = -1
    label_map[:, :, -1] = -1

    # Filter small noisy regions
    unique, counts = np.unique(label_map, return_counts=True)
    for value in unique[counts < filt_noisy]:
        if value > 0:  # don't touch -1 or 0 (background/dams)
            label_map[label_map == int(value)] = -1

    # Statistics
    valid_labels = unique[unique > 0]
    valid_labels = [v for v in valid_labels
                    if np.sum(label_map == v) >= filt_noisy]
    print(f"Watershed result: {len(valid_labels)} fragments (labels: {sorted(valid_labels)[:20]}...)")

    # Save label map
    save_nii(label_map.astype(np.float32),
             os.path.join(work_path, "watershed_labels.nii"), "label map")

    # Save label map GIF
    save_voxel_gif(label_map.astype(np.float32),
                   os.path.join(work_path, "watershed_labels.gif"),
                   title_prefix="Labels ")

    return label_map


def compute_interior_mask(raw_volume, field_type, gt_obj_path=None, resolution=None,
                          cage_offset=0.03):
    """
    Compute a boolean mask of voxels that are INSIDE the object.

    For signed fields (FFB/signed_udf): (raw_volume + cage_offset) < 0 means inside.
      This matches the cage offset shift used in the watershed preprocessing.
    For unsigned fields (UDF): use gt_obj (squirrel.obj) contains test.

    Returns:
        np.ndarray of bool, same shape as raw_volume
    """
    if field_type == 'signed':
        # (raw - cage_offset) < 0 → raw < cage_offset → inside + thin outer layer
        shifted = raw_volume - cage_offset
        mask = shifted < 0
        n_inside = mask.sum()
        print(f"Interior mask (signed field): {n_inside} / {mask.size} voxels "
              f"({100*n_inside/mask.size:.1f}%) inside")
        return mask
    else:
        # UDF: no sign info, need GT mesh for inside/outside
        if gt_obj_path is None or not os.path.exists(gt_obj_path):
            print("WARNING: UDF field but no GT obj for interior mask. Using all voxels.")
            return np.ones(raw_volume.shape, dtype=bool)

        # Use vedo binarize (VTK C++) — fast voxelization of watertight mesh
        res = raw_volume.shape[0]
        spacing = (2.0 / (res - 1),) * 3
        origin = (-1.0, -1.0, -1.0)

        gt_vedo = vd.Mesh(gt_obj_path)
        print(f"Computing interior mask from {os.path.basename(gt_obj_path)} "
              f"({gt_vedo.npoints} verts) via vedo binarize...")
        binary_vol = gt_vedo.binarize(values=(1, 0), spacing=spacing,
                                       dims=(res, res, res), origin=origin)
        mask = binary_vol.tonumpy().astype(bool)
        n_inside = mask.sum()
        print(f"Interior mask (UDF + vedo binarize): {n_inside} / {mask.size} voxels "
              f"({100*n_inside/mask.size:.1f}%) inside")
        return mask


def labels_to_meshes(label_map, resolution, work_path):
    """
    Per-label isosurface: extract mesh for each watershed fragment.

    Uses vedo Volume.isosurface() (VTK marching cubes) with proper
    origin/spacing to output meshes in [-1,1]³ coordinate space.

    Returns:
        list of (label_id, vedo.Mesh) tuples
    """
    print("\n--- Per-Label Isosurface ---")

    unique_labels = sorted(set(np.unique(label_map)) - {-1, 0})
    print(f"Extracting meshes for {len(unique_labels)} labels...")

    # Spacing and origin for [-1,1]³ mapping
    spacing = (2.0 / (resolution - 1),) * 3
    origin = (-1.0, -1.0, -1.0)

    fragment_meshes = []
    for label_id in unique_labels:
        binary_vol = (label_map == label_id).astype(np.float32)
        voxel_count = int(binary_vol.sum())

        try:
            vol = vd.Volume(binary_vol, spacing=spacing, origin=origin)
            mesh = vol.isosurface(value=0.5)
            mesh.smooth(niter=50)

            npts = mesh.npoints if hasattr(mesh, 'npoints') else mesh.N()
            if npts > 0:
                fragment_meshes.append((label_id, mesh))
                print(f"  Label {label_id}: {voxel_count} voxels → {npts} verts")
            else:
                print(f"  Label {label_id}: {voxel_count} voxels → empty mesh, skipped")
        except Exception as e:
            print(f"  Label {label_id}: isosurface failed: {e}")

    print(f"Extracted {len(fragment_meshes)} fragment meshes")

    # Save individual fragments
    frag_dir = os.path.join(work_path, "fragments")
    os.makedirs(frag_dir, exist_ok=True)
    for label_id, mesh in fragment_meshes:
        mesh.write(os.path.join(frag_dir, f"fragment_{label_id}.obj"))

    return fragment_meshes


def mesh_boolean_with_gt(fragment_meshes, gt_obj_path, work_path):
    """
    Mesh boolean intersection: each fragment ∩ GT object.
    Removes the cover cage and clips fragments to the GT boundary.
    """
    print(f"\n--- Mesh Boolean ∩ {os.path.basename(gt_obj_path)} ---")

    if not os.path.exists(gt_obj_path):
        print(f"  WARNING: GT object not found: {gt_obj_path}")
        print("  Skipping mesh boolean, returning raw fragments")
        return fragment_meshes

    try:
        import trimesh
    except ImportError:
        print("  WARNING: trimesh not installed, skipping mesh boolean")
        return fragment_meshes

    gt_mesh = trimesh.load(gt_obj_path)
    print(f"  GT mesh: {len(gt_mesh.vertices)} verts, "
          f"bounds=[{gt_mesh.vertices.min(axis=0).round(3)}, {gt_mesh.vertices.max(axis=0).round(3)}]")

    result_meshes = []
    bool_dir = os.path.join(work_path, "boolean_fragments")
    os.makedirs(bool_dir, exist_ok=True)

    for label_id, vedo_mesh in fragment_meshes:
        try:
            # Convert vedo mesh to trimesh
            verts = vedo_mesh.vertices
            faces_raw = vedo_mesh.cells
            frag_tri = trimesh.Trimesh(vertices=verts, faces=faces_raw)

            # Boolean intersection
            result = frag_tri.intersection(gt_mesh)

            if hasattr(result, 'vertices') and len(result.vertices) > 0:
                # Convert back to vedo
                result_vedo = vd.Mesh([result.vertices, result.faces])
                result_meshes.append((label_id, result_vedo))
                result.export(os.path.join(bool_dir, f"fragment_{label_id}.obj"))
                print(f"  Label {label_id}: {len(frag_tri.vertices)} → "
                      f"{len(result.vertices)} verts after boolean")
            else:
                print(f"  Label {label_id}: empty after boolean, skipped")
        except Exception as e:
            print(f"  Label {label_id}: boolean failed ({e}), keeping raw")
            result_meshes.append((label_id, vedo_mesh))

    print(f"Boolean complete: {len(result_meshes)} fragments")
    return result_meshes


# ========== Main Pipeline ==========

def extract_mesh_flooding(args):
    """Full flooding pipeline.

    Accepts volume input via:
      --volume: path to .nii or .npz file containing pre-computed volume
      --ckpt + --model_type: load old-style model and infer (legacy)
    """
    # Create output directories
    out_dir = os.path.dirname(args.output) or '.'
    out_stem = os.path.splitext(os.path.basename(args.output))[0]
    work_path = os.path.join(out_dir, f"flooding_work_{out_stem}")
    os.makedirs(work_path, exist_ok=True)

    # === Step 1: Get raw volume ===
    if args.volume:
        # Load pre-computed volume from file
        print(f"Loading volume from: {args.volume}")
        if args.volume.endswith('.nii') or args.volume.endswith('.nii.gz'):
            raw_volume = nib.load(args.volume).get_fdata().astype(np.float32)
        elif args.volume.endswith('.npz'):
            data = np.load(args.volume)
            # Try common key names
            for key in ['volume', 'vol', 'data', data.files[0]]:
                if key in data:
                    raw_volume = data[key].astype(np.float32)
                    break
        else:
            raise ValueError(f"Unsupported volume format: {args.volume}")
        print(f"Volume shape: {raw_volume.shape}, range: [{raw_volume.min():.4f}, {raw_volume.max():.4f}]")
    else:
        # Legacy: load model and infer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        model = load_model(args.model_type, args.ckpt, device)
        raw_volume = infer_volume(model, args.resolution, args.batch_size, device)

    args.resolution = raw_volume.shape[0]
    print(f"Resolution: {args.resolution}³")
    print(f"Field type: {args.field_type}")

    # Save raw volume
    save_nii(raw_volume, os.path.join(work_path, "raw_volume.nii"), "raw volume")
    save_voxel_gif(raw_volume, os.path.join(work_path, "raw_volume.gif"),
                   title_prefix="Raw ")

    # === Step 2: Convert to watershed-ready volume ===
    # Watershed finds LOCAL MINIMA as basin seeds.
    # For signed fields (FFB/signed_udf):
    #   1. Add cage_offset (+0.03) → shift zero-crossing outward (cover cage)
    #   2. abs() → unified UDF (surface=0, distance grows inward & outward)
    #   3. Invert: (max - abs_vol) → fragment centers (farthest from surface) become
    #      the LOWEST values = basin seeds for watershed. Surface becomes HIGH = ridges.
    #   4. Scale to 0–250 for ImageJ
    # For unsigned fields (UDF): already UDF, just invert and scale.
    cage_offset = args.cage_offset  # default 0.03

    if args.field_type == 'signed':
        print(f"Signed field → UDF with cover cage (offset={cage_offset})")
        print(f"  Raw range: [{raw_volume.min():.4f}, {raw_volume.max():.4f}]")

        # Step a: subtract cage offset (shifts zero-crossing outward to raw=+cage_offset)
        shifted = raw_volume.copy() - cage_offset
        print(f"  After -{cage_offset}: [{shifted.min():.4f}, {shifted.max():.4f}]")

        # Step b: abs → UDF (surface=0, grows with distance)
        abs_vol = np.abs(shifted)
        print(f"  After abs(): [{abs_vol.min():.4f}, {abs_vol.max():.4f}]")

        # Step c: invert so fragment centers (high UDF) → low values (basins)
        invert_ceiling = abs_vol.max()
        ws_volume = (invert_ceiling - abs_vol).astype(np.float32)
        print(f"  After invert ({invert_ceiling:.4f} - vol): [{ws_volume.min():.4f}, {ws_volume.max():.4f}]")
        print(f"  Fragment centers (large abs) → low values ≈ {ws_volume.min():.4f} (basins)")
        print(f"  Surface (small abs) → high values ≈ {ws_volume.max():.4f} (ridges)")
    else:
        # UDF: surface=0, increasing outward. Invert so centers become basins.
        print(f"UDF volume: range [{raw_volume.min():.4f}, {raw_volume.max():.4f}]")
        invert_ceiling = raw_volume.max()
        ws_volume = (invert_ceiling - raw_volume).astype(np.float32)
        print(f"  After invert: [{ws_volume.min():.4f}, {ws_volume.max():.4f}]")

    # Save watershed-ready volume
    save_nii(ws_volume, os.path.join(work_path, "ws_volume.nii"), "watershed-ready volume")
    save_voxel_gif(ws_volume, os.path.join(work_path, "ws_volume.gif"),
                   title_prefix="WS ")

    # === Step 3: Direct marching cubes (intermediate visualization) ===
    direct_mesh = direct_marching_cubes(raw_volume, args.resolution,
                                        isolevel=cage_offset,
                                        field_type=args.field_type)
    direct_path = os.path.join(work_path, "direct_mc.obj")
    direct_mesh.write(direct_path)
    print(f"  Saved direct MC mesh: {direct_path}")

    # === Step 4: Normalize for ImageJ (scale to 0–250) ===
    # ws_volume is already inverted: high=basins, low=ridges
    # normalize_udf_for_imagej scales [0, max] → [0, 250]
    discrete, scale_factor = normalize_udf_for_imagej(ws_volume,
                                                       target_max=args.target_max)

    # Save normalized volume
    save_nii(discrete, os.path.join(work_path, "discrete_volume.nii"),
             "normalized for ImageJ")
    save_voxel_gif(discrete, os.path.join(work_path, "discrete_volume.gif"),
                   title_prefix="Discrete ")

    # === Step 5: ImageJ watershed ===
    label_map = flooding_with_imagej(discrete, work_path,
                                      tolerance=args.tolerance,
                                      conn=args.conn,
                                      filt_noisy=args.filt_noisy)

    if label_map is None:
        print("\nWatershed failed. Using direct marching cubes result.")
        direct_mesh.write(args.output)
        print(f"Saved fallback mesh: {args.output}")
        return

    # === Step 6: Mask out exterior labels ===
    interior_mask = compute_interior_mask(raw_volume, args.field_type,
                                          gt_obj_path=args.gt_obj,
                                          resolution=args.resolution,
                                          cage_offset=cage_offset)
    exterior_count = np.sum((label_map > 0) & ~interior_mask)
    total_labeled = np.sum(label_map > 0)
    label_map[~interior_mask] = 0
    print(f"Masked exterior voxels: {exterior_count} / {total_labeled} labeled voxels removed")

    # Save masked label map
    save_nii(label_map.astype(np.float32),
             os.path.join(work_path, "labels_interior.nii"), "interior-only labels")

    # === Step 7: Per-label marching cubes (interior only) ===
    fragment_meshes = labels_to_meshes(label_map, args.resolution, work_path)

    if not fragment_meshes:
        print("No fragments extracted from watershed. Using direct MC.")
        direct_mesh.write(args.output)
        print(f"Saved fallback mesh: {args.output}")
        return

    # === Step 8: Mesh boolean ∩ GT object (smooth outer shell) ===
    if args.gt_obj:
        fragment_meshes = mesh_boolean_with_gt(fragment_meshes, args.gt_obj,
                                                work_path)

    # === Step 9: Filter small fragments and merge ===
    min_faces = args.min_faces
    before_count = len(fragment_meshes)
    filtered = []
    for label_id, mesh in fragment_meshes:
        ncells = mesh.ncells if hasattr(mesh, 'ncells') else mesh.NCells()
        if ncells >= min_faces:
            filtered.append((label_id, mesh))
    fragment_meshes = filtered
    print(f"\n--- Filtered: {before_count} → {len(fragment_meshes)} fragments "
          f"(removed {before_count - len(fragment_meshes)} with faces < {min_faces}) ---")

    if not fragment_meshes:
        print("No fragments remaining after filtering. Using direct MC.")
        direct_mesh.write(args.output)
        print(f"Saved fallback mesh: {args.output}")
        return

    all_meshes = [m for _, m in fragment_meshes]
    if len(all_meshes) == 1:
        merged = all_meshes[0]
    else:
        merged = vd.merge(all_meshes)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    merged.write(args.output)
    print(f"\nFinal mesh saved: {args.output}")

    # Print statistics
    try:
        npts = merged.npoints if hasattr(merged, 'npoints') else merged.N()
        ncells = merged.ncells if hasattr(merged, 'ncells') else merged.NCells()
        print(f"  Vertices: {npts}")
        print(f"  Faces: {ncells}")
    except Exception:
        pass

    # === Summary of intermediate files ===
    print(f"\n{'=' * 60}")
    print(f"Intermediate files in: {work_path}/")
    print(f"  raw_volume.nii / .gif     — model output before processing")
    print(f"  ws_volume.nii / .gif      — watershed-ready volume")
    print(f"  discrete_volume.nii / .gif — normalized 0–{args.target_max:.0f} for ImageJ")
    print(f"  input_discrete.nii        — actual input to watershed")
    print(f"  watershed_labels.nii / .gif — label map from watershed")
    print(f"  labels_interior.nii       — label map after exterior masking")
    print(f"  direct_mc.obj             — direct MC at isolevel={args.cage_offset}")
    print(f"  fragments/                — individual fragment meshes (interior only)")
    if args.gt_obj:
        print(f"  boolean_fragments/        — fragments after ∩ GT object")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract mesh using Flooding (Watershed) Algorithm",
        epilog="""
Examples:

  # From pre-computed volume (NII from infer_vq_mlp.py)
  python src/extract_mesh_flooding.py \\
      --volume output/voxels/ffb/ffb_1_voxel_128.nii \\
      --field_type signed \\
      --gt_obj data/obj/1.obj \\
      --output output/meshes/ffb_1_flooding.obj

  # UDF volume
  python src/extract_mesh_flooding.py \\
      --volume output/voxels/udf/udf_1_voxel_128.nii \\
      --field_type unsigned \\
      --gt_obj data/obj/1.obj \\
      --output output/meshes/udf_1_flooding.obj

  # Legacy: load old model checkpoint directly
  python src/extract_mesh_flooding.py \\
      --model_type ffb_mlp \\
      --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \\
      --field_type signed \\
      --output output/meshes/ffb_flooding.obj
        """
    )

    # Volume input (preferred)
    parser.add_argument('--volume', type=str, default=None,
                        help='Path to pre-computed volume (.nii or .npz)')

    # Field type
    parser.add_argument('--field_type', type=str, required=True,
                        choices=['signed', 'unsigned'],
                        help='signed (FFB/signed_udf): inside<0, outside>0; '
                             'unsigned (UDF/truncated_udf): surface=0, all>=0')

    # Legacy model loading (only used if --volume is not set)
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['udf_mlp', 'ffb_mlp', 'neuraludf_mlp'],
                        help='(Legacy) Type of model to load')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='(Legacy) Path to model checkpoint')

    # GT object for mesh boolean
    parser.add_argument('--gt_obj', type=str, default=None,
                        help='Path to GT object (.obj) for mesh boolean intersection')

    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output mesh path (.obj or .obj)')

    # Inference
    parser.add_argument('--resolution', type=int, default=128,
                        help='Volume resolution (default: 128)')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Batch size for inference (default: 100000)')

    # FFB cover cage
    parser.add_argument('--cage_offset', type=float, default=0.03,
                        help='Cover cage offset for FFB (default: 0.03)')

    # Normalization
    parser.add_argument('--target_max', type=float, default=250.0,
                        help='Target max value for ImageJ normalization (default: 250)')

    # Watershed parameters
    parser.add_argument('--tolerance', type=int, default=25,
                        help='Watershed tolerance in normalized units (default: 10, ≈0.1 original)')
    parser.add_argument('--conn', type=int, default=6, choices=[6, 26],
                        help='Watershed connectivity: 6=face, 26=full (default: 26)')
    parser.add_argument('--filt_noisy', type=int, default=50,
                        help='Min voxels per label, smaller removed (default: 50)')
    parser.add_argument('--min_faces', type=int, default=250,
                        help='Min faces per fragment after boolean, smaller removed (default: 150)')

    args = parser.parse_args()

    if not args.volume and not args.ckpt:
        parser.error("Either --volume or --ckpt must be provided")

    extract_mesh_flooding(args)


if __name__ == "__main__":
    main()
