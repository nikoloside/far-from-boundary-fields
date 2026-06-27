"""
VQ-MLP training for UDF encoding comparison.
Refactored from VQ-mlp-origin-siren.py to support:
  - CSV-based collision condition input (replaces JSON)
  - 4 encoding types: ffb, udf, truncated_udf, signed_udf
  - Point-level batching: all shapes' points flattened, batch_size = num points
  - CLI-driven configuration

Architecture (identical to VQ-mlp-origin-siren.py):
  - MultiLatentEncoder: Siren(7D -> z_latent_dim) for collision conditions
  - PosEncoder: Siren(3D -> pos_encode_dim) for spatial coordinates
  - ImplicitFunction: 8-layer MLP with skip connection at layer 5
  - DeepSDFLoss: L1(clamp(pred), clamp(gt)) + latent_norm_reg

Usage:
    python src/train_vq_mlp.py --encoding_type ffb --epochs 100
    python src/train_vq_mlp.py --encoding_type udf --epochs 100
"""
import argparse
import os
import sys
import json
import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from siren_pytorch import Siren


# ========== Encoding type config ==========

ENCODING_CONFIG = {
    'ffb': {
        'npz_dir': 'npz-resample',
        'npz_dir_uniform': 'npz-resample-uniform',
        'value_key': 'sdf_values',
        'delta': None,       # no clamping — FFB range [-1, ~1] needs full range
        'signed': True,
    },
    'udf': {
        'npz_dir': 'npz-udf',
        'npz_dir_uniform': 'npz-udf-uniform',
        'value_key': 'udf_values',
        'delta': None,       # no clamping — UDF range [0, ~1] needs full range
        'signed': False,
    },
    'truncated_udf': {
        'npz_dir': 'npz-truncated-udf',
        'npz_dir_uniform': None,
        'value_key': 'udf_values',
        'delta': 0.1,        # truncated UDF is already clamped to [0, 0.1]
        'signed': False,
    },
    'signed_udf': {
        'npz_dir': 'npz-signed-udf',
        'npz_dir_uniform': None,
        'value_key': 'sdf_values',
        'delta': None,       # no clamping — raw signed distance [-~1, ~1]
        'signed': True,
    },
}


# ========== CSV Parsing ==========

def parse_csv_condition(csv_path):
    """
    Parse collision condition from CSV file.
    Row 2 (sphere1) contains: pos(3D), dir(3D), impulse(derived from dir norm).

    Returns: pos (3,), direction (3,), impulse (1,) — all float32 numpy arrays
    """
    maxImpulse = 304527.0

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Row 2 = sphere1 collision (index 1)
    row = lines[1].strip().split(';')
    # Fields 2-4 (0-indexed): position
    pos = np.array([float(row[2]), float(row[3]), float(row[4])], dtype=np.float32)
    # Fields 8-10: direction
    direction = np.array([float(row[8]), float(row[9]), float(row[10])], dtype=np.float32)
    # Impulse: norm of direction, normalized
    impulse = np.array([np.linalg.norm(direction) / maxImpulse], dtype=np.float32)

    return pos, direction, impulse


# ========== Stratified NB Sampling ==========

def stratified_nb_sample(points, values, sample_limit, signed, encoding_type='ffb'):
    """
    Stratified near-boundary (NB) sampling to ensure balanced value-range coverage.

    For FFB (signed, normalized interior [-1, 0)):
      - < -0.2:     15%  (deep interior)
      - [-0.2, 0):  35%  (near-surface inside)
      - [0, 0.2]:   35%  (near-surface outside)
      - > 0.2:      15%  (far exterior)

    For signed_udf (signed, raw interior [-0.37, 0)):
      - < -0.05:    15%  (deep interior, adaptive to shallow range)
      - [-0.05, 0): 35%  (near-surface inside)
      - [0, 0.05]:  35%  (near-surface outside)
      - > 0.05:     15%  (far exterior)

    For UDF (unsigned):
      - [0, 0.2]:   60%  (near-surface)
      - > 0.2:      40%  (far from surface)

    If a bin has fewer points than its target, extra quota is redistributed
    proportionally to the other bins.
    """
    N = len(values)
    if N <= sample_limit:
        return points, values

    if encoding_type == 'signed_udf':
        # Signed-UDF: raw SDF, shallow interior (~-0.05 to -0.37)
        # Use narrower bins around 0
        bins = [
            (values < -0.05,  0.15),   # deep interior
            ((values >= -0.05) & (values < 0), 0.35),  # near-surface inside
            ((values >= 0) & (values <= 0.05), 0.35),   # near-surface outside
            (values > 0.05,   0.15),   # far exterior
        ]
    elif signed:
        # FFB: normalized interior [-1, 0), wider bins
        bins = [
            (values < -0.2,  0.15),   # deep interior
            ((values >= -0.2) & (values < 0), 0.35),  # near-surface inside
            ((values >= 0) & (values <= 0.2), 0.35),   # near-surface outside
            (values > 0.2,   0.15),   # far exterior
        ]
    else:
        # UDF: 2 bins
        bins = [
            (values <= 0.2,  0.60),   # near-surface
            (values > 0.2,   0.40),   # far from surface
        ]

    # Compute target counts and available indices per bin
    bin_indices = []
    bin_targets = []
    for mask, ratio in bins:
        idx = np.where(mask)[0]
        bin_indices.append(idx)
        bin_targets.append(int(sample_limit * ratio))

    # Redistribute if any bin has fewer points than target
    selected = []
    deficit = 0
    surplus_bins = []
    for i, (idx, target) in enumerate(zip(bin_indices, bin_targets)):
        available = len(idx)
        if available <= target:
            # Take all points from this bin
            selected.append(idx)
            deficit += target - available
        else:
            surplus_bins.append(i)
            selected.append(None)  # placeholder

    # Distribute deficit proportionally among surplus bins
    if deficit > 0 and surplus_bins:
        surplus_total = sum(bin_targets[i] for i in surplus_bins)
        for i in surplus_bins:
            extra = int(deficit * bin_targets[i] / surplus_total) if surplus_total > 0 else 0
            bin_targets[i] += extra

    # Sample from surplus bins
    for i in surplus_bins:
        idx = bin_indices[i]
        target = min(bin_targets[i], len(idx))
        chosen = np.random.choice(idx, size=target, replace=False)
        selected[i] = chosen

    indices = np.concatenate([s for s in selected if len(s) > 0])
    # Trim to exact sample_limit if rounding caused overshoot
    if len(indices) > sample_limit:
        indices = np.random.choice(indices, size=sample_limit, replace=False)

    return points[indices], values[indices]


# ========== Dataset (point-level) ==========

class PointLevelDataset(Dataset):
    """
    Point-level dataset: all shapes' points are flattened into one big dataset.
    Each __getitem__ returns ONE point with its shape's condition metadata.

    With 5 shapes × 130,000 points = 650,000 items.
    batch_size=8 → 81,250 batches per epoch.
    """

    def __init__(self, data_dir, encoding_type, max_samples=350,
                 sample_limit=130000, noisy_dim=8, sampling='nb'):
        super().__init__()
        self.data_dir = data_dir
        self.encoding_type = encoding_type
        self.noisy_dim = noisy_dim

        cfg = ENCODING_CONFIG[encoding_type]
        if sampling == 'uniform':
            if cfg['npz_dir_uniform'] is None:
                raise ValueError(
                    f"Uniform sampling is not supported for encoding_type='{encoding_type}'")
            npz_dir_name = cfg['npz_dir_uniform']
        else:
            npz_dir_name = cfg['npz_dir']
        npz_dir = os.path.join(data_dir, npz_dir_name)
        value_key = cfg['value_key']

        # Find all NPZ files
        npz_files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        self.shape_ids = [os.path.basename(f).split('.')[0] for f in npz_files]
        self.shape_ids = self.shape_ids[:max_samples]
        num_shapes = len(self.shape_ids)

        # Pre-load all shapes' data and flatten
        all_points = []
        all_values = []
        all_shape_idx = []

        # Per-shape condition vectors (stored once, indexed per point)
        self.shape_pos = torch.zeros(num_shapes, 3)
        self.shape_dir = torch.zeros(num_shapes, 3)
        self.shape_imp = torch.zeros(num_shapes, 1)

        for si, shape_id in enumerate(self.shape_ids):
            # Load CSV condition
            csv_path = os.path.join(data_dir, 'csv', f'{shape_id}.csv')
            if os.path.exists(csv_path):
                pos, direction, impulse = parse_csv_condition(csv_path)
            else:
                pos = np.zeros(3, dtype=np.float32)
                direction = np.zeros(3, dtype=np.float32)
                impulse = np.zeros(1, dtype=np.float32)
            self.shape_pos[si] = torch.from_numpy(pos)
            self.shape_dir[si] = torch.from_numpy(direction)
            self.shape_imp[si] = torch.from_numpy(impulse)

            # Load NPZ points
            npz_path = os.path.join(npz_dir, f'{shape_id}.npz')
            data = np.load(npz_path)
            points = data['poisson_grid_points']
            values = data[value_key]

            # Subsample per shape
            use_stratified = (sampling == 'nb' and
                              encoding_type in ('ffb', 'udf', 'signed_udf'))
            if use_stratified:
                points, values = stratified_nb_sample(
                    points, values, sample_limit, signed=cfg['signed'],
                    encoding_type=encoding_type)
            elif len(values) > sample_limit:
                indices = np.random.choice(len(values), size=sample_limit, replace=False)
                points = points[indices]
                values = values[indices]

            all_points.append(points.astype(np.float32))
            all_values.append(values.astype(np.float32))
            all_shape_idx.append(np.full(len(values), si, dtype=np.int64))

        # Flatten into single arrays
        self.points = torch.from_numpy(np.concatenate(all_points))     # (N_total, 3)
        self.values = torch.from_numpy(np.concatenate(all_values))     # (N_total,)
        self.shape_idx = torch.from_numpy(np.concatenate(all_shape_idx))  # (N_total,)

        # Per-shape latent vectors (VQ codebook) — optimized jointly
        latent_sd = 0.01
        self.latent_vectors = torch.randn(num_shapes, noisy_dim) * latent_sd
        self.latent_vectors.requires_grad = True

        print(f"PointLevelDataset: {num_shapes} shapes, {len(self.points)} total points, "
              f"encoding={encoding_type}, npz_dir={npz_dir}")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        si = self.shape_idx[idx].item()
        return (self.shape_pos[si],          # (3,)
                self.shape_dir[si],          # (3,)
                self.shape_imp[si],          # (1,)
                self.points[idx],            # (3,)
                self.values[idx],            # scalar
                self.latent_vectors[si],     # (noisy_dim,)
                torch.tensor(si, dtype=torch.long))  # shape index


# ========== Network (identical to VQ-mlp-origin-siren.py) ==========

class MultiLatentEncoder(nn.Module):
    def __init__(self, z_latent_dim=120):
        super().__init__()
        self.neuron_input = Siren(dim_in=7, dim_out=z_latent_dim)

    def forward(self, pos, direct, imp):
        input_encoded = torch.concat((pos, direct, imp), -1)
        return self.neuron_input(input_encoded)


class PosEncoder(nn.Module):
    def __init__(self, pos_encode_dim=128):
        super().__init__()
        self.neuron_input = Siren(dim_in=3, dim_out=pos_encode_dim)

    def forward(self, pos):
        return self.neuron_input(pos)


class ImplicitFunction(nn.Module):
    def __init__(self, z_latent_dim=120, pos_encode_dim=128, noisy_dim=8, mlp_dim=512,
                 activation='silu'):
        super().__init__()
        self.activation = activation
        input_dim = z_latent_dim + pos_encode_dim + noisy_dim

        self.input_layer = self._block(input_dim, mlp_dim)
        self.layer2 = self._block(mlp_dim, mlp_dim)
        self.layer3 = self._block(mlp_dim, mlp_dim)
        self.layer4 = self._block(mlp_dim, mlp_dim - input_dim)
        self.layer5 = self._block(mlp_dim, mlp_dim)  # skip connection
        self.layer6 = self._block(mlp_dim, mlp_dim)
        self.layer7 = self._block(mlp_dim, mlp_dim)
        self.layer8 = nn.Linear(mlp_dim, 1)  # No activation on output layer

    def _block(self, in_dim, out_dim):
        if self.activation == 'siren':
            return Siren(dim_in=in_dim, dim_out=out_dim)
        elif self.activation == 'softplus':
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Softplus())
        else:  # silu (default)
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU())

    def forward(self, feature_z, latent_vec, coords):
        # Supports both:
        #   - Point-level: feature_z (B, z_dim), latent_vec (B, noisy_dim), coords (B, pos_dim)
        #   - Shape-level: feature_z (B, z_dim), latent_vec (B, noisy_dim), coords (B, N, pos_dim)
        if coords.dim() == 3:
            # Shape-level: broadcast feature_z and latent_vec to match num_points
            latent_vec = latent_vec.unsqueeze(1).repeat(1, coords.shape[1], 1)
            feature_vec = feature_z.unsqueeze(1).repeat(1, coords.shape[1], 1)
        else:
            # Point-level: all are (B, dim), no broadcast needed
            feature_vec = feature_z

        x = torch.cat([feature_vec, latent_vec, coords], dim=-1)
        skip_x = x

        x = self.input_layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(torch.cat([x, skip_x], dim=-1))  # skip connection
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        if coords.dim() == 3:
            return x.squeeze(-1).tanh()
        else:
            return x.squeeze(-1).tanh()


class DeepSDFLoss:
    def __init__(self, delta=1.0, sd=0.01, loss_type='l1'):
        self.delta = delta
        self.sd = sd
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type in ('l2', 'weighted_l2'):
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def __call__(self, yhat, y, latent):
        if self.delta is not None:
            yhat_clamped = torch.clamp(yhat, -self.delta, self.delta)
            y_clamped = torch.clamp(y, -self.delta, self.delta)
        else:
            yhat_clamped = yhat
            y_clamped = y

        if self.loss_type == 'l1':
            l = self.loss_fn(yhat_clamped, y_clamped)
        elif self.loss_type == 'l2':
            l = self.loss_fn(yhat_clamped, y_clamped).mean()
        elif self.loss_type == 'weighted_l2':
            mse = self.loss_fn(yhat_clamped, y_clamped)
            weights = torch.where(torch.abs(y) < 0.1, 10.0, 1.0)
            l = (mse * weights).mean()

        latent_norm = torch.pow(latent, 2).sum(dim=-1).mean() * (1 / (self.sd ** 2))
        return l + latent_norm


# ========== Training ==========

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = ENCODING_CONFIG[args.encoding_type]

    # Dataset (point-level)
    dataset = PointLevelDataset(
        data_dir=args.data_dir,
        encoding_type=args.encoding_type,
        max_samples=args.max_shapes,
        sample_limit=args.sample_limit,
        noisy_dim=args.noisy_dim,
        sampling=args.sampling,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,   # must be 0 for latent_vectors gradient flow
        drop_last=False,
    )

    # Models
    featureEncoder = MultiLatentEncoder(z_latent_dim=args.z_latent_dim).to(device)
    encoder = PosEncoder(pos_encode_dim=args.pos_encode_dim).to(device)
    decoder = ImplicitFunction(
        z_latent_dim=args.z_latent_dim,
        pos_encode_dim=args.pos_encode_dim,
        noisy_dim=args.noisy_dim,
        mlp_dim=args.mlp_dim,
        activation=args.activation,
    ).to(device)

    crit = DeepSDFLoss(delta=cfg['delta'], sd=0.01, loss_type=args.loss_type)

    # Optimizer: jointly optimize latent vectors + all network params
    optimizer = torch.optim.Adam(
        [dataset.latent_vectors] + list(decoder.parameters())
        + list(encoder.parameters()) + list(featureEncoder.parameters()),
        lr=args.lr,
    )

    num_params = sum(p.numel() for p in decoder.parameters())
    num_params += sum(p.numel() for p in encoder.parameters())
    num_params += sum(p.numel() for p in featureEncoder.parameters())
    print(f"Total parameters: {num_params:,}")
    print(f"Encoding: {args.encoding_type}, delta={cfg['delta']}, signed={cfg['signed']}")
    print(f"Batches per epoch: {len(dataloader)} (batch_size={args.batch_size})")

    # Save directory
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "networks"), exist_ok=True)

    # Resume from checkpoint if requested
    start_epoch = 1
    loss_epochs = []
    loss_values = []

    if args.resume:
        proj = args.proj_name
        resume_pt = os.path.join(args.save_path, f"{proj}-resume.pt")
        decoder_pt = os.path.join(args.save_path, f"{proj}-decoder.pt")
        if os.path.exists(decoder_pt):
            print(f"Resuming from {args.save_path}...")
            decoder = torch.load(os.path.join(args.save_path, f"{proj}-decoder.pt"),
                                 map_location=device, weights_only=False)
            encoder = torch.load(os.path.join(args.save_path, f"{proj}-encoder.pt"),
                                 map_location=device, weights_only=False)
            featureEncoder = torch.load(os.path.join(args.save_path, f"{proj}-featureEncoder.pt"),
                                        map_location=device, weights_only=False)
            codes = np.load(os.path.join(args.save_path, f"{proj}-codes.npz"))
            saved_latents = torch.from_numpy(codes[codes.files[0]])
            if saved_latents.shape == dataset.latent_vectors.shape:
                dataset.latent_vectors = torch.nn.Parameter(saved_latents)
            else:
                print(f"  Warning: latent shape mismatch ({saved_latents.shape} vs {dataset.latent_vectors.shape}), using fresh latents")

            # Rebuild optimizer with loaded params
            optimizer = torch.optim.Adam(
                [dataset.latent_vectors] + list(decoder.parameters())
                + list(encoder.parameters()) + list(featureEncoder.parameters()),
                lr=args.lr,
            )

            if os.path.exists(resume_pt):
                resume_data = torch.load(resume_pt, map_location=device, weights_only=False)
                start_epoch = resume_data.get('epoch', 0) + 1
                if 'optimizer_state_dict' in resume_data:
                    try:
                        optimizer.load_state_dict(resume_data['optimizer_state_dict'])
                        print(f"  Optimizer state restored")
                    except Exception as e:
                        print(f"  Warning: could not restore optimizer state: {e}")
            else:
                # No resume.pt, try to infer epoch from loss_history
                loss_hist_path = os.path.join(args.save_path, "loss_history.json")
                if os.path.exists(loss_hist_path):
                    with open(loss_hist_path) as f:
                        hist = json.load(f)
                    if hist.get('epochs'):
                        start_epoch = hist['epochs'][-1] + 1

            # Restore previous loss history (append new epochs after resume)
            loss_hist_path = os.path.join(args.save_path, "loss_history.json")
            if os.path.exists(loss_hist_path):
                with open(loss_hist_path) as f:
                    hist = json.load(f)
                # Keep only entries before start_epoch
                for ep, lv in zip(hist.get('epochs', []), hist.get('losses', [])):
                    if ep < start_epoch:
                        loss_epochs.append(ep)
                        loss_values.append(lv)

            print(f"  Resuming from epoch {start_epoch}")
        else:
            print(f"  No checkpoint found in {args.save_path}, starting fresh")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        decoder.train()
        encoder.train()
        featureEncoder.train()
        epoch_loss = 0.0
        num_batches = 0

        loop = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch}')
        for pos, direction, imp, coords, sdfs, latents, shape_idx in loop:
            optimizer.zero_grad()

            pos = pos.to(device)           # (B, 3)
            direction = direction.to(device)  # (B, 3)
            imp = imp.to(device)           # (B, 1)
            coords = coords.to(device)    # (B, 3)
            sdfs = sdfs.to(device)         # (B,)
            latents = latents.to(device)   # (B, noisy_dim)

            # Point-level forward: all tensors are (B, dim)
            feature_z = featureEncoder(pos, direction, imp)   # (B, z_latent_dim)
            feature_coords = encoder(coords)                   # (B, pos_encode_dim)
            predicted_sdf = decoder(feature_z, latents, feature_coords)  # (B,)

            loss = crit(predicted_sdf, sdfs, latents)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            loop.set_postfix(loss=f'{loss.item():.6f}')

        epoch_loss /= max(num_batches, 1)
        loss_epochs.append(epoch)
        loss_values.append(epoch_loss)

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{args.epochs} - avg_loss: {epoch_loss:.6f}")

        # Save periodically
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            _save_model(args, decoder, encoder, featureEncoder, dataset, epoch, optimizer)

    # Final save
    _save_model(args, decoder, encoder, featureEncoder, dataset, args.epochs, optimizer)

    # Save loss history
    loss_path = os.path.join(args.save_path, "loss_history.json")
    with open(loss_path, 'w') as f:
        json.dump({"epochs": loss_epochs, "losses": loss_values}, f)
    print(f"Loss history saved to {loss_path}")

    # Save config
    config_path = os.path.join(args.save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training complete. Models saved to {args.save_path}")


def _save_model(args, decoder, encoder, featureEncoder, dataset, epoch, optimizer=None):
    """Save in same format as VQ-mlp-origin-siren.py for backward compatibility."""
    proj = args.proj_name

    # Latest checkpoint
    codeName = os.path.join(args.save_path, f"{proj}-codes.npz")
    np.savez(codeName, dataset.latent_vectors.detach().numpy())
    torch.save(decoder, os.path.join(args.save_path, f"{proj}-decoder.pt"))
    torch.save(encoder, os.path.join(args.save_path, f"{proj}-encoder.pt"))
    torch.save(featureEncoder, os.path.join(args.save_path, f"{proj}-featureEncoder.pt"))

    # Save resume checkpoint (optimizer state + epoch)
    resume_path = os.path.join(args.save_path, f"{proj}-resume.pt")
    resume_data = {'epoch': epoch}
    if optimizer is not None:
        resume_data['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(resume_data, resume_path)

    # Epoch checkpoint
    net_dir = os.path.join(args.save_path, "networks")
    codeName = os.path.join(net_dir, f"{proj}{epoch}-codes.npz")
    np.savez(codeName, dataset.latent_vectors.detach().numpy())
    torch.save(decoder, os.path.join(net_dir, f"{proj}{epoch}-decoder.pt"))
    torch.save(encoder, os.path.join(net_dir, f"{proj}{epoch}-encoder.pt"))
    torch.save(featureEncoder, os.path.join(net_dir, f"{proj}{epoch}-featureEncoder.pt"))


def main():
    parser = argparse.ArgumentParser(description="Train VQ-MLP for UDF encoding comparison")
    parser.add_argument('--encoding_type', type=str, required=True,
                        choices=['ffb', 'udf', 'truncated_udf', 'signed_udf'],
                        help='Encoding type')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Data directory (default: data/)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Save directory (default: auto from encoding_type)')
    parser.add_argument('--proj_name', type=str, default='vqmlp',
                        help='Project name for save files')

    # Sampling, loss, activation
    parser.add_argument('--sampling', type=str, default='nb',
                        choices=['nb', 'uniform'],
                        help='Sampling strategy (default: nb)')
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'l2', 'weighted_l2'],
                        help='Loss type (default: l1)')
    parser.add_argument('--activation', type=str, default='silu',
                        choices=['silu', 'siren', 'softplus'],
                        help='Activation function for ImplicitFunction (default: silu)')

    # Architecture (matching VQ-mlp-origin-siren.py defaults)
    parser.add_argument('--z_latent_dim', type=int, default=120)
    parser.add_argument('--pos_encode_dim', type=int, default=128)
    parser.add_argument('--noisy_dim', type=int, default=8)
    parser.add_argument('--mlp_dim', type=int, default=512)

    # Training
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Points per batch (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_shapes', type=int, default=350)
    parser.add_argument('--sample_limit', type=int, default=130000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from existing checkpoint in save_path')

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join('data', 'ckpts', f'vq_mlp_{args.encoding_type}')

    train(args)


if __name__ == "__main__":
    main()
