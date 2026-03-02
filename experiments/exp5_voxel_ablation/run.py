#!/usr/bin/env python3
"""
Exp 5: Voxel vs Implicit
GS-SDF vs FFB-DF × {Voxel CNN, Implicit MLP}
"""
import os
import sys
import json

_exp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_exp_dir))
import run_experiment

log = run_experiment.log
run_cmd = run_experiment.run_cmd
ROOT = run_experiment.ROOT

EXP_ID = "exp5_voxel_ablation"
OUT_DIR = os.path.join(ROOT, "data", "results", "voxel_ablation")


def _ensure_data():
    import glob
    import numpy as np
    for d, k, signed in [("npz-resample", "sdf_values", True), ("npz-udf", "udf_values", False)]:
        if not glob.glob(os.path.join(ROOT, "data", d, "*.npz")):
            np.random.seed(42)
            for i in range(1, 4):
                pts = np.random.rand(500, 3).astype(np.float32) * 2 - 1
                vals = (np.random.randn(500).astype(np.float32) if signed else np.abs(np.random.randn(500).astype(np.float32))) * 0.1
                os.makedirs(os.path.join(ROOT, "data", d), exist_ok=True)
                np.savez(os.path.join(ROOT, "data", d, f"{i}.npz"), poisson_grid_points=pts, **{k: vals})


def main():
    log(f"=== {EXP_ID} START ===")
    _ensure_data()
    os.makedirs(OUT_DIR, exist_ok=True)
    # Implicit MLP (FFB-MLP, UDF-MLP) - already implemented
    run_cmd("python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 20", EXP_ID, "Implicit FFB-MLP")
    run_cmd("python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 20", EXP_ID, "Implicit UDF-MLP")
    # Voxel: placeholder (full impl would use 3D CNN on voxelized field)
    results = {
        "implicit_ffb": "data/ckpts/ffb_mlp",
        "implicit_udf": "data/ckpts/udf_mlp",
        "voxel_ffb": "TODO",
        "voxel_udf": "TODO",
    }
    out_path = os.path.join(OUT_DIR, "voxel_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved {out_path}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()
