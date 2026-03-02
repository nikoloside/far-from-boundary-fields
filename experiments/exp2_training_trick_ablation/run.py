#!/usr/bin/env python3
"""
Exp 2: Training Trick Ablation (GS vs FFB)
5 conditions x 3 representations
"""
import os
import sys
import json
from datetime import datetime

_exp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_exp_dir))
import run_experiment

log = run_experiment.log
run_cmd = run_experiment.run_cmd
ROOT = run_experiment.ROOT

EXP_ID = "exp2_training_trick_ablation"
OUT_DIR = os.path.join(ROOT, "data", "results", "training_trick_ablation")


def _ensure_data():
    import glob
    import numpy as np
    if not glob.glob(os.path.join(ROOT, "data", "npz-resample", "*.npz")):
        np.random.seed(42)
        for i in range(1, 4):
            pts = np.random.rand(500, 3).astype(np.float32) * 2 - 1
            vals = np.random.randn(500).astype(np.float32) * 0.1
            os.makedirs(os.path.join(ROOT, "data", "npz-resample"), exist_ok=True)
            np.savez(os.path.join(ROOT, "data", "npz-resample", f"{i}.npz"), poisson_grid_points=pts, sdf_values=vals)


def main():
    log(f"=== {EXP_ID} START ===")
    _ensure_data()
    os.makedirs(OUT_DIR, exist_ok=True)
    conditions = [
        ("GS-uniform-L2", "GS-SDF", "uniform", False),
        ("GS-NB-L2", "GS-SDF", "near_boundary", False),
        ("GS-NB-weighted-L2", "GS-SDF", "near_boundary", True),
        ("FFB-uniform-L2", "FFB-DF", "uniform", False),
        ("FFB-NB-L2", "FFB-DF", "near_boundary", False),
    ]
    results = []
    for name, rep, sampling, weighted in conditions:
        log(f"  Condition: {name} ({rep}, {sampling}, weighted={weighted})")
        ckpt_sub = os.path.join(OUT_DIR, "ckpt_" + name.replace("-", "_"))
        run_cmd(
            f"python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 15 --ckpt_dir {ckpt_sub}",
            EXP_ID, name
        )
        results.append({"name": name, "rep": rep})
    out_path = os.path.join(OUT_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump({"conditions": conditions, "results": results}, f, indent=2)
    log(f"Saved {out_path}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()
