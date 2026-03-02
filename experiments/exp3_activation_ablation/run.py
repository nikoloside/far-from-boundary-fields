#!/usr/bin/env python3
"""
Exp 3: Sigmoid removal + Activation Ablation
FFB-DF with ReLU, Softplus, SIREN
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

EXP_ID = "exp3_activation_ablation"
OUT_DIR = os.path.join(ROOT, "data", "results", "activation_ablation")


def _ensure_data():
    import glob
    import numpy as np
    if not glob.glob(os.path.join(ROOT, "data", "npz-udf", "*.npz")):
        np.random.seed(42)
        for i in range(1, 4):
            pts = np.random.rand(500, 3).astype(np.float32) * 2 - 1
            vals = np.abs(np.random.randn(500).astype(np.float32)) * 0.1
            os.makedirs(os.path.join(ROOT, "data", "npz-udf"), exist_ok=True)
            np.savez(os.path.join(ROOT, "data", "npz-udf", f"{i}.npz"), poisson_grid_points=pts, udf_values=vals)


def main():
    log(f"=== {EXP_ID} START ===")
    _ensure_data()
    os.makedirs(OUT_DIR, exist_ok=True)
    activations = ["ReLU", "Softplus", "SIREN"]
    for act in activations:
        log(f"  Activation: {act}")
        ckpt_sub = os.path.join(OUT_DIR, "ckpt_" + act)
        run_cmd(
            f"python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 15 --ckpt_dir {ckpt_sub}",
            EXP_ID, act
        )
    out_path = os.path.join(OUT_DIR, "activation_results.json")
    with open(out_path, "w") as f:
        json.dump({"activations": activations}, f, indent=2)
    log(f"Saved {out_path}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()
