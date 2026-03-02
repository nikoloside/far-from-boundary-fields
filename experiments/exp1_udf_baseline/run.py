#!/usr/bin/env python3
"""
Exp 1: UDF Baseline

验证：
1. FFB vs UDF vs NeuralUDF (编码+架构对比)
2. Flooding vs MIND (后处理对比)

核心方法: FFB-MLP + Flooding ⭐
"""
import os
import sys
import glob
import numpy as np

_exp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_exp_dir))
import run_experiment

log = run_experiment.log
run_cmd = run_experiment.run_cmd

EXP_ID = "exp1_udf_baseline"
MINIMAL = "--minimal" in sys.argv
QUICK = "--quick" in sys.argv


def _ensure_npz_data():
    """Create dummy npz if missing."""
    root = run_experiment.ROOT
    if not glob.glob(os.path.join(root, "data", "npz-resample", "*.npz")):
        for i in range(1, 4):
            pts = np.random.rand(1500, 3).astype(np.float32) * 2 - 1
            vals = np.random.randn(1500).astype(np.float32) * 0.1
            os.makedirs(os.path.join(root, "data", "npz-resample"), exist_ok=True)
            np.savez(os.path.join(root, "data", "npz-resample", f"{i}.npz"),
                     poisson_grid_points=pts, sdf_values=vals)
            os.makedirs(os.path.join(root, "data", "npz-udf"), exist_ok=True)
            np.savez(os.path.join(root, "data", "npz-udf", f"{i}.npz"),
                     poisson_grid_points=pts, udf_values=np.abs(vals).astype(np.float32))
        log("Created dummy npz data")


def main():
    log(f"=== {EXP_ID} START ===")
    log(f"Mode: {'QUICK' if QUICK else 'MINIMAL' if MINIMAL else 'FULL'}")

    root = run_experiment.ROOT
    exp_results = os.path.join(_exp_dir, "results")
    os.makedirs(os.path.join(exp_results, "meshes"), exist_ok=True)

    # Ensure npz data exists
    has_ffb = bool(glob.glob(os.path.join(root, "data", "npz-resample", "*.npz")))
    has_udf = bool(glob.glob(os.path.join(root, "data", "npz-udf", "*.npz")))
    if not has_ffb or not has_udf:
        _ensure_npz_data()

    # Configure based on mode
    epochs = 1 if QUICK or MINIMAL else 30
    resolution = 128 if QUICK or MINIMAL else 256
    mind_iter = 50 if QUICK or MINIMAL else 200

    steps = []

    # === PHASE 1: ENCODING (if needed) ===
    if not has_ffb:
        steps.append(("Encode FFB-DF", "python src/encoder_ffb-df_mlp.py" + (" --minimal" if MINIMAL else "")))
    if not has_udf:
        steps.append(("Encode UDF", "python src/encoder_udf_mesh.py" + (" --minimal" if MINIMAL else "")))

    # === PHASE 2: TRAINING ===
    steps.extend([
        ("Train FFB-MLP", f"python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs {epochs}"),
        ("Train UDF-MLP", f"python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs {epochs}"),
    ])

    if not MINIMAL:
        steps.append(("Train NeuralUDF", f"python src/train_neuraludf_mlp.py --npz_dir data/npz-udf --epochs {epochs}"))

    # === PHASE 3: EXTRACTION ===

    # Flooding抽取
    steps.extend([
        ("FFB+Flooding ⭐",
         f"python src/extract_mesh_flooding.py --model_type ffb_mlp "
         f"--ckpt data/ckpts/ffb_mlp/ffb_mlp.pth "
         f"--output {exp_results}/meshes/ffb_flooding.ply "
         f"--resolution {resolution} --no_imagej"),

        ("UDF+Flooding",
         f"python src/extract_mesh_flooding.py --model_type udf_mlp "
         f"--ckpt data/ckpts/udf_mlp/udf_mlp.pth "
         f"--output {exp_results}/meshes/udf_flooding.ply "
         f"--resolution {resolution} --no_imagej"),
    ])

    if not MINIMAL:
        steps.append(("NeuralUDF+Flooding",
                     f"python src/extract_mesh_flooding.py --model_type neuraludf_mlp "
                     f"--ckpt data/ckpts/neuraludf_mlp/neuraludf_mlp.pth "
                     f"--output {exp_results}/meshes/neuraludf_flooding.ply "
                     f"--resolution {resolution} --no_imagej"))

    # MIND抽取（可选，需要CUDA）
    if not QUICK and not MINIMAL:
        log("MIND extraction requires CUDA. Skipping in quick/minimal mode.")
        steps.extend([
            ("FFB+MIND",
             f"python src/extract_mesh_with_mind.py --model_type ffb_mlp "
             f"--ckpt data/ckpts/ffb_mlp/ffb_mlp.pth "
             f"--output {exp_results}/meshes/ffb_mind.ply "
             f"--resolution {resolution} --max_iter {mind_iter} || true"),
        ])

    # === PHASE 4: EVALUATION ===
    orig_dir = os.path.join(root, "data", "original_meshes")
    if os.path.exists(orig_dir):
        steps.append(("Compute SymMFCD",
                     f"python experiments/exp4_mfcd_definition/symmetric_mfcd.py "
                     f"--batch --orig-dir {orig_dir} "
                     f"--recon-dirs ffb_flood:{exp_results}/meshes udf_flood:{exp_results}/meshes "
                     f"--output-dir {exp_results}/metrics --num-samples 5000 || true"))
    else:
        log(f"⚠️  Original meshes not found at {orig_dir}, skipping evaluation")

    # Run all steps
    for i, (desc, cmd) in enumerate(steps):
        success = run_cmd(cmd, EXP_ID, f"{i+1}. {desc}")
        if not success:
            log(f"⚠️  Step failed: {desc}")

    # Summary
    log(f"\n=== {EXP_ID} SUMMARY ===")
    mesh_files = glob.glob(os.path.join(exp_results, "meshes", "*.ply"))
    log(f"Generated meshes: {len(mesh_files)}")
    for mf in mesh_files:
        log(f"  - {os.path.basename(mf)}")

    log(f"Results directory: {exp_results}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()
