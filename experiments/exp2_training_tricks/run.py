#!/usr/bin/env python3
"""
Exp2: Training Technique Ablation — Sampling + Loss on FFB and UDF

Tests NB vs Uniform sampling × L1 vs L2 vs Weighted-L2 loss on two encodings.
2 encodings × 4 techniques = 8 conditions.

Usage:
    python experiments/exp2_training_tricks/run.py
    python experiments/exp2_training_tricks/run.py --minimal
    python experiments/exp2_training_tricks/run.py --quick
    python experiments/exp2_training_tricks/run.py --eval-only
"""
import os
import sys
import json
import glob
import numpy as np

_exp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_exp_dir))
import run_experiment
from eval_utils import (compute_symmfcd, plot_loss_curves, plot_mfcd_bar_chart,
                        render_mesh_comparison)

log = run_experiment.log
run_cmd = run_experiment.run_cmd
ROOT = run_experiment.ROOT

EXP_ID = "exp2_training_tricks"
OUT_DIR = os.path.join(_exp_dir, "output")

MINIMAL = "--minimal" in sys.argv
QUICK = "--quick" in sys.argv
EVAL_ONLY = "--eval-only" in sys.argv

ALL_SHAPE_IDS = ['1', '2', '3', '4', '5']
GT_OBJ_DIR = os.path.join(ROOT, "data", "obj")
SQUIRREL_OBJ = os.path.join(ROOT, "data", "squirrel.obj")

# 2 encodings × 4 technique combos = 8 conditions
ENCODINGS = ['ffb', 'udf']
CONDITIONS = [
    {'sampling': 'nb',      'loss_type': 'l1',          'label': 'NB_L1'},
    {'sampling': 'nb',      'loss_type': 'l2',          'label': 'NB_L2'},
    {'sampling': 'nb',      'loss_type': 'weighted_l2', 'label': 'NB_WL2'},
    {'sampling': 'uniform', 'loss_type': 'l1',          'label': 'Uniform_L1'},
]


def _get_available_shapes():
    return [sid for sid in ALL_SHAPE_IDS
            if os.path.exists(os.path.join(GT_OBJ_DIR, f"{sid}.obj"))]


def main():
    log(f"=== {EXP_ID} START ===")
    log(f"Mode: {'MINIMAL' if MINIMAL else 'QUICK' if QUICK else 'FULL'}")
    run_experiment.check_dependencies()

    shape_ids = _get_available_shapes()
    if MINIMAL:
        shape_ids = shape_ids[:1]
    log(f"Shapes: {shape_ids}")

    os.makedirs(os.path.join(OUT_DIR, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "voxels"), exist_ok=True)

    epochs = 2 if MINIMAL else (15 if QUICK else 1500)
    resolution = 64 if MINIMAL else (128 if QUICK else 128)
    batch_size = 2500 if MINIMAL else (128 if QUICK else 128)
    save_interval = 1 if MINIMAL else (5 if QUICK else 100)
    sample_limit = 5000 if MINIMAL else (30000 if QUICK else 130000)

    encodings = ['ffb'] if MINIMAL else ENCODINGS
    conditions = CONDITIONS[:2] if MINIMAL else CONDITIONS

    if not EVAL_ONLY:
        # === PHASE 1: TRAINING ===
        log("\n--- PHASE 1: Training ---")
        for enc_type in encodings:
            for cond in conditions:
                tag = f"{enc_type}_{cond['label']}"
                ckpt_dir = os.path.join(OUT_DIR, "ckpts", tag)
                if run_experiment.is_training_complete(ckpt_dir, epochs):
                    log(f"  SKIP {tag}: already trained to {epochs} epochs")
                    continue
                resume_flag = ""
                if os.path.exists(os.path.join(ckpt_dir, "vqmlp-decoder.pt")):
                    resume_flag = "--resume "
                    log(f"  Resuming: {tag}")
                else:
                    log(f"  Training: {tag}")
                run_cmd(
                    f"python src/train_vq_mlp.py "
                    f"--encoding_type {enc_type} "
                    f"--sampling {cond['sampling']} "
                    f"--loss_type {cond['loss_type']} "
                    f"--data_dir data/ "
                    f"--save_path {ckpt_dir} "
                    f"--epochs {epochs} "
                    f"--batch_size {batch_size} "
                    f"--save_interval {save_interval} "
                    f"--sample_limit {sample_limit} "
                    f"--max_shapes {len(shape_ids)} "
                    f"{resume_flag}"
                    f"--proj_name vqmlp",
                    EXP_ID, f"Train {tag}")

        # === PHASE 2: MESH EXTRACTION ===
        log("\n--- PHASE 2: Mesh Extraction ---")
        for enc_type in encodings:
            for cond in conditions:
                tag = f"{enc_type}_{cond['label']}"
                ckpt_dir = os.path.join(OUT_DIR, "ckpts", tag)
                if not os.path.exists(os.path.join(ckpt_dir, "vqmlp-decoder.pt")):
                    log(f"  Skipping {tag}: no model")
                    continue
                for sid in shape_ids:
                    mesh_path = os.path.join(OUT_DIR, "meshes", f"{tag}_{sid}.obj")
                    run_cmd(
                        f"python src/infer_vq_mlp.py "
                        f"--encoding_type {enc_type} "
                        f"--model_dir {ckpt_dir} "
                        f"--data_dir data/ "
                        f"--shape_id {sid} "
                        f"--output {mesh_path} "
                        f"--resolution {resolution} "
                        f"--proj_name vqmlp "
                        f"--batch_chunks {16 if MINIMAL else 64} "
                        f"--voxel_gif --voxel_nii "
                        f"--voxel_resolution 64 "
                        f"--output_dir {os.path.join(OUT_DIR, 'voxels', tag)} "
                        f"--render_mesh",
                        EXP_ID, f"Infer {tag} shape {sid}")

        # === PHASE 2b: FLOODING (Fragment Extraction) ===
        log("\n--- PHASE 2b: Flooding (Fragment Extraction) ---")
        os.makedirs(os.path.join(OUT_DIR, "flooding"), exist_ok=True)
        field_type_map = {'ffb': 'signed', 'udf': 'unsigned'}
        for enc_type in encodings:
            for cond in conditions:
                tag = f"{enc_type}_{cond['label']}"
                for sid in shape_ids:
                    nii_path = os.path.join(OUT_DIR, "voxels", tag,
                                            f"{enc_type}_{sid}_voxel_{resolution}.nii")
                    if not os.path.exists(nii_path):
                        log(f"  Skipping flooding {tag} {sid}: no volume NII")
                        continue
                    flood_output = os.path.join(OUT_DIR, "flooding", f"{tag}_{sid}.obj")
                    gt_flag = f"--gt_obj {SQUIRREL_OBJ}" if os.path.exists(SQUIRREL_OBJ) else ""
                    tol = 25 if enc_type == 'ffb' else 7
                    run_cmd(
                        f"python src/extract_mesh_flooding.py "
                        f"--volume {nii_path} "
                        f"--field_type {field_type_map[enc_type]} "
                        f"--output {flood_output} "
                        f"--tolerance {tol} "
                        f"--conn 6 "
                        f"{gt_flag}",
                        EXP_ID, f"Flooding {tag} shape {sid}")
    else:
        log("\n--- EVAL_ONLY mode: skipping training, inference, flooding ---")

    # === PHASE 3: EVALUATION (using flooding results) ===
    log("\n--- PHASE 3: Evaluation (flooding meshes) ---")
    all_results = {}
    for enc_type in encodings:
        for cond in conditions:
            tag = f"{enc_type}_{cond['label']}"
            all_results[tag] = {}
            for sid in shape_ids:
                gt_obj = os.path.join(GT_OBJ_DIR, f"{sid}.obj")
                mesh_path = os.path.join(OUT_DIR, "flooding", f"{tag}_{sid}.obj")
                if not os.path.exists(mesh_path):
                    log(f"  SKIP {tag} shape {sid}: flooding mesh not found: {mesh_path}")
                    continue
                if not os.path.exists(gt_obj):
                    log(f"  SKIP {tag} shape {sid}: GT obj not found: {gt_obj}")
                    continue
                try:
                    result = compute_symmfcd(gt_obj, mesh_path, num_samples=2500)
                    all_results[tag][sid] = result
                    log(f"  {tag} shape {sid}: SymMFCD = {result['symmetric_mfcd']:.6f}")
                except Exception as e:
                    log(f"  {tag} shape {sid}: Failed: {e}")

    metrics_path = os.path.join(OUT_DIR, "metrics", "symmfcd_results.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # === PHASE 4: VISUALIZATION ===
    log("\n--- PHASE 4: Visualization ---")

    # Loss curves grouped by encoding
    for enc_type in encodings:
        loss_files, loss_labels = [], []
        for cond in conditions:
            tag = f"{enc_type}_{cond['label']}"
            lf = os.path.join(OUT_DIR, "ckpts", tag, "loss_history.json")
            if os.path.exists(lf):
                loss_files.append(lf)
                loss_labels.append(cond['label'])
        if loss_files:
            plot_loss_curves(loss_files, loss_labels,
                             os.path.join(OUT_DIR, "figures", f"loss_curves_{enc_type}.png"),
                             title=f"Training Loss: {enc_type.upper()} Technique Ablation")

    # SymMFCD bar chart (all conditions)
    avg_mfcd = {}
    for enc_type in encodings:
        for cond in conditions:
            tag = f"{enc_type}_{cond['label']}"
            vals = [r['symmetric_mfcd'] for r in all_results.get(tag, {}).values()
                    if 'symmetric_mfcd' in r and not np.isnan(r['symmetric_mfcd'])]
            if vals:
                avg_mfcd[tag] = float(np.mean(vals))
    if avg_mfcd:
        plot_mfcd_bar_chart(
            list(avg_mfcd.keys()), list(avg_mfcd.values()),
            os.path.join(OUT_DIR, "figures", "symmfcd_comparison.png"),
            title="SymMFCD: Training Technique Ablation (avg)")

    # === SUMMARY ===
    log(f"\n=== {EXP_ID} SUMMARY ===")
    tags = [f"{e}_{c['label']}" for e in encodings for c in conditions]
    log(f"Conditions: {tags}")
    mesh_files = glob.glob(os.path.join(OUT_DIR, "meshes", "*.obj"))
    log(f"Meshes: {len(mesh_files)}")
    if avg_mfcd:
        for tag, val in avg_mfcd.items():
            log(f"  {tag}: {val:.6f}")
    log(f"Results: {OUT_DIR}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()
