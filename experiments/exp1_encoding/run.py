#!/usr/bin/env python3
"""
Exp1: Encoding Comparison — FFB vs UDF vs Truncated-UDF vs Signed-UDF

All 4 encodings trained with VQ-MLP + SiLU + NB sampling + L1 loss.
5 shapes trained together with CSV→z_feature conditioning.

Pipeline:
1. Ensure NPZ encodings exist
2. Train VQ-MLP per encoding type
3. Extract meshes + voxel GIF/NII
4. Evaluate SymMFCD
5. Visualize: loss curves, SymMFCD comparison, mesh renderings

Usage:
    python experiments/exp1_encoding/run.py
    python experiments/exp1_encoding/run.py --minimal
    python experiments/exp1_encoding/run.py --quick
    python experiments/exp1_encoding/run.py --reboolean
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
                        visualize_sampling, render_mesh_comparison)

log = run_experiment.log
run_cmd = run_experiment.run_cmd
ROOT = run_experiment.ROOT

EXP_ID = "exp1_encoding"
OUT_DIR = os.path.join(_exp_dir, "output")

MINIMAL = "--minimal" in sys.argv
QUICK = "--quick" in sys.argv
REBOOLEAN = "--reboolean" in sys.argv

ENCODING_TYPES = ['ffb', 'udf', 'truncated_udf', 'signed_udf']
ENCODER_SCRIPTS = {
    'ffb': 'src/encoder_ffb-df_mlp.py',
    'udf': 'src/encoder_udf_mesh.py',
    'truncated_udf': 'src/encoder_truncated_udf.py',
    'signed_udf': 'src/encoder_signed_udf.py',
}
NPZ_DIRS = {
    'ffb': 'data/npz-resample',
    'udf': 'data/npz-udf',
    'truncated_udf': 'data/npz-truncated-udf',
    'signed_udf': 'data/npz-signed-udf',
}
FIELD_TYPE = {
    'ffb': 'signed',
    'udf': 'unsigned',
    'truncated_udf': 'unsigned',
    'signed_udf': 'signed',
}

ALL_SHAPE_IDS = ['1', '2', '3', '4', '5']
GT_OBJ_DIR = os.path.join(ROOT, "data", "obj")
SQUIRREL_OBJ = os.path.join(ROOT, "data", "squirrel.obj")


def _get_available_shapes():
    available = []
    for sid in ALL_SHAPE_IDS:
        if os.path.exists(os.path.join(GT_OBJ_DIR, f"{sid}.obj")):
            available.append(sid)
    return available


def _ensure_encoding(encoding_type, shape_ids):
    npz_dir = os.path.join(ROOT, NPZ_DIRS[encoding_type])
    missing = [sid for sid in shape_ids
               if not os.path.exists(os.path.join(npz_dir, f"{sid}.npz"))]
    if not missing:
        log(f"  {encoding_type}: all NPZ files present")
        return
    log(f"  {encoding_type}: encoding {len(missing)} shapes: {missing}")
    script = ENCODER_SCRIPTS[encoding_type]
    # Match the run mode: MINIMAL/QUICK use a small sample count so a fresh clone
    # (no precomputed npz) still encodes in seconds/minutes instead of hours.
    enc_flag = " --minimal" if MINIMAL else (" --fast" if QUICK else "")
    for sid in missing:
        run_cmd(f"python {script} --shape_id {sid}{enc_flag}",
                EXP_ID, f"Encode {encoding_type} shape {sid}")


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

    encoding_types = ['ffb', 'udf'] if MINIMAL else ENCODING_TYPES

    if REBOOLEAN:
        # === REBOOLEAN MODE: redo boolean + merge using existing fragments ===
        log("\n--- REBOOLEAN: Redo boolean ∩ squirrel.obj from existing fragments ---")
        import trimesh
        import vedo as vd

        if not os.path.exists(SQUIRREL_OBJ):
            log(f"  ERROR: squirrel.obj not found: {SQUIRREL_OBJ}")
            sys.exit(1)

        min_faces = 250
        for enc_type in encoding_types:
            for sid in shape_ids:
                out_stem = f"{enc_type}_{sid}"
                work_path = os.path.join(OUT_DIR, "flooding", f"flooding_work_{out_stem}")
                frag_dir = os.path.join(work_path, "fragments")
                flood_output = os.path.join(OUT_DIR, "flooding", f"{out_stem}.obj")

                if not os.path.isdir(frag_dir):
                    log(f"  SKIP {out_stem}: no fragments dir")
                    continue

                # Load existing fragment meshes
                frag_files = sorted(glob.glob(os.path.join(frag_dir, "fragment_*.obj")))
                if not frag_files:
                    # Try .ply fallback
                    frag_files = sorted(glob.glob(os.path.join(frag_dir, "fragment_*.ply")))
                if not frag_files:
                    log(f"  SKIP {out_stem}: no fragment files")
                    continue

                fragment_meshes = []
                for fp in frag_files:
                    label_id = int(os.path.basename(fp).split('_')[1].split('.')[0])
                    m = vd.Mesh(fp)
                    if m.npoints > 0:
                        fragment_meshes.append((label_id, m))

                log(f"  {out_stem}: loaded {len(fragment_meshes)} fragments")

                # Redo boolean with new squirrel.obj
                gt_mesh = trimesh.load(SQUIRREL_OBJ, force='mesh', process=False)
                bool_dir = os.path.join(work_path, "boolean_fragments")
                os.makedirs(bool_dir, exist_ok=True)

                result_meshes = []
                for label_id, vedo_mesh in fragment_meshes:
                    try:
                        verts = vedo_mesh.vertices
                        faces_raw = vedo_mesh.cells
                        frag_tri = trimesh.Trimesh(vertices=verts, faces=faces_raw)
                        result = frag_tri.intersection(gt_mesh)
                        if hasattr(result, 'vertices') and len(result.vertices) > 0:
                            result_vedo = vd.Mesh([result.vertices, result.faces])
                            result_meshes.append((label_id, result_vedo))
                            result.export(os.path.join(bool_dir, f"fragment_{label_id}.obj"))
                        else:
                            log(f"    Label {label_id}: empty after boolean")
                    except Exception as e:
                        log(f"    Label {label_id}: boolean failed ({e}), keeping raw")
                        result_meshes.append((label_id, vedo_mesh))

                # Filter small fragments
                filtered = [(lid, m) for lid, m in result_meshes
                            if (m.ncells if hasattr(m, 'ncells') else m.NCells()) >= min_faces]
                log(f"  {out_stem}: {len(result_meshes)} → {len(filtered)} after filter (>={min_faces} faces)")

                if not filtered:
                    log(f"  {out_stem}: no fragments remaining")
                    continue

                # Merge and save
                all_m = [m for _, m in filtered]
                merged = all_m[0] if len(all_m) == 1 else vd.merge(all_m)
                merged.write(flood_output)
                npts = merged.npoints if hasattr(merged, 'npoints') else merged.N()
                log(f"  {out_stem}: saved {flood_output} ({npts} verts)")

        log("--- REBOOLEAN complete, continuing to evaluation ---\n")
    else:
        # === PHASE 0: ENCODING ===
        log("\n--- PHASE 0: Ensure Encodings ---")
        for enc_type in encoding_types:
            _ensure_encoding(enc_type, shape_ids)

        # Sampling visualization
        for enc_type in encoding_types:
            for sid in shape_ids[:1]:
                npz_path = os.path.join(ROOT, NPZ_DIRS[enc_type], f"{sid}.npz")
                if os.path.exists(npz_path):
                    visualize_sampling(
                        npz_path,
                        os.path.join(OUT_DIR, "figures", f"sampling_{enc_type}_{sid}.png"))

        # === PHASE 1: TRAINING ===
        log("\n--- PHASE 1: Training ---")
        for enc_type in encoding_types:
            ckpt_dir = os.path.join(OUT_DIR, "ckpts", enc_type)
            if run_experiment.is_training_complete(ckpt_dir, epochs):
                log(f"  SKIP {enc_type}: already trained to {epochs} epochs")
                continue
            # Auto-resume if partial checkpoint exists
            resume_flag = ""
            if os.path.exists(os.path.join(ckpt_dir, "vqmlp-decoder.pt")):
                resume_flag = "--resume "
                log(f"  Resuming VQ-MLP: encoding={enc_type}")
            else:
                log(f"  Training VQ-MLP: encoding={enc_type}")
            run_cmd(
                f"python src/train_vq_mlp.py "
                f"--encoding_type {enc_type} "
                f"--data_dir data/ "
                f"--save_path {ckpt_dir} "
                f"--epochs {epochs} "
                f"--batch_size {batch_size} "
                f"--save_interval {save_interval} "
                f"--sample_limit {sample_limit} "
                f"--max_shapes {len(shape_ids)} "
                f"{resume_flag}"
                f"--proj_name vqmlp",
                EXP_ID, f"Train {enc_type}")

        # === PHASE 2: MESH EXTRACTION + VOXEL VIS ===
        log("\n--- PHASE 2: Mesh Extraction + Voxel Visualization ---")
        for enc_type in encoding_types:
            ckpt_dir = os.path.join(OUT_DIR, "ckpts", enc_type)
            if not os.path.exists(os.path.join(ckpt_dir, "vqmlp-decoder.pt")):
                log(f"  Skipping {enc_type}: no trained model")
                continue
            for sid in shape_ids:
                mesh_path = os.path.join(OUT_DIR, "meshes", f"{enc_type}_{sid}.obj")
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
                    f"--output_dir {os.path.join(OUT_DIR, 'voxels', enc_type)} "
                    f"--render_mesh",
                    EXP_ID, f"Infer {enc_type} shape {sid}")

        # === PHASE 2b: FLOODING (Fragment Extraction) ===
        log("\n--- PHASE 2b: Flooding (Fragment Extraction) ---")
        os.makedirs(os.path.join(OUT_DIR, "flooding"), exist_ok=True)
        for enc_type in encoding_types:
            for sid in shape_ids:
                voxel_dir = os.path.join(OUT_DIR, "voxels", enc_type)
                nii_path = os.path.join(voxel_dir, f"{enc_type}_{sid}_voxel_{resolution}.nii")
                if not os.path.exists(nii_path):
                    log(f"  Skipping flooding {enc_type} {sid}: no volume NII")
                    continue
                flood_output = os.path.join(OUT_DIR, "flooding", f"{enc_type}_{sid}.obj")
                gt_flag = f"--gt_obj {SQUIRREL_OBJ}" if os.path.exists(SQUIRREL_OBJ) else ""
                tol = 25 if enc_type == 'ffb' else 7
                run_cmd(
                    f"python src/extract_mesh_flooding.py "
                    f"--volume {nii_path} "
                    f"--field_type {FIELD_TYPE[enc_type]} "
                    f"--output {flood_output} "
                    f"--tolerance {tol} "
                    f"--conn 6 "
                    f"{gt_flag}",
                    EXP_ID, f"Flooding {enc_type} shape {sid}")

    # === PHASE 3: EVALUATION (using flooding results) ===
    log("\n--- PHASE 3: Evaluation (flooding meshes) ---")
    all_results = {}
    for enc_type in encoding_types:
        all_results[enc_type] = {}
        for sid in shape_ids:
            gt_obj = os.path.join(GT_OBJ_DIR, f"{sid}.obj")
            mesh_path = os.path.join(OUT_DIR, "flooding", f"{enc_type}_{sid}.obj")
            if not os.path.exists(mesh_path):
                log(f"  SKIP {enc_type} shape {sid}: flooding mesh not found: {mesh_path}")
                continue
            if not os.path.exists(gt_obj):
                log(f"  SKIP {enc_type} shape {sid}: GT obj not found: {gt_obj}")
                continue
            log(f"  SymMFCD: {enc_type} shape {sid}...")
            try:
                result = compute_symmfcd(gt_obj, mesh_path, num_samples=4000)
                all_results[enc_type][sid] = result
                log(f"    SymMFCD = {result['symmetric_mfcd']:.6f}")
            except Exception as e:
                log(f"    Failed: {e}")

    metrics_path = os.path.join(OUT_DIR, "metrics", "symmfcd_results.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # === PHASE 4: VISUALIZATION ===
    log("\n--- PHASE 4: Visualization ---")

    # Loss curves
    loss_files, loss_labels = [], []
    for enc_type in encoding_types:
        lf = os.path.join(OUT_DIR, "ckpts", enc_type, "loss_history.json")
        if os.path.exists(lf):
            loss_files.append(lf)
            loss_labels.append(enc_type)
    if loss_files:
        plot_loss_curves(loss_files, loss_labels,
                         os.path.join(OUT_DIR, "figures", "loss_curves.png"),
                         title="Training Loss: Encoding Comparison")

    # SymMFCD bar chart (average)
    avg_mfcd = {}
    for enc_type in encoding_types:
        vals = [r['symmetric_mfcd'] for r in all_results.get(enc_type, {}).values()
                if 'symmetric_mfcd' in r and not np.isnan(r['symmetric_mfcd'])]
        if vals:
            avg_mfcd[enc_type] = float(np.mean(vals))
    if avg_mfcd:
        plot_mfcd_bar_chart(
            list(avg_mfcd.keys()), list(avg_mfcd.values()),
            os.path.join(OUT_DIR, "figures", "symmfcd_comparison.png"),
            title="SymMFCD: Encoding Comparison (avg)")

    # Per-shape bar charts
    for sid in shape_ids:
        shape_mfcd = {}
        for enc_type in encoding_types:
            r = all_results.get(enc_type, {}).get(sid)
            if r and not np.isnan(r.get('symmetric_mfcd', float('nan'))):
                shape_mfcd[enc_type] = r['symmetric_mfcd']
        if shape_mfcd:
            plot_mfcd_bar_chart(
                list(shape_mfcd.keys()), list(shape_mfcd.values()),
                os.path.join(OUT_DIR, "figures", f"symmfcd_shape_{sid}.png"),
                title=f"SymMFCD: Shape {sid}")

    # Mesh comparison per shape (flooding results) — never let visualization
    # failures (e.g. empty meshes from an undertrained smoke run) kill the run.
    for sid in shape_ids:
        paths = [os.path.join(OUT_DIR, "flooding", f"{et}_{sid}.obj")
                 for et in encoding_types]
        try:
            render_mesh_comparison(
                paths, encoding_types,
                os.path.join(OUT_DIR, "figures", f"mesh_comparison_{sid}.png"),
                title=f"Mesh Comparison: Shape {sid}")
        except Exception as e:
            log(f"  Mesh comparison for shape {sid} skipped: {e}")

    # === SUMMARY ===
    log(f"\n=== {EXP_ID} SUMMARY ===")
    log(f"Shapes: {shape_ids}, Encodings: {encoding_types}")
    mesh_files = glob.glob(os.path.join(OUT_DIR, "meshes", "*.obj"))
    log(f"Meshes: {len(mesh_files)}")
    if avg_mfcd:
        for et, val in avg_mfcd.items():
            log(f"  {et}: {val:.6f}")
    log(f"Results: {OUT_DIR}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()
