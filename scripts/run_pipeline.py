#!/usr/bin/env python3
"""
Run full pipeline: encode -> train -> visualize.
FFB-MLP, UDF (mesh), MIND-style UDF MLP, NeuralUDF-style UDF MLP.
Output: PNG visualizations in data/results/udf_baseline/qualitative/
"""
import os
import sys
import subprocess
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def run(cmd, desc=""):
    print(f"\n{'='*60}\n{desc}\n{cmd}\n{'='*60}")
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"Failed: {cmd}")


def main():
    fast = "--fast" in sys.argv
    minimal = "--minimal" in sys.argv
    epochs = 30 if minimal else (50 if fast else 150)
    enc_extra = " --minimal" if minimal else (" --fast" if fast else "")
    print("Pipeline: FFB-MLP | UDF-mesh | MIND | NeuralUDF -> PNG")

    # 1. FFB-DF encoder (npz-resample)
    run(f"python src/encoder_ffb-df_mlp.py{enc_extra}", "1. Encode FFB-DF (SDF)")
    npz_ffb = glob.glob("data/npz-resample/*.npz")
    if not npz_ffb:
        run("python src/encoder_udf_mesh.py", "1b. UDF encoder (fallback: npz-udf)")
        raise SystemExit("No npz-resample; check encoder_ffb-df_mlp.py paths")

    # 2. UDF mesh encoder
    run(f"python src/encoder_udf_mesh.py{enc_extra}", "2. Encode UDF from mesh")

    # 3. Train FFB-MLP
    run(f"python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs {epochs}", "3. Train FFB-MLP")

    # 4. Train UDF-MLP (for MIND & NeuralUDF)
    run(f"python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs {epochs}", "4. Train UDF-MLP")

    # 5. Generate encodings from trained models & render
    os.makedirs("data/results/udf_baseline/qualitative", exist_ok=True)
    run("python scripts/generate_and_render.py", "5. Generate model outputs & render PNG")


if __name__ == "__main__":
    main()
