#!/usr/bin/env python3
"""
Quick test for Far-From-Boundary Fields (FFB).

A single-command sanity run, in the spirit of TEBP's `predict-runtime.py --auto-run`.
It drives the encoding-comparison experiment (`experiments/exp1_encoding/run.py`):
ensure NPZ encodings → train the VQ-MLP for a few steps → extract meshes →
evaluate SymMFCD → render the comparison figure. Small settings let it finish in a
few minutes.

Usage:
    python quick_test.py            # minimal smoke run
    python quick_test.py --quick    # medium run

Outputs (meshes, metrics, figures) are written under
`experiments/exp1_encoding/output/`.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join("experiments", "exp1_encoding", "run.py")


def main():
    parser = argparse.ArgumentParser(description="FFB quick end-to-end test")
    parser.add_argument("--quick", action="store_true",
                        help="Use --quick mode (medium run); default is --minimal")
    args = parser.parse_args()

    entry_abs = os.path.join(ROOT, ENTRY)
    if not os.path.exists(entry_abs):
        sys.exit(f"[quick_test] missing entry script: {entry_abs}")

    mode = "--quick" if args.quick else "--minimal"
    cmd = [sys.executable, ENTRY, mode]

    print(f"[quick_test] running: {' '.join(cmd)} (cwd={ROOT})")
    print("[quick_test] encode -> train (VQ-MLP) -> extract -> SymMFCD -> figure. "
          "Results go under experiments/exp1_encoding/output/")
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        sys.exit(f"[quick_test] experiment exited with code {rc}")

    out = os.path.join(ROOT, "experiments", "exp1_encoding", "output")
    print(f"\n[quick_test] done. Inspect results under: {out}")
    print("  figures/symmfcd_comparison.png  — the SymMFCD comparison figure")


if __name__ == "__main__":
    main()
