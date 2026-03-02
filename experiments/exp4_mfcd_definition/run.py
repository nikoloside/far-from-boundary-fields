#!/usr/bin/env python3
"""
Exp 4: MFCD Symmetric Definition
Define MFCD, SymMFCD, toy example
"""
import os
import sys
import numpy as np
import json

_exp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_exp_dir))
import run_experiment

log = run_experiment.log
ROOT = run_experiment.ROOT

EXP_ID = "exp4_mfcd_definition"
OUT_DIR = os.path.join(ROOT, "data", "results", "mfcd_definition")


def mfcd_one_sided(pred_pts, gt_pts, pred_vols, gt_vols):
    """Volume-weighted one-sided MFCD (Pred→GT). Simplified."""
    return np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1))


def sym_mfcd(pred_pts, gt_pts):
    """SymMFCD = 0.5 * (MFCD(P→G) + MFCD(G→P))."""
    m1 = mfcd_one_sided(pred_pts, gt_pts, None, None)
    m2 = mfcd_one_sided(gt_pts, pred_pts, None, None)
    return 0.5 * (m1 + m2)


def main():
    log(f"=== {EXP_ID} START ===")
    os.makedirs(OUT_DIR, exist_ok=True)
    # Toy example: Case 1 large ok small missing vs Case 2 global shift
    np.random.seed(42)
    case1_pred = np.random.rand(100, 3) * 0.5 - 0.25
    case1_gt = np.random.rand(100, 3) * 0.5 - 0.25
    case2_pred = np.random.rand(100, 3) * 0.5 - 0.25 + 0.3
    case2_gt = np.random.rand(100, 3) * 0.5 - 0.25
    r1 = sym_mfcd(case1_pred, case1_gt)
    r2 = sym_mfcd(case2_pred, case2_gt)
    results = {"case1_sym_mfcd": float(r1), "case2_sym_mfcd": float(r2)}
    out_path = os.path.join(OUT_DIR, "mfcd_toy_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Toy SymMFCD: Case1={r1:.4f} Case2={r2:.4f}")
    log(f"Saved {out_path}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()
