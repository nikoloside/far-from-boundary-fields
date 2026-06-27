#!/usr/bin/env python3
"""
Exp5: Legacy Method MFCD Comparison

Compare all methods in exp5_legacy/legacy/*.obj against:
    exp5_legacy/squirrel-gt.obj

Outputs:
    - exp5_legacy/output/metrics/mfcd_results.json
    - exp5_legacy/output/metrics/mfcd_summary.txt
    - exp5_legacy/output/metrics/mfcd_summary.csv

Usage:
    python exp5_legacy/run.py
"""
import csv
import glob
import json
import math
import os
import sys
from datetime import datetime

_exp_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_exp_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Optional project helpers (reuse if available)
try:
    import run_experiment  # type: ignore
    log = run_experiment.log
except Exception:
    def log(msg: str):
        print(msg)

try:
    from eval_utils import compute_symmfcd  # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to import compute_symmfcd from eval_utils. "
        "Please make sure this script is placed inside the project tree so that "
        "its parent directory contains eval_utils.py."
    ) from e

EXP_ID = "exp5_legacy"
LEGACY_DIR = os.path.join(_exp_dir, "legacy")
ALT_LEGACY_DIR = os.path.join(_exp_dir, "legecy")  # legacy fallback for the old typo'd dir
GT_OBJ = os.path.join(_exp_dir, "squirrel-gt.obj")
OUT_DIR = os.path.join(_exp_dir, "output")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")


def _pick_legacy_dir() -> str:
    if os.path.isdir(LEGACY_DIR):
        return LEGACY_DIR
    if os.path.isdir(ALT_LEGACY_DIR):
        log(f"[WARN] '{LEGACY_DIR}' not found, fallback to '{ALT_LEGACY_DIR}'.")
        return ALT_LEGACY_DIR
    raise FileNotFoundError(
        f"Neither '{LEGACY_DIR}' nor '{ALT_LEGACY_DIR}' exists."
    )


def _safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _summarize(rows):
    valid = [r for r in rows if r.get("symmetric_mfcd") is not None]
    failed = [r for r in rows if r.get("symmetric_mfcd") is None]
    best = min(valid, key=lambda r: r["symmetric_mfcd"]) if valid else None
    worst = max(valid, key=lambda r: r["symmetric_mfcd"]) if valid else None
    avg = (sum(r["symmetric_mfcd"] for r in valid) / len(valid)) if valid else None

    ordered = sorted(
        valid,
        key=lambda r: (r["symmetric_mfcd"], r["method"].lower())
    )

    return {
        "num_methods_total": len(rows),
        "num_methods_valid": len(valid),
        "num_methods_failed": len(failed),
        "mean_symmetric_mfcd": avg,
        "best_method": best["method"] if best else None,
        "best_symmetric_mfcd": best["symmetric_mfcd"] if best else None,
        "worst_method": worst["method"] if worst else None,
        "worst_symmetric_mfcd": worst["symmetric_mfcd"] if worst else None,
        "ranking": [
            {"rank": i + 1, "method": r["method"], "symmetric_mfcd": r["symmetric_mfcd"]}
            for i, r in enumerate(ordered)
        ],
    }


def main():
    log(f"=== {EXP_ID} START ===")
    os.makedirs(METRICS_DIR, exist_ok=True)

    legacy_dir = _pick_legacy_dir()
    if not os.path.exists(GT_OBJ):
        raise FileNotFoundError(f"GT OBJ not found: {GT_OBJ}")

    method_paths = sorted(glob.glob(os.path.join(legacy_dir, "*.obj")))
    if not method_paths:
        raise FileNotFoundError(f"No OBJ files found in: {legacy_dir}")

    log(f"GT: {GT_OBJ}")
    log(f"Methods dir: {legacy_dir}")
    log(f"Found methods: {len(method_paths)}")

    rows = []
    for path in method_paths:
        method = os.path.splitext(os.path.basename(path))[0]
        log(f"  Evaluating: {method}")
        row = {
            "method": method,
            "path": path,
            "gt_obj": GT_OBJ,
            "symmetric_mfcd": None,
            "raw_result": None,
            "status": "ok",
            "error": None,
        }
        try:
            result = compute_symmfcd(GT_OBJ, path)
            mfcd = _safe_float(result.get("symmetric_mfcd"))
            row["symmetric_mfcd"] = mfcd
            row["raw_result"] = result
            if mfcd is None:
                row["status"] = "invalid"
                row["error"] = "symmetric_mfcd missing/NaN/Inf"
                log(f"    -> INVALID (missing/NaN/Inf symmetric_mfcd)")
            else:
                log(f"    -> SymMFCD = {mfcd:.6f}")
        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
            log(f"    -> FAILED: {e}")
        rows.append(row)

    summary = _summarize(rows)
    payload = {
        "exp_id": EXP_ID,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gt_obj": GT_OBJ,
        "methods_dir": legacy_dir,
        "results": rows,
        "summary": summary,
    }

    json_path = os.path.join(METRICS_DIR, "mfcd_results.json")
    csv_path = os.path.join(METRICS_DIR, "mfcd_summary.csv")
    txt_path = os.path.join(METRICS_DIR, "mfcd_summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "method", "symmetric_mfcd"])
        writer.writeheader()
        for item in summary["ranking"]:
            writer.writerow(item)

    lines = []
    lines.append(f"=== {EXP_ID} SUMMARY ===")
    lines.append(f"GT OBJ: {GT_OBJ}")
    lines.append(f"Methods DIR: {legacy_dir}")
    lines.append(f"Total methods: {summary['num_methods_total']}")
    lines.append(f"Valid methods: {summary['num_methods_valid']}")
    lines.append(f"Failed methods: {summary['num_methods_failed']}")
    if summary["mean_symmetric_mfcd"] is not None:
        lines.append(f"Mean SymMFCD: {summary['mean_symmetric_mfcd']:.6f}")
    if summary["best_method"] is not None:
        lines.append(
            f"Best: {summary['best_method']} ({summary['best_symmetric_mfcd']:.6f})"
        )
    if summary["worst_method"] is not None:
        lines.append(
            f"Worst: {summary['worst_method']} ({summary['worst_symmetric_mfcd']:.6f})"
        )
    lines.append("")
    lines.append("Ranking (lower SymMFCD is better):")
    if summary["ranking"]:
        for item in summary["ranking"]:
            lines.append(
                f"  {item['rank']:>2}. {item['method']}: {item['symmetric_mfcd']:.6f}"
            )
    else:
        lines.append("  No valid methods.")

    failed_rows = [r for r in rows if r.get("status") != "ok"]
    if failed_rows:
        lines.append("")
        lines.append("Failures:")
        for r in failed_rows:
            lines.append(f"  - {r['method']}: {r['status']} | {r.get('error')}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for line in lines:
        log(line)
    log(f"JSON: {json_path}")
    log(f"CSV: {csv_path}")
    log(f"TXT: {txt_path}")
    log(f"=== {EXP_ID} END ===")


if __name__ == "__main__":
    main()

