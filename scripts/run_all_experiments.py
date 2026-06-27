#!/usr/bin/env python3
"""
Run all 5 experiments and append logs to docs/experiments_log.md
Usage: python scripts/run_all_experiments.py [--minimal]
"""
import os
import sys
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(ROOT, "docs", "experiments_log.md")
os.chdir(ROOT)

EXPERIMENTS = [
    "experiments/exp1_udf_baseline/run.py",
    "experiments/exp2_training_trick_ablation/run.py",
    "experiments/exp3_activation_ablation/run.py",
    "experiments/exp4_mfcd_definition/run.py",
    "experiments/exp5_voxel_ablation/run.py",
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(msg)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    minimal = " --minimal" if "--minimal" in sys.argv else ""
    log("=" * 60)
    log("RUN ALL EXPERIMENTS START")
    log("=" * 60)
    for i, exp_script in enumerate(EXPERIMENTS):
        log(f"\n>>> Experiment {i+1}/5: {exp_script}")
        r = subprocess.run(
            f"python {exp_script}{minimal}",
            shell=True,
            cwd=ROOT,
        )
        log(f"Exit code: {r.returncode}")
    log("\n" + "=" * 60)
    log("RUN ALL EXPERIMENTS END")
    log("=" * 60)


if __name__ == "__main__":
    main()
