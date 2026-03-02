#!/usr/bin/env python3
"""
Base runner for experiments. Logs to docs/experiments_log.md.
Usage: python experiments/run_experiment.py <exp_id> [--minimal]
"""
import os
import sys
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LOG_FILE = os.path.join(ROOT, "docs", "experiments_log.md")


def log(msg, to_file=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(msg)
    if to_file:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)


def run_cmd(cmd, exp_id, step):
    log(f"\n--- {exp_id} | Step: {step} ---\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    log(f"Exit code: {r.returncode}")
    return r.returncode == 0
