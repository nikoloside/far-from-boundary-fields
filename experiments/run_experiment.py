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


def is_training_complete(ckpt_dir, target_epochs, proj_name='vqmlp'):
    """Check if training already finished (last saved epoch >= target).

    Returns True only if the model exists AND was trained to completion.
    Partial checkpoints (e.g. interrupted at epoch 300/3000) return False.
    """
    import json
    decoder_path = os.path.join(ckpt_dir, f"{proj_name}-decoder.pt")
    if not os.path.exists(decoder_path):
        return False
    # Check loss_history for actual last epoch
    loss_path = os.path.join(ckpt_dir, "loss_history.json")
    if os.path.exists(loss_path):
        try:
            with open(loss_path) as f:
                hist = json.load(f)
            last_epoch = max(hist.get('epochs', [0]))
            return last_epoch >= target_epochs
        except Exception:
            pass
    # Check resume checkpoint for epoch
    resume_path = os.path.join(ckpt_dir, f"{proj_name}-resume.pt")
    if os.path.exists(resume_path):
        try:
            import torch
            data = torch.load(resume_path, map_location='cpu', weights_only=False)
            return data.get('epoch', 0) >= target_epochs
        except Exception:
            pass
    return False


def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    for pkg, import_name in [
        ("vedo", "vedo"),
        ("trimesh", "trimesh"),
        ("nibabel", "nibabel"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("torch", "torch"),
        ("tqdm", "tqdm"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        log(f"ERROR: Missing packages: {', '.join(missing)}")
        log(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)
