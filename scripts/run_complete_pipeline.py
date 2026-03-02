"""
Complete Pipeline for UDF Baseline Experiments

This script runs the entire pipeline for all 5 data objects:
1. Data encoding (FFB-DF and UDF)
2. Model training (FFB-MLP, UDF-MLP, NeuralUDF-MLP)
3. Mesh extraction (with and without MIND)
4. Metric computation (SymMFCD, Fragment-wise, Boundary recall)
5. Visualization (comparison plots)

Usage:
    # Full pipeline
    python scripts/run_complete_pipeline.py

    # Quick test (1 epoch, low resolution)
    python scripts/run_complete_pipeline.py --quick_test

    # Custom configuration
    python scripts/run_complete_pipeline.py --epochs 50 --resolution 128
"""

import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path


class PipelineRunner:
    """Manages the complete experimental pipeline."""

    def __init__(self, args):
        self.args = args
        self.root_dir = Path(__file__).parent.parent
        self.results_dir = self.root_dir / "data" / "results" / "complete_pipeline"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Log file
        self.log_file = self.results_dir / f"pipeline_log_{self.timestamp}.txt"

        # Status tracking
        self.status = {
            'start_time': None,
            'end_time': None,
            'phases': {}
        }

    def log(self, message):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')

    def run_command(self, cmd, phase_name, check=True):
        """Run a shell command and log output."""
        self.log(f"\n{'='*60}")
        self.log(f"Phase: {phase_name}")
        self.log(f"Command: {cmd}")
        self.log(f"{'='*60}\n")

        phase_start = time.time()

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=check,
                capture_output=True,
                text=True
            )

            phase_time = time.time() - phase_start

            self.log(f"\n[STDOUT]\n{result.stdout}")
            if result.stderr:
                self.log(f"\n[STDERR]\n{result.stderr}")

            self.status['phases'][phase_name] = {
                'status': 'success',
                'time': phase_time,
                'returncode': result.returncode
            }

            self.log(f"\n✅ Phase '{phase_name}' completed in {phase_time:.1f}s\n")

            return result

        except subprocess.CalledProcessError as e:
            phase_time = time.time() - phase_start

            self.log(f"\n[STDOUT]\n{e.stdout}")
            self.log(f"\n[STDERR]\n{e.stderr}")

            self.status['phases'][phase_name] = {
                'status': 'failed',
                'time': phase_time,
                'returncode': e.returncode,
                'error': str(e)
            }

            self.log(f"\n❌ Phase '{phase_name}' failed in {phase_time:.1f}s\n")

            if check:
                raise
            return None

    def phase_1_encoding(self):
        """Phase 1: Data Encoding (FFB-DF and UDF)."""
        self.log("\n" + "="*80)
        self.log("PHASE 1: DATA ENCODING")
        self.log("="*80 + "\n")

        # Check if encoding already exists
        ffb_dir = self.root_dir / "data" / "npz-resample"
        udf_dir = self.root_dir / "data" / "npz-udf"

        if self.args.skip_encoding and ffb_dir.exists() and udf_dir.exists():
            self.log("⏭️  Skipping encoding (data already exists)")
            return

        # FFB-DF encoding
        self.run_command(
            f"python {self.root_dir}/src/encoder_ffb-df_mlp.py",
            "1.1_FFB_Encoding"
        )

        # UDF encoding
        self.run_command(
            f"python {self.root_dir}/src/encoder_udf_mesh.py",
            "1.2_UDF_Encoding"
        )

    def phase_2_training(self):
        """Phase 2: Model Training."""
        self.log("\n" + "="*80)
        self.log("PHASE 2: MODEL TRAINING")
        self.log("="*80 + "\n")

        epochs = self.args.epochs

        # 2.1 FFB-MLP training
        if not self.args.skip_ffb:
            self.run_command(
                f"python {self.root_dir}/src/train_ffb_mlp.py "
                f"--npz_dir {self.root_dir}/data/npz-resample "
                f"--epochs {epochs} "
                f"--output_dir {self.root_dir}/data/ckpts/ffb_mlp",
                "2.1_FFB_MLP_Training"
            )

        # 2.2 UDF-MLP training
        if not self.args.skip_udf:
            self.run_command(
                f"python {self.root_dir}/src/train_udf_mlp.py "
                f"--npz_dir {self.root_dir}/data/npz-udf "
                f"--epochs {epochs} "
                f"--output_dir {self.root_dir}/data/ckpts/udf_mlp",
                "2.2_UDF_MLP_Training"
            )

        # 2.3 NeuralUDF-MLP training
        if not self.args.skip_neuraludf:
            self.run_command(
                f"python {self.root_dir}/src/train_neuraludf_mlp.py "
                f"--npz_dir {self.root_dir}/data/npz-udf "
                f"--epochs {epochs} "
                f"--d_hidden 256 "
                f"--n_layers 6 "
                f"--skip_in 4 "
                f"--multires 6 "
                f"--output_dir {self.root_dir}/data/ckpts/neuraludf_mlp",
                "2.3_NeuralUDF_MLP_Training"
            )

    def phase_3_mesh_extraction(self):
        """Phase 3: Mesh Extraction (with and without MIND)."""
        self.log("\n" + "="*80)
        self.log("PHASE 3: MESH EXTRACTION")
        self.log("="*80 + "\n")

        resolution = self.args.resolution
        mind_iter = self.args.mind_iter

        mesh_dir = self.root_dir / "data" / "results" / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        # Define models to extract
        models = []

        if not self.args.skip_udf:
            models.append({
                'name': 'UDF-MLP',
                'type': 'udf_mlp',
                'ckpt': f"{self.root_dir}/data/ckpts/udf_mlp/udf_mlp.pth",
                'output': f"{mesh_dir}/udf_mlp.ply"
            })

        if not self.args.skip_ffb:
            models.append({
                'name': 'FFB-MLP',
                'type': 'ffb_mlp',
                'ckpt': f"{self.root_dir}/data/ckpts/ffb_mlp/ffb_mlp.pth",
                'output': f"{mesh_dir}/ffb_mlp.ply"
            })

        if not self.args.skip_neuraludf:
            models.append({
                'name': 'NeuralUDF-MLP',
                'type': 'neuraludf_mlp',
                'ckpt': f"{self.root_dir}/data/ckpts/neuraludf_mlp/neuraludf_mlp.pth",
                'output': f"{mesh_dir}/neuraludf_mlp.ply"
            })

        # Extract meshes with MIND
        for model in models:
            if not self.args.skip_mind:
                self.run_command(
                    f"python {self.root_dir}/src/extract_mesh_with_mind.py "
                    f"--model_type {model['type']} "
                    f"--ckpt {model['ckpt']} "
                    f"--output {model['output'].replace('.ply', '_mind.ply')} "
                    f"--resolution {resolution} "
                    f"--max_iter {mind_iter}",
                    f"3_Extract_{model['name']}_MIND",
                    check=False  # Don't fail if MIND has issues
                )

    def phase_4_evaluation(self):
        """Phase 4: Metric Computation."""
        self.log("\n" + "="*80)
        self.log("PHASE 4: METRIC COMPUTATION")
        self.log("="*80 + "\n")

        if self.args.skip_evaluation:
            self.log("⏭️  Skipping evaluation")
            return

        # Symmetric MFCD computation
        mesh_dir = self.root_dir / "data" / "results" / "meshes"
        orig_dir = self.root_dir / "data" / "original_meshes"

        if not orig_dir.exists():
            self.log("⚠️  Original meshes not found, skipping MFCD evaluation")
            return

        # Build recon_dirs argument
        recon_dirs = []
        if not self.args.skip_udf and (mesh_dir / "udf_mlp_mind.ply").exists():
            recon_dirs.append(f"udf_mlp:{mesh_dir}")
        if not self.args.skip_ffb and (mesh_dir / "ffb_mlp_mind.ply").exists():
            recon_dirs.append(f"ffb_mlp:{mesh_dir}")
        if not self.args.skip_neuraludf and (mesh_dir / "neuraludf_mlp_mind.ply").exists():
            recon_dirs.append(f"neuraludf:{mesh_dir}")

        if recon_dirs:
            self.run_command(
                f"python {self.root_dir}/experiments/mfcd_definition/symmetric_mfcd.py "
                f"--batch "
                f"--orig-dir {orig_dir} "
                f"--recon-dirs {' '.join(recon_dirs)} "
                f"--output-dir {self.results_dir}/mfcd "
                f"--num-samples 5000",
                "4_SymMFCD_Evaluation",
                check=False
            )

    def phase_5_visualization(self):
        """Phase 5: Visualization."""
        self.log("\n" + "="*80)
        self.log("PHASE 5: VISUALIZATION")
        self.log("="*80 + "\n")

        if self.args.skip_visualization:
            self.log("⏭️  Skipping visualization")
            return

        # Create comparison visualizations
        # (This would call a visualization script once it's created)
        self.log("📊 Visualization phase - TODO: Create comparison plots")

    def save_status(self):
        """Save pipeline status to JSON."""
        status_file = self.results_dir / f"pipeline_status_{self.timestamp}.json"
        with open(status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
        self.log(f"\n📄 Status saved to: {status_file}")

    def print_summary(self):
        """Print pipeline summary."""
        self.log("\n" + "="*80)
        self.log("PIPELINE SUMMARY")
        self.log("="*80 + "\n")

        total_time = self.status['end_time'] - self.status['start_time']

        self.log(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
        self.log(f"\nPhase breakdown:")

        for phase, info in self.status['phases'].items():
            status_emoji = "✅" if info['status'] == 'success' else "❌"
            self.log(f"  {status_emoji} {phase}: {info['time']:.1f}s ({info['status']})")

        success_count = sum(1 for p in self.status['phases'].values() if p['status'] == 'success')
        total_count = len(self.status['phases'])

        self.log(f"\nSuccess rate: {success_count}/{total_count}")

        self.log(f"\n📁 Results saved to: {self.results_dir}")
        self.log(f"📄 Log file: {self.log_file}")

    def run(self):
        """Run the complete pipeline."""
        self.status['start_time'] = time.time()

        self.log("\n" + "="*80)
        self.log("STARTING COMPLETE PIPELINE")
        self.log("="*80)
        self.log(f"Timestamp: {self.timestamp}")
        self.log(f"Configuration:")
        for key, value in vars(self.args).items():
            self.log(f"  {key}: {value}")
        self.log("="*80 + "\n")

        try:
            # Run all phases
            if not self.args.skip_phase_1:
                self.phase_1_encoding()

            if not self.args.skip_phase_2:
                self.phase_2_training()

            if not self.args.skip_phase_3:
                self.phase_3_mesh_extraction()

            if not self.args.skip_phase_4:
                self.phase_4_evaluation()

            if not self.args.skip_phase_5:
                self.phase_5_visualization()

        except Exception as e:
            self.log(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            self.log(traceback.format_exc())

        finally:
            self.status['end_time'] = time.time()
            self.save_status()
            self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Run complete UDF baseline experimental pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Full pipeline (default)
  python scripts/run_complete_pipeline.py

  # Quick test (1 epoch, low resolution)
  python scripts/run_complete_pipeline.py --quick_test

  # Custom epochs and resolution
  python scripts/run_complete_pipeline.py --epochs 50 --resolution 128

  # Skip certain phases
  python scripts/run_complete_pipeline.py --skip_phase_1 --skip_phase_4

  # Skip certain models
  python scripts/run_complete_pipeline.py --skip_ffb --skip_neuraludf
        """
    )

    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test mode (1 epoch, low resolution)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')

    # Mesh extraction parameters
    parser.add_argument('--resolution', type=int, default=256,
                        help='Grid resolution for MIND (default: 256)')
    parser.add_argument('--mind_iter', type=int, default=200,
                        help='MIND optimization iterations (default: 200)')

    # Skip phases
    parser.add_argument('--skip_phase_1', action='store_true',
                        help='Skip Phase 1 (Encoding)')
    parser.add_argument('--skip_phase_2', action='store_true',
                        help='Skip Phase 2 (Training)')
    parser.add_argument('--skip_phase_3', action='store_true',
                        help='Skip Phase 3 (Mesh Extraction)')
    parser.add_argument('--skip_phase_4', action='store_true',
                        help='Skip Phase 4 (Evaluation)')
    parser.add_argument('--skip_phase_5', action='store_true',
                        help='Skip Phase 5 (Visualization)')

    # Skip specific models
    parser.add_argument('--skip_ffb', action='store_true',
                        help='Skip FFB-MLP')
    parser.add_argument('--skip_udf', action='store_true',
                        help='Skip UDF-MLP')
    parser.add_argument('--skip_neuraludf', action='store_true',
                        help='Skip NeuralUDF-MLP')
    parser.add_argument('--skip_mind', action='store_true',
                        help='Skip MIND optimization')

    # Skip operations
    parser.add_argument('--skip_encoding', action='store_true',
                        help='Skip encoding if data already exists')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip metric evaluation')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip visualization generation')

    args = parser.parse_args()

    # Quick test mode overrides
    if args.quick_test:
        args.epochs = 1
        args.resolution = 64
        args.mind_iter = 10
        print("🚀 Quick test mode activated")

    # Run pipeline
    runner = PipelineRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
