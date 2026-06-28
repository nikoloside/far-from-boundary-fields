<p align="center">
  <h2 align="center">Far-From-Boundary Fields for Learning Segmented Implicit Solids</h2>
  <p align="center">
    <a href="https://nikoloside.graphics/"><strong>Yuhang Huang</strong></a>
    ·
    <a href="https://graphics.c.u-tokyo.ac.jp/hp/en/kanai"><strong>Takashi Kanai</strong></a>
    <br><b>The University of Tokyo</b><br><br>
    <a href="https://www.sciencedirect.com/science/article/pii/S0097849326001196"><img src="https://img.shields.io/badge/Paper-C&G-red" height=22.5></a>
    <a href="https://github.com/nikoloside/far-from-boundary-fields"><img src="https://img.shields.io/badge/Code-FFB-blue" height=22.5></a>
    <a href="https://colab.research.google.com/github/nikoloside/far-from-boundary-fields/blob/main/notebooks/colab_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>
    <a href="https://www.replicabilitystamp.org/"><img src="https://img.shields.io/badge/GRSI-pending-lightgrey" height=22.5></a>
  </p>
</p>

Reference implementation and experiments for the SMI 2026 paper *"Far-From-Boundary
Fields for Learning Segmented Implicit Solids"* — a VQ-MLP framework for fractured solid
reconstruction that compares UDF encoding strategies, training techniques, activation
functions, and external SOTA methods.

## Quick test (reproduce a figure)

- **Online (zero setup):** open the Colab notebook
  [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) via the badge
  above — it clones the repo, installs dependencies (incl. ImageJ), and runs the pipeline.
- **Local** (needs [Git LFS](https://git-lfs.com) for the bundled data):
  ```bash
  git lfs install
  git clone https://github.com/nikoloside/far-from-boundary-fields
  cd far-from-boundary-fields
  pip install -r requirements.txt
  python quick_test.py          # smoke test: 2 epochs / 1 shape (just checks it runs)
  python quick_test.py --quick  # produces a real symmfcd_comparison.png (minutes on GPU)
  ```
  Outputs (meshes, metrics, and `symmfcd_comparison.png`) are written under
  `experiments/exp1_encoding/output/`.

### ImageJ (watershed segmentation)

The paper's reconstruction segments the implicit field into fragments with an ImageJ
3D watershed (`pyimagej` + a JDK). This is **required to reproduce the paper's exact
numbers**; without it the pipeline falls back to direct marching cubes (still produces a
figure, but not the watershed result). On Debian/Ubuntu/Colab:
```bash
apt-get install -y openjdk-11-jdk maven
pip install pyimagej          # already in requirements.txt
```
The first run downloads Fiji (a few hundred MB). The full publication-scale result is
`python experiments/exp1_encoding/run.py` (no flag).

## Architecture

**VQ-MLP** (from `VQ-mlp-origin-siren.py`):
- **MultiLatentEncoder**: Siren(7D -> 120) — encodes collision conditions (pos, dir, impulse) from CSV
- **PosEncoder**: Siren(3D -> 128) — encodes spatial coordinates
- **ImplicitFunction**: 8-layer MLP (512 hidden, skip at layer 5) — predicts distance field
- **DeepSDFLoss**: L1/L2/Weighted-L2 with clamping + latent norm regularization

**Training**: Point-level batching. All shapes' points flattened (5 shapes x 130k pts = 650k items). `batch_size=8` = 8 points per batch = ~81k batches/epoch.

## Quick Start

```bash
# Train with FFB encoding (default: NB sampling, L1 loss, SiLU activation)
python src/train_vq_mlp.py --encoding_type ffb --data_dir data/ --epochs 80000

# Extract mesh
python src/infer_vq_mlp.py --encoding_type ffb --model_dir data/ckpts/vq_mlp_ffb \
    --shape_id 1 --output mesh_1.ply --resolution 256

# Run experiments (add --minimal for quick smoke test, --quick for medium run)
python experiments/exp1_encoding/run.py --minimal
python experiments/exp2_training_tricks/run.py --minimal
python experiments/exp3_external_methods/run.py --minimal
python experiments/exp4_activation/run.py --minimal
```

## Repository Structure

```
src/
  train_vq_mlp.py              # Core training (point-level batching, all encodings)
  infer_vq_mlp.py              # Inference: grid query + marching cubes + voxel vis
  encoder_ffb-df_mlp.py        # FFB encoding: per-fragment normalized SDF
  encoder_udf_mesh.py          # UDF encoding: abs(SDF)
  encoder_truncated_udf.py     # Truncated UDF: min(abs(SDF), 0.1)
  encoder_flip_truncated_udf.py # Flip-Truncated UDF: sign(SDF)*min(abs(SDF), 0.1)
  FFB-MLP_VQ-MLP/              # Original reference implementations

experiments/
  run_experiment.py             # Shared utilities (log, run_cmd)
  eval_utils.py                 # SymMFCD, loss curves, bar charts, mesh rendering
  exp1_encoding/run.py          # Exp1: Encoding comparison (4 types)
  exp2_training_tricks/run.py   # Exp2: Sampling x Loss ablation (8 conditions)
  exp3_external_methods/run.py  # Exp3: NDC / MeshUDF / CAP-UDF vs ours
  exp4_activation/run.py        # Exp4: SiLU / Siren / SoftPlus ablation

data/
  obj/                          # Ground truth OBJ meshes (1.obj - 5.obj)
  csv/                          # Collision conditions per shape (1.csv - 5.csv)
  npz-resample/                 # FFB encoded points (NB sampling)
  npz-resample-uniform/         # FFB encoded points (uniform sampling)
  npz-udf/                      # UDF encoded points (NB sampling)
  npz-truncated-udf/            # Truncated UDF encoded points
  npz-flip-truncated-udf/       # Flip-Truncated UDF encoded points

docs/
  ARCHITECTURE.md               # VQ-MLP architecture details
  EXPERIMENTS.md                # 4-layer experiment design
  ENCODINGS.md                  # Encoding type specifications
```

## Evaluation

**SymMFCD** (Symmetric Mean Fragment Chamfer Distance): Primary metric comparing reconstructed mesh to ground truth OBJ. Lower = better.

All experiments produce:
- `output/figures/loss_curves.png` — training loss over epochs
- `output/figures/symmfcd_comparison.png` — SymMFCD bar chart
- `output/meshes/*.ply` — extracted meshes
- `output/voxels/*.gif` + `*.nii` — voxel visualizations

## Data provenance

The solids in `data/obj/` (`1.obj`–`5.obj`) are **samples** bundled for reproduction
(tracked via Git LFS). The full dataset of fractured solids is produced by the
brittle-fracture simulation in our DeepFracture / TEBP project and cooked into
far-from-boundary fields:

- Fracture simulation: [`TEBP-DeepFracture/01.Data-generation`](https://github.com/nikoloside/TEBP-DeepFracture/tree/main/01.Data-generation)
- FFBDF cooking (the encoding this paper builds on): [`TEBP-DeepFracture/02.CookData/create_input_output.py`](https://github.com/nikoloside/TEBP-DeepFracture/blob/main/02.CookData/create_input_output.py)

## License

Released under the [MIT License](LICENSE) (non-commercial use permitted). All
dependencies are free for academic / research use.

## Citation

```bibtex
@inproceedings{huang2026ffb,
  author    = {Huang, Yuhang and Kanai, Takashi},
  title     = {Far-From-Boundary Fields for Learning Segmented Implicit Solids},
  booktitle = {Shape Modeling International (SMI)},
  year      = {2026}
}
```

## Acknowledgements

- Shared fracture data + runtime: [TEBP / DeepFracture](https://github.com/nikoloside/TEBP-DeepFracture).
- Thanks to the Graphics Replicability Stamp Initiative (GRSI).
