# 実験実行ログ (Experiments Log)

全実験の実行ログを記録。`python scripts/run_all_experiments.py` または各 `experiments/expN_*/run.py` 実行時に追記される。

---
[2026-02-28 01:02:53] === exp1_udf_baseline START ===
[2026-02-28 01:02:53] 
--- exp1_udf_baseline | Step: 1. Encode GS-SDF / FFB-DF ---
$ python src/encoder_ffb-df_mlp.py --minimal
[2026-02-28 01:04:59] === exp4_mfcd_definition START ===
[2026-02-28 01:05:00] Toy SymMFCD: Case1=0.3625 Case2=0.5710
[2026-02-28 01:05:00] Saved /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/mfcd_definition/mfcd_toy_results.json
[2026-02-28 01:05:00] === exp4_mfcd_definition END ===
[2026-02-28 01:05:00] === exp1_udf_baseline START ===
[2026-02-28 01:05:00] 
--- exp1_udf_baseline | Step: 1. Train FFB-MLP ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 30
[2026-02-28 01:05:37] Exit code: 0
[2026-02-28 01:05:37] 
--- exp1_udf_baseline | Step: 2. Train UDF-MLP ---
$ python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 30
[2026-02-28 01:06:40] ============================================================
[2026-02-28 01:06:40] RUN ALL EXPERIMENTS START
[2026-02-28 01:06:41] ============================================================
[2026-02-28 01:06:41] 
>>> Experiment 1/5: experiments/exp1_udf_baseline/run.py
[2026-02-28 01:06:41] === exp1_udf_baseline START ===
[2026-02-28 01:06:41] 
--- exp1_udf_baseline | Step: 1. Train FFB-MLP ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 30
[2026-02-28 01:07:16] Exit code: 0
[2026-02-28 01:07:16] 
--- exp1_udf_baseline | Step: 2. Train UDF-MLP ---
$ python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 30
[2026-02-28 01:13:33] Exit code: 0
[2026-02-28 01:13:33] 
--- exp1_udf_baseline | Step: 3. Visualize ---
$ python scripts/generate_and_render.py
[2026-02-28 01:14:51] Exit code: 0
[2026-02-28 01:14:51] === exp1_udf_baseline END ===
[2026-02-28 01:15:15] Exit code: 0
[2026-02-28 01:15:15] 
--- exp1_udf_baseline | Step: 3. Visualize ---
$ python scripts/generate_and_render.py
[2026-02-28 01:16:19] Exit code: 0
[2026-02-28 01:16:19] === exp1_udf_baseline END ===
[2026-02-28 01:16:19] Exit code: 0
[2026-02-28 01:16:19] 
>>> Experiment 2/5: experiments/exp2_training_trick_ablation/run.py
[2026-02-28 01:16:19] === exp2_training_trick_ablation START ===
[2026-02-28 01:16:19]   Condition: GS-uniform-L2 (GS-SDF, uniform, weighted=False)
[2026-02-28 01:16:19] 
--- exp2_training_trick_ablation | Step: GS-uniform-L2 ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 15 --ckpt_dir /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/training_trick_ablation/ckpt_GS_uniform_L2
[2026-02-28 01:16:25] Exit code: 0
[2026-02-28 01:16:25]   Condition: GS-NB-L2 (GS-SDF, near_boundary, weighted=False)
[2026-02-28 01:16:25] 
--- exp2_training_trick_ablation | Step: GS-NB-L2 ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 15 --ckpt_dir /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/training_trick_ablation/ckpt_GS_NB_L2
[2026-02-28 01:16:30] Exit code: 0
[2026-02-28 01:16:30]   Condition: GS-NB-weighted-L2 (GS-SDF, near_boundary, weighted=True)
[2026-02-28 01:16:30] 
--- exp2_training_trick_ablation | Step: GS-NB-weighted-L2 ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 15 --ckpt_dir /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/training_trick_ablation/ckpt_GS_NB_weighted_L2
[2026-02-28 01:16:38] Exit code: 0
[2026-02-28 01:16:38]   Condition: FFB-uniform-L2 (FFB-DF, uniform, weighted=False)
[2026-02-28 01:16:38] 
--- exp2_training_trick_ablation | Step: FFB-uniform-L2 ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 15 --ckpt_dir /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/training_trick_ablation/ckpt_FFB_uniform_L2
[2026-02-28 01:17:02] Exit code: 0
[2026-02-28 01:17:02]   Condition: FFB-NB-L2 (FFB-DF, near_boundary, weighted=False)
[2026-02-28 01:17:02] 
--- exp2_training_trick_ablation | Step: FFB-NB-L2 ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 15 --ckpt_dir /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/training_trick_ablation/ckpt_FFB_NB_L2
[2026-02-28 01:17:23] Exit code: 0
[2026-02-28 01:17:23] Saved /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/training_trick_ablation/ablation_results.json
[2026-02-28 01:17:23] === exp2_training_trick_ablation END ===
[2026-02-28 01:17:23] Exit code: 0
[2026-02-28 01:17:23] 
>>> Experiment 3/5: experiments/exp3_activation_ablation/run.py
[2026-02-28 01:17:24] === exp3_activation_ablation START ===
[2026-02-28 01:17:24]   Activation: ReLU
[2026-02-28 01:17:24] 
--- exp3_activation_ablation | Step: ReLU ---
$ python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 15 --ckpt_dir /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/activation_ablation/ckpt_ReLU
[2026-02-28 01:21:07] Exit code: 0
[2026-02-28 01:21:07]   Activation: Softplus
[2026-02-28 01:21:07] 
--- exp3_activation_ablation | Step: Softplus ---
$ python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 15 --ckpt_dir /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/activation_ablation/ckpt_Softplus
[2026-02-28 01:24:37] Exit code: 0
[2026-02-28 01:24:37]   Activation: SIREN
[2026-02-28 01:24:37] 
--- exp3_activation_ablation | Step: SIREN ---
$ python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 15 --ckpt_dir /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/activation_ablation/ckpt_SIREN
[2026-02-28 01:27:37] Exit code: 0
[2026-02-28 01:27:37] Saved /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/activation_ablation/activation_results.json
[2026-02-28 01:27:37] === exp3_activation_ablation END ===
[2026-02-28 01:27:37] Exit code: 0
[2026-02-28 01:27:37] 
>>> Experiment 4/5: experiments/exp4_mfcd_definition/run.py
[2026-02-28 01:27:37] === exp4_mfcd_definition START ===
[2026-02-28 01:27:37] Toy SymMFCD: Case1=0.3625 Case2=0.5710
[2026-02-28 01:27:37] Saved /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/mfcd_definition/mfcd_toy_results.json
[2026-02-28 01:27:37] === exp4_mfcd_definition END ===
[2026-02-28 01:27:37] Exit code: 0
[2026-02-28 01:27:37] 
>>> Experiment 5/5: experiments/exp5_voxel_ablation/run.py
[2026-02-28 01:27:37] === exp5_voxel_ablation START ===
[2026-02-28 01:27:37] 
--- exp5_voxel_ablation | Step: Implicit FFB-MLP ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 20
[2026-02-28 01:28:04] Exit code: 0
[2026-02-28 01:28:04] 
--- exp5_voxel_ablation | Step: Implicit UDF-MLP ---
$ python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 20
[2026-02-28 01:32:58] Exit code: 0
[2026-02-28 01:32:58] Saved /Users/yuhanghuang/Workspaces/01.research-proj/breaking-workingspace/data/results/voxel_ablation/voxel_comparison.json
[2026-02-28 01:32:58] === exp5_voxel_ablation END ===
[2026-02-28 01:32:58] Exit code: 0
[2026-02-28 01:32:58] 
============================================================
[2026-02-28 01:32:58] RUN ALL EXPERIMENTS END
[2026-02-28 01:32:58] ============================================================
[2026-02-28 01:47:50] Exit code: 0
[2026-02-28 01:47:50] 
--- exp1_udf_baseline | Step: 2. Encode UDF ---
$ python src/encoder_udf_mesh.py --minimal
[2026-02-28 01:48:20] Exit code: 0
[2026-02-28 01:48:20] 
--- exp1_udf_baseline | Step: 3. Train FFB-MLP ---
$ python src/train_ffb_mlp.py --npz_dir data/npz-resample --epochs 30
[2026-02-28 01:48:25] Exit code: 0
[2026-02-28 01:48:25] 
--- exp1_udf_baseline | Step: 4. Train UDF-MLP ---
$ python src/train_udf_mlp.py --npz_dir data/npz-udf --epochs 30
[2026-02-28 01:48:33] Exit code: 0
[2026-02-28 01:48:33] 
--- exp1_udf_baseline | Step: 5. Visualize ---
$ python scripts/generate_and_render.py
[2026-02-28 01:48:56] Exit code: 0
[2026-02-28 01:48:56] === exp1_udf_baseline END ===
