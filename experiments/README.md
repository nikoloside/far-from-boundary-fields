# 5 実験

3 表現 (GS-SDF, UDF, FFB-DF) を用いた 5 実験の実行コード。

## 実行

```bash
# 全実験を順に実行（ログは docs/experiments_log.md）
python scripts/run_all_experiments.py

# 高速テスト
python scripts/run_all_experiments.py --minimal

# 個別実行
python experiments/exp1_udf_baseline/run.py
python experiments/exp2_training_trick_ablation/run.py
python experiments/exp3_activation_ablation/run.py
python experiments/exp4_mfcd_definition/run.py
python experiments/exp5_voxel_ablation/run.py
```

## 実験一覧

| # | フォルダ | 説明 |
|---|----------|------|
| 1 | exp1_udf_baseline | GS-SDF / UDF / FFB-DF ベースライン比較 |
| 2 | exp2_training_trick_ablation | 5 条件 Training Trick Ablation |
| 3 | exp3_activation_ablation | ReLU / Softplus / SIREN |
| 4 | exp4_mfcd_definition | MFCD, SymMFCD 定義・トイ例 |
| 5 | exp5_voxel_ablation | Voxel vs Implicit |

## ログ・TODO

- `docs/experiments_log.md` — 実行ログ
- `docs/todo.md` — TODO と進行状況
