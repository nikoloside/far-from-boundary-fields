# Breaking Workingspace (FFB-DF 論文改訂用)

実験タスク管理と結果整理用リポジトリ。

## 構成

```
├── docs/
│   └── TODO.md           # 論文改訂用実験 TODO
├── experiments/
│   ├── udf_baseline/
│   ├── training_trick_ablation/
│   ├── activation_ablation/
│   ├── mfcd_definition/
│   └── voxel_ablation/
├── data/
│   └── results/          # 各実験の出力先
│       ├── udf_baseline/
│       ├── training_trick_ablation/
│       ├── activation_ablation/
│       ├── mfcd_definition/
│       └── voxel_ablation/
└── scripts/
    └── aggregate_results.py  # 結果集約スクリプト
```

## 使い方

1. `docs/TODO.md` のチェックリストに従って実験を進める
2. 各実験の結果を `data/results/<experiment_name>/` に出力
3. `python scripts/aggregate_results.py` で `data/results/aggregated_table.json` を生成
