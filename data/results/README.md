# 実験結果の格納先

各実験の数値結果・ログ・可視化をここに格納。

## ディレクトリ構成

```
data/
├── encodings/              # UDF/SDF 事前計算エンコーディング（学習用）
│   ├── npz-resample/       # FFB-DF (encoder_ffb-df_mlp.py)
│   ├── npz-udf/           # UDF from mesh (encoder_udf_mesh.py)
│   ├── npz-udf-neuraludf/ # UDF from NeuralUDF model
│   └── npz-udf-mind/      # UDF from MIND model
│
└── results/
    ├── udf_baseline/           # CD, SymMFCD, 小フラグメント指標, 定性比較
    ├── training_trick_ablation/# 5条件比較（テーブル, Loss曲線, 定性可視化）
    ├── activation_ablation/    # ReLU vs Softplus vs SIREN
    ├── mfcd_definition/        # MFCD vs SymMFCD 定義, トイ例
    └── voxel_ablation/         # GS-SDF vs FFB-DF × Voxel/Implicit
```

## NPZ フォーマット（統一）

- `poisson_grid_points`: (N, 3) サンプル点
- `sdf_values` または `udf_values`: (N,) 距離値

## 推奨ファイル形式（results/）

各実験フォルダ内：

- `metrics.csv` / `metrics.json` — 定量指標
- `loss_curves/` — 学習曲線（PNG 等）
- `qualitative/` — 可視化画像・メッシュ
- `config.yaml`（任意） — 実験設定の記録
