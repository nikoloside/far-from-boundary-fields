# 1. UDF Baseline

UDF（Unsigned Distance Field）ベースライン比較。同じ MLP 構成で UDF を学習し、GS-SDF / FFB-DF と定性的・定量的に比較。

## UDF 比較対象

| 手法 | リポジトリ |
|------|-----------|
| **NeuralUDF** | https://github.com/xxlong0/NeuralUDF |
| **MIND** | https://github.com/jjjkkyz/MIND |

## エンコーディング生成（学習用データ）

FFB-DF と同様のサンプリング・出力形式で UDF を生成：

| 方式 | スクリプト | 出力先 |
|------|-----------|--------|
| UDF from mesh (GT) | `python src/encoder_udf_mesh.py` | `data/npz-udf/` |
| FFB-DF (SDF) | `python src/encoder_ffb-df_mlp.py` | `data/npz-resample/` |
| NeuralUDF | `python experiments/udf_baseline/encode_neuraludf.py --exp_dir <path> --npz <points.npz> --out <out.npz>` | `data/npz-udf-neuraludf/` |
| MIND | `encode_mind.py`（query_func 実装要） | `data/npz-udf-mind/` |

統一フォーマット：`npz(poisson_grid_points, udf_values)` または `(poisson_grid_points, sdf_values)`

## 指標

- **Global**: CD, 対称 MFCD
- **Fragment-wise**: 小フラグメントの再現率（per-fragment CD, IoU）
- **Boundary**: 内部境界のリコール

## 定性比較

GT vs GS-SDF vs UDF (NeuralUDF/MIND) vs FFB-DF（特に細いクラックと小片）

## 実行方法

```bash
# フルパイプライン（エンコード → 学習 → 可視化）
./scripts/run_full_pipeline.sh

# 最小限テスト（--minimal: エンコーダは高速だが粗い）
./scripts/run_full_pipeline.sh --minimal
```

## 出力

- `data/results/udf_baseline/qualitative/*.png` — 点群可視化（FFB, UDF mesh, FFB-MLP, UDF-MLP）
- FFB-MLP / UDF-MLP は同一 UDF アーキテクチャ（MIND・NeuralUDF 互換）で学習

## 結果出力先

`data/results/udf_baseline/`
