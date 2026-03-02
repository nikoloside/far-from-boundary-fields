# TODO と進行状況

## 3 つの表現 (Representations)

| ID | 表現 | 説明 |
|----|------|------|
| **GS-SDF** | Global Signed Distance Field | 単一 SDF、境界非考慮 |
| **UDF** | Unsigned Distance Field | NeuralUDF / MIND スタイル |
| **FFB-DF** | Fragment-aware Boundary DF | 境界考慮、多フラグメント正規化 |

---

## 5 実験一覧

| # | 実験 | フォルダ | 状態 |
|---|------|----------|------|
| 1 | UDF ベースライン | `experiments/exp1_udf_baseline/` | ⏳ 進行中 |
| 2 | Training Trick Ablation | `experiments/exp2_training_trick_ablation/` | [ ] |
| 3 | Activation Ablation | `experiments/exp3_activation_ablation/` | [ ] |
| 4 | MFCD 対称版定義 | `experiments/exp4_mfcd_definition/` | ✅ 実施済 |
| 5 | Voxel vs Implicit | `experiments/exp5_voxel_ablation/` | [ ] |

---

## 詳細 TODO

### Exp 1: UDF ベースライン
- [ ] GS-SDF / FFB-DF / UDF を同一 MLP で学習
- [ ] 指標：CD, SymMFCD, 小フラグメント再現率, 内部境界リコール
- [ ] 定性比較：細いクラック・小片
- [ ] NeuralUDF / MIND との比較

### Exp 2: Training Trick Ablation
- [ ] GS-uniform-L2
- [ ] GS-NB-L2
- [ ] GS-NB-weighted-L2
- [ ] FFB-uniform-L2
- [ ] FFB-NB-L2（提案）
- [ ] 各条件：CD, SymMFCD, Loss 曲線

### Exp 3: Activation Ablation
- [ ] Sigmoid 削除、線形正規化 FFB-DF
- [ ] ReLU / Softplus / SIREN 比較
- [ ] 学習安定性（Loss 曲線）

### Exp 4: MFCD 定義
- [ ] MFCD（一方向）明確化
- [ ] SymMFCD 定義
- [ ] トイ例：フラグメント欠落 vs 位置ずれ

### Exp 5: Voxel Ablation
- [ ] GS-SDF vs FFB-DF × {Voxel CNN, Implicit MLP}
- [ ] 小テーブル比較

---

## 実行コマンド

```bash
# 全実験を順に実行（ログは docs/experiments_log.md に追記）
python scripts/run_all_experiments.py

# 高速テスト（--minimal）
python scripts/run_all_experiments.py --minimal

# 個別実行
python experiments/exp1_udf_baseline/run.py
python experiments/exp2_training_trick_ablation/run.py
python experiments/exp3_activation_ablation/run.py
python experiments/exp4_mfcd_definition/run.py
python experiments/exp5_voxel_ablation/run.py
```

---

## データ・成果物

| パス | 内容 |
|------|------|
| `data/npz-resample/` | FFB-DF エンコーディング |
| `data/npz-udf/` | UDF エンコーディング |
| `data/ckpts/` | 学習済み MLP |
| `data/results/*/` | 各実験の出力 |
| `docs/experiments_log.md` | 実行ログ |
