# TODO と進行状況（更新版）

**更新日**: 2026-03-02
**更新理由**: FFB編码方式的澄清 + NeuralUDF/MIND対比追加

---

## 🎯 3つの表現 (Representations)

| ID | 表現 | 説明 | 実装状況 |
|----|------|------|---------|
| **FFB-DF** | Fragment-aware Boundary DF | 内部正規化(÷max_dist)、外部原距離<br>**重要**: 標準SDFではなく混合エンコード | ✅ 実装済 |
| **純UDF** | Pure Unsigned Distance Field | 常に非負、内外区別なし | ✅ 実装済 |
| **GS-SDF** | Global Signed Distance Field | 単一SDF、境界非考慮（参考用） | ⚠️ 現在はFFB実装 |

**重要な発見**:
- 現在の`encoder_ffb-df_mlp.py`が生成する`sdf_values`は標準SDFではない
- FFB編码 = 内部正規化([-1,0]) + 外部原距離([0,+∞))
- これは意図的な設計で、fragment-basedシナリオに適している

---

## 📋 5実験一覧（更新版）

| # | 実験 | フォルダ | 主な変更点 | 状態 |
|---|------|----------|-----------|------|
| **1** | **UDF/FFB/NeuralUDF/MIND 対比** | `experiments/exp1_udf_baseline/` | **大幅更新**：NeuralUDF訓練+MIND追加 | ⏳ 更新中 |
| **2** | Training Trick Ablation | `experiments/exp2_training_trick_ablation/` | 維持 | ✅ 完了 |
| **3** | Activation Ablation | `experiments/exp3_activation_ablation/` | 維持 | ✅ 完了 |
| **4** | MFCD 対称版定義 | `experiments/exp4_mfcd_definition/` | 維持 | ✅ 完了 |
| **5** | Voxel vs Implicit | `experiments/exp5_voxel_ablation/` | 維持 | ✅ 完了 |

---

## 📌 Exp 1: UDF/FFB/NeuralUDF/MIND 対比 (更新版)

### 目標

4種類の方法を対比：
1. **純UDF** (UDF-MLP, 4層, 128次元)
2. **正規化FFB** (FFB-MLP, 4層, 128次元)
3. **NeuralUDF** (完整架構, 6層, 256次元, skip+init+norm)
4. **UDF+MIND / NeuralUDF+MIND** (非流形mesh抽出)

### 対比軸

```
編码方式:
├─ 純UDF (無向距離場)
└─ 正規化FFB (fragment-based, 内部正規化)

ネットワーク架構:
├─ 簡化MLP (4層, 128次元, multires=4)
└─ 完整NeuralUDF (6層, 256次元, multires=6, skip+init+norm)

後処理:
├─ 無 (直接抽出)
└─ MIND優化 (非流形対応)
```

### 実装タスク

#### ✅ 已完成

- [x] FFB-DF エンコード: `data/npz-resample/` (正規化FFB)
- [x] UDF エンコード: `data/npz-udf/` (純UDF)
- [x] FFB-MLP 訓練: `data/ckpts/ffb_mlp/` (4層, 128次元)
- [x] UDF-MLP 訓練: `data/ckpts/udf_mlp/` (4層, 128次元)
- [x] 基本可視化

#### ⏳ 追加実装必要

**1. NeuralUDF-MLP 訓練**
```bash
# 新規スクリプト作成
src/train_neuraludf_mlp.py

# 訓練コマンド
python src/train_neuraludf_mlp.py --npz_dir data/npz-udf --epochs 100

# 出力
data/ckpts/neuraludf_mlp/neuraludf_mlp.pth
```

**2. MIND mesh抽出**
```bash
# 新規スクリプト作成
src/extract_mesh_with_mind.py

# 使用例
python src/extract_mesh_with_mind.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_mlp_mind.ply

python src/extract_mesh_with_mind.py \
    --model_type neuraludf_mlp \
    --ckpt data/ckpts/neuraludf_mlp/neuraludf_mlp.pth \
    --output data/results/meshes/neuraludf_mlp_mind.ply
```

**3. 定量評価**
```bash
# 指標計算スクリプト
scripts/compute_metrics.py

# 指標:
- Chamfer Distance (CD)
- 対称MFCD
- Fragment-wise 再現率
- Boundary recall
```

**4. 定性比較**
```bash
# 可視化スクリプト
scripts/visualize_comparison.py

# 対比:
- 細いクラック
- 小片保留
- 内部境界
```

### 実験マトリクス

| 方法 | 編码 | 架構 | 後処理 | 出力 | 状態 |
|------|------|------|--------|------|------|
| UDF-MLP | 純UDF | 簡化 | - | UDF関数 | ✅ |
| FFB-MLP | 正規化FFB | 簡化 | - | FFB関数 | ✅ |
| NeuralUDF-MLP | 純UDF | 完整 | - | UDF関数 | ⏳ |
| UDF-MLP + MIND | 純UDF | 簡化 | MIND | Mesh | ⏳ |
| NeuralUDF + MIND | 純UDF | 完整 | MIND | Mesh | ⏳ |
| (参考) FFB-MLP + MIND | 正規化FFB | 簡化 | MIND | Mesh | ⏳ |

### 重要な対比

**対比A: 編码方式の影響**
```
UDF-MLP vs FFB-MLP
→ 純UDF vs 正規化FFB
→ 架構同じ、編码のみ異なる
```

**対比B: ネットワーク架構の影響**
```
UDF-MLP vs NeuralUDF-MLP
→ 簡化 vs 完整
→ 編码同じ、架構のみ異なる
```

**対比C: MIND後処理の影響**
```
UDF-MLP vs UDF-MLP + MIND
NeuralUDF-MLP vs NeuralUDF-MLP + MIND
→ 非流形処理の効果
```

**対比D: 総合比較**
```
全方法 → 最適組合せを特定
```

---

## 📌 Exp 2: Training Trick Ablation (維持)

### 目標
採樣策略と損失関数の影響を調査

### 5条件

| ID | 名称 | 表現 | 採樣 | 重み |
|----|------|------|------|------|
| 1 | GS-uniform-L2 | GS-SDF | uniform | False |
| 2 | GS-NB-L2 | GS-SDF | near_boundary | False |
| 3 | GS-NB-weighted-L2 | GS-SDF | near_boundary | True |
| 4 | FFB-uniform-L2 | FFB-DF | uniform | False |
| 5 | FFB-NB-L2 | FFB-DF | near_boundary | False |

### 状態
- ✅ 実装済み
- ✅ 実験完了
- 結果: `data/results/training_trick_ablation/`

**更新不要**（既に完了）

---

## 📌 Exp 3: Activation Ablation (維持)

### 目標
活性化関数の影響を調査

### 3種類

| ID | 活性化関数 | 特徴 |
|----|-----------|------|
| 1 | ReLU | 標準、線形領域 |
| 2 | Softplus | 滑らか、微分可能 |
| 3 | SIREN | 周期的、高周波対応 |

### 状態
- ✅ 実装済み
- ✅ 実験完了
- 結果: `data/results/activation_ablation/`

**更新不要**（既に完了）

---

## 📌 Exp 4: MFCD 対称版定義 (維持)

### 目標
MFCDとSymMFCDの定義を明確化

### タスク
- [x] MFCD（一方向）定義
- [x] SymMFCD定義
- [x] トイ例: fragment欠落 vs 位置ずれ

### 状態
- ✅ 実装済み
- ✅ 実験完了
- 結果: `data/results/mfcd_definition/`

**更新不要**（既に完了）

---

## 📌 Exp 5: Voxel vs Implicit (維持)

### 目標
Voxel CNNとImplicit MLPを比較

### 比較

| 方法 | 表現 | モデル | 出力 |
|------|------|--------|------|
| Implicit FFB | FFB-DF | MLP | 連続場 |
| Implicit UDF | UDF | MLP | 連続場 |
| (将来) Voxel FFB | FFB-DF | CNN | 離散場 |
| (将来) Voxel UDF | UDF | CNN | 離散場 |

### 状態
- ✅ Implicit部分完了
- ⏳ Voxel部分は将来実装

**更新不要**（現状維持）

---

## 🚀 実行コマンド（更新版）

### 全実験実行
```bash
# 全実験を順に実行
python scripts/run_all_experiments.py

# 高速テスト
python scripts/run_all_experiments.py --minimal
```

### 個別実験実行

**Exp 1: UDF Baseline（更新版）**
```bash
# 従来の実験（既完了）
python experiments/exp1_udf_baseline/run.py

# 新規追加：NeuralUDF訓練
python src/train_neuraludf_mlp.py --epochs 100

# 新規追加：MIND mesh抽出
python src/extract_mesh_with_mind.py --model_type udf_mlp ...
python src/extract_mesh_with_mind.py --model_type neuraludf_mlp ...

# 新規追加：総合評価
python scripts/compare_all_methods.py
```

**Exp 2-5: 維持（変更なし）**
```bash
python experiments/exp2_training_trick_ablation/run.py
python experiments/exp3_activation_ablation/run.py
python experiments/exp4_mfcd_definition/run.py
python experiments/exp5_voxel_ablation/run.py
```

---

## 📊 データ・成果物（更新版）

### 既存データ ✅

| パス | 内容 | サイズ |
|------|------|--------|
| `data/npz-resample/` | FFB-DF エンコード (5 objects) | ~55MB |
| `data/npz-udf/` | UDF エンコード (5 objects) | ~44MB |
| `data/ckpts/ffb_mlp/` | FFB-MLP weights | ~150KB |
| `data/ckpts/udf_mlp/` | UDF-MLP weights | ~150KB |
| `data/results/udf_baseline/` | 基本可視化 | ~10MB |

### 新規生成予定 ⏳

| パス | 内容 | 予想サイズ |
|------|------|----------|
| `data/ckpts/neuraludf_mlp/` | NeuralUDF-MLP weights | ~2MB |
| `data/results/meshes/` | MIND抽出mesh (3-6個) | ~50MB |
| `data/results/comparison/` | 総合比較図 | ~20MB |
| `data/results/metrics/` | 定量指標CSV | ~1MB |

---

## ⏱️ 実装スケジュール

### Phase 1: NeuralUDF訓練 (30-60分)
- [ ] `src/train_neuraludf_mlp.py` 作成
- [ ] NeuralUDF-MLP訓練実行
- [ ] checkpoint保存確認

### Phase 2: MIND統合 (1-2時間)
- [ ] `src/extract_mesh_with_mind.py` 作成
- [ ] UDF-MLP + MIND実行
- [ ] NeuralUDF-MLP + MIND実行
- [ ] FFB-MLP + MIND実行（オプション）

### Phase 3: 評価スクリプト (30分)
- [ ] `scripts/compute_metrics.py` 作成
- [ ] `scripts/visualize_comparison.py` 作成
- [ ] 指標計算実行

### Phase 4: 文書化 (30分)
- [ ] 実験結果まとめ
- [ ] `experiments/exp1_udf_baseline/RESULTS.md` 更新
- [ ] 論文用figure生成

**総計**: 約3-4時間

---

## 📝 重要な注意点

### FFB編码の理解

**以前の誤解**:
- ❌ `sdf_values` = 標準SDF

**正しい理解**:
- ✅ `sdf_values` = FFB編码（混合編码）
  - 外部: 原距離（正値、無限界）
  - 内部: 正規化距離（負値、[-1,0]）

**影響**:
- FFB-MLPは「SDF MLP」という名前だが、実際は「FFB MLP」
- UDF-MLPとFFB-MLPの対比は「UDF vs SDF」ではなく「UDF vs FFB」
- これは破碎形状に適した意図的な設計

### 実験の公平性

**対比A (編码)**: 架構を統一
```
UDF-MLP vs FFB-MLP
→ 両方とも 4層, 128次元, multires=4
```

**対比B (架構)**: 編码を統一
```
UDF-MLP vs NeuralUDF-MLP
→ 両方とも純UDF、架構のみ異なる
```

**対比C (後処理)**: 基底方法を統一
```
Method vs Method + MIND
→ 同じ基底、後処理のみ異なる
```

---

## 🎯 最終目標

### 学術貢献

1. **FFB編码の有効性**: 正規化fragment-based fieldの優位性を示す
2. **架構の影響**: skip connection等の重要性を定量化
3. **MIND統合**: 非流形処理の効果を検証
4. **総合評価**: 破碎形状再構成の最適な組合せを特定

### 実用価値

- 破碎物体のデジタル化
- 考古学的遺物の復元
- 製造業の破損検出
- 衝撃シミュレーションの検証

---

**文書版本**: v2.0 - 大幅更新
**更新日**: 2026-03-02
**次のステップ**: Phase 1 (NeuralUDF訓練) から開始
