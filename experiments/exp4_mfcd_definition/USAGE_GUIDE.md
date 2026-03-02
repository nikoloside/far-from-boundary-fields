# Symmetric MFCD - Usage Guide

**创建日期**: 2026-03-02
**目的**: 对称MFCD的完整使用指南

---

## 快速开始

### 1. 运行玩具示例 (5分钟)

**目的**: 理解单向和对称MFCD的区别

```bash
cd experiments/mfcd_definition/
python toy_example_sym_mfcd.py
```

**输出**:
- 4个场景的数值对比
- 可视化保存在 `toy_examples/`

---

### 2. 比较两个mesh (1分钟)

```bash
python symmetric_mfcd.py \
    --orig path/to/original.obj \
    --recon path/to/reconstructed.obj \
    --num-samples 5000 \
    --output-dir results/
```

---

### 3. 批量对比多个方法 (Exp 1使用)

```bash
python symmetric_mfcd.py \
    --batch \
    --orig-dir data/original_meshes/ \
    --recon-dirs \
        udf_mlp:data/results/meshes/udf_mlp/ \
        ffb_mlp:data/results/meshes/ffb_mlp/ \
        neuraludf:data/results/meshes/neuraludf/ \
    --output-dir results/mfcd_comparison/ \
    --num-samples 10000
```

---

## 核心概念

### 单向MFCD的问题

**原始实现** (`vc_plot_charmer_distance_error.py`):
- ✅ Chamfer Distance是双向的
- ❌ Fragment匹配是单向的（只计算Orig→Recon）

**缺陷**: 无法检测多余碎片！

### Symmetric MFCD的解决

```
SymMFCD(A, B) = MFCD(A→B) + MFCD(B→A)
```

**优势**:
- ✅ 检测缺失碎片（A→B高）
- ✅ 检测多余碎片（B→A高）
- ✅ 对称性
- ✅ 符合CD的定义

---

## 玩具示例结果

### Scenario 1: 缺失碎片
```
原始: 5个碎片
重建: 3个碎片

One-dir MFCD: 高（缺失2个）
SymMFCD:
  MFCD(O→R): 高 ← 缺失碎片
  MFCD(R→O): 低 ← 所有重建都有对应
```

### Scenario 2: 多余碎片 (关键！)
```
原始: 3个碎片
重建: 5个碎片（2个噪声）

One-dir MFCD: 低 ← 错过问题！
SymMFCD:
  MFCD(O→R): 低 ← 原始都有对应
  MFCD(R→O): 高 ← 噪声碎片无对应
```

---

## 批量对比输出

### 1. 控制台总结

```
SUMMARY
========================================
udf_mlp:
  SymMFCD: 0.012345
  MFCD (Orig→Recon): 0.006789
  MFCD (Recon→Orig): 0.005556

ffb_mlp:
  SymMFCD: 0.010234
  MFCD (Orig→Recon): 0.005123
  MFCD (Recon→Orig): 0.005111
```

### 2. JSON文件

`symmetric_mfcd_results.json`:
```json
{
  "method_name": {
    "per_mesh": [...],
    "symmetric_mfcd_mean": 0.012345,
    "mfcd_orig_to_recon_mean": 0.006789,
    "mfcd_recon_to_orig_mean": 0.005556
  }
}
```

### 3. 可视化图表

`symmetric_mfcd_comparison.png`:
- 3个子图并排
- SymMFCD / MFCD(O→R) / MFCD(R→O)

---

## 实验应用

### Exp 1: UDF Baseline

**评估方法**: UDF-MLP, FFB-MLP, NeuralUDF, +MIND

```bash
python experiments/mfcd_definition/symmetric_mfcd.py \
    --batch \
    --orig-dir data/original_meshes/ \
    --recon-dirs \
        udf_mlp:experiments/udf_baseline/results/meshes/udf_mlp/ \
        ffb_mlp:experiments/udf_baseline/results/meshes/ffb_mlp/ \
        neuraludf:experiments/udf_baseline/results/meshes/neuraludf/ \
        udf_mind:experiments/udf_baseline/results/meshes/udf_mlp_mind/ \
        neuraludf_mind:experiments/udf_baseline/results/meshes/neuraludf_mind/ \
    --output-dir experiments/udf_baseline/results/mfcd/
```

### Exp 4: MFCD定义验证

**验证对称性**:

```bash
python toy_example_sym_mfcd.py
# 检查: SymMFCD(A,B) = SymMFCD(B,A)
```

---

## 重要文件

| 文件 | 用途 |
|------|------|
| `symmetric_mfcd.py` | 主实现 |
| `SYMMETRIC_MFCD_EXPLANATION.md` | 详细说明 |
| `toy_example_sym_mfcd.py` | 玩具示例 |
| `USAGE_GUIDE.md` | 本文档 |
| `vc_plot_charmer_distance_error.py` | 原始单向实现 |

---

## 建议

### 论文报告

报告3个指标:
1. **SymMFCD** (主要指标)
2. **MFCD(Orig→Recon)** (缺失碎片影响)
3. **MFCD(Recon→Orig)** (多余碎片影响)

### 分析方法

- 如果 MFCD(O→R) > MFCD(R→O): 缺失碎片
- 如果 MFCD(R→O) > MFCD(O→R): 多余碎片
- 如果两者接近: 整体偏移或平衡的错误

---

**完成日期**: 2026-03-02
**状态**: ✅ 可用
