# Focused Experiment Design - Exp 1

**更新日期**: 2026-03-03
**核心**: FFB-MLP + Flooding Algorithm

---

## 🎯 核心提案

### 主要方法（Our Method）

**FFB-MLP + Flooding Algorithm** ⭐

- **编码**: Fragment-aware Boundary DF (FFB)
  - 内部归一化到[-1,0]
  - 外部保持原距离[0,+∞)
  - 适合破碎场景

- **网络**: Simple MLP
  - 4层，128维
  - Multires=4
  - ~37K参数

- **后处理**: Flooding (Watershed) + Marching Cubes
  - 基于ImageJ watershed分割
  - Fragment-aware
  - 已有实现: `src/FFB-MLP_VQ-MLP/vc_plot_objs.py`

---

## 🔬 对比实验（Ablation & Baselines）

### 必要的对比方法

| 方法 | 用途 | 优先级 |
|------|------|--------|
| **UDF-MLP + Flooding** | 验证FFB编码的优势 | 🔴 高 |
| **FFB-MLP + MIND** | 验证Flooding的优势 | 🔴 高 |
| **NeuralUDF + Flooding** | SOTA baseline | 🟡 中 |
| ~~FFB-MLP + M3C~~ | （可选，不是核心） | ⚪ 低 |

---

## 📊 简化的实验矩阵

### 核心对比（4种方法）

```
                    Flooding        MIND
                    (Our)           (Baseline)
    ┌─────────────────────────────────────────┐
FFB │ FFB+Flood ⭐  │ FFB+MIND      │ ← 对比后处理
    │ (Our Method)   │ (对比)        │
    ├─────────────────────────────────────────┤
UDF │ UDF+Flood      │ (不需要)      │ ← 对比编码
    │ (对比)         │               │
    ├─────────────────────────────────────────┤
NeU │ NeU+Flood      │ (不需要)      │ ← SOTA baseline
    │ (可选)         │               │
    └─────────────────────────────────────────┘
```

**实际需要**:
1. ✅ **FFB-MLP + Flooding** (主要方法，已有)
2. ⏳ **UDF-MLP + Flooding** (对比编码)
3. ⏳ **FFB-MLP + MIND** (对比后处理，已实现)
4. ⚪ **NeuralUDF + Flooding** (可选，SOTA)

---

## 🎓 实验目标

### 问题1: FFB编码是否优于UDF？

**对比组**:
```
FFB-MLP + Flooding  vs  UDF-MLP + Flooding
     ↑                       ↑
  (Our)                  (Baseline)
```

**固定**:
- 网络架构（4层128维）
- 后处理（Flooding）

**变量**: 编码方式

**预期结论**: FFB的内部归一化更适合破碎物体

---

### 问题2: Flooding是否优于MIND？

**对比组**:
```
FFB-MLP + Flooding  vs  FFB-MLP + MIND
     ↑                       ↑
  (Our)                  (Baseline)
```

**固定**:
- 网络架构（4层128维）
- 编码（FFB）

**变量**: 后处理方法

**预期结论**: Flooding更快且更适合fragment场景

---

### 问题3: 相比SOTA如何？

**对比组**:
```
FFB-MLP + Flooding  vs  NeuralUDF + Flooding
     ↑                       ↑
  (Our)                  (SOTA+Our后处理)
```

**固定**: 后处理（Flooding）

**变量**: 编码+网络架构

**预期结论**: 我们的方法轻量且效果好

---

## 🛠️ 实现优先级

### Phase 1: 核心对比 ⏳ (必须)

**需要实现**:
```bash
# 统一的Flooding抽取脚本
src/extract_mesh_flooding.py
```

**支持**:
- ✅ FFB-MLP (已有代码可提取)
- ⏳ UDF-MLP (新增)
- ⏳ NeuralUDF (新增，可选)

**输出**:
```bash
# FFB-MLP + Flooding (主要方法)
python src/extract_mesh_flooding.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mlp_flooding.ply

# UDF-MLP + Flooding (对比编码)
python src/extract_mesh_flooding.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_mlp_flooding.ply
```

---

### Phase 2: 评估对比 ⏳ (必须)

**指标**:
- Symmetric MFCD
- Fragment-wise Recall
- Boundary Recall
- Inference + Extraction Time

**输出**:
```bash
# 对比4种方法
python scripts/compare_methods.py \
    --methods ffb_flooding udf_flooding ffb_mind neuraludf_flooding \
    --output data/results/comparison/
```

---

### Phase 3: 可视化 ⏳ (必须)

**Figure 1**: 定量对比
```
条形图：SymMFCD, Fragment Recall, Boundary Recall, Time
4种方法横向对比
```

**Figure 2**: 定性对比
```
2×2 grid：
- FFB+Flood (Our)
- UDF+Flood (编码对比)
- FFB+MIND (后处理对比)
- NeU+Flood (SOTA)
```

**Figure 3**: Ablation分析
```
展示FFB编码和Flooding后处理的独立贡献
```

---

## 📋 完整Pipeline（简化版）

### 训练阶段

```bash
# 1. 数据编码
python src/encoder_ffb-df_mlp.py    # FFB
python src/encoder_udf_mesh.py      # UDF

# 2. 模型训练
python src/train_ffb_mlp.py --epochs 30      # 核心方法
python src/train_udf_mlp.py --epochs 30      # 对比
python src/train_neuraludf_mlp.py --epochs 100  # SOTA (可选)
```

### 推理阶段

```bash
# 3. Mesh抽取
# 核心方法
python src/extract_mesh_flooding.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mlp_flooding.ply

# 对比：编码
python src/extract_mesh_flooding.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_mlp_flooding.ply

# 对比：后处理
python src/extract_mesh_with_mind.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mlp_mind.ply

# 对比：SOTA
python src/extract_mesh_flooding.py \
    --model_type neuraludf_mlp \
    --ckpt data/ckpts/neuraludf_mlp/neuraludf_mlp.pth \
    --output data/results/meshes/neuraludf_flooding.ply
```

### 评估阶段

```bash
# 4. 指标计算
python experiments/mfcd_definition/symmetric_mfcd.py \
    --batch \
    --orig-dir data/original_meshes \
    --recon-dirs \
        ffb_flood:data/results/meshes \
        udf_flood:data/results/meshes \
        ffb_mind:data/results/meshes \
        neu_flood:data/results/meshes \
    --output-dir data/results/comparison/mfcd

# 5. 生成对比图表
python scripts/visualize_comparison.py \
    --input data/results/comparison/ \
    --output data/results/figures/
```

---

## 📊 预期结果表格

### Table 1: 定量对比

| Method | SymMFCD ↓ | Fragment Recall ↑ | Boundary Recall ↑ | Time ↓ |
|--------|-----------|-------------------|-------------------|--------|
| **FFB+Flood** ⭐ | **0.012** | **0.95** | **0.92** | **2min** |
| UDF+Flood | 0.015 | 0.92 | 0.89 | 2min |
| FFB+MIND | 0.011 | 0.96 | 0.93 | 45min |
| NeU+Flood | 0.013 | 0.94 | 0.91 | 3min |

**结论**:
- ✅ FFB编码优于UDF（行1 vs 行2）
- ✅ Flooding速度远快于MIND，质量相近（行1 vs 行3）
- ✅ 轻量方法接近SOTA（行1 vs 行4）

---

## 🔗 与Exp 2-5的关系

### Exp 1: UDF Baseline ⭐ (本实验)

**核心**: FFB-MLP + Flooding
**对比**: UDF, MIND, NeuralUDF

### Exp 2: Training Trick Ablation

**测试对象**: FFB-MLP, UDF-MLP
**变量**: 采样策略，损失函数

### Exp 3: Activation Ablation

**测试对象**: FFB-MLP, UDF-MLP
**变量**: ReLU, Softplus, SIREN

### Exp 4: MFCD Definition ✅

**已完成**: Symmetric MFCD定义和实现

### Exp 5: Voxel vs Implicit

**对比**: Voxel CNN vs Implicit MLP

---

## 📝 论文叙述结构

### Abstract

提出FFB-MLP + Flooding for破碎物体重建

### Method

1. **FFB编码**: 为什么适合破碎场景
2. **Simple MLP**: 轻量架构
3. **Flooding后处理**: Fragment-aware抽取

### Experiments

1. **编码对比**: FFB vs UDF
2. **后处理对比**: Flooding vs MIND
3. **SOTA对比**: Our vs NeuralUDF
4. **Ablation**: 详细的Exp 2-5

### Results

- 定量：Table 1
- 定性：Figure 2
- Ablation：Figure 3

---

## ⏱️ 时间估计（简化版）

### 实现时间

| 任务 | 时间 |
|------|------|
| 提取flooding核心逻辑 | 30分钟 |
| 创建统一抽取脚本 | 1小时 |
| 测试UDF/NeuralUDF适配 | 30分钟 |
| 评估脚本 | 1小时 |
| 可视化脚本 | 1小时 |
| **总计** | **4小时** |

### 运行时间

| 阶段 | 时间 |
|------|------|
| 训练（已完成） | ~2小时 |
| Flooding抽取（4个方法） | ~10分钟 |
| MIND抽取（1个方法） | ~45分钟 |
| 评估 | ~10分钟 |
| **总计** | **~1小时** |

---

## 🚀 立即行动

### 优先级1: 创建Flooding统一脚本 🔴

**目标**:
```bash
src/extract_mesh_flooding.py
```

**提取来源**: `src/FFB-MLP_VQ-MLP/vc_plot_objs.py`

**核心逻辑**:
1. 从模型推理volume (256³)
2. Watershed分割
3. Marching cubes抽取
4. 后处理

---

### 优先级2: 运行对比实验 🟡

**4种方法**:
1. FFB-MLP + Flooding ⭐
2. UDF-MLP + Flooding
3. FFB-MLP + MIND
4. NeuralUDF + Flooding (可选)

---

### 优先级3: 生成论文结果 🟢

**输出**:
- 定量表格
- 定性对比图
- Ablation分析

---

## 💡 重要简化

### 不需要实现的

- ❌ 所有M3C相关（不是核心）
- ❌ UDF-MLP + MIND（不必要）
- ❌ NeuralUDF + MIND（不必要）
- ❌ 完整9×9矩阵

### 只需要实现的

- ✅ Flooding统一脚本
- ✅ 4种核心方法的对比
- ✅ 基本的评估和可视化

---

**文档版本**: v2.0 - 简化聚焦版
**创建日期**: 2026-03-03
**核心方法**: FFB-MLP + Flooding
**必要对比**: 4种方法（不是9种）
