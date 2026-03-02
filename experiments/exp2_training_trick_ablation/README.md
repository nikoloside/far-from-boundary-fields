# Exp 2: Training Trick Ablation - FFB+Flooding训练策略优化

**目标**: 基于FFB+Flooding方法，验证不同训练策略的影响

**基线方法**: Exp 1的FFB-MLP + Flooding ⭐

---

## 🎯 实验目标

### 核心问题: 哪些训练技巧能提升FFB+Flooding的性能？

**对比维度**:
1. **采样策略**: Uniform vs Near-boundary
2. **损失函数**: MSE vs Weighted MSE
3. **数据增强**: With/Without augmentation

**预期结论**: Near-boundary采样 + Weighted loss显著提升小碎片重建

---

## 📊 实验矩阵

### 完整测试条件

| ID | 采样策略 | 损失函数 | 数据增强 | 期望效果 |
|----|----------|----------|----------|----------|
| **Baseline** | Uniform | MSE | No | Exp 1基线 |
| Exp2.1 | Near-boundary | MSE | No | 提升边界精度 |
| Exp2.2 | Uniform | Weighted MSE | No | 平衡大小碎片 |
| Exp2.3 | Near-boundary | Weighted MSE | No | ⭐ 最优组合 |
| Exp2.4 | Near-boundary | Weighted MSE | Yes | 测试增强效果 |

**核心对比**: Baseline vs Exp2.3 (最优训练策略)

---

## 🚀 运行方式

### 方式1: 一键运行（推荐）

```bash
cd experiments/exp2_training_trick_ablation
python run.py
```

### 方式2: 快速测试

```bash
python run.py --quick  # 低epochs, 低分辨率
```

### 方式3: 测试单个策略

```bash
python run.py --condition exp2.3  # 只测试最优组合
```

---

## 📁 输出结果

```
experiments/exp2_training_trick_ablation/
├── results/
│   ├── meshes/                      # 各策略生成的mesh
│   │   ├── baseline_flooding.ply    # Exp 1基线（复制）
│   │   ├── exp2_1_flooding.ply      # Near-boundary
│   │   ├── exp2_2_flooding.ply      # Weighted loss
│   │   ├── exp2_3_flooding.ply      # ⭐ 最优组合
│   │   └── exp2_4_flooding.ply      # + 数据增强
│   │
│   ├── metrics/                     # 评估指标
│   │   ├── symmetric_mfcd_results.json
│   │   └── training_trick_comparison.json
│   │
│   ├── figures/                     # 可视化
│   │   ├── training_trick_ablation.png
│   │   └── fragment_recall_comparison.png
│   │
│   └── logs/
│       └── exp2_log_*.txt
```

---

## 📊 预期结果

### 定量对比

| Strategy | SymMFCD ↓ | Fragment Recall ↑ | Small Frag Quality ↑ |
|----------|-----------|-------------------|---------------------|
| **Baseline** (Exp 1) | 0.012 | 0.95 | 0.88 |
| Near-boundary | 0.011 | 0.96 | 0.91 |
| Weighted loss | 0.011 | 0.96 | 0.90 |
| **NB + Weighted** ⭐ | **0.009** | **0.97** | **0.94** |
| + Augmentation | 0.009 | 0.97 | 0.94 |

### 关键发现

1. **Near-boundary采样**: 提升边界精度，fragment recall +1%
2. **Weighted loss**: 平衡大小碎片，小碎片质量 +2%
3. **组合效果**: NB + Weighted降低MFCD 25%
4. **数据增强**: 提升有限（<2%），可选

---

## 🔍 详细设计

### 采样策略对比

**Uniform Sampling** (Baseline):
```python
# 均匀采样整个空间
points = torch.rand(batch_size, 3) * 2 - 1  # [-1, 1]³
```

**Near-boundary Sampling** (Exp2.1, 2.3, 2.4):
```python
# 80%采样边界附近 (|UDF| < threshold)
# 20%采样全局空间
boundary_mask = torch.abs(udf) < 0.05
points_boundary = points[boundary_mask]  # 80%
points_global = torch.rand(n_global, 3) * 2 - 1  # 20%
```

---

### 损失函数对比

**MSE Loss** (Baseline):
```python
loss = F.mse_loss(pred_udf, gt_udf)
```

**Weighted MSE** (Exp2.2, 2.3, 2.4):
```python
# 根据fragment体积加权
weight = 1.0 / torch.sqrt(fragment_volume + 1e-6)
loss = (weight * (pred_udf - gt_udf)**2).mean()
```

---

### 数据增强 (Exp2.4)

```python
# 随机旋转 + 平移 + 缩放
def augment(points, udf):
    # Rotation: random SO(3)
    R = random_rotation_matrix()
    points = points @ R.T

    # Translation: [-0.1, 0.1]³
    t = torch.rand(3) * 0.2 - 0.1
    points = points + t

    # Scale: [0.9, 1.1]
    s = torch.rand(1) * 0.2 + 0.9
    points = points * s
    udf = udf * s  # UDF也需要缩放

    return points, udf
```

---

## 🛠️ 依赖资源

### 数据（来自Exp 1）
- `data/npz-resample/` - FFB编码数据
- `data/original_meshes/` - Ground truth

### 基线模型（来自Exp 1）
- `experiments/exp1_udf_baseline/results/meshes/ffb_flooding.ply` - Baseline结果
- `data/ckpts/ffb_mlp/ffb_mlp.pth` - Baseline checkpoint

### 工具脚本
- `src/train_ffb_mlp.py` - 训练（需支持参数：--sampling, --loss_type, --augment）
- `src/extract_mesh_flooding.py` - Flooding抽取
- `experiments/exp4_mfcd_definition/symmetric_mfcd.py` - MFCD计算

---

## ⏱️ 运行时间

| 阶段 | 时间 |
|------|------|
| 训练 (5个策略) | ~3小时 |
| Flooding抽取 (5个) | ~15分钟 |
| 评估 | ~20分钟 |
| **总计** | **~3.5小时** |

**快速测试** (--quick): ~40分钟

---

## 📝 注意事项

### 前置条件

1. **必须先运行Exp 1**，需要基线结果：
   ```bash
   cd experiments/exp1_udf_baseline
   python run.py
   ```

2. 确保`src/train_ffb_mlp.py`支持训练策略参数：
   ```bash
   python src/train_ffb_mlp.py --help
   # 应该有: --sampling [uniform|near_boundary]
   #         --loss_type [mse|weighted_mse]
   #         --augment [flag]
   ```

### 实现要求

如果`train_ffb_mlp.py`尚未支持这些参数，需要先扩展：

```python
# 在train_ffb_mlp.py中添加
parser.add_argument('--sampling', default='uniform',
                    choices=['uniform', 'near_boundary'])
parser.add_argument('--loss_type', default='mse',
                    choices=['mse', 'weighted_mse'])
parser.add_argument('--augment', action='store_true',
                    help='Enable data augmentation')
```

---

## 📚 相关文档

- **Exp 1基线**: `../exp1_udf_baseline/README.md`
- **实验总览**: `../EXPERIMENTS_OVERVIEW.md`
- **训练脚本**: `../../src/train_ffb_mlp.py`
- **评估工具**: `../exp4_mfcd_definition/symmetric_mfcd.py`

---

## 🔗 后续实验

- **Exp 3**: 基于本实验最优策略测试不同激活函数
- **Exp 4**: 使用本实验结果验证MFCD定义

---

**实验版本**: v1.0
**创建日期**: 2026-03-03
**基线方法**: Exp 1 FFB+Flooding
**核心改进**: Near-boundary + Weighted Loss ⭐
**状态**: ⏳ 需要扩展train_ffb_mlp.py参数支持
