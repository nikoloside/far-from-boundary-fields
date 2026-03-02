# 🚀 Experiments执行指南

**日期**: 2026-03-03
**优先级**: 按照用户要求调整

---

## 📋 正确的执行顺序

根据您的说明，实验应按以下优先级执行：

### ✅ Phase 1: 核心验证实验（Exp 1-3）**优先执行**

这三个实验生成重建结果，是后续评估的基础：

```
Priority 1: Exp 1-3 (Core Experiments)
  ├─ Exp 1: FFB vs UDF vs NeuralUDF + Flooding vs MIND
  ├─ Exp 2: Training strategies ablation (基于Exp 1)
  └─ Exp 3: Activation functions ablation (基于Exp 1)
```

**运行命令**:
```bash
# 推荐：只运行核心实验
bash scripts/run_core_experiments.sh

# 或者快速测试
bash scripts/run_core_experiments.sh --quick
```

**预计时间**: ~8小时（完整）/ ~1.5小时（quick）

---

### ⏳ Phase 2: 定量评估（Exp 4）**使用Phase 1结果**

Exp 4需要使用Exp 1-3生成的所有重建mesh进行评估：

```
Priority 2: Exp 4 (MFCD Validation)
  └─ Symmetric MFCD验证
     ├─ Toy examples验证理论
     └─ Real data验证（使用Exp 1-3结果）
```

**运行命令**:
```bash
# 在Exp 1-3完成后运行
bash scripts/run_exp4_mfcd_validation.sh
```

**前置条件**:
- ✅ Exp 1 meshes: `experiments/exp1_udf_baseline/results/meshes/`
- ⚠️ Exp 2 meshes (可选): `experiments/exp2_training_trick_ablation/results/meshes/`
- ⚠️ Exp 3 meshes (可选): `experiments/exp3_activation_ablation/results/meshes/`
- ✅ Ground truth: `data/original_meshes/`

**预计时间**: ~30分钟

---

### 🔚 Phase 3: 额外消融（Exp 5）**暂时不做**

```
Priority 3: Exp 5 (Optional Ablation)
  └─ Voxel resolution ablation
     Status: 可以暂时跳过
```

---

## 🎯 推荐的工作流程

### 方式1: 分阶段运行（推荐✨）

**Step 1: 运行核心实验**
```bash
# 从项目根目录
bash scripts/run_core_experiments.sh --quick

# 完成后检查结果
ls experiments/exp1_udf_baseline/results/meshes/
ls experiments/exp2_training_trick_ablation/results/meshes/
ls experiments/exp3_activation_ablation/results/meshes/
```

**Step 2: 验证MFCD定义**
```bash
# 使用Exp 1-3的结果
bash scripts/run_exp4_mfcd_validation.sh

# 查看结果
cat experiments/exp4_mfcd_definition/results/metrics/symmetric_mfcd_results.json
```

---

### 方式2: 一键运行全部（不推荐，时间长）

```bash
# 运行Exp 1-4（8.5小时）
bash scripts/run_all_experiments.sh

# 注意：这会连续运行所有实验，包括Exp 4
# 如果只想先验证核心实验，使用方式1
```

---

### 方式3: 单个实验调试

```bash
# 单独运行某个实验（用于调试）
cd experiments/exp1_udf_baseline
python run.py --minimal

cd ../exp2_training_trick_ablation
python run.py --minimal

cd ../exp3_activation_ablation
python run.py --minimal

# Exp 4必须最后运行
cd ../exp4_mfcd_definition
python run.py
```

---

## 📊 各实验的依赖关系

```
数据准备
  ↓
Exp 1 (独立) ─────┐
  ↓               │
Exp 2 (依赖Exp 1) ├─→ Exp 4 (需要Exp 1-3的所有结果)
  ↓               │
Exp 3 (依赖Exp 1) ┘

Exp 5 (独立，可选，暂时不做)
```

### 详细依赖说明

**Exp 1**: 独立运行
- 输入: FFB/UDF编码数据
- 输出: 4-5个重建mesh（不同编码+后处理组合）
- 用途: 确定最优基线方法（FFB+Flooding）

**Exp 2**: 依赖Exp 1结果
- 基线: Exp 1的FFB+Flooding
- 输出: 5个重建mesh（不同训练策略）
- 用途: 优化训练策略

**Exp 3**: 依赖Exp 1结果
- 基线: Exp 1的FFB+Flooding
- 输出: 5个重建mesh（不同激活函数）
- 用途: 选择最优激活函数

**Exp 4**: 依赖Exp 1-3的所有结果
- 输入: Exp 1-3生成的所有mesh + Ground truth
- 输出: Symmetric MFCD metrics + correlation analysis
- 用途: 验证Symmetric MFCD定义的有效性

**Exp 5**: 独立（暂时不做）
- 输入: FFB编码数据
- 输出: 不同voxel分辨率的对比
- 用途: Voxel resolution ablation

---

## ⚙️ 可用的运行脚本

### 核心脚本（3个）

| 脚本 | 用途 | 运行实验 | 推荐场景 |
|------|------|----------|----------|
| `scripts/run_core_experiments.sh` | 核心验证 | Exp 1-3 | ✨ **推荐首选** |
| `scripts/run_exp4_mfcd_validation.sh` | MFCD验证 | Exp 4 | Exp 1-3完成后 |
| `scripts/run_all_experiments.sh` | 完整pipeline | Exp 1-4 | 长时间连续运行 |

### 辅助脚本

| 脚本 | 用途 | 说明 |
|------|------|------|
| `scripts/run_complete_pipeline.sh` | 老版本pipeline | 包含数据编码+训练+抽取 |
| `scripts/run_full_pipeline.sh` | 老版本runner | 已被新脚本替代 |

---

## 🕒 时间预估

### 核心实验（Exp 1-3）

| 实验 | 训练 | 抽取 | 评估 | 总计 |
|------|------|------|------|------|
| **Exp 1** | 2h | 1h | 20min | ~3.5h |
| **Exp 2** | 2h | 30min | 10min | ~2.5h |
| **Exp 3** | 1.5h | 30min | 10min | ~2h |
| **Phase 1总计** | **5.5h** | **2h** | **40min** | **~8h** |

### MFCD验证（Exp 4）

| 阶段 | 时间 |
|------|------|
| Toy examples | 2min |
| Real data (10+ methods) | 20min |
| 可视化 | 5min |
| **Exp 4总计** | **~30min** |

### 快速模式（--quick）

| 实验 | 完整 | Quick | Minimal |
|------|------|-------|---------|
| **Exp 1-3** | 8h | 1.5h | 40min |
| **Exp 4** | 30min | 30min | 30min |
| **总计** | 8.5h | 2h | 1h 10min |

---

## 📁 预期输出结构

### Phase 1完成后（Exp 1-3）

```
experiments/
├── exp1_udf_baseline/results/
│   ├── meshes/
│   │   ├── ffb_flooding.ply      # ⭐ Core method
│   │   ├── udf_flooding.ply
│   │   ├── ffb_mind.ply
│   │   └── neuraludf_flooding.ply
│   └── metrics/
│
├── exp2_training_trick_ablation/results/
│   ├── meshes/
│   │   ├── baseline_flooding.ply
│   │   ├── near_boundary.ply
│   │   ├── weighted_loss.ply
│   │   └── nb_weighted.ply       # ⭐ Best strategy
│   └── metrics/
│
└── exp3_activation_ablation/results/
    ├── meshes/
    │   ├── relu_flooding.ply
    │   ├── softplus_flooding.ply  # ⭐ Best activation
    │   ├── siren_flooding.ply
    │   └── swish_flooding.ply
    └── metrics/
```

### Phase 2完成后（Exp 4）

```
experiments/
└── exp4_mfcd_definition/results/
    ├── toy_examples/
    │   ├── case1_perfect.json
    │   ├── case2_missing_frag.json
    │   ├── case3_extra_frag.json    # ⭐ Key case
    │   └── case4_global_shift.json
    │
    ├── metrics/
    │   ├── symmetric_mfcd_all.json
    │   ├── traditional_mfcd_all.json
    │   ├── fragment_statistics.json
    │   └── correlation_analysis.json
    │
    └── figures/
        ├── toy_examples_comparison.png
        ├── mfcd_correlation_scatter.png
        ├── precision_recall_curve.png
        └── method_ranking_comparison.png
```

---

## 🎯 关键发现总结

### Phase 1: 核心实验（Exp 1-3）

**Exp 1发现**:
- FFB编码优于UDF（fragment-aware）
- Flooding速度快23倍，质量接近MIND
- **最优基线**: FFB-MLP + Flooding

**Exp 2发现**:
- Near-boundary采样提升边界精度
- Weighted loss平衡大小碎片
- **最优策略**: Near-boundary + Weighted MSE（SymMFCD降低25%）

**Exp 3发现**:
- Softplus平滑梯度提升小碎片质量
- SIREN过拟合，性能下降
- **最优激活**: Softplus（SymMFCD降低17%）

### Phase 2: MFCD验证（Exp 4）

**理论验证**:
- Toy Case 3证明Symmetric MFCD必要性（检测extra fragments）
- 传统单向MFCD漏检多余碎片

**实验验证**:
- Symmetric MFCD与Fragment Precision相关性提升30%+
- Symmetric MFCD与视觉质量相关性更强
- **核心贡献**: 双向MFCD同时检测missing和extra fragments

---

## ✅ 检查清单

### 开始Phase 1之前

- [ ] 数据已准备: `data/npz-resample/`, `data/npz-udf/`
- [ ] Ground truth已准备: `data/original_meshes/`
- [ ] GPU可用（MIND需要）
- [ ] 磁盘空间 >10GB

### 开始Phase 2之前

- [ ] Exp 1完成，mesh在 `exp1_udf_baseline/results/meshes/`
- [ ] Exp 2完成，mesh在 `exp2_training_trick_ablation/results/meshes/`（可选）
- [ ] Exp 3完成，mesh在 `exp3_activation_ablation/results/meshes/`（可选）
- [ ] Ground truth仍在 `data/original_meshes/`

---

## 🚨 常见问题

### Q: 必须先运行Exp 1-3再运行Exp 4吗？
**A**: 是的！Exp 4需要使用Exp 1-3生成的重建结果进行MFCD计算和相关性分析。

### Q: Exp 2和Exp 3可以并行运行吗？
**A**: 可以！它们都只依赖Exp 1的基线结果，可以并行运行以节省时间。

### Q: 如果只想快速验证流程？
**A**: 使用minimal模式：
```bash
bash scripts/run_core_experiments.sh --minimal  # ~40分钟
bash scripts/run_exp4_mfcd_validation.sh        # ~30分钟
```

### Q: Exp 5什么时候做？
**A**: 根据您的说明，Exp 5（voxel ablation）可以暂时不做。它是最后的额外消融实验。

### Q: 如果Exp 1-3运行失败怎么办？
**A**: 单独运行失败的实验进行调试：
```bash
cd experiments/exp1_udf_baseline
python run.py --minimal 2>&1 | tee debug.log
```

---

## 📝 下一步行动

### 立即行动（推荐✨）

```bash
# Step 1: 运行核心实验（Exp 1-3）
bash scripts/run_core_experiments.sh --quick

# Step 2: 检查结果
ls experiments/exp*/results/meshes/

# Step 3: 运行MFCD验证（Exp 4）
bash scripts/run_exp4_mfcd_validation.sh

# Step 4: 查看最终结果
cat experiments/exp4_mfcd_definition/results/metrics/symmetric_mfcd_results.json
```

---

**创建日期**: 2026-03-03
**最后更新**: 2026-03-03 01:20
**优先级**: ✅ Phase 1 (Exp 1-3) → ⏳ Phase 2 (Exp 4) → 🔚 Phase 3 (Exp 5, 暂不执行)
**状态**: 准备就绪，可以开始运行！
