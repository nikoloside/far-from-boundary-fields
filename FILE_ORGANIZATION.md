# File Organization Guide

**日期**: 2026-03-03
**最后更新**: 2026-03-03 01:10
**状态**: ✅ **合并完成！**

---

## 📁 当前统一结构（已完成整理）

**标准experiments/结构**:
```
experiments/
├── exp1_udf_baseline/           ✅ 包含MIND/、NeuralUDF/和所有文档
├── exp2_training_trick_ablation/ ✅ 包含README.md
├── exp3_activation_ablation/     ✅ 包含README.md
├── exp4_mfcd_definition/        ✅ 包含symmetric_mfcd.py和所有工具
└── exp5_voxel_ablation/
```

**✅ 已删除的重复文件夹**:
- ~~`udf_baseline/`~~ → 已合并到 `exp1_udf_baseline/`
- ~~`mfcd_definition/`~~ → 已合并到 `exp4_mfcd_definition/`
- ~~`training_trick_ablation/`~~ → 已移除
- ~~`activation_ablation/`~~ → 已移除
- ~~`voxel_ablation/`~~ → 已移除

---

## ✅ 已执行的整理方案（方案B：移动文件）

### 整理内容

**exp1_udf_baseline/** - 合并内容：
```bash
# 从udf_baseline/移动
✓ MIND/                        # MIND子模块
✓ NeuralUDF/                   # NeuralUDF子模块
✓ FFB_ENCODING_CLARIFICATION.md
✓ EXPERIMENT_PLAN.md
✓ FOCUSED_EXPERIMENT_DESIGN.md
✓ FULL_EXPERIMENT_MATRIX.md
✓ METHOD_COMPARISON*.md
✓ PIPELINE_GUIDE.md
✓ encode_mind.py
✓ encode_neuraludf.py
```

**exp4_mfcd_definition/** - 合并内容：
```bash
# 从mfcd_definition/移动
✓ symmetric_mfcd.py            # 核心实现
✓ toy_example_sym_mfcd.py      # Toy examples
✓ SYMMETRIC_MFCD_EXPLANATION.md
✓ USAGE_GUIDE.md
✓ vc_plot_charmer_distance_error.py
```

**清理**:
```bash
✓ rmdir udf_baseline/
✓ rmdir mfcd_definition/
✓ rm -rf training_trick_ablation/
✓ rm -rf activation_ablation/
✓ rm -rf voxel_ablation/
```

### 优势

✅ **单一真实来源** - 所有文件都在exp*标准结构下
✅ **清晰的层次** - 5个实验文件夹，各自完整
✅ **易于导航** - 不再有重复路径
✅ **完整文档** - 每个exp都有comprehensive README

---

## 📂 推荐的最终结构

```
experiments/
├── EXPERIMENTS_OVERVIEW.md     # ✅ 已创建 - 总览

├── exp1_udf_baseline/
│   ├── run.py                  # ✅ 已更新 - 主runner
│   ├── README.md               # ✅ 已创建 - 实验说明
│   ├── MIND/                   # MIND实现（保留原位置）
│   ├── NeuralUDF/              # NeuralUDF实现（保留原位置）
│   └── results/                # 实验结果输出
│
├── exp2_training_trick_ablation/
│   ├── run.py                  # 原有
│   ├── README.md               # TODO: 创建
│   └── results/
│
├── exp3_activation_ablation/
│   ├── run.py                  # 原有
│   ├── README.md               # TODO: 创建
│   └── results/
│
├── exp4_mfcd_definition/
│   ├── run.py                  # 原有
│   ├── README.md               # TODO: 创建
│   ├── symmetric_mfcd.py       # ✅ 在mfcd_definition/
│   ├── toy_example_sym_mfcd.py # ✅ 在mfcd_definition/
│   ├── SYMMETRIC_MFCD_EXPLANATION.md  # ✅ 在mfcd_definition/
│   └── results/
│
└── exp5_voxel_ablation/
    ├── run.py
    └── README.md
```

---

## 🚀 核心脚本位置

### 数据和模型
```
src/
├── encoder_ffb-df_mlp.py       # FFB编码
├── encoder_udf_mesh.py         # UDF编码
├── train_ffb_mlp.py            # FFB-MLP训练
├── train_udf_mlp.py            # UDF-MLP训练
├── train_neuraludf_mlp.py      # ✅ 已创建 - NeuralUDF训练
├── extract_mesh_flooding.py    # ✅ 已创建 - Flooding抽取
└── extract_mesh_with_mind.py   # ✅ 已创建 - MIND抽取
```

### Pipeline脚本
```
scripts/
├── run_all_experiments.sh      # ✅ 已创建 - Master runner
├── run_core_experiments.sh     # ✅ 已创建 - 核心实验
├── run_complete_pipeline.sh    # ✅ 已创建 - 完整pipeline
└── run_complete_pipeline.py    # ✅ 已创建 - Python版
```

---

## 📊 实验运行方式

### 方式1: 运行单个实验（推荐）

```bash
# Exp 1: UDF Baseline
cd experiments/exp1_udf_baseline
python run.py                    # 完整运行
python run.py --quick            # 快速测试
python run.py --minimal          # 最小测试

# Exp 2-4: 同样方式
cd experiments/exp2_training_trick_ablation
python run.py
```

### 方式2: Master runner

```bash
# 从项目根目录
bash scripts/run_all_experiments.sh          # 完整
bash scripts/run_all_experiments.sh --quick  # 快速
bash scripts/run_all_experiments.sh --minimal # 最小
```

### 方式3: 核心实验（Exp 1重点）

```bash
bash scripts/run_core_experiments.sh
```

---

## 🔗 文档引用路径（已更新）

### 主要文档

| 文档 | 当前位置 | 使用方式 |
|------|----------|---------|
| **实验总览** | `experiments/EXPERIMENTS_OVERVIEW.md` | 主入口文档 |
| **Exp 1说明** | `experiments/exp1_udf_baseline/README.md` | ✅ 详细实验说明 |
| **Exp 2说明** | `experiments/exp2_training_trick_ablation/README.md` | ✅ 训练策略消融 |
| **Exp 3说明** | `experiments/exp3_activation_ablation/README.md` | ✅ 激活函数消融 |
| **Exp 4说明** | `experiments/exp4_mfcd_definition/README.md` | ✅ MFCD定义验证 |
| **FFB编码说明** | `experiments/exp1_udf_baseline/FFB_ENCODING_CLARIFICATION.md` | ✅ 已移动 |
| **SymMFCD说明** | `experiments/exp4_mfcd_definition/SYMMETRIC_MFCD_EXPLANATION.md` | ✅ 已移动 |

### 更新的导入路径

```python
# 使用SymMFCD（路径已更新）
from experiments.exp4_mfcd_definition.symmetric_mfcd import symmetric_mfcd

# 或者直接运行
subprocess.run("python experiments/exp4_mfcd_definition/symmetric_mfcd.py ...")

# MIND和NeuralUDF在exp1下
sys.path.append("experiments/exp1_udf_baseline/MIND/src")
from mind import MIND
```

---

## ✅ 已完成的整理

### 文档结构
1. ✅ 创建`EXPERIMENTS_OVERVIEW.md` - 实验总览
2. ✅ 创建`exp1_udf_baseline/README.md` - Exp 1详细说明
3. ✅ 创建`exp2_training_trick_ablation/README.md` - Exp 2详细说明
4. ✅ 创建`exp3_activation_ablation/README.md` - Exp 3详细说明
5. ✅ 创建`exp4_mfcd_definition/README.md` - Exp 4详细说明

### 脚本和工具
6. ✅ 更新`exp1_udf_baseline/run.py` - 整合新功能
7. ✅ 创建`scripts/run_all_experiments.sh` - Master runner
8. ✅ 所有核心脚本在`src/`下就位

### 文件组织
9. ✅ **合并重复文件夹** - 执行方案B（移动文件）
10. ✅ 移动`udf_baseline/*`到`exp1_udf_baseline/`
11. ✅ 移动`mfcd_definition/*`到`exp4_mfcd_definition/`
12. ✅ 删除所有重复和临时文件夹

---

## ⏳ 待完成的整理

1. ⏳ 更新`exp2-4/run.py`以使用新的评估工具（symmetric_mfcd.py）
2. ⏳ 扩展`src/train_ffb_mlp.py`支持Exp 2的训练策略参数
3. ⏳ 扩展`src/train_ffb_mlp.py`支持Exp 3的激活函数参数
4. ⏳ 测试完整pipeline运行

---

## 💡 使用指南

### 立即可用

文件结构已清理完成，可以开始运行实验：

```bash
# 运行Exp 1（完整测试）
cd experiments/exp1_udf_baseline
python run.py --quick

# 运行所有实验
cd ../..  # 返回项目根目录
bash scripts/run_all_experiments.sh --minimal

# 使用SymMFCD（路径已更新）
python experiments/exp4_mfcd_definition/symmetric_mfcd.py --help
```

### 后续开发

1. **扩展训练脚本**: 为Exp 2和Exp 3添加参数支持
2. **完善run.py**: 更新exp2-4的runner使用symmetric_mfcd.py
3. **测试pipeline**: 运行完整实验流程验证

---

**文档版本**: v2.0
**创建日期**: 2026-03-03
**最后更新**: 2026-03-03 01:10
**状态**: ✅ **文件组织已完成！所有重复文件夹已合并清理**
