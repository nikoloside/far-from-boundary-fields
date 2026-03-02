# Complete Pipeline Guide - UDF Baseline Experiments

**更新日期**: 2026-03-03
**目的**: 完整的pipeline运行指南

---

## 🎯 概述

这个pipeline针对5个data object完成以下所有步骤：

1. **数据编码** - FFB-DF和UDF编码
2. **模型训练** - FFB-MLP, UDF-MLP, NeuralUDF-MLP
3. **Mesh抽取** - 使用MIND优化
4. **指标计算** - SymMFCD, Fragment-wise, Boundary recall
5. **可视化** - 对比图表

---

## 🚀 快速开始

### 方式1: Bash脚本（推荐）

```bash
# 完整pipeline
bash scripts/run_complete_pipeline.sh

# 快速测试（1 epoch, 低分辨率）
bash scripts/run_complete_pipeline.sh --quick

# 自定义参数
bash scripts/run_complete_pipeline.sh --epochs 50 --resolution 128
```

### 方式2: Python脚本

```bash
# 完整pipeline
python scripts/run_complete_pipeline.py

# 快速测试
python scripts/run_complete_pipeline.py --quick_test

# 跳过某些阶段
python scripts/run_complete_pipeline.py --skip_phase_1 --skip_phase_4
```

---

## 📋 Pipeline详细步骤

### Phase 1: 数据编码 (约10-15分钟)

**目的**: 生成FFB-DF和UDF编码数据

```bash
# FFB-DF编码
python src/encoder_ffb-df_mlp.py

# UDF编码
python src/encoder_udf_mesh.py
```

**输出**:
- `data/npz-resample/` - FFB-DF数据 (5个对象，~55MB)
- `data/npz-udf/` - UDF数据 (5个对象，~44MB)

**数据特征**:
```
FFB-DF:
  - 352K 点/对象
  - 值域: [-1.00, 0.95]
  - 内部归一化 + 外部原距离

UDF:
  - 352K 点/对象
  - 值域: [0, 0.92]
  - 始终非负
```

---

### Phase 2: 模型训练 (约30-90分钟，取决于epochs)

#### 2.1 FFB-MLP训练

```bash
python src/train_ffb_mlp.py \
    --npz_dir data/npz-resample \
    --epochs 30 \
    --output_dir data/ckpts/ffb_mlp
```

**架构**:
- 4层, 128维
- Multires=4 (位置编码)
- ReLU激活
- ~37K参数

**输出**: `data/ckpts/ffb_mlp/ffb_mlp.pth`

---

#### 2.2 UDF-MLP训练

```bash
python src/train_udf_mlp.py \
    --npz_dir data/npz-udf \
    --epochs 30 \
    --output_dir data/ckpts/udf_mlp
```

**架构**:
- 4层, 128维
- Multires=4
- ReLU激活
- ~37K参数

**输出**: `data/ckpts/udf_mlp/udf_mlp.pth`

---

#### 2.3 NeuralUDF-MLP训练

```bash
python src/train_neuraludf_mlp.py \
    --npz_dir data/npz-udf \
    --epochs 100 \
    --d_hidden 256 \
    --n_layers 6 \
    --skip_in 4 \
    --multires 6 \
    --output_dir data/ckpts/neuraludf_mlp
```

**架构**:
- 6层, 256维
- Multires=6
- Skip connection at layer 4
- Softplus激活
- Geometric initialization
- Weight normalization
- ~285K参数

**输出**: `data/ckpts/neuraludf_mlp/neuraludf_mlp.pth`

---

### Phase 3: Mesh抽取 (约1-2小时)

使用MIND优化从UDF field抽取mesh。

#### 3.1 UDF-MLP + MIND

```bash
python src/extract_mesh_with_mind.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_mlp_mind.ply \
    --resolution 256 \
    --max_iter 200
```

#### 3.2 FFB-MLP + MIND

```bash
python src/extract_mesh_with_mind.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mlp_mind.ply \
    --resolution 256 \
    --max_iter 200
```

#### 3.3 NeuralUDF-MLP + MIND

```bash
python src/extract_mesh_with_mind.py \
    --model_type neuraludf_mlp \
    --ckpt data/ckpts/neuraludf_mlp/neuraludf_mlp.pth \
    --output data/results/meshes/neuraludf_mlp_mind.ply \
    --resolution 256 \
    --max_iter 200
```

**MIND参数**:
- `resolution`: 网格分辨率 (128-512)
- `max_iter`: 优化迭代次数 (100-300)
- `laplacian_weight`: Laplacian正则化权重 (1000.0)
- `learning_rate`: 学习率 (0.0005)

**输出**: `data/results/meshes/*.ply`

---

### Phase 4: 指标计算 (约10-20分钟)

#### 4.1 Symmetric MFCD

```bash
python experiments/mfcd_definition/symmetric_mfcd.py \
    --batch \
    --orig-dir data/original_meshes \
    --recon-dirs \
        udf_mlp:data/results/meshes \
        ffb_mlp:data/results/meshes \
        neuraludf:data/results/meshes \
    --output-dir data/results/complete_pipeline/mfcd \
    --num-samples 5000
```

**指标**:
- **SymMFCD**: 对称多碎片Chamfer距离
- **MFCD(Orig→Recon)**: 缺失碎片的影响
- **MFCD(Recon→Orig)**: 多余碎片的影响

**输出**:
- `symmetric_mfcd_results.json` - 详细结果
- `symmetric_mfcd_comparison.png` - 对比图表

---

### Phase 5: 可视化 (TODO)

生成以下对比图表：
- 定量指标对比
- 定性结果对比
- Fragment-wise分析
- Boundary recall分析

---

## 📊 实验矩阵

| 方法 | 编码 | 架构 | 后处理 | 状态 |
|------|------|------|--------|------|
| **UDF-MLP** | 纯UDF | 简化(4层,128维) | - | ✅ 可训练 |
| **FFB-MLP** | 归一化FFB | 简化(4层,128维) | - | ✅ 可训练 |
| **NeuralUDF-MLP** | 纯UDF | 完整(6层,256维) | - | ✅ 可训练 |
| **UDF-MLP + MIND** | 纯UDF | 简化 | MIND优化 | ✅ 可抽取 |
| **FFB-MLP + MIND** | 归一化FFB | 简化 | MIND优化 | ✅ 可抽取 |
| **NeuralUDF + MIND** | 纯UDF | 完整 | MIND优化 | ✅ 可抽取 |

---

## 🔍 关键对比

### 对比A: 编码方式

```
UDF-MLP vs FFB-MLP
→ 纯UDF vs 归一化FFB
→ 架构相同（4层,128维）
→ 只有编码不同
```

### 对比B: 网络架构

```
UDF-MLP vs NeuralUDF-MLP
→ 简化 vs 完整
→ 编码相同（纯UDF）
→ 架构不同（4层128维 vs 6层256维）
```

### 对比C: MIND后处理

```
Method vs Method + MIND
→ 基底方法相同
→ 只有后处理不同
→ 检验MIND的效果
```

---

## ⏱️ 时间估计

### 完整pipeline (30 epochs)

| 阶段 | 时间 | 说明 |
|------|------|------|
| Phase 1 | 10-15分钟 | 数据编码 |
| Phase 2 | 30-90分钟 | 3个模型训练 |
| Phase 3 | 60-120分钟 | MIND抽取（3个mesh） |
| Phase 4 | 10-20分钟 | 指标计算 |
| Phase 5 | 5-10分钟 | 可视化 |
| **总计** | **2-4小时** | 完整pipeline |

### 快速测试 (1 epoch, 低分辨率)

| 阶段 | 时间 |
|------|------|
| Phase 1 | 10分钟 |
| Phase 2 | 5分钟 |
| Phase 3 | 10分钟 |
| Phase 4 | 跳过 |
| Phase 5 | 跳过 |
| **总计** | **~25分钟** |

---

## 📁 输出文件结构

```
data/
├── npz-resample/              # FFB-DF编码数据
│   ├── 1.npz
│   ├── 2.npz
│   └── ...
│
├── npz-udf/                   # UDF编码数据
│   ├── 1.npz
│   ├── 2.npz
│   └── ...
│
├── ckpts/                     # 模型checkpoints
│   ├── ffb_mlp/
│   │   └── ffb_mlp.pth
│   ├── udf_mlp/
│   │   └── udf_mlp.pth
│   └── neuraludf_mlp/
│       └── neuraludf_mlp.pth
│
└── results/
    ├── meshes/                # 抽取的mesh
    │   ├── udf_mlp_mind.ply
    │   ├── ffb_mlp_mind.ply
    │   └── neuraludf_mlp_mind.ply
    │
    └── complete_pipeline/     # Pipeline结果
        ├── mfcd/              # MFCD指标
        │   ├── symmetric_mfcd_results.json
        │   └── symmetric_mfcd_comparison.png
        ├── pipeline_log_*.txt
        └── pipeline_status_*.json
```

---

## 🛠️ 故障排除

### 问题1: CUDA内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
```bash
# 减小batch size
python src/train_neuraludf_mlp.py --batch_size 4096

# 或使用CPU
export CUDA_VISIBLE_DEVICES=""
```

### 问题2: MIND抽取失败

**症状**: MIND运行崩溃

**解决**:
```bash
# 降低分辨率
python src/extract_mesh_with_mind.py --resolution 128

# 减少迭代
python src/extract_mesh_with_mind.py --max_iter 100

# 增大batch size限制
python src/extract_mesh_with_mind.py --max_batch 50000
```

### 问题3: 原始mesh文件不存在

**症状**: `⚠️  Original meshes not found`

**解决**:
```bash
# 创建目录并放置原始mesh
mkdir -p data/original_meshes
# 将5个原始OBJ文件放入此目录
```

### 问题4: 训练loss不收敛

**症状**: Loss始终很高或NaN

**解决**:
```bash
# 降低学习率
python src/train_neuraludf_mlp.py --lr 1e-5

# 禁用geometric init
python src/train_neuraludf_mlp.py --no_geometric_init

# 禁用weight norm
python src/train_neuraludf_mlp.py --no_weight_norm
```

---

## 📝 检查点 (Checklist)

在运行pipeline之前：

- [ ] ✅ 安装所有依赖 (`pip install -r requirements.txt`)
- [ ] ✅ 准备5个data object的OBJ文件
- [ ] ✅ 确保有足够的磁盘空间 (至少10GB)
- [ ] ✅ 确认CUDA可用 (如果使用GPU)
- [ ] ✅ 创建`data/original_meshes/`目录（如果需要评估）

在pipeline运行后：

- [ ] ✅ 检查编码数据生成
- [ ] ✅ 检查模型训练完成
- [ ] ✅ 检查mesh抽取成功
- [ ] ✅ 检查指标计算完成
- [ ] ✅ 查看可视化结果

---

## 🎓 学术用途

### 论文实验复现

```bash
# 1. 完整pipeline (推荐配置)
bash scripts/run_complete_pipeline.sh --epochs 100

# 2. 查看结果
ls data/results/complete_pipeline/mfcd/

# 3. 生成论文图表
# (TODO: 创建论文figure生成脚本)
```

### 消融实验

```bash
# 只训练特定模型
python scripts/run_complete_pipeline.py --skip_ffb --skip_neuraludf

# 跳过MIND
python scripts/run_complete_pipeline.py --skip_mind

# 只评估
python scripts/run_complete_pipeline.py --skip_phase_1 --skip_phase_2 --skip_phase_3
```

---

## 📚 相关文档

- **方法对比**: `METHOD_COMPARISON.md`
- **FFB编码说明**: `FFB_ENCODING_CLARIFICATION.md`
- **MIND分析**: `METHOD_COMPARISON_MIND.md`
- **实验计划**: `EXPERIMENT_PLAN.md`
- **SymMFCD说明**: `../mfcd_definition/SYMMETRIC_MFCD_EXPLANATION.md`

---

## 💡 最佳实践

### 首次运行

1. **先运行快速测试**
   ```bash
   bash scripts/run_complete_pipeline.sh --quick
   ```
   验证环境配置正确

2. **检查中间结果**
   - 编码后检查NPZ文件
   - 训练后检查checkpoint
   - 抽取后检查mesh

3. **运行完整pipeline**
   ```bash
   bash scripts/run_complete_pipeline.sh
   ```

### 调试技巧

- 使用Python脚本的`--skip_phase_*`参数跳过已完成的阶段
- 查看日志文件了解详细错误信息
- 使用`--quick_test`快速迭代

### 性能优化

- GPU训练比CPU快10-20倍
- 降低MIND分辨率可显著加速（质量略降）
- 并行运行独立的mesh抽取任务

---

**文档版本**: v1.0
**创建日期**: 2026-03-03
**维护者**: Claude Code
