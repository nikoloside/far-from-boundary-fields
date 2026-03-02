# Quick Start Guide

**更新日期**: 2026-03-03
**核心方法**: FFB-MLP + Flooding Algorithm ⭐

---

## 🎯 核心实验（推荐）

### 一键运行 4 种核心方法

```bash
# 完整运行 (约1-2小时)
bash scripts/run_core_experiments.sh

# 快速测试 (约20分钟)
bash scripts/run_core_experiments.sh --quick

# 跳过SOTA对比 (更快)
bash scripts/run_core_experiments.sh --skip-sota
```

**4种方法**:
1. ✅ **FFB-MLP + Flooding** ⭐ (Our Method)
2. ✅ **UDF-MLP + Flooding** (对比编码)
3. ✅ **FFB-MLP + MIND** (对比后处理)
4. ✅ **NeuralUDF + Flooding** (对比SOTA，可选)

---

## 📋 完整Pipeline（可选）

如果需要运行所有phases（包括数据编码）：

```bash
# 完整pipeline
bash scripts/run_complete_pipeline.sh

# 快速测试
bash scripts/run_complete_pipeline.sh --quick
```

---

## 🔬 单独运行各个步骤

### Step 1: 数据编码 (首次运行需要)

```bash
# FFB-DF编码
python src/encoder_ffb-df_mlp.py

# UDF编码
python src/encoder_udf_mesh.py
```

### Step 2: 模型训练

```bash
# FFB-MLP (Our encoding)
python src/train_ffb_mlp.py --epochs 30

# UDF-MLP (Comparison)
python src/train_udf_mlp.py --epochs 30

# NeuralUDF (SOTA, optional)
python src/train_neuraludf_mlp.py --epochs 100
```

### Step 3: Mesh抽取

#### 方法1: Flooding (Our Method)

```bash
# FFB-MLP + Flooding ⭐
python src/extract_mesh_flooding.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_flooding.ply \
    --resolution 256 \
    --no_imagej

# UDF-MLP + Flooding (对比)
python src/extract_mesh_flooding.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_flooding.ply \
    --resolution 256 \
    --no_imagej
```

#### 方法2: MIND (Baseline)

```bash
# FFB-MLP + MIND (对比)
python src/extract_mesh_with_mind.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mind.ply \
    --resolution 256 \
    --max_iter 200
```

### Step 4: 评估

```bash
# Symmetric MFCD
python experiments/mfcd_definition/symmetric_mfcd.py \
    --batch \
    --orig-dir data/original_meshes \
    --recon-dirs \
        ffb_flood:data/results/meshes \
        udf_flood:data/results/meshes \
        ffb_mind:data/results/meshes \
    --output-dir data/results/mfcd \
    --num-samples 5000
```

---

## 📊 查看结果

### 查看Mesh

```bash
# 列出所有生成的mesh
ls -lh data/results/core_experiments/meshes/

# 在MeshLab中打开
meshlab data/results/core_experiments/meshes/ffb_flooding.ply
```

### 查看指标

```bash
# 查看JSON结果
cat data/results/core_experiments/mfcd/symmetric_mfcd_results.json

# 查看对比图
open data/results/core_experiments/mfcd/symmetric_mfcd_comparison.png
```

### 查看日志

```bash
# 最新的日志
ls -t data/results/core_experiments/logs/*.txt | head -1 | xargs cat
```

---

## 🎓 实验对比

### 我们的方法 vs 其他

| 方法 | 用途 | 预期结论 |
|------|------|----------|
| **FFB+Flood** ⭐ | Our Method | 最优 |
| **UDF+Flood** | 对比编码 | FFB > UDF |
| **FFB+MIND** | 对比后处理 | Flood更快，质量相近 |
| **NeU+Flood** | 对比SOTA | 轻量方法接近SOTA |

---

## 🛠️ 常用命令

### 快速测试（推荐首次运行）

```bash
# 20分钟快速测试
bash scripts/run_core_experiments.sh --quick
```

### 只运行训练

```bash
# 使用已有数据训练
python src/train_ffb_mlp.py --epochs 30
```

### 只运行抽取（假设已训练）

```bash
# 跳过训练，直接抽取
bash scripts/run_core_experiments.sh --skip-training
```

### 检查checkpoint是否存在

```bash
ls data/ckpts/*/**.pth
```

---

## 📁 文件结构

```
项目根目录/
├── src/
│   ├── encoder_ffb-df_mlp.py       # FFB编码
│   ├── encoder_udf_mesh.py         # UDF编码
│   ├── train_ffb_mlp.py            # FFB-MLP训练
│   ├── train_udf_mlp.py            # UDF-MLP训练
│   ├── train_neuraludf_mlp.py      # NeuralUDF训练
│   ├── extract_mesh_flooding.py    # ✨ Flooding抽取
│   └── extract_mesh_with_mind.py   # ✨ MIND抽取
│
├── scripts/
│   ├── run_core_experiments.sh     # ⭐ 核心实验
│   └── run_complete_pipeline.sh    # 完整pipeline
│
├── experiments/
│   ├── udf_baseline/
│   │   ├── FOCUSED_EXPERIMENT_DESIGN.md  # 实验设计
│   │   └── PIPELINE_GUIDE.md             # 详细指南
│   └── mfcd_definition/
│       ├── symmetric_mfcd.py             # SymMFCD计算
│       └── USAGE_GUIDE.md                # 使用指南
│
├── data/
│   ├── npz-resample/               # FFB-DF数据
│   ├── npz-udf/                    # UDF数据
│   ├── ckpts/                      # 模型checkpoints
│   └── results/
│       └── core_experiments/       # 核心实验结果
│           ├── meshes/             # 生成的mesh
│           ├── mfcd/               # 指标结果
│           └── logs/               # 日志
│
├── QUICK_START.md                  # ⭐ 本文档
├── IMPLEMENTATION_SUMMARY.md       # 实现总结
└── README.md                       # 项目README
```

---

## 📚 详细文档

### 核心文档

1. **快速开始**: `QUICK_START.md` (本文档)
2. **实验设计**: `experiments/udf_baseline/FOCUSED_EXPERIMENT_DESIGN.md`
3. **实现总结**: `IMPLEMENTATION_SUMMARY.md`

### 技术文档

1. **完整Pipeline**: `experiments/udf_baseline/PIPELINE_GUIDE.md`
2. **FFB编码**: `experiments/udf_baseline/FFB_ENCODING_CLARIFICATION.md`
3. **SymMFCD**: `experiments/mfcd_definition/SYMMETRIC_MFCD_EXPLANATION.md`
4. **方法对比**: `experiments/udf_baseline/METHOD_COMPARISON.md`

---

## ⚠️ 常见问题

### Q1: CUDA内存不足

**解决**:
```bash
# 降低batch size
python src/train_ffb_mlp.py --batch_size 4096

# 降低resolution
python src/extract_mesh_flooding.py --resolution 128
```

### Q2: MIND抽取失败

**原因**: MIND需要CUDA

**解决**: 使用CPU或跳过MIND
```bash
bash scripts/run_core_experiments.sh --skip-mind
```

### Q3: ImageJ不可用

**原因**: ImageJ需要单独安装

**解决**: 使用`--no_imagej`参数（已默认）
```bash
python src/extract_mesh_flooding.py --no_imagej
```

### Q4: 原始mesh不存在

**解决**: 创建目录并放置原始OBJ文件
```bash
mkdir -p data/original_meshes
# 将5个原始OBJ文件放入此目录
```

---

## ✅ 检查清单

运行前确认：

- [ ] Python 3.8+
- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 有足够磁盘空间 (>5GB)
- [ ] (可选) CUDA可用

运行后检查：

- [ ] Checkpoints生成: `ls data/ckpts/*/**.pth`
- [ ] Meshes生成: `ls data/results/*/meshes/*.ply`
- [ ] 指标计算: `ls data/results/*/mfcd/*.json`

---

## 🚀 推荐工作流

### 首次使用

```bash
# 1. 快速测试（20分钟）
bash scripts/run_core_experiments.sh --quick

# 2. 检查结果
ls data/results/core_experiments/meshes/

# 3. 如果成功，运行完整版本
bash scripts/run_core_experiments.sh
```

### 日常使用

```bash
# 只运行核心4种方法
bash scripts/run_core_experiments.sh --skip-sota

# 查看结果
cat data/results/core_experiments/mfcd/symmetric_mfcd_results.json
```

---

## 📞 获取帮助

- **详细指南**: `experiments/udf_baseline/PIPELINE_GUIDE.md`
- **实验设计**: `experiments/udf_baseline/FOCUSED_EXPERIMENT_DESIGN.md`
- **故障排除**: 查看log文件 `data/results/*/logs/*.txt`

---

**文档版本**: v1.0
**创建日期**: 2026-03-03
**推荐命令**: `bash scripts/run_core_experiments.sh`
