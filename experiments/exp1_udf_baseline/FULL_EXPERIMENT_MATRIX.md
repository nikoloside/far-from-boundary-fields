# Complete Experiment Matrix - Exp 1

**更新日期**: 2026-03-03
**重要发现**: 实验矩阵是 3×3 = 9种组合

---

## 🎯 实验矩阵

### 完整对比矩阵

|  | **MIND** | **Flooding** | **M3C** |
|---|----------|--------------|---------|
| **FFB-MLP** | FFB + MIND | FFB + Flooding ✅已有 | FFB + M3C |
| **UDF-MLP** | UDF + MIND ✅已实现 | UDF + Flooding | UDF + M3C |
| **NeuralUDF** | NeuralUDF + MIND ✅已实现 | NeuralUDF + Flooding | NeuralUDF + M3C |

**说明**:
- ✅ 已有代码：FFB-MLP + Flooding (`vc_plot_objs.py`)
- ✅ 已实现：MIND相关方法 (`extract_mesh_with_mind.py`)
- ⏳ 需要实现：UDF/NeuralUDF + Flooding，所有M3C方法

---

## 📐 两个对比维度

### 维度1: 编码/模型架构

| 方法 | 编码 | 架构 | 参数 |
|------|------|------|------|
| **FFB-MLP** | 归一化FFB（内部[-1,0]，外部[0,+∞)） | 4层×128维，multires=4 | ~37K |
| **UDF-MLP** | 纯UDF（始终非负[0,+∞)） | 4层×128维，multires=4 | ~37K |
| **NeuralUDF** | 纯UDF | 6层×256维，skip@4，multires=6 | ~285K |

### 维度2: 后处理/Mesh抽取方法

| 方法 | 类型 | 核心算法 | 特点 |
|------|------|----------|------|
| **MIND** | 优化 | Vertex optimization + Laplacian | 非流形支持，vertex可学习 |
| **Flooding** | 分割 | Watershed + Marching Cubes | Fragment segmentation，基于ImageJ |
| **M3C** | 直接抽取 | Multiple Material Marching Cubes | Multi-material，基于DREAM.3D |

---

## 🔍 关键对比

### 对比A: 编码方式（固定后处理）

**问题**: 不同编码对mesh质量的影响？

**组合** (使用相同后处理):
```
FFB-MLP + MIND  vs  UDF-MLP + MIND  vs  NeuralUDF + MIND
FFB-MLP + Flood vs  UDF-MLP + Flood vs  NeuralUDF + Flood
FFB-MLP + M3C   vs  UDF-MLP + M3C   vs  NeuralUDF + M3C
```

**假设**:
- FFB的内部归一化可能更适合fragment场景
- UDF的对称性可能更容易学习
- NeuralUDF的完整架构应该效果最好

---

### 对比B: 后处理方法（固定编码）

**问题**: 不同后处理方法对同一field的抽取质量？

**组合** (使用相同模型):
```
FFB-MLP + MIND  vs  FFB-MLP + Flood  vs  FFB-MLP + M3C
UDF-MLP + MIND  vs  UDF-MLP + Flood  vs  UDF-MLP + M3C
NeuralUDF + MIND vs NeuralUDF + Flood vs NeuralUDF + M3C
```

**假设**:
- MIND的优化应该最精确但最慢
- Flooding的分割适合碎片化场景
- M3C应该最快但可能丢失细节

---

### 对比C: 架构复杂度（固定编码和后处理）

**问题**: 网络架构的影响？

**组合** (都是UDF编码):
```
UDF-MLP (4层128维) vs NeuralUDF (6层256维+skip)
```

每种后处理都测试一遍：
- + MIND
- + Flooding
- + M3C

---

## 📊 实现状态

### ✅ 已有/已实现

| 组合 | 文件/脚本 | 状态 |
|------|----------|------|
| **FFB-MLP 训练** | `src/train_ffb_mlp.py` | ✅ |
| **UDF-MLP 训练** | `src/train_udf_mlp.py` | ✅ |
| **NeuralUDF 训练** | `src/train_neuraludf_mlp.py` | ✅ |
| **FFB-MLP + Flooding** | `src/FFB-MLP_VQ-MLP/vc_plot_objs.py` | ✅ 已有 |
| **UDF-MLP + MIND** | `src/extract_mesh_with_mind.py` | ✅ |
| **FFB-MLP + MIND** | `src/extract_mesh_with_mind.py` | ✅ |
| **NeuralUDF + MIND** | `src/extract_mesh_with_mind.py` | ✅ |

### ⏳ 需要实现

| 组合 | 需要创建 | 优先级 |
|------|----------|--------|
| **UDF-MLP + Flooding** | `src/extract_mesh_flooding.py` | 🔴 高 |
| **NeuralUDF + Flooding** | 同上 | 🔴 高 |
| **FFB-MLP + M3C** | `src/extract_mesh_m3c.py` | 🟡 中 |
| **UDF-MLP + M3C** | 同上 | 🟡 中 |
| **NeuralUDF + M3C** | 同上 | 🟡 中 |

---

## 🛠️ 实现计划

### Phase 1: Flooding Algorithm统一化 ⏳

**目标**: 创建统一的Flooding抽取接口，支持所有模型

**步骤**:
1. 提取`vc_plot_objs.py`中的核心flooding逻辑
2. 创建`src/extract_mesh_flooding.py`
3. 支持从任意模型checkpoint加载
4. 支持FFB/UDF/NeuralUDF

**预期输出**:
```bash
python src/extract_mesh_flooding.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_mlp_flooding.ply
```

---

### Phase 2: M3C Algorithm集成 ⏳

**目标**: 集成M3C (Multiple Material Marching Cubes)

**参考**:
- https://dream3d.bluequartz.net/Help/Filters/SurfaceMeshingFilters/M3CSliceBySlice/
- DREAM.3D软件包

**步骤**:
1. 研究M3C算法实现
2. 创建`src/extract_mesh_m3c.py`
3. 从model推理volume
4. 使用M3C抽取multi-material mesh

**预期输出**:
```bash
python src/extract_mesh_m3c.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mlp_m3c.ply
```

---

### Phase 3: 完整Pipeline更新 ⏳

**目标**: 更新pipeline支持所有9种组合

**修改文件**:
- `scripts/run_complete_pipeline.py`
- `scripts/run_complete_pipeline.sh`

**新增参数**:
```bash
--methods MIND,Flooding,M3C     # 选择后处理方法
--models FFB,UDF,NeuralUDF      # 选择模型
```

**完整运行**:
```bash
# 运行所有9种组合
bash scripts/run_complete_pipeline.sh --methods MIND,Flooding,M3C

# 只运行特定组合
bash scripts/run_complete_pipeline.sh --methods MIND --models UDF,NeuralUDF
```

---

## 📋 评估指标

### 所有9种方法都计算

| 指标 | 说明 |
|------|------|
| **Symmetric MFCD** | 双向多碎片Chamfer距离 |
| **MFCD(Orig→Recon)** | 缺失碎片影响 |
| **MFCD(Recon→Orig)** | 多余碎片影响 |
| **Fragment-wise Recall** | 碎片级别召回率 |
| **Boundary Recall** | 边界重建质量 |
| **Vertex Count** | 顶点数量 |
| **Face Count** | 面片数量 |
| **Inference Time** | 推理时间 |
| **Extraction Time** | 抽取时间 |

---

## 🎯 预期结果

### 定量对比表

|  | SymMFCD | Fragment Recall | Boundary Recall | Time |
|---|---------|-----------------|-----------------|------|
| **FFB + MIND** | ? | ? | ? | 慢 |
| **FFB + Flooding** | ? | ? | ? | 中 |
| **FFB + M3C** | ? | ? | ? | 快 |
| **UDF + MIND** | ? | ? | ? | 慢 |
| **UDF + Flooding** | ? | ? | ? | 中 |
| **UDF + M3C** | ? | ? | ? | 快 |
| **NeuralUDF + MIND** | ? | ? | ? | 慢 |
| **NeuralUDF + Flooding** | ? | ? | ? | 中 |
| **NeuralUDF + M3C** | ? | ? | ? | 快 |

### 定性对比

为每种方法生成：
- 细裂缝重建效果
- 小碎片保留情况
- 内部边界清晰度
- 整体mesh质量

---

## 📊 可视化

### Figure 1: 完整对比矩阵

```
3×3网格，每个格子展示一个方法的结果
横轴: MIND, Flooding, M3C
纵轴: FFB-MLP, UDF-MLP, NeuralUDF
```

### Figure 2: 编码对比

```
固定后处理（例如MIND），对比3种编码
```

### Figure 3: 后处理对比

```
固定编码（例如UDF-MLP），对比3种后处理
```

### Figure 4: 定量指标对比

```
条形图或热力图，展示所有指标
```

---

## 🔗 与其他实验的关系

### Exp 1: UDF Baseline (本实验)

**完整矩阵**: 3×3 = 9种方法

### Exp 2-5: 其他实验

根据`docs/TODO_UPDATED.md`:

- **Exp 2**: Training Trick Ablation
  - 对FFB-MLP和UDF-MLP测试不同训练策略

- **Exp 3**: Activation Ablation
  - 对FFB-MLP和UDF-MLP测试不同激活函数

- **Exp 4**: MFCD Definition
  - 验证SymMFCD定义（已完成）

- **Exp 5**: Voxel vs Implicit
  - 对比voxel CNN和implicit MLP

**注意**: Exp 2-5主要针对FFB-MLP和UDF-MLP，不一定需要NeuralUDF

---

## 🚀 下一步行动

### 优先级1: Flooding统一化 🔴

```bash
# 需要创建
src/extract_mesh_flooding.py
```

**为什么优先**:
- 已有FFB-MLP + Flooding的实现
- 只需要适配到UDF-MLP和NeuralUDF
- 快速扩展到6种方法（3种模型×2种后处理）

---

### 优先级2: M3C集成 🟡

```bash
# 需要创建
src/extract_mesh_m3c.py
```

**为什么第二**:
- 需要研究M3C算法
- 可能需要外部依赖（DREAM.3D）
- 完成后达到完整9种方法

---

### 优先级3: 完整评估 🟢

```bash
# 需要更新
scripts/compute_all_metrics.py
scripts/visualize_full_comparison.py
```

**包含**:
- 所有9种方法的指标计算
- 完整对比可视化
- 论文figure生成

---

## 📝 代码架构

### 统一的抽取接口

```python
# 所有抽取方法的统一接口
class MeshExtractor:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type

    def infer_volume(self, resolution=256):
        """从模型推理volume"""
        pass

    def extract_mesh(self, volume, method='MIND'):
        """从volume抽取mesh"""
        if method == 'MIND':
            return self.extract_with_mind(volume)
        elif method == 'Flooding':
            return self.extract_with_flooding(volume)
        elif method == 'M3C':
            return self.extract_with_m3c(volume)
```

### 文件组织

```
src/
├── train_*.py               # 训练脚本
├── extract_mesh_mind.py     # ✅ MIND抽取
├── extract_mesh_flooding.py # ⏳ Flooding抽取
├── extract_mesh_m3c.py      # ⏳ M3C抽取
└── mesh_extractor.py        # ⏳ 统一接口

experiments/udf_baseline/
├── FULL_EXPERIMENT_MATRIX.md  # 本文档
├── PIPELINE_GUIDE.md          # 使用指南
└── results/
    └── all_methods/           # 9种方法的结果
```

---

## 💡 重要发现

### 1. 现有代码可重用

**FFB-MLP + Flooding已有**:
- `src/FFB-MLP_VQ-MLP/VQ-mlp-origin-siren.py` - inference
- `src/FFB-MLP_VQ-MLP/vc_plot_objs.py` - flooding + marching cubes

**可以提取核心逻辑**:
- `processSDFWithImageJ()` - watershed算法
- ImageJ脚本 - 分割逻辑
- vedo isosurface - marching cubes

### 2. Flooding vs MIND的本质区别

**Flooding (Watershed)**:
- 基于图像分割
- 先分割label，再marching cubes
- 适合碎片化场景
- 不优化vertex位置

**MIND**:
- 基于优化
- 直接优化vertex位置
- 支持非流形
- 使用UDF query function

### 3. M3C的独特性

**M3C (Multiple Material Marching Cubes)**:
- 支持多材质
- slice-by-slice处理
- 直接从volume抽取
- 可能需要DREAM.3D库

---

**文档版本**: v1.0
**创建日期**: 2026-03-03
**实验范围**: Exp 1 - UDF Baseline
**完整矩阵**: 3 models × 3 methods = 9 combinations
