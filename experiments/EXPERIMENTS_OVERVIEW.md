# Experiments Overview

**更新日期**: 2026-03-03
**验证顺序**: 按照研究问题的逻辑顺序

---

## 🎯 实验组织结构

```
experiments/
├── exp1_udf_baseline/           # Exp 1: 编码+架构+后处理对比
│   ├── run.py                   # 主运行脚本
│   ├── README.md                # 实验说明
│   └── results/                 # 结果目录
│
├── exp2_training_trick_ablation/ # Exp 2: 训练技巧消融
│   ├── run.py
│   ├── README.md
│   └── results/
│
├── exp3_activation_ablation/    # Exp 3: 激活函数消融
│   ├── run.py
│   ├── README.md
│   └── results/
│
├── exp4_mfcd_definition/        # Exp 4: MFCD定义验证
│   ├── run.py
│   ├── symmetric_mfcd.py        # ✅ 已实现
│   ├── README.md
│   └── results/
│
└── exp5_voxel_ablation/         # Exp 5: Voxel消融（暂不需要）
    ├── run.py
    └── README.md
```

---

## 📊 验证顺序

### Phase 1: 核心方法对比（Exp 1）

**验证问题**:
1. **编码方式**: FFB vs UDF vs NeuralUDF
2. **后处理**: Flooding vs MIND

**实验矩阵**:
```
           Flooding      MIND
FFB-MLP    FFB+Flood⭐   FFB+MIND
UDF-MLP    UDF+Flood     (可选)
NeuralUDF  NeU+Flood     NeU+MIND
```

**运行命令**:
```bash
cd experiments/exp1_udf_baseline
python run.py
```

**预期结果**:
- FFB编码优于UDF（fragment-aware）
- Flooding速度快，MIND精度高
- FFB+Flood为最优组合

---

### Phase 2: 训练技巧消融（Exp 2）

**验证问题**: FFB+Flooding vs 不同训练策略

**测试变量**:
1. 采样策略: uniform vs near_boundary
2. 损失函数: MSE vs weighted MSE
3. 数据增强: with/without

**运行命令**:
```bash
cd experiments/exp2_training_trick_ablation
python run.py --method ffb_flooding
```

**对比基线**: Exp 1的FFB+Flooding结果

---

### Phase 3: 激活函数消融（Exp 3）

**验证问题**: 不同激活函数对FFB的影响

**测试激活函数**:
1. ReLU (baseline)
2. Softplus
3. SIREN
4. Swish/Mish

**运行命令**:
```bash
cd experiments/exp3_activation_ablation
python run.py --model ffb_mlp --postprocess flooding
```

**对比基线**: Exp 1的FFB+Flooding结果

---

### Phase 4: MFCD定义验证（Exp 4）

**验证问题**: Symmetric MFCD vs One-directional MFCD

**测试内容**:
1. Toy examples (4个场景)
2. 真实数据对比
3. 对称性验证

**运行命令**:
```bash
cd experiments/exp4_mfcd_definition
python run.py
```

**已实现**: ✅ `symmetric_mfcd.py`, `toy_example_sym_mfcd.py`

---

### Phase 5: Voxel消融（暂不需要）

暂时跳过

---

## 🚀 完整Pipeline运行

### 一键运行所有实验

```bash
# 在项目根目录
bash scripts/run_all_experiments.sh
```

### 分步运行

```bash
# Step 1: 核心对比
cd experiments/exp1_udf_baseline
python run.py

# Step 2: 训练技巧
cd ../exp2_training_trick_ablation
python run.py

# Step 3: 激活函数
cd ../exp3_activation_ablation
python run.py

# Step 4: MFCD验证
cd ../exp4_mfcd_definition
python run.py
```

---

## 📈 结果汇总

### 定量指标对比

| 实验 | 对比内容 | 指标 |
|------|----------|------|
| **Exp 1** | FFB vs UDF vs NeU | SymMFCD, Fragment Recall |
| **Exp 1** | Flood vs MIND | Time, Quality |
| **Exp 2** | Training tricks | SymMFCD |
| **Exp 3** | Activations | SymMFCD |
| **Exp 4** | MFCD definitions | Toy + Real |

### 论文Figure

**Figure 1**: Exp 1核心对比（3×2 grid）
**Figure 2**: Exp 2训练技巧消融（bar chart）
**Figure 3**: Exp 3激活函数消融（bar chart）
**Figure 4**: Exp 4 MFCD对比（scatter plot）

---

## 🔗 依赖关系

```
Exp 1 (基线)
  ↓
  ├─→ Exp 2 (以Exp 1的FFB+Flood为基线)
  ├─→ Exp 3 (以Exp 1的FFB+Flood为基线)
  └─→ Exp 4 (使用Exp 1-3的所有结果)
```

**建议顺序**: Exp 1 → Exp 2 → Exp 3 → Exp 4

---

## 📁 共享资源

### 数据
- `data/npz-resample/` - FFB编码数据
- `data/npz-udf/` - UDF编码数据
- `data/original_meshes/` - Ground truth

### 模型Checkpoints
- `data/ckpts/ffb_mlp/` - FFB-MLP
- `data/ckpts/udf_mlp/` - UDF-MLP
- `data/ckpts/neuraludf_mlp/` - NeuralUDF

### 工具脚本
- `src/extract_mesh_flooding.py` - Flooding抽取
- `src/extract_mesh_with_mind.py` - MIND抽取
- `experiments/exp4_mfcd_definition/symmetric_mfcd.py` - MFCD计算

---

## ⏱️ 预计时间

| 实验 | 训练时间 | 抽取时间 | 评估时间 | 总计 |
|------|----------|----------|----------|------|
| **Exp 1** | 2小时 | 1小时 | 20分钟 | ~3.5小时 |
| **Exp 2** | 2小时 | 30分钟 | 10分钟 | ~2.5小时 |
| **Exp 3** | 1.5小时 | 30分钟 | 10分钟 | ~2小时 |
| **Exp 4** | - | - | 30分钟 | ~30分钟 |
| **总计** | | | | **~8.5小时** |

---

**文档版本**: v1.0
**创建日期**: 2026-03-03
**维护者**: 请保持此文档与各exp的README.md同步
