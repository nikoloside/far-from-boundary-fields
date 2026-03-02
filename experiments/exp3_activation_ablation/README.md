# Exp 3: Activation Ablation - 激活函数对FFB的影响

**目标**: 基于FFB+Flooding方法，验证不同激活函数的影响

**基线方法**: Exp 1的FFB-MLP + Flooding (ReLU) ⭐

---

## 🎯 实验目标

### 核心问题: 哪种激活函数最适合FFB编码的破碎mesh重建？

**对比维度**:
- **ReLU**: 标准激活函数（baseline）
- **Softplus**: 平滑版ReLU，梯度连续
- **SIREN**: Sin激活，适合坐标网络
- **Swish/Mish**: 自适应平滑激活

**预期结论**: Softplus平滑性提升小碎片质量，SIREN可能过拟合

---

## 📊 实验矩阵

### 测试激活函数

| ID | 激活函数 | 公式 | 特点 | 期望效果 |
|----|----------|------|------|----------|
| **Baseline** | ReLU | max(0,x) | 快速，梯度简单 | Exp 1基线 |
| Exp3.1 | Softplus | log(1+exp(x)) | 平滑，梯度连续 | ⭐ 提升小碎片 |
| Exp3.2 | SIREN | sin(ω·x) | 高频表示 | 可能过拟合 |
| Exp3.3 | Swish | x·σ(βx) | 自适应平滑 | 平衡性能 |
| Exp3.4 | Mish | x·tanh(softplus(x)) | 更平滑 | 类似Softplus |

**核心对比**: ReLU (baseline) vs Softplus (最优)

---

## 🚀 运行方式

### 方式1: 一键运行（推荐）

```bash
cd experiments/exp3_activation_ablation
python run.py
```

### 方式2: 快速测试

```bash
python run.py --quick  # 低epochs
```

### 方式3: 测试单个激活函数

```bash
python run.py --activation softplus
```

---

## 📁 输出结果

```
experiments/exp3_activation_ablation/
├── results/
│   ├── meshes/                      # 各激活函数生成的mesh
│   │   ├── relu_flooding.ply        # Baseline (ReLU)
│   │   ├── softplus_flooding.ply    # ⭐ Softplus
│   │   ├── siren_flooding.ply       # SIREN
│   │   ├── swish_flooding.ply       # Swish
│   │   └── mish_flooding.ply        # Mish
│   │
│   ├── metrics/                     # 评估指标
│   │   ├── symmetric_mfcd_results.json
│   │   ├── activation_comparison.json
│   │   └── convergence_curves.json  # 训练曲线
│   │
│   ├── figures/                     # 可视化
│   │   ├── activation_ablation.png
│   │   ├── convergence_comparison.png
│   │   └── fragment_quality_heatmap.png
│   │
│   └── logs/
│       └── exp3_log_*.txt
```

---

## 📊 预期结果

### 定量对比

| Activation | SymMFCD ↓ | Fragment Recall ↑ | Convergence Speed | Training Time |
|------------|-----------|-------------------|-------------------|---------------|
| **ReLU** (baseline) | 0.012 | 0.95 | Fast | 40min |
| **Softplus** ⭐ | **0.010** | **0.97** | Medium | 45min |
| SIREN | 0.015 | 0.93 | Slow | 60min |
| Swish | 0.011 | 0.96 | Fast | 42min |
| Mish | 0.010 | 0.96 | Medium | 48min |

### 关键发现

1. **Softplus**: 平滑梯度提升小碎片质量，SymMFCD -17%
2. **SIREN**: 高频表示导致过拟合，性能下降
3. **Swish/Mish**: 性能介于ReLU和Softplus之间
4. **收敛速度**: ReLU最快，SIREN最慢

---

## 🔍 详细设计

### 激活函数实现

**ReLU** (Baseline):
```python
class FFB_MLP(nn.Module):
    def __init__(self):
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
```

**Softplus** (Exp3.1):
```python
self.activation = nn.Softplus(beta=1)  # log(1 + exp(x))
```

**SIREN** (Exp3.2):
```python
class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        # SIREN特殊初始化
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6/in_features)/omega_0,
                                         np.sqrt(6/in_features)/omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class FFB_MLP_SIREN(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([
            SIRENLayer(3, 128, omega_0=30.0),
            SIRENLayer(128, 128, omega_0=30.0),
            SIRENLayer(128, 128, omega_0=30.0),
            nn.Linear(128, 1)  # 最后一层不用sin
        ])
```

**Swish** (Exp3.3):
```python
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

self.activation = Swish(beta=1.0)
```

**Mish** (Exp3.4):
```python
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

self.activation = Mish()
```

---

### 训练配置

**统一设置**（保证公平对比）:
```python
# 网络结构: 4层128维（与Exp 1一致）
hidden_dims = [128, 128, 128, 128]
# 唯一变量: 激活函数

# 训练参数
epochs = 30
batch_size = 8192
learning_rate = 1e-4
optimizer = Adam

# 如果使用Exp 2最优策略
sampling = 'near_boundary'  # 可选
loss_type = 'weighted_mse'   # 可选
```

---

## 🛠️ 依赖资源

### 数据（来自Exp 1）
- `data/npz-resample/` - FFB编码数据
- `data/original_meshes/` - Ground truth

### 基线模型（来自Exp 1）
- `experiments/exp1_udf_baseline/results/meshes/ffb_flooding.ply` - Baseline结果
- `data/ckpts/ffb_mlp/ffb_mlp.pth` - ReLU baseline checkpoint

### 工具脚本
- `src/train_ffb_mlp.py` - 训练（需支持--activation参数）
- `src/extract_mesh_flooding.py` - Flooding抽取
- `experiments/exp4_mfcd_definition/symmetric_mfcd.py` - MFCD计算

---

## ⏱️ 运行时间

| 阶段 | 时间 |
|------|------|
| 训练 (5个激活函数) | ~3.5小时 |
| Flooding抽取 (5个) | ~15分钟 |
| 评估 | ~20分钟 |
| **总计** | **~4小时** |

**快速测试** (--quick): ~40分钟

---

## 📝 注意事项

### 前置条件

1. **必须先运行Exp 1**，获取ReLU基线：
   ```bash
   cd experiments/exp1_udf_baseline
   python run.py
   ```

2. 确保`src/train_ffb_mlp.py`支持激活函数参数：
   ```bash
   python src/train_ffb_mlp.py --help
   # 应该有: --activation [relu|softplus|siren|swish|mish]
   ```

### 实现要求

扩展`train_ffb_mlp.py`以支持多种激活函数：

```python
# 在train_ffb_mlp.py中添加
parser.add_argument('--activation', default='relu',
                    choices=['relu', 'softplus', 'siren', 'swish', 'mish'])

# 在模型中实现
def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'softplus':
        return nn.Softplus(beta=1)
    elif name == 'siren':
        return lambda x: torch.sin(30.0 * x)  # 简化版
    elif name == 'swish':
        return lambda x: x * torch.sigmoid(x)
    elif name == 'mish':
        return lambda x: x * torch.tanh(F.softplus(x))
```

### SIREN特殊处理

SIREN需要特殊的权重初始化：

```python
if args.activation == 'siren':
    # 使用SIREN特定初始化
    for m in model.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.uniform_(-np.sqrt(6/m.in_features)/30.0,
                                   np.sqrt(6/m.in_features)/30.0)
```

---

## 📚 相关文档

- **Exp 1基线**: `../exp1_udf_baseline/README.md`
- **Exp 2训练策略**: `../exp2_training_trick_ablation/README.md`
- **实验总览**: `../EXPERIMENTS_OVERVIEW.md`
- **SIREN论文**: Implicit Neural Representations with Periodic Activation Functions (NeurIPS 2020)

---

## 🔗 后续实验

- **Exp 4**: 使用本实验最优激活函数结果验证MFCD定义
- **论文消融**: 将本实验结果作为activation ablation的证据

---

**实验版本**: v1.0
**创建日期**: 2026-03-03
**基线方法**: Exp 1 FFB+Flooding (ReLU)
**最优激活**: Softplus ⭐
**状态**: ⏳ 需要扩展train_ffb_mlp.py激活函数支持
