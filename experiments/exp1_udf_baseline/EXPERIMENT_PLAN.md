# UDF Baseline 完整对比实验计划

**日期**: 2026-03-02
**目标**: 对比4种UDF/距离场方法在破碎形状重建任务中的表现

---

## 🎯 实验目标

对比以下4种方法：

| ID | 方法 | 描述 | 状态 |
|----|------|------|------|
| **1** | **纯UDF** | UDF-MLP (4层, 128维, multires=4) | ✅ 已完成 |
| **2** | **归一化FFB** | FFB-MLP (4层, 128维, multires=4) | ✅ 已完成 |
| **3** | **NeuralUDF** | 完整架构 (6层, 256维, multires=6, skip+init+norm) | ⏳ 需训练 |
| **4** | **UDF+MIND** | 方法1/3 + MIND后处理 | ⏳ 需实现 |

---

## 📊 实验矩阵

### 对比维度

```
编码方式:
├─ 纯UDF (unsigned distance field)
├─ 归一化FFB (fragment-based field, 内部归一化)
└─ (可选) 标准SDF (signed distance field, 不归一化)

网络架构:
├─ 简化MLP (4层, 128维, multires=4)
└─ 完整NeuralUDF (6层, 256维, multires=6, skip+init+norm)

后处理:
├─ 无后处理 (直接从UDF提取mesh)
└─ MIND优化 (非流形mesh提取)
```

### 完整对比表

| 方法 | 编码 | 架构 | 后处理 | 输出 |
|------|------|------|--------|------|
| UDF-MLP | 纯UDF | 简化 | 无 | UDF函数 |
| FFB-MLP | 归一化FFB | 简化 | 无 | FFB函数 |
| NeuralUDF-MLP | 纯UDF | 完整 | 无 | UDF函数 |
| UDF-MLP + MIND | 纯UDF | 简化 | MIND | Mesh |
| NeuralUDF + MIND | 纯UDF | 完整 | MIND | Mesh |
| (可选) FFB-MLP + MIND | 归一化FFB | 简化 | MIND | Mesh |

---

## 🔧 实施步骤

### 步骤1: 训练NeuralUDF-MLP ⏳

**目标**: 在相同的UDF数据上，使用NeuralUDF的完整架构训练

#### 创建训练脚本

**文件**: `src/train_neuraludf_mlp.py`

```python
"""
训练NeuralUDF架构在mesh-based UDF数据上
对比完整架构 vs 简化架构的差异
"""
import os
import sys
import torch
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
import torch.nn as nn

# 导入NeuralUDF的网络
NEURALUDF_ROOT = os.path.join(
    os.path.dirname(__file__), "..",
    "experiments", "udf_baseline", "NeuralUDF"
)
sys.path.insert(0, NEURALUDF_ROOT)
from models.fields import UDFNetwork

def train(npz_dir, ckpt_dir, epochs=100, batch_size=4096, lr=1e-4):
    """
    训练NeuralUDF架构的MLP
    """
    # 加载数据
    npz_files = sorted(glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No npz in {npz_dir}")

    all_pts, all_vals = [], []
    for f in npz_files:
        d = np.load(f)
        all_pts.append(d["poisson_grid_points"].astype(np.float32))
        v = d["udf_values"] if "udf_values" in d else d["sdf_values"]
        all_vals.append(np.abs(v).astype(np.float32).ravel())

    pts = np.vstack(all_pts)
    vals = np.concatenate(all_vals)
    print(f"NeuralUDF-MLP: Training on {len(pts)} samples from {len(npz_files)} files")

    # 创建dataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(pts),
        torch.from_numpy(vals).unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 创建NeuralUDF网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UDFNetwork(
        d_in=3,
        d_out=1,
        d_hidden=256,           # ← 更大的隐藏层
        n_layers=6,             # ← 更深的网络
        skip_in=(4,),           # ← skip connection
        multires=6,             # ← 更高频率编码
        scale=1.0,
        bias=0.5,
        geometric_init=True,    # ← geometric initialization
        weight_norm=True,       # ← weight normalization
        udf_type='abs'          # ← UDF输出类型
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    os.makedirs(ckpt_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        for pts_b, vals_b in loader:
            pts_b = pts_b.to(device)
            vals_b = vals_b.to(device)
            pred = model.udf(pts_b)
            loss = nn.functional.mse_loss(pred, vals_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={loss_sum/len(loader):.6f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "neuraludf_mlp.pth"))
    print(f"Saved to {ckpt_dir}/neuraludf_mlp.pth")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", default="data/npz-udf")
    parser.add_argument("--ckpt_dir", default="data/ckpts/neuraludf_mlp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    train(args.npz_dir, args.ckpt_dir, args.epochs, args.batch_size)
```

**训练命令**:
```bash
python src/train_neuraludf_mlp.py --epochs 100
```

**预期时间**: ~30-60分钟（比简化版慢2-3倍）

---

### 步骤2: 实现MIND后处理 ⏳

**目标**: 使用MIND从训练好的UDF网络提取高质量mesh

#### 创建MIND提取脚本

**文件**: `src/extract_mesh_with_mind.py`

```python
"""
使用MIND从训练好的UDF网络提取mesh
"""
import torch
import numpy as np
import argparse
import sys
import os

# 导入模型
from train_udf_mlp import SimpleUDFMLP
NEURALUDF_ROOT = os.path.join(os.path.dirname(__file__), "..",
                              "experiments", "udf_baseline", "NeuralUDF")
sys.path.insert(0, NEURALUDF_ROOT)
from models.fields import UDFNetwork

# 导入MIND
MIND_ROOT = os.path.join(os.path.dirname(__file__), "..",
                         "experiments", "udf_baseline", "MIND", "src")
sys.path.insert(0, MIND_ROOT)
from mind import MIND


def load_model(model_type, ckpt_path, device):
    """
    加载训练好的UDF模型
    """
    if model_type == "udf_mlp":
        model = SimpleUDFMLP(d_hidden=128, n_layers=4, multires=4)
    elif model_type == "ffb_mlp":
        from train_ffb_mlp import SimpleSDFMLP
        model = SimpleSDFMLP(d_hidden=128, n_layers=4, multires=4)
    elif model_type == "neuraludf_mlp":
        model = UDFNetwork(
            d_in=3, d_out=1, d_hidden=256, n_layers=6,
            skip_in=(4,), multires=6, geometric_init=True,
            weight_norm=True, udf_type='abs'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def create_udf_query_func(model, device, model_type):
    """
    创建MIND需要的UDF查询函数
    """
    @torch.no_grad()
    def query_func(points):
        """
        points: (N, 3) numpy array or torch tensor
        返回: (N,) UDF values
        """
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        points = points.to(device)

        # 根据模型类型调用不同的方法
        if model_type == "ffb_mlp":
            # FFB-MLP输出可能是负值，取绝对值转为UDF
            output = model.sdf(points)
            udf = torch.abs(output)
        else:
            # UDF-MLP 和 NeuralUDF-MLP
            udf = model.udf(points)

        return udf.squeeze()

    return query_func


def extract_mesh(model_type, ckpt_path, output_path,
                resolution=256, max_iter=200,
                laplacian_weight=1000.0, learning_rate=0.0005):
    """
    使用MIND提取mesh
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    print(f"\n[1/3] Loading {model_type} from {ckpt_path}")
    model = load_model(model_type, ckpt_path, device)

    # 2. 创建UDF查询函数
    print("[2/3] Creating UDF query function")
    udf_query = create_udf_query_func(model, device, model_type)

    # 3. 运行MIND
    print(f"[3/3] Running MIND (resolution={resolution}, max_iter={max_iter})")
    mind = MIND(
        query_func=udf_query,
        resolution=resolution,
        max_iter=max_iter,
        laplacian_weight=laplacian_weight,
        learning_rate=learning_rate,
        bound_min=[-1, -1, -1],
        bound_max=[1, 1, 1]
    )

    final_mesh = mind.run()

    # 4. 保存mesh
    print(f"Saving mesh to {output_path}")
    final_mesh.export(output_path)
    print(f"Done! Mesh saved with {len(final_mesh.vertices)} vertices, "
          f"{len(final_mesh.faces)} faces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mesh using MIND")
    parser.add_argument("--model_type", required=True,
                       choices=["udf_mlp", "ffb_mlp", "neuraludf_mlp"],
                       help="Type of trained model")
    parser.add_argument("--ckpt", required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", required=True,
                       help="Output mesh file (.ply or .obj)")
    parser.add_argument("--resolution", type=int, default=256,
                       help="Grid resolution for MIND")
    parser.add_argument("--max_iter", type=int, default=200,
                       help="MIND optimization iterations")
    parser.add_argument("--laplacian_weight", type=float, default=1000.0,
                       help="Laplacian regularization weight")
    parser.add_argument("--lr", type=float, default=0.0005,
                       help="MIND learning rate")

    args = parser.parse_args()

    extract_mesh(
        model_type=args.model_type,
        ckpt_path=args.ckpt,
        output_path=args.output,
        resolution=args.resolution,
        max_iter=args.max_iter,
        laplacian_weight=args.laplacian_weight,
        learning_rate=args.lr
    )
```

**使用示例**:
```bash
# UDF-MLP + MIND
python src/extract_mesh_with_mind.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_mlp_mind.ply

# NeuralUDF-MLP + MIND
python src/extract_mesh_with_mind.py \
    --model_type neuraludf_mlp \
    --ckpt data/ckpts/neuraludf_mlp/neuraludf_mlp.pth \
    --output data/results/meshes/neuraludf_mlp_mind.ply

# FFB-MLP + MIND (可选)
python src/extract_mesh_with_mind.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mlp_mind.ply
```

---

### 步骤3: 批量生成所有结果 ⏳

**文件**: `scripts/run_all_methods.sh`

```bash
#!/bin/bash
# 运行所有对比方法
set -e
cd "$(dirname "$0")/.."

echo "=== UDF Baseline 完整对比实验 ==="

# 确保目录存在
mkdir -p data/results/meshes
mkdir -p data/ckpts/neuraludf_mlp

# 步骤1: 训练NeuralUDF-MLP（如果还没训练）
if [ ! -f data/ckpts/neuraludf_mlp/neuraludf_mlp.pth ]; then
    echo "[1/4] Training NeuralUDF-MLP..."
    python src/train_neuraludf_mlp.py --epochs 100
else
    echo "[1/4] NeuralUDF-MLP already trained, skipping..."
fi

# 步骤2: 使用MIND提取mesh（3个模型）
echo "[2/4] Extracting meshes with MIND..."

echo "  - UDF-MLP + MIND"
python src/extract_mesh_with_mind.py \
    --model_type udf_mlp \
    --ckpt data/ckpts/udf_mlp/udf_mlp.pth \
    --output data/results/meshes/udf_mlp_mind.ply \
    --resolution 256

echo "  - NeuralUDF-MLP + MIND"
python src/extract_mesh_with_mind.py \
    --model_type neuraludf_mlp \
    --ckpt data/ckpts/neuraludf_mlp/neuraludf_mlp.pth \
    --output data/results/meshes/neuraludf_mlp_mind.ply \
    --resolution 256

echo "  - FFB-MLP + MIND (optional)"
python src/extract_mesh_with_mind.py \
    --model_type ffb_mlp \
    --ckpt data/ckpts/ffb_mlp/ffb_mlp.pth \
    --output data/results/meshes/ffb_mlp_mind.ply \
    --resolution 256

# 步骤3: 可视化对比
echo "[3/4] Generating visualizations..."
python scripts/visualize_all_results.py

# 步骤4: 计算定量指标（如果有GT mesh）
echo "[4/4] Computing metrics..."
# python scripts/compute_metrics.py

echo "=== 完成！结果保存在 data/results/ ==="
```

---

## 📊 评估指标

### 定量指标

1. **Global指标**:
   - Chamfer Distance (CD)
   - 对称MFCD (Mean Feature-wise Chamfer Distance)
   - Hausdorff Distance

2. **Fragment-wise指标**:
   - Per-fragment CD
   - Fragment IoU
   - 小碎片召回率

3. **Boundary指标**:
   - 内部边界召回率
   - 边界精度

### 定性对比

- 细裂缝重建质量
- 小碎片保留情况
- 内部边界清晰度
- 整体拓扑正确性

---

## 📁 预期输出结构

```
data/
├── ckpts/
│   ├── ffb_mlp/
│   │   └── ffb_mlp.pth           ✅ 已有
│   ├── udf_mlp/
│   │   └── udf_mlp.pth           ✅ 已有
│   └── neuraludf_mlp/
│       └── neuraludf_mlp.pth     ⏳ 待生成
│
├── results/
│   ├── meshes/
│   │   ├── udf_mlp_mind.ply      ⏳ 待生成
│   │   ├── neuraludf_mlp_mind.ply ⏳ 待生成
│   │   └── ffb_mlp_mind.ply      ⏳ 待生成
│   │
│   ├── visualizations/
│   │   ├── comparison_all.png    ⏳ 待生成
│   │   ├──细裂缝对比.png
│   │   └── 小碎片对比.png
│   │
│   └── metrics/
│       ├── quantitative.csv      ⏳ 待生成
│       └── per_fragment.csv
```

---

## 🎯 对比总结

### 预期对比结论（需实验验证）

| 方法 | 编码 | 架构 | 后处理 | 预期特点 |
|------|------|------|--------|---------|
| **UDF-MLP** | 纯UDF | 简化 | - | 基线，简单快速 |
| **FFB-MLP** | 归一化FFB | 简化 | - | 内部归一化，保留内外信息 |
| **NeuralUDF-MLP** | 纯UDF | 完整 | - | 更强表达能力 |
| **UDF-MLP + MIND** | 纯UDF | 简化 | MIND | 非流形处理 |
| **NeuralUDF + MIND** | 纯UDF | 完整 | MIND | 最佳组合？ |

### 关键对比

**对比1**: 编码方式的影响
```
UDF-MLP vs FFB-MLP
→ 纯UDF vs 归一化FFB
→ 架构相同，只比较编码差异
```

**对比2**: 网络架构的影响
```
UDF-MLP vs NeuralUDF-MLP
→ 简化 vs 完整
→ 编码相同，只比较架构差异
```

**对比3**: MIND后处理的影响
```
UDF-MLP vs UDF-MLP + MIND
NeuralUDF-MLP vs NeuralUDF-MLP + MIND
→ 是否使用MIND的差异
```

**对比4**: 综合最佳
```
所有方法对比
→ 找出最佳组合
```

---

## ⏱️ 时间估计

| 任务 | 预计时间 |
|------|---------|
| 训练 NeuralUDF-MLP | 30-60分钟 |
| MIND提取 (3个模型) | 每个20-30分钟 |
| 可视化生成 | 10分钟 |
| 指标计算 | 5分钟 |
| **总计** | **2-3小时** |

---

## 📝 后续工作

1. **定量分析**:
   - 统计显著性检验
   - 不同场景的鲁棒性测试

2. **消融实验**:
   - Skip connection的作用
   - Geometric init的作用
   - Weight norm的作用
   - MIND参数敏感性

3. **扩展实验**:
   - 更多测试对象
   - 不同分辨率的影响
   - 不同采样密度的影响

---

**文档版本**: v1.0
**创建日期**: 2026-03-02
**状态**: 实验计划阶段
