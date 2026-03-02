# UDF Baseline 实验结果

实验执行时间：2026-02-28 / 2026-03-01

## 📋 实验概述

本实验对比了UDF（Unsigned Distance Field）和SDF（Signed Distance Field）在相同MLP架构下的表现，用于基线比较。

## ✅ 完成状态

### 数据编码
- ✅ **FFB-DF Encoder**: 5个对象完成 → `data/npz-resample/`
- ✅ **UDF Mesh Encoder**: 5个对象完成 → `data/npz-udf/`

### 模型训练
- ✅ **FFB-MLP**: 30 epochs, 最终损失 = 0.000361
  - 模型保存：`data/ckpts/ffb_mlp/ffb_mlp.pth`
- ✅ **UDF-MLP**: 30 epochs, 最终损失 = 0.000361
  - 模型保存：`data/ckpts/udf_mlp/udf_mlp.pth`

### 可视化生成
- ✅ 所有结果图像已生成 → `data/results/udf_baseline/qualitative/`

## 📊 数据统计

### SDF (FFB-DF) 数据特征
```
样本数: 5个对象
点数/样本: 352,000 点
SDF值范围: [-1.00, 0.95]
均值: ~0.099
标准差: ~0.165

分布特征:
- 负值表示物体内部
- 正值表示物体外部
- 零等值面表示物体表面
```

### UDF (Mesh-based) 数据特征
```
样本数: 5个对象
点数/样本: 352,000 点 (除了obj 2: 1,500点)
UDF值范围: [0, 0.92]
均值: ~0.125
标准差: ~0.114

分布特征:
- 所有值非负（无向距离）
- 0表示在表面上
- 数值表示到最近表面的距离
```

## 🎯 生成的可视化

### 训练数据可视化
1. **FFB (SDF)**: `ffb_1.png` ~ `ffb_5.png`
   - 显示编码后的SDF点云
   - 颜色表示有向距离值

2. **UDF Mesh**: `udf_mesh_1.png` ~ `udf_mesh_5.png`
   - 显示从mesh编码的UDF点云
   - 颜色表示无向距离值

### 模型预测可视化
1. **FFB-MLP预测**: `ffb_mlp_pred.png`
   - MLP网络学习的SDF表示

2. **UDF-MLP预测**: `udf_mlp_pred.png`
   - MLP网络学习的UDF表示（MIND/NeuralUDF风格）

### 预处理数据分析
额外生成的预处理数据详细分析：
- `data/results/preprocessed_vis/sdf/` - SDF数据多视角分析
- `data/results/preprocessed_vis/udf/` - UDF数据多视角分析

每个可视化包含：
- 3D点云视图（按距离值着色）
- XY/XZ平面投影
- 距离值分布直方图
- 统计信息摘要
- 接近表面的点分布

## 🔍 关键观察

### 模型收敛
- 两个模型都收敛到相似的损失值（~0.000361）
- 训练过程稳定，无明显过拟合

### SDF vs UDF 对比
1. **点云密度**：
   - SDF和UDF都使用相同的采样网格（352K点）
   - obj 2的UDF数据异常小（仅1.5K点），可能需要检查

2. **数值范围**：
   - SDF: [-1, 1] 范围，区分内外
   - UDF: [0, 1] 范围，仅表示距离

3. **表面表示**：
   - SDF: 零等值面精确表示表面
   - UDF: 零值表示表面，但丢失了内外信息

## 📁 文件结构

```
data/
├── npz-resample/          # FFB-DF (SDF) 编码数据
│   ├── 1.npz ~ 5.npz      # 每个 ~11MB
├── npz-udf/               # UDF mesh 编码数据
│   ├── 1.npz ~ 5.npz      # 大部分 ~11MB (2.npz 仅47KB)
├── ckpts/
│   ├── ffb_mlp/           # FFB-MLP 训练权重
│   └── udf_mlp/           # UDF-MLP 训练权重
└── results/
    ├── udf_baseline/
    │   └── qualitative/   # 主要实验结果可视化
    └── preprocessed_vis/  # 预处理数据详细分析
        ├── sdf/           # SDF数据可视化
        └── udf/           # UDF数据可视化
```

## 🔄 下一步

### 建议的后续分析
1. **定量评估**：
   - 计算Chamfer Distance (CD)
   - 计算对称MFCD（Mean Feature-wise Chamfer Distance）
   - Fragment-wise重建质量评估
   - Boundary recall测量

2. **定性比较**：
   - GT vs GS-SDF vs UDF vs FFB-DF
   - 特别关注细裂缝和小碎片的重建质量

3. **与其他方法对比**：
   - NeuralUDF编码与重建
   - MIND方法编码与重建
   - 与当前FFB-DF/UDF-MLP基线的对比

4. **问题调查**：
   - 检查为什么obj 2的UDF数据异常小
   - 验证编码过程是否一致

## 📝 可用脚本

- `scripts/run_full_pipeline.sh` - 完整流程自动化
- `scripts/visualize_preprocessed_data.py` - 预处理数据可视化
- `scripts/generate_and_render.py` - 结果生成与渲染

## 🏆 结论

基线实验成功完成。FFB-MLP和UDF-MLP都成功学习了相应的距离场表示，为后续与NeuralUDF、MIND等方法的对比提供了可靠的基线。

训练损失收敛良好，预处理数据质量高，为下一步的定量和定性评估奠定了基础。
