# FFB-DF 编码方式澄清

**重要发现**: FFB-DF编码器生成的不是标准SDF，而是一种**混合编码**

**日期**: 2026-03-02

---

## 问题发现

用户指出：FFB-DF里面的encoding可能是**融合两个方式的一个encoding**

**验证结果**：✅ 完全正确！

---

## FFB-DF 编码的实际实现

### 关键代码

**文件**: `src/encoder_ffb-df_mlp.py:136-144, 192-201`

```python
if bool_ffbdf:
    # FFBDF: compute SDF using multiple objects
    sdf_values = np.full(len(poisson_grid_points), np.inf)

    for i, (obj_vertices, obj_faces, max_dist) in enumerate(
        zip(all_vertices, all_faces, max_distances)
    ):
        # 1. 计算原始SDF
        obj_sdf = igl.signed_distance(
            poisson_grid_points, obj_vertices, obj_faces
        )[0]

        # 2. ⚠️ 关键归一化：内部点除以max_dist
        inside_mask = obj_sdf < 0
        obj_sdf[inside_mask] = obj_sdf[inside_mask] / max_dist

        # 3. 取所有碎片的最小值
        sdf_values = np.minimum(sdf_values, obj_sdf)
```

### max_dist 的计算

**文件**: `src/encoder_ffb-df_mlp.py:68-97`

```python
# 对每个碎片计算max_dist
for obj in objs:
    vertices = np.array(obj.points)
    faces = np.array(obj.cells)

    # 在网格上采样SDF
    grid_points = ...  # 128³ 网格
    sdf_values = igl.signed_distance(grid_points, vertices, faces)[0]

    # max_dist = 碎片内部最远点到表面的距离
    inside_mask = sdf_values < 0
    if np.any(inside_mask):
        farthest_dist = np.max(-sdf_values[inside_mask])
    else:
        farthest_dist = np.linalg.norm(bbox_max - bbox_min)

    max_distances.append(farthest_dist)
```

**max_dist 含义**：
- 每个碎片内部，距离表面最远的点的距离
- 用于将内部SDF归一化到 [-1, 0] 范围

---

## FFB编码的数学定义

### 对于单个碎片 i

```
SDF_i(p) = signed_distance(p, Fragment_i)

FFB_i(p) = {
    SDF_i(p),                    如果 SDF_i(p) >= 0  (外部)
    SDF_i(p) / max_dist_i,       如果 SDF_i(p) < 0   (内部)
}
```

### 对于多个碎片

```
FFB(p) = min_i FFB_i(p)
```

**特点**：
- 外部点：保持原始SDF（未归一化，单位：米或场景单位）
- 内部点：归一化到 [-1, 0] 范围
- 多碎片：取最小值（最接近的碎片）

---

## 编码范围分析

### 理论范围

```python
FFB(p) ∈ {
    [-1, 0]      如果 p 在某个碎片内部  ← 归一化，有界
    [0, +∞)      如果 p 在所有碎片外部  ← 未归一化，无界
}
```

### 实际数据范围（来自实验）

```python
# 来自实验输出：experiments/udf_baseline/RESULTS.md

SDF (FFB) 数据特征:
- 样本数: 5个对象
- 点数/样本: 352,000 点
- SDF值范围: [-1.00, 0.95]  ← 注意：内部最小-1.0
- 均值: ~0.099
- 标准差: ~0.165
```

**观察**：
- 内部最小值 ≈ -1.0 → 符合归一化预期
- 外部最大值 ≈ 0.95 → 说明采样主要集中在表面附近
- 均值为正 → 外部点多于内部点

---

## 与标准SDF/UDF的对比

| 特性 | 标准SDF | FFB编码 | 标准UDF |
|------|---------|---------|---------|
| **外部** | 正值，原始距离 | ✅ 正值，原始距离 | 正值，原始距离 |
| **内部** | 负值，原始距离 | ⚠️ 负值，**归一化** | N/A（无内部） |
| **正负号** | ✅ 有 | ✅ 有 | ❌ 无 |
| **值域** | $(-\infty, +\infty)$ | $[-1, +\infty)$ | $[0, +\infty)$ |
| **内部有界** | ❌ | ✅ [-1, 0] | N/A |
| **外部有界** | ❌ | ❌ | ❌ |

**FFB编码的独特性**：
- ✅ 区分内外（有正负号）← 像SDF
- ✅ 内部归一化（有界）← 不像SDF
- ⚠️ 外部未归一化 ← 不像典型的normalized field

---

## 训练时的使用方式

### FFB-MLP 训练

**文件**: `src/train_ffb_mlp.py:46-50`

```python
all_pts, all_vals = [], []
for f in npz_files:
    d = np.load(f)
    all_pts.append(d["poisson_grid_points"].astype(np.float32))
    all_vals.append(d["sdf_values"].astype(np.float32).ravel())
    # ← 直接使用FFB编码，没有任何转换
```

**FFB-MLP学习的函数**：
```
f_FFB-MLP(p) ≈ FFB(p)
```

**特点**：
- 直接拟合混合编码
- 网络需要学习不对称的距离场（内部归一化，外部不归一化）

---

### UDF-MLP 训练

**文件**: `src/train_udf_mlp.py:70-75`

```python
all_pts, all_vals = [], []
for f in npz_files:
    d = np.load(f)
    all_pts.append(d["poisson_grid_points"].astype(np.float32))

    # ⚠️ 关键：取绝对值！
    v = d["udf_values"] if "udf_values" in d else d["sdf_values"]
    all_vals.append(np.abs(v).astype(np.float32).ravel())
```

**情况1**: 如果使用 `data/npz-udf/` 数据
```python
v = d["udf_values"]  # 已经是UDF（非负）
all_vals.append(np.abs(v))  # abs不改变值
```

**情况2**: 如果使用 `data/npz-resample/` 数据（fallback）
```python
v = d["sdf_values"]  # FFB编码（有正负）
all_vals.append(np.abs(v))  # abs转换为UDF
```

**UDF-MLP学习的函数**：
```
f_UDF-MLP(p) ≈ |FFB(p)|  或  UDF(p)
```

**特点**：
- 丢失内外信息
- 学习对称的距离场

---

## 数据流图

```
Mesh (with fragments)
    ↓
vedo.split() → 多个碎片
    ↓
对每个碎片:
├─ 计算 max_dist (内部最远距离)
├─ 计算 SDF
└─ 归一化内部: SDF[inside] /= max_dist
    ↓
取最小值: min(FFB_1, FFB_2, ...)
    ↓
保存: npz(poisson_grid_points, sdf_values)
    ↓              ↓
    ↓              ├─→ FFB-MLP: 直接使用
    ↓              └─→ UDF-MLP: abs(sdf_values)
    ↓
也生成: data/npz-udf/ (纯UDF，通过unsigned distance)
    ↓
    └─→ UDF-MLP: 直接使用
```

---

## 为什么这样设计？

### FFB编码的动机

1. **内部归一化的好处**：
   - 不同大小的碎片有统一的内部值域 [-1, 0]
   - 避免大碎片的内部距离主导训练
   - 网络更容易学习（内部有界）

2. **外部不归一化的原因**：
   - 保持外部距离的实际意义
   - 远离表面的点可以用大的距离值表示
   - 适合fragment-based场景

3. **Fragment-Based的意义**：
   - 每个碎片有自己的max_dist
   - 小碎片和大碎片都能被网络学习
   - 适合破碎物体重建

---

## 对实验的影响

### 当前实验设置

```python
# 已完成的训练
FFB-MLP:  学习 FFB编码 (内部归一化, 外部原始)
UDF-MLP:  学习 |FFB编码| (始终非负)

# 数据来源
FFB-MLP ← data/npz-resample/  (FFB编码)
UDF-MLP ← data/npz-udf/       (纯UDF) 或 abs(data/npz-resample/)
```

### 实际训练的是

| 模型 | 输入 | 目标 | 学习的函数 |
|------|------|------|-----------|
| **FFB-MLP** | (x,y,z) | FFB编码 | 混合距离场（内部归一化） |
| **UDF-MLP** | (x,y,z) | 纯UDF | 无向距离场（对称） |

### 对比的公平性

**问题**：FFB-MLP和UDF-MLP学习的不是同一种field！

```
FFB-MLP学习:  FFB(p) = { SDF/max_dist if inside, SDF if outside }
UDF-MLP学习:  UDF(p) = distance_to_surface (unsigned)
```

**它们的区别**：
1. FFB保留内外信息，UDF不保留
2. FFB内部归一化，UDF不归一化
3. FFB是不对称的，UDF是对称的

**结论**：
- 这不是"SDF vs UDF"的对比
- 而是"FFB (normalized fragment-based) vs UDF"的对比

---

## 正确的理解

### 编码器生成的数据

| 编码器 | 输出文件 | 数据键 | 实际内容 |
|--------|---------|--------|---------|
| `encoder_ffb-df_mlp.py` | `data/npz-resample/*.npz` | `sdf_values` | **FFB编码**（混合） |
| `encoder_udf_mesh.py` | `data/npz-udf/*.npz` | `udf_values` | **纯UDF** |

### 训练的模型

| 模型 | 使用的数据 | 实际学习的 | 命名问题 |
|------|-----------|-----------|---------|
| `train_ffb_mlp.py` | `npz-resample/` | **FFB编码** | ⚠️ 命名为"SDF MLP"但学的是FFB |
| `train_udf_mlp.py` | `npz-udf/` | **纯UDF** | ✅ 确实是UDF MLP |

---

## 建议

### 1. 更新文档和代码注释

明确说明：
- `encoder_ffb-df_mlp.py` 生成的是 **FFB编码**，不是标准SDF
- `train_ffb_mlp.py` 中的 `SimpleSDFMLP` 实际上学习的是 **FFB**

### 2. 实验对比的准确描述

```
当前实验对比的是:
✅ FFB-MLP (归一化fragment-based field)
    vs
✅ UDF-MLP (unsigned distance field)

而不是:
❌ SDF-MLP (signed distance field)
    vs
❌ UDF-MLP
```

### 3. 可选：增加纯SDF基线

如果要对比标准SDF，可以：
```python
# 新增编码器：encoder_sdf.py
# 不对内部归一化
obj_sdf = igl.signed_distance(points, vertices, faces)[0]
# 直接使用，不除以max_dist
sdf_values = np.minimum(sdf_values, obj_sdf)
```

---

## 总结

1. **FFB-DF编码器** 生成的确实是**融合编码**：
   - 外部：标准SDF（正值，原始距离）
   - 内部：归一化SDF（负值，除以max_dist）

2. **FFB-MLP** 学习的是这个混合编码，不是纯SDF

3. **UDF-MLP** 学习的是纯UDF（通过取绝对值转换）

4. **对比实验** 实际是：
   ```
   归一化Fragment-Based Field (FFB)
        vs
   Unsigned Distance Field (UDF)
   ```

5. **这是合理的设计**，特别适合破碎物体场景

---

**文档版本**: v1.0 - FFB编码澄清
**最后更新**: 2026-03-02
**感谢用户发现这个重要细节！**
