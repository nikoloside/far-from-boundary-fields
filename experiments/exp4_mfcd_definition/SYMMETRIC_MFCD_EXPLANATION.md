# Symmetric Multi-Fragment Chamfer Distance (SymMFCD)

**日期**: 2026-03-02
**目的**: 澄清单向和双向MFCD的区别

---

## 问题：当前实现是单向的

### Chamfer Distance vs Multi-Fragment CD的层次

需要区分两个层次：

1. **点云层次的Chamfer Distance (CD)**
   - 比较两个点云P1和P2
   - 可以是单向或双向

2. **碎片层次的Multi-Fragment CD (MFCD)**
   - 比较两组碎片A={a₁, a₂, ...}和B={b₁, b₂, ...}
   - 涉及碎片匹配问题
   - 可以是单向或双向

---

## 当前实现分析

### 当前代码：`vc_plot_charmer_distance_error.py`

```python
# Lines 24-30: CD函数（双向✅）
def chamfer_distance(points1, points2):
    dist_matrix = cdist(points1, points2)
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)  # P1→P2
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)  # P2→P1
    chamfer_dist = np.mean(min_dist_1_to_2) + np.mean(min_dist_2_to_1)
    return chamfer_dist
```

**结论**: Chamfer Distance本身是**双向的**✅

---

### 当前代码：MFCD计算（单向❌）

```python
# Lines 126-134: MFCD计算
for orig_shape in orig_shapes:  # 对原始的每个碎片
    orig_points = sample_points(orig_shape, num_sample_points)
    min_error = float('inf')

    # 在重建结果中找最接近的碎片
    for filtered_shape in filtered_shapes:
        filtered_points = sample_points(filtered_shape, num_sample_points)
        error = chamfer_distance(orig_points, filtered_points)
        min_error = min(min_error, error)

    method_errors.append(min_error)  # 记录这个碎片的误差

# 平均所有原始碎片的误差
method_avg_errors[method] = np.nanmean(method_errors)
```

**实际计算的是**:
```
MFCD(Orig→Recon) = (1/|Orig|) * Σᵢ minⱼ CD(orig_i, recon_j)
```

**缺失的方向**:
```
MFCD(Recon→Orig) = (1/|Recon|) * Σⱼ minᵢ CD(recon_j, orig_i)
```

**结论**: MFCD是**单向的**❌（只计算Orig→Recon）

---

## 为什么需要对称MFCD？

### 单向MFCD的问题

**情况1: 重建缺失碎片**
```
原始: 10个碎片
重建: 5个碎片（缺失5个）

MFCD(Orig→Recon): 高误差（原始碎片找不到对应）
MFCD(Recon→Orig): 低误差（重建的都能找到对应）

单向MFCD只报告第一种，无法检测到"多余碎片"的问题
```

**情况2: 重建多余碎片**
```
原始: 10个碎片
重建: 15个碎片（多5个噪声碎片）

MFCD(Orig→Recon): 低误差（原始碎片都能找到对应）
MFCD(Recon→Orig): 高误差（多余碎片找不到好的对应）

单向MFCD无法检测到"多余碎片"的问题
```

### 对称MFCD的优势

```
SymMFCD(A, B) = MFCD(A→B) + MFCD(B→A)
```

**优点**:
- ✅ 检测缺失碎片（A→B方向）
- ✅ 检测多余碎片（B→A方向）
- ✅ 对称性：SymMFCD(A,B) = SymMFCD(B,A)
- ✅ 全面评估重建质量

---

## 数学定义

### 单向MFCD（当前实现）

**定义**:
```
MFCD(A→B) = (1/|A|) * Σᵢ minⱼ CD(aᵢ, bⱼ)
```

**含义**:
- 对A中的每个碎片aᵢ
- 找到B中最接近的碎片bⱼ
- 计算CD(aᵢ, bⱼ)
- 平均所有原始碎片的误差

**特点**:
- 反映原始碎片能否在重建中找到对应
- 不关心重建中是否有多余碎片

---

### 对称MFCD（新实现）

**定义**:
```
SymMFCD(A, B) = MFCD(A→B) + MFCD(B→A)

其中:
MFCD(A→B) = (1/|A|) * Σᵢ minⱼ CD(aᵢ, bⱼ)
MFCD(B→A) = (1/|B|) * Σⱼ minᵢ CD(bⱼ, aᵢ)
```

**含义**:
- **方向1 (A→B)**: 原始碎片的再现质量
- **方向2 (B→A)**: 重建碎片的准确性（是否有多余碎片）

**特点**:
- ✅ 对称性：SymMFCD(A,B) = SymMFCD(B,A)
- ✅ 全面性：同时考虑缺失和多余
- ✅ 标准性：类似标准Chamfer Distance的定义

---

## 实现对比

### 当前实现（单向）

**文件**: `vc_plot_charmer_distance_error.py`

**流程**:
```python
# 只计算一个方向
method_errors = []
for orig_shape in orig_shapes:
    min_error = min([CD(orig_shape, recon_shape)
                     for recon_shape in recon_shapes])
    method_errors.append(min_error)

mfcd = np.mean(method_errors)
```

---

### 新实现（对称）

**文件**: `symmetric_mfcd.py`

**流程**:
```python
# 方向1: Orig → Recon
errors_o2r = []
for orig_shape in orig_shapes:
    min_error = min([CD(orig_shape, recon_shape)
                     for recon_shape in recon_shapes])
    errors_o2r.append(min_error)
mfcd_o2r = np.mean(errors_o2r)

# 方向2: Recon → Orig
errors_r2o = []
for recon_shape in recon_shapes:
    min_error = min([CD(recon_shape, orig_shape)
                     for orig_shape in orig_shapes])
    errors_r2o.append(min_error)
mfcd_r2o = np.mean(errors_r2o)

# 对称MFCD
sym_mfcd = mfcd_o2r + mfcd_r2o
```

---

## 使用方法

### 1. 比较两个mesh

```bash
python symmetric_mfcd.py \
    --orig data/original/1.obj \
    --recon data/results/udf_mlp/1.obj \
    --num-samples 5000 \
    --output-dir results/mfcd/
```

**输出**:
```
Results:
  SymMFCD: 0.012345
  MFCD (Orig→Recon): 0.006789
  MFCD (Recon→Orig): 0.005556
```

---

### 2. 批量对比多个方法

```bash
python symmetric_mfcd.py \
    --batch \
    --orig-dir data/original/ \
    --recon-dirs \
        udf_mlp:data/results/udf_mlp/ \
        ffb_mlp:data/results/ffb_mlp/ \
        neuraludf:data/results/neuraludf/ \
    --output-dir results/mfcd_comparison/ \
    --num-samples 5000
```

**输出**:
```
SUMMARY
========================================
udf_mlp:
  SymMFCD: 0.012345
  MFCD (Orig→Recon): 0.006789
  MFCD (Recon→Orig): 0.005556

ffb_mlp:
  SymMFCD: 0.010234
  MFCD (Orig→Recon): 0.005123
  MFCD (Recon→Orig): 0.005111

neuraludf:
  SymMFCD: 0.008765
  MFCD (Orig→Recon): 0.004321
  MFCD (Recon→Orig): 0.004444
```

---

## 实验应用

### Exp 1: UDF Baseline对比

**使用SymMFCD评估**:
```bash
# 假设已经生成了mesh结果
python symmetric_mfcd.py \
    --batch \
    --orig-dir data/original_meshes/ \
    --recon-dirs \
        udf_mlp:data/results/meshes/udf_mlp/ \
        ffb_mlp:data/results/meshes/ffb_mlp/ \
        neuraludf:data/results/meshes/neuraludf/ \
        udf_mind:data/results/meshes/udf_mlp_mind/ \
        neuraludf_mind:data/results/meshes/neuraludf_mind/ \
    --output-dir experiments/udf_baseline/results/mfcd/ \
    --num-samples 10000
```

**预期结果**:
- 量化每种方法的重建质量
- 分析缺失碎片的影响（Orig→Recon高）
- 分析多余碎片的影响（Recon→Orig高）

---

### Exp 4: MFCD定义验证

**使用toy example验证**:

**Example 1: 缺失碎片**
```python
orig_fragments = [A, B, C, D, E]  # 5个碎片
recon_fragments = [A', B', C']     # 3个碎片（缺D, E）

SymMFCD分析:
- MFCD(Orig→Recon): 高（D,E找不到对应）
- MFCD(Recon→Orig): 低（A',B',C'都能找到对应）
```

**Example 2: 位置偏移**
```python
orig_fragments = [A, B, C]
recon_fragments = [A', B', C']  # 所有碎片都有，但位置偏移

SymMFCD分析:
- MFCD(Orig→Recon): 中等（能找到对应，但有偏移）
- MFCD(Recon→Orig): 中等（双向都有偏移）
- SymMFCD平衡反映整体质量
```

**Example 3: 多余碎片**
```python
orig_fragments = [A, B, C]
recon_fragments = [A', B', C', X, Y]  # 多2个噪声碎片

SymMFCD分析:
- MFCD(Orig→Recon): 低（A,B,C都能找到对应）
- MFCD(Recon→Orig): 高（X,Y找不到好的对应）
```

---

## 与标准Chamfer Distance的对应

| 层次 | 标准CD | MFCD类比 |
|------|--------|----------|
| **基本元素** | 点 | 碎片 |
| **单向定义** | mean(min_dist(P1→P2)) | mean(min_CD(Frag_A→Frag_B)) |
| **双向定义** | +mean(min_dist(P2→P1)) | +mean(min_CD(Frag_B→Frag_A)) |
| **对称性** | CD(P1,P2) ≠ CD(P2,P1) (single) | MFCD单向不对称 |
| **对称性** | CD_sym(P1,P2) = CD_sym(P2,P1) | SymMFCD对称 |

**结论**: SymMFCD是MFCD在碎片层次上的自然推广，类似CD的双向定义

---

## 输出文件

### JSON结果 (`symmetric_mfcd_results.json`)

```json
{
  "udf_mlp": {
    "per_mesh": [
      {
        "obj_name": "1.obj",
        "symmetric_mfcd": 0.012345,
        "mfcd_orig_to_recon": 0.006789,
        "mfcd_recon_to_orig": 0.005556,
        "num_orig_fragments": 10,
        "num_recon_fragments": 8,
        "fragment_errors_orig_to_recon": [0.001, 0.002, ...],
        "fragment_errors_recon_to_orig": [0.0015, 0.0025, ...]
      },
      ...
    ],
    "symmetric_mfcd_mean": 0.012345,
    "mfcd_orig_to_recon_mean": 0.006789,
    "mfcd_recon_to_orig_mean": 0.005556
  },
  "ffb_mlp": { ... },
  ...
}
```

### 可视化 (`symmetric_mfcd_comparison.png`)

- 三个子图并排
- 子图1: SymMFCD对比
- 子图2: MFCD(Orig→Recon)对比
- 子图3: MFCD(Recon→Orig)对比

---

## 总结

### 关键区别

| 特性 | 当前实现（单向） | 新实现（对称） |
|------|----------------|---------------|
| **Chamfer Distance** | ✅ 双向 | ✅ 双向 |
| **Fragment匹配** | ❌ 单向（Orig→Recon） | ✅ 双向（双向） |
| **检测缺失碎片** | ✅ | ✅ |
| **检测多余碎片** | ❌ | ✅ |
| **对称性** | ❌ | ✅ |
| **标准化** | - | ✅ 符合CD定义 |

### 建议

1. **实验中使用SymMFCD**：更全面、更标准
2. **保留单向MFCD**：分析缺失/多余碎片的具体影响
3. **报告三个指标**：
   - SymMFCD（主要指标）
   - MFCD(Orig→Recon)（缺失碎片影响）
   - MFCD(Recon→Orig)（多余碎片影响）

---

**文档版本**: v1.0
**最后更新**: 2026-03-02
**作者**: Claude Code
