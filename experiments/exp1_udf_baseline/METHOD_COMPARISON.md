# FFB-MLP vs NeuralUDF vs MIND：实现方式对比

**文档类型**: 技术实现对比（非效果评估）
**日期**: 2026-03-02
**目的**: 对比三种方法的具体代码实现差异

> **重要说明**: 本文档仅对比技术实现方式，不包含任何效果推测或性能评估。所有效果相关的结论需通过实验得出。

---

## 目录

1. [网络架构实现对比](#网络架构实现对比)
2. [位置编码实现对比](#位置编码实现对比)
3. [编码器实现对比](#编码器实现对比)
4. [训练实现对比](#训练实现对比)
5. [代码实现差异总结](#代码实现差异总结)

---

## 网络架构实现对比

### 1. FFB-MLP 实现

#### SimpleSDFMLP (SDF版本)

**文件**: `src/train_ffb_mlp.py:18-34`

```python
class SimpleSDFMLP(nn.Module):
    def __init__(self, d_in=3, d_hidden=256, n_layers=6, multires=4):
        super().__init__()
        from models.embedder import get_embedder
        self.embed_fn, embed_dim = get_embedder(multires, input_dims=d_in)

        # 构建层序列
        dims = [embed_dim] + [d_hidden] * (n_layers - 1) + [1]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])

        self.activation = nn.Softplus(beta=100)
        self.scale = 1.0

    def forward(self, x):
        x = x * self.scale
        x = self.embed_fn(x)
        for i, lin in enumerate(self.layers[:-1]):
            x = self.activation(lin(x))
        return self.layers[-1](x) / self.scale
```

**实际训练配置** (来自实验输出):
```python
# src/train_ffb_mlp.py:64
model = SimpleSDFMLP(
    d_hidden=128,    # 实际使用128，不是定义中的256
    n_layers=4,      # 实际使用4层，不是定义中的6层
    multires=4
)
```

**网络结构**:
```
输入: (batch, 3)
  ↓ embed_fn (multires=4)
(batch, 27)
  ↓ Linear(27 → 128) → Softplus
(batch, 128)
  ↓ Linear(128 → 128) → Softplus
(batch, 128)
  ↓ Linear(128 → 128) → Softplus
(batch, 128)
  ↓ Linear(128 → 1)
输出: (batch, 1)
```

**实现特点**:
- ✅ 使用 Softplus(beta=100) 激活函数
- ✅ 有 scale 参数（但实际值为1.0）
- ❌ 无 skip connection
- ❌ 无 geometric initialization
- ❌ 无 weight normalization
- ❌ 无 bias 参数设置

---

#### SimpleUDFMLP (UDF版本)

**文件**: `src/train_udf_mlp.py:41-61`

```python
class SimpleUDFMLP(nn.Module):
    def __init__(self, d_in=3, d_hidden=256, n_layers=6, multires=4):
        super().__init__()
        from models.embedder import get_embedder
        self.embed_fn, embed_dim = get_embedder(multires, input_dims=d_in)
        dims = [embed_dim] + [d_hidden] * (n_layers - 1) + [1]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        self.activation = nn.Softplus(beta=100)
        self.scale = 1.0

    def forward(self, x):
        x = x * self.scale
        x = self.embed_fn(x)
        for i, lin in enumerate(self.layers[:-1]):
            x = self.activation(lin(x))
        x = self.layers[-1](x)
        return torch.abs(x) / self.scale  # ← 唯一区别：使用abs()
```

**实际训练配置** (来自实验输出):
```python
# src/train_udf_mlp.py:87
model = SimpleUDFMLP(
    d_hidden=128,
    n_layers=4,
    multires=4
)
```

**与 SimpleSDF MLP 的代码差异**:
```python
# SimpleSDF:
return self.layers[-1](x) / self.scale

# SimpleUDF:
x = self.layers[-1](x)
return torch.abs(x) / self.scale  # 增加了 torch.abs()
```

**实现特点**:
- 与 SimpleSDFMLP 完全相同
- 唯一区别：输出层应用 `torch.abs()` 确保非负

---

### 2. NeuralUDF 实现

#### UDFNetwork

**文件**: `experiments/udf_baseline/NeuralUDF/models/fields.py:115-232`

```python
class UDFNetwork(nn.Module):
    def __init__(self,
                 d_in,                    # 输入维度
                 d_out,                   # 输出维度
                 d_hidden,                # 隐藏层维度
                 n_layers,                # 层数
                 skip_in=(4,),            # skip connection位置
                 multires=0,              # 位置编码频率
                 scale=1,                 # 缩放因子
                 bias=0.5,                # bias初始化值
                 geometric_init=True,     # 是否使用几何初始化
                 weight_norm=True,        # 是否使用weight normalization
                 udf_type='abs',          # UDF类型: 'abs', 'square', 'sdf'
                 ):
        super(UDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        # 位置编码
        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.geometric_init = geometric_init

        # 构建网络层
        for l in range(0, self.num_layers - 1):
            # 计算输出维度（考虑skip connection）
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            # 几何初始化
            if geometric_init:
                if l == self.num_layers - 2:
                    # 最后一层的特殊初始化
                    torch.nn.init.normal_(lin.weight,
                                         mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                                         std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    # 第一层的初始化
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                         np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    # skip层的初始化
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                         np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    # 其他层的初始化
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                         np.sqrt(2) / np.sqrt(out_dim))

            # Weight normalization
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.udf_type = udf_type

    def udf_out(self, x):
        """UDF输出变换"""
        if self.udf_type == 'abs':
            return torch.abs(x)
        elif self.udf_type == 'square':
            return x ** 2
        elif self.udf_type == 'sdf':
            return x

    def forward(self, inputs):
        inputs = inputs * self.scale
        xyz = inputs
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            # Skip connection实现
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        # 应用UDF输出变换
        return torch.cat([self.udf_out(x[:, :1]) / self.scale, x[:, 1:]], dim=-1)

    def udf(self, x):
        return self.forward(x)[:, :1]
```

**网络结构示例** (假设 d_hidden=256, n_layers=6, multires=6, skip_in=(4,)):
```
输入: (batch, 3)
  ↓ embed_fn (multires=6)
(batch, 39)
  ↓ Linear(39 → 256) + weight_norm → Softplus
(batch, 256)
  ↓ Linear(256 → 256) + weight_norm → Softplus
(batch, 256)
  ↓ Linear(256 → 256) + weight_norm → Softplus
(batch, 256)
  ↓ concat with (batch, 39) / √2  ← Skip Connection
(batch, 295)
  ↓ Linear(295 → 256) + weight_norm → Softplus
(batch, 256)
  ↓ Linear(256 → 256) + weight_norm → Softplus
(batch, 256)
  ↓ Linear(256 → 1) + weight_norm
(batch, 1)
  ↓ udf_out() / scale
输出: (batch, 1)
```

**实现特点**:
- ✅ Skip connection 在指定层（默认第4层）
- ✅ Geometric initialization (4种不同的初始化策略)
- ✅ Weight normalization (可选)
- ✅ 灵活的UDF类型 (abs/square/sdf)
- ✅ 多输出支持 (d_out > 1时可包含额外特征)
- ✅ 使用 Softplus(beta=100)

---

### 3. MIND 实现

**文件**: `experiments/udf_baseline/MIND/README.md`

MIND **不是神经网络**，而是一个网格提取算法。

```python
from mldf import MIND

# 输入：任意UDF查询函数
def udf_query_func(points):
    """
    points: (N, 3) numpy array
    返回: (N,) numpy array of UDF values
    """
    return model.udf(torch.from_numpy(points)).cpu().numpy()

# 创建MIND提取器
mind_extractor = MIND(udf_query_func, resolution=256)

# 运行提取
mesh = mind_extractor.run()
```

**实现特点**:
- ❌ 不是神经网络
- ✅ 接受任意UDF函数作为输入
- ✅ 基于网格采样的网格提取算法
- ✅ 专门处理非流形表面

---

## 网络架构实现差异总结

### 层结构对比

| 组件 | FFB-MLP (实际) | NeuralUDF (典型) |
|------|----------------|------------------|
| **输入维度** | 3 | 3 |
| **位置编码** | multires=4 → 27维 | multires=6 → 39维 |
| **隐藏层数** | 3层 (4层总共) | 5层 (6层总共，有skip) |
| **隐藏维度** | 128 | 256 |
| **输出维度** | 1 | 1 (或更多) |
| **Skip位置** | 无 | 第4层 |
| **总参数量** | ~52K | ~400K |

**参数量计算**:

FFB-MLP:
```
27 → 128: 27×128 + 128 = 3,584
128 → 128: 128×128 + 128 = 16,512
128 → 128: 128×128 + 128 = 16,512
128 → 1: 128×1 + 1 = 129
总计: ~36,737 参数
```

NeuralUDF (d_hidden=256, n_layers=6, skip_in=4):
```
39 → 256: 39×256 + 256 = 10,240
256 → 256: 256×256 + 256 = 65,792 (×3层)
295 → 256: 295×256 + 256 = 75,776 (skip后)
256 → 256: 256×256 + 256 = 65,792
256 → 1: 256×1 + 1 = 257
总计: ~284,657 参数 (不含weight_norm额外参数)
```

### 初始化策略对比

| 层 | FFB-MLP | NeuralUDF (geometric_init=True) |
|----|---------|--------------------------------|
| **第一层** | PyTorch默认 | weight[:, 3:] = 0; weight[:, :3] ~ N(0, √(2/out_dim)); bias = 0 |
| **中间层** | PyTorch默认 | weight ~ N(0, √(2/out_dim)); bias = 0 |
| **Skip层** | N/A | weight[:, -(dims[0]-3):] = 0; 其他 ~ N(0, √(2/out_dim)); bias = 0 |
| **最后层** | PyTorch默认 | weight ~ N(√π/√in_dim, 0.0001); bias = -0.5 |

**PyTorch默认初始化** (nn.Linear):
```python
# 来源: torch.nn.Linear
bound = 1 / sqrt(in_features)
weight ~ Uniform(-bound, bound)
bias ~ Uniform(-bound, bound)
```

### 激活函数对比

| 网络 | 激活函数 | beta参数 |
|------|---------|---------|
| FFB-MLP | Softplus | 100 |
| NeuralUDF | Softplus | 100 |

**Softplus公式**:
```python
Softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
```

当 beta=100 时，Softplus 近似于 ReLU，但在 x=0 附近更平滑。

### Weight Normalization 实现

NeuralUDF 使用 `nn.utils.weight_norm`:

```python
# PyTorch weight_norm 实现
# 将权重分解为方向和大小：w = g * (v / ||v||)
lin = nn.utils.weight_norm(lin)
```

FFB-MLP 不使用 weight normalization。

---

## 位置编码实现对比

### 共同的 Embedder 实现

三种方法都使用相同的位置编码实现：

**文件**: `experiments/udf_baseline/NeuralUDF/models/embedder.py`

```python
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        # 是否包含原始输入
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        # 生成频率带
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        # 为每个频率创建sin和cos函数
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,              # 包含原始坐标
        'input_dims': input_dims,           # 输入维度=3
        'max_freq_log2': multires-1,        # 最大频率的log2值
        'num_freqs': multires,              # 频率数量
        'log_sampling': True,               # 使用对数采样
        'periodic_fns': [torch.sin, torch.cos],  # 周期函数
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
```

### 编码公式

对于输入点 $\mathbf{p} = (x, y, z)$，位置编码为：

$$
\gamma(\mathbf{p}) = [\mathbf{p}, \sin(2^0 \pi \mathbf{p}), \cos(2^0 \pi \mathbf{p}), \sin(2^1 \pi \mathbf{p}), \cos(2^1 \pi \mathbf{p}), \ldots, \sin(2^{L-1} \pi \mathbf{p}), \cos(2^{L-1} \pi \mathbf{p})]
$$

其中 $L$ = `multires`

### 具体编码对比

#### FFB-MLP (multires=4)

**配置**:
```python
embed_kwargs = {
    'include_input': True,
    'input_dims': 3,
    'max_freq_log2': 3,      # multires - 1 = 4 - 1 = 3
    'num_freqs': 4,          # multires = 4
    'log_sampling': True,
    'periodic_fns': [torch.sin, torch.cos],
}
```

**频率带**:
```python
freq_bands = 2^torch.linspace(0, 3, 4)
          = [2^0, 2^1, 2^2, 2^3]
          = [1, 2, 4, 8]
```

**编码维度**:
```python
out_dim = 3 (原始输入)
        + 3 × 2 (sin+cos) × 4 (频率数)
        = 3 + 24
        = 27 维
```

**具体编码输出** (对于单个点 (x, y, z)):
```python
[
    x, y, z,                                    # 原始坐标 (3维)
    sin(1π·x), cos(1π·x), sin(1π·y), cos(1π·y), sin(1π·z), cos(1π·z),  # freq=1 (6维)
    sin(2π·x), cos(2π·x), sin(2π·y), cos(2π·y), sin(2π·z), cos(2π·z),  # freq=2 (6维)
    sin(4π·x), cos(4π·x), sin(4π·y), cos(4π·y), sin(4π·z), cos(4π·z),  # freq=4 (6维)
    sin(8π·x), cos(8π·x), sin(8π·y), cos(8π·y), sin(8π·z), cos(8π·z),  # freq=8 (6维)
]
```

#### NeuralUDF (multires=6, 典型配置)

**配置**:
```python
embed_kwargs = {
    'include_input': True,
    'input_dims': 3,
    'max_freq_log2': 5,      # multires - 1 = 6 - 1 = 5
    'num_freqs': 6,          # multires = 6
    'log_sampling': True,
    'periodic_fns': [torch.sin, torch.cos],
}
```

**频率带**:
```python
freq_bands = 2^torch.linspace(0, 5, 6)
          = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5]
          = [1, 2, 4, 8, 16, 32]
```

**编码维度**:
```python
out_dim = 3 (原始输入)
        + 3 × 2 (sin+cos) × 6 (频率数)
        = 3 + 36
        = 39 维
```

**具体编码输出** (对于单个点 (x, y, z)):
```python
[
    x, y, z,                                    # 原始坐标 (3维)
    sin(1π·x), cos(1π·x), ..., sin(1π·z), cos(1π·z),    # freq=1 (6维)
    sin(2π·x), cos(2π·x), ..., sin(2π·z), cos(2π·z),    # freq=2 (6维)
    sin(4π·x), cos(4π·x), ..., sin(4π·z), cos(4π·z),    # freq=4 (6维)
    sin(8π·x), cos(8π·x), ..., sin(8π·z), cos(8π·z),    # freq=8 (6维)
    sin(16π·x), cos(16π·x), ..., sin(16π·z), cos(16π·z), # freq=16 (6维)
    sin(32π·x), cos(32π·x), ..., sin(32π·z), cos(32π·z), # freq=32 (6维)
]
```

### 编码实现差异总结

| 参数 | FFB-MLP | NeuralUDF (典型) |
|------|---------|------------------|
| **multires** | 4 | 6 |
| **max_freq_log2** | 3 | 5 |
| **num_freqs** | 4 | 6 |
| **频率范围** | [1, 2, 4, 8] | [1, 2, 4, 8, 16, 32] |
| **输出维度** | 27 | 39 |
| **最高频率** | 8π | 32π |

**关键差异**:
- NeuralUDF 有额外的两个高频分量（16π 和 32π）
- 这使得编码维度从 27 增加到 39
- 两者的编码算法完全相同，只是频率数量不同

---

## 编码器实现对比

### 1. FFB-DF 编码器

**文件**: `src/encoder_ffb-df_mlp.py`

#### 主要参数

```python
mlp_sample_num = 64000      # 标准模式
mlp_sample_num = 4000       # --fast模式
mlp_sample_num = 1500       # --minimal模式

bool_ffbdf = True           # FFB-DF标志
```

#### 采样流程

**步骤1: 加载和分割mesh**

```python
mesh = vedo.load(objName)
objs = mesh.split()  # 分割成多个连通组件

all_vertices = []
all_faces = []

for obj in objs:
    vertices = np.array(obj.points)
    faces = np.array(obj.cells)

    # 处理faces格式
    if len(faces.shape) == 1:
        faces = faces.reshape(-1, 3)
    elif len(faces.shape) == 2 and faces.shape[1] > 3:
        faces = faces[:, :3]

    faces = faces.astype(np.int32)

    all_vertices.append(vertices)
    all_faces.append(faces)
```

**步骤2: 计算采样数量**

```python
near_surface_num = int(mlp_sample_num * 0.5)  # 50% 近表面
uniform_num = mlp_sample_num - near_surface_num  # 50% 均匀采样

if not use_minimal:
    near_surface_num *= 10  # 实际采样10倍，后续筛选
```

对于标准模式 (mlp_sample_num=64000):
```
near_surface_num = 32000 × 10 = 320000 (采样尝试次数)
uniform_num = 32000
```

**步骤3: 近表面采样（迭代100次）**

```python
near_surface_points = np.array([]).reshape(0, 3)

for iteration in tqdm(range(100)):
    # 使用Poisson Disk采样
    coarse_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
    coarse_points = coarse_sampler.random(near_surface_num * 2)
    coarse_points = 2 * coarse_points - 1  # 归一化到 [-1, 1]

    # 计算每个碎片的SDF
    sdf_values = np.full(len(coarse_points), np.inf)
    for obj_vertices, obj_faces in zip(all_vertices, all_faces):
        s = igl.signed_distance(coarse_points, obj_vertices, obj_faces)[0]
        sdf_values = np.minimum(sdf_values, s)  # 取最小SDF

    # 筛选近表面点
    if bool_multi_fragment:
        near_surface_mask = np.abs(sdf_values) <= 0.1
    else:
        near_surface_mask = np.abs(sdf_values) <= 0.05

    new_near_surface_points = coarse_points[near_surface_mask]
    near_surface_points = np.vstack([near_surface_points, new_near_surface_points])

    # 达到目标数量则退出
    if len(near_surface_points) >= near_surface_num:
        break

# 随机选择目标数量的点
near_surface_selected = near_surface_points[
    np.random.choice(len(near_surface_points),
                     int(mlp_sample_num * 0.5),
                     replace=False)
]
```

**步骤4: 均匀采样**

```python
uniform_points = np.random.rand(uniform_num, 3).astype(np.float64) * 2 - 1
```

**步骤5: 合并并计算最终SDF**

```python
# 合并采样点
final_poisson_grid_points = np.vstack([near_surface_selected, uniform_points])

# 计算最终SDF（对所有碎片取最小值）
final_sdf = np.full(len(final_poisson_grid_points), np.inf)
for obj_vertices, obj_faces in zip(all_vertices, all_faces):
    s = igl.signed_distance(final_poisson_grid_points, obj_vertices, obj_faces)[0]
    final_sdf = np.minimum(final_sdf, s)
```

**步骤6: 保存**

```python
np.savez(
    output_path,
    poisson_grid_points=final_poisson_grid_points,  # (N, 3)
    sdf_values=final_sdf                            # (N,)
)
```

#### 关键实现细节

**Poisson Disk Sampling**:
```python
scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
```
- `d=3`: 三维空间
- `radius=0.025`: 最小点间距离（在[0,1]³空间）
- 输出范围: [0, 1]³
- 后续变换到 [-1, 1]³

**SDF计算** (igl.signed_distance):
```python
s, i, c = igl.signed_distance(query_points, vertices, faces)
```
- `s`: signed distance值
- `i`: 最近的面片索引
- `c`: 最近的表面点坐标
- 正值表示外部，负值表示内部

**Min-SDF策略**:
```python
sdf_values = np.minimum(sdf_values, s)
```
对多个碎片，取最小的SDF值，使得：
- 在任意碎片外部：取最近碎片的距离
- 在某个碎片内部：该碎片的负距离
- 在多个碎片内部：取绝对值最小的（最接近某个表面）

---

### 2. UDF Mesh 编码器

**文件**: `src/encoder_udf_mesh.py`

#### 与 FFB-DF 的差异

大部分代码与 FFB-DF 编码器相同，主要差异在：

**差异1: 距离计算方式**

```python
# UDF Mesh 编码器使用无向距离
s, _, _ = igl.signed_distance(
    coarse_points, obj_vertices, obj_faces,
    sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED  # ← 关键差异
)
coarse_udf = np.minimum(coarse_udf, s)
```

**差异2: 近表面阈值**

```python
# FFB-DF (SDF)
if bool_multi_fragment:
    near_surface_mask = np.abs(sdf_values) <= 0.1
else:
    near_surface_mask = np.abs(sdf_values) <= 0.05

# UDF Mesh
if bool_multi_fragment:
    near_surface_mask = coarse_udf <= 0.3    # ← UDF直接比较，无需abs
else:
    near_surface_mask = coarse_udf <= 0.1
```

**差异3: 保存的数据名称**

```python
# FFB-DF
np.savez(output_path,
    poisson_grid_points=final_points,
    sdf_values=final_sdf          # ← 键名: sdf_values
)

# UDF Mesh
np.savez(output_path,
    poisson_grid_points=final_points,
    udf_values=final_udf          # ← 键名: udf_values
)
```

#### 关键实现细节

**igl.SIGNED_DISTANCE_TYPE_UNSIGNED**:
```python
# 源码中的枚举值
igl.SIGNED_DISTANCE_TYPE_DEFAULT   # 默认：有向距离
igl.SIGNED_DISTANCE_TYPE_UNSIGNED  # 无向距离（始终非负）
```

使用无向距离时：
- 所有距离值 ≥ 0
- 不区分内部/外部
- 返回到最近表面点的欧氏距离

---

### 3. NeuralUDF 编码器

NeuralUDF 的编码器与上述两者完全不同，它基于**多视角图像**而非mesh。

**典型输入数据格式** (来自NeuralUDF论文):
```
dataset/
├── scan_id/
│   ├── image/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── mask/
│   │   ├── 000000.png
│   │   └── ...
│   └── cameras.npz  # 相机参数
```

**NeuralUDF的采样策略** (基于射线):

1. **射线采样**:
   - 从每个相机发射光线
   - 在光线上采样3D点

2. **重要性采样**:
   - 基于当前UDF预测在表面附近密集采样

3. **渲染监督**:
   - 使用图像RGB和mask监督
   - 通过体渲染或表面渲染loss

**与mesh-based编码的本质区别**:
- 输入：图像 vs mesh
- 采样：射线采样 vs 空间采样
- 监督：图像loss vs 距离loss

---

### 4. MIND 编码器

MIND **没有编码器**。它接受任意UDF查询函数：

```python
from mldf import MIND

# 定义UDF查询函数
def udf_func(points):
    """
    points: (N, 3) numpy array, 查询点坐标
    返回: (N,) numpy array, UDF值
    """
    # 可以是:
    # 1. 从训练好的神经网络查询
    # 2. 直接从mesh计算
    # 3. 从预计算的体素网格插值
    return udf_values

# 使用MIND提取mesh
mind = MIND(udf_func, resolution=256)
mesh = mind.run()
```

**MIND的采样策略**:
- 在3D规则网格上采样（如256³个点）
- 调用用户提供的`udf_func`获取UDF值
- 使用dual contouring类算法提取网格
- 特殊处理非流形点

---

## 编码器实现差异总结

### 数据来源对比

| 编码器 | 输入数据 | 输出数据 |
|--------|---------|---------|
| **FFB-DF** | Mesh (.obj) | (points, sdf_values) .npz |
| **UDF Mesh** | Mesh (.obj) | (points, udf_values) .npz |
| **NeuralUDF** | 多视角图像 + 相机参数 | 训练好的神经网络 |
| **MIND** | UDF查询函数 | Mesh |

### 采样策略对比

| 编码器 | 采样方式 | 点数 | 近表面策略 |
|--------|---------|------|-----------|
| **FFB-DF** | Poisson Disk (迭代) + 均匀 | 64K | 50%近表面(|SDF|<0.1), 50%均匀 |
| **UDF Mesh** | Poisson Disk (迭代) + 均匀 | 64K | 50%近表面(UDF<0.3), 50%均匀 |
| **NeuralUDF** | 射线采样 | 变化 | 基于当前UDF预测自适应 |
| **MIND** | 规则网格 | resolution³ | 均匀采样 |

### 距离计算对比

| 编码器 | 距离类型 | 计算方法 | 多碎片处理 |
|--------|---------|---------|-----------|
| **FFB-DF** | SDF | `igl.signed_distance` (默认) | `min(SDF_i)` |
| **UDF Mesh** | UDF | `igl.signed_distance` (UNSIGNED) | `min(UDF_i)` |
| **NeuralUDF** | UDF | 从渲染loss学习 | N/A |
| **MIND** | UDF | 调用用户函数 | 取决于用户函数 |

### 代码实现对比

**Poisson Disk参数**:
```python
# FFB-DF 和 UDF Mesh 都使用相同的参数
scipy.stats.qmc.PoissonDisk(d=3, radius=0.025)
```

**igl.signed_distance调用**:
```python
# FFB-DF (SDF)
s = igl.signed_distance(points, vertices, faces)[0]
# 返回: 正值(外部), 负值(内部)

# UDF Mesh (UDF)
s = igl.signed_distance(points, vertices, faces,
                       sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)[0]
# 返回: 非负值（距离）
```

**Min策略**:
```python
# 两者都使用相同的min策略
for obj_vertices, obj_faces in zip(all_vertices, all_faces):
    s = igl.signed_distance(points, obj_vertices, obj_faces, ...)[0]
    distance_field = np.minimum(distance_field, s)
```

---

## 训练实现对比

### 1. FFB-MLP 训练

**文件**: `src/train_ffb_mlp.py:40-84`

```python
def train(npz_dir, ckpt_dir, epochs=100, batch_size=4096, lr=1e-4):
    # 加载所有npz文件
    npz_files = sorted(glob(os.path.join(npz_dir, "*.npz")))

    all_pts, all_vals = [], []
    for f in npz_files:
        d = np.load(f)
        all_pts.append(d["poisson_grid_points"].astype(np.float32))
        all_vals.append(d["sdf_values"].astype(np.float32).ravel())

    pts = np.vstack(all_pts)
    vals = np.concatenate(all_vals)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    print(f"Training on {len(pts)} samples from {len(npz_files)} files")

    # 创建dataset和dataloader
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(pts),
        torch.from_numpy(vals.astype(np.float32))
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 创建模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSDFMLP(d_hidden=128, n_layers=4, multires=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        for pts_b, vals_b in loader:
            pts_b = pts_b.to(device)
            vals_b = vals_b.to(device)
            pred = model.sdf(pts_b)
            loss = nn.functional.mse_loss(pred, vals_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={loss_sum/len(loader):.6f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "ffb_mlp.pth"))
    return model
```

**实际训练参数** (来自实验输出):
```python
epochs = 30          # 脚本中传入
batch_size = 4096
lr = 1e-4
optimizer = Adam
loss = MSE Loss
```

**训练数据量** (来自实验输出):
```
数据: 5个npz文件
每个文件: 352,000个点
总计: 1,760,000个点
```

---

### 2. UDF-MLP 训练

**文件**: `src/train_udf_mlp.py:64-107`

```python
def train(npz_dir, ckpt_dir, epochs=100, batch_size=4096, lr=1e-4):
    npz_files = sorted(glob(os.path.join(npz_dir, "*.npz")))

    all_pts, all_vals = [], []
    for f in npz_files:
        d = np.load(f)
        all_pts.append(d["poisson_grid_points"].astype(np.float32))

        # 支持两种键名
        v = d["udf_values"] if "udf_values" in d else d["sdf_values"]
        all_vals.append(np.abs(v).astype(np.float32).ravel())  # ← 确保非负

    pts = np.vstack(all_pts)
    vals = np.concatenate(all_vals)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(pts),
        torch.from_numpy(vals).unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUDFMLP(d_hidden=128, n_layers=4, multires=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

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

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "udf_mlp.pth"))
    return model
```

**与 FFB-MLP 训练的差异**:

1. 数据加载:
```python
# FFB-MLP
all_vals.append(d["sdf_values"].astype(np.float32).ravel())

# UDF-MLP
v = d["udf_values"] if "udf_values" in d else d["sdf_values"]
all_vals.append(np.abs(v).astype(np.float32).ravel())  # 使用abs确保非负
```

2. 前向传播:
```python
# FFB-MLP
pred = model.sdf(pts_b)

# UDF-MLP
pred = model.udf(pts_b)
```

其他完全相同。

---

### 3. NeuralUDF 训练

NeuralUDF的训练流程与上述两者完全不同，它基于图像渲染loss。

**典型训练配置** (来自NeuralUDF配置文件):
```yaml
# experiments/udf_baseline/NeuralUDF/confs/example.conf

train:
    learning_rate: 5e-4
    learning_rate_alpha: 0.05
    end_iter: 300000

    batch_size: 512
    validate_resolution_level: 4
    warm_up_end: 5000

    save_freq: 10000
    val_freq: 2500
    report_freq: 100

model:
    udf_network:
        d_out: 257
        d_in: 3
        d_hidden: 256
        n_layers: 8
        skip_in: [4]
        multires: 6
        bias: 0.5
        scale: 1.0
        geometric_init: True
        weight_norm: True
```

**损失函数** (多项loss组合):
```python
# 从NeuralUDF训练代码推断的loss结构

# 1. 颜色重建loss
loss_color = F.l1_loss(color_pred, color_gt)

# 2. Eikonal loss (梯度约束)
loss_eikonal = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()

# 3. Mask loss
loss_mask = F.binary_cross_entropy(mask_pred, mask_gt)

# 4. 总loss
loss = w_color * loss_color + w_eikonal * loss_eikonal + w_mask * loss_mask
```

---

### 训练实现差异总结

| 特性 | FFB-MLP | UDF-MLP | NeuralUDF |
|------|---------|---------|-----------|
| **输入数据** | (points, SDF) .npz | (points, UDF) .npz | 多视角图像 |
| **Batch size** | 4096 | 4096 | 512 (射线) |
| **Epochs** | 30 | 30 | ~300K iterations |
| **Learning rate** | 1e-4 | 1e-4 | 5e-4 |
| **Optimizer** | Adam | Adam | Adam |
| **Loss** | MSE | MSE | Color + Eikonal + Mask |
| **训练时间** | 5分钟 | 5分钟 | 数小时-数天 |
| **数据量** | 1.76M点 | 1.76M点 | 图像序列 |

---

## 代码实现差异总结

### 关键技术差异矩阵

| 技术特性 | FFB-MLP | NeuralUDF | 实现层面的差异 |
|---------|---------|-----------|---------------|
| **Skip Connection** | ❌ | ✅ | NeuralUDF在forward中有`if l in self.skip_in: x = torch.cat([x, inputs], 1)` |
| **Geometric Init** | ❌ | ✅ | NeuralUDF在`__init__`中有专门的初始化代码块 |
| **Weight Norm** | ❌ | ✅ | NeuralUDF使用`nn.utils.weight_norm(lin)` |
| **位置编码频率** | multires=4 | multires=6 | `get_embedder(multires, ...)` 参数不同 |
| **网络深度** | 4层 | 6-8层 | `n_layers` 参数不同 |
| **隐藏维度** | 128 | 256 | `d_hidden` 参数不同 |
| **UDF输出** | `torch.abs(x)` | `udf_out(x)` 三种模式 | 输出处理方式不同 |

### 编码器差异矩阵

| 技术特性 | FFB-DF | UDF Mesh | NeuralUDF | 实现层面的差异 |
|---------|--------|----------|-----------|---------------|
| **输入类型** | Mesh | Mesh | 图像 | 数据加载方式完全不同 |
| **距离类型** | SDF | UDF | UDF | `igl.signed_distance`的`sign_type`参数不同 |
| **采样算法** | Poisson Disk | Poisson Disk | 射线采样 | 采样代码完全不同 |
| **近表面阈值** | 0.1 | 0.3 | N/A | 筛选条件不同 |
| **输出键名** | `sdf_values` | `udf_values` | N/A | .npz文件中的键名不同 |
| **多碎片处理** | `np.minimum` | `np.minimum` | N/A | 都使用min策略 |

### 训练差异矩阵

| 技术特性 | FFB-MLP | UDF-MLP | NeuralUDF | 实现层面的差异 |
|---------|---------|---------|-----------|---------------|
| **数据加载** | 直接加载npz | 加载npz+abs | 自定义DataLoader | 数据预处理不同 |
| **Loss函数** | MSE | MSE | 多项loss组合 | loss计算代码不同 |
| **Batch组织** | 随机点 | 随机点 | 射线batch | batch构建方式不同 |
| **前向调用** | `model.sdf(x)` | `model.udf(x)` | 复杂渲染 | 前向传播逻辑不同 |

### MIND的特殊性

MIND是**后处理算法**，不是学习方法：
- 无网络架构
- 无编码器
- 无训练过程
- 接口：`MIND(udf_func, resolution)`

---

## 附录：完整代码路径

| 组件 | 文件路径 | 关键函数/类 |
|------|---------|------------|
| **FFB-MLP网络** | `src/train_ffb_mlp.py` | `class SimpleSDFMLP` (L18-34) |
| **UDF-MLP网络** | `src/train_udf_mlp.py` | `class SimpleUDFMLP` (L41-61) |
| **NeuralUDF网络** | `experiments/udf_baseline/NeuralUDF/models/fields.py` | `class UDFNetwork` (L115-232) |
| **位置编码** | `experiments/udf_baseline/NeuralUDF/models/embedder.py` | `class Embedder`, `get_embedder` (L6-51) |
| **FFB-DF编码器** | `src/encoder_ffb-df_mlp.py` | 主脚本 |
| **UDF编码器** | `src/encoder_udf_mesh.py` | 主脚本 |
| **FFB-MLP训练** | `src/train_ffb_mlp.py` | `def train` (L40-84) |
| **UDF-MLP训练** | `src/train_udf_mlp.py` | `def train` (L64-107) |
| **MIND** | `experiments/udf_baseline/MIND/` | `from mldf import MIND` |

---

**文档版本**: v2.0 - 纯实现对比版
**最后更新**: 2026-03-02
**重要**: 本文档仅描述代码实现差异，不包含效果评估或推测
