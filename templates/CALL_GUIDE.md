# 预调用模板指令文件

> 本文件说明如何使用 cos_comparison 项目的直接调用模板。
> 所有模板均支持通过配置文件或代码参数设置训练数据位置。

## 一、项目核心API概览

### 1.1 后端选择

```python
from cos_comparison.core import set_mode, get_mode, get_available_backends

# 查看可用后端
print(get_available_backends())  # ('.cos_comparison_pydll', '.cos_comparison_c', '.cos_comparison')

# 切换后端（优先级从高到低）
set_mode(['cos_comparison_c', 'cos_comparison'])  # 优先ctypes，回退纯Python
set_mode('cos_comparison')  # 强制纯Python
```

### 1.2 核心函数

| 函数 | 用途 | 关键参数 |
|------|------|---------|
| `cos(a, b, algorithm=...)` | 全张量余弦相似度 | a,b: 同形状张量 |
| `cos_comparison_passive(data, ...)` | 被动模式（滑动窗口自相似） | window_size, w1,w2,b1,b2, start/end/step |
| `cos_comparison_active(data, kernel, ...)` | 主动模式（模板匹配） | kernel: 模板张量, w1,w2,b1,b2 |
| `mean_local(data, local_size, ...)` | 局部均值 | local_size: 窗口大小, step |
| `local_variance(data, local_size, ...)` | 局部方差 | local_size: 窗口大小, step |
| `infer_shape(data)` | 推断张量形状 | data: 任意张量 |
| `create_void_list(shape, default)` | 创建空张量 | shape: 维度元组 |
| `load_as_default_data(data, ...)` | 加载为标准数据格式 | data, start, shape, step |
| `threshold_filter(data, low, high, ...)` | 阈值过滤 | low/high: 阈值范围 |
| `threshold_map(data, pairs, ...)` | 阈值映射 | pairs: (low,high,value)列表 |
| `data_filter(data, callback, ...)` | 自定义过滤 | callback: 过滤函数 |
| `data_mapping(data, callback, ...)` | 自定义映射 | callback: 映射函数 |
| `elementwise(*tensors, func, ...)` | 逐元素操作 | func: 操作函数 |

### 1.3 张量类 `vector_map_as_tensor`

```python
from cos_comparison.core import vector_map_as_tensor

t = vector_map_as_tensor(vector=[1,2,3,4], shape=(2,2), strides=(2,1))
t.mean()       # 均值
t.variance()   # 方差
t[0, 1]        # 索引访问
t[0, 1] = 5.0  # 索引赋值
len(t)         # 元素总数
for x in t: ... # 迭代
```

### 1.4 算法选择

`cos()` 和 `cos_comparison_passive/active` 的 `algorithm` 参数可选：

- `_cos` — 标准余弦相似度（默认）
- `_mod` — 模长比
- `_cosmod` — 余弦×模长比

## 二、调用模板使用方法

### 2.1 训练模板 `training_template.py`

复制模板后，修改顶部配置区：

```python
# ===== 配置区 =====
DATA_DIR = r"C:\path\to\training\data"   # 训练数据位置
BACKEND = 'cos_comparison_c'              # 后端选择
WINDOW_SIZE = (3, 3)                      # 被动模式窗口
KERNEL = None                             # 主动模式模板（None=自动生成）
OUTPUT_DIR = r"C:\path\to\output"         # 输出位置
# ==================
```

运行：
```bash
python training_template.py
```

### 2.2 匹配模板 `matching_template.py`

用于两张量/多张量匹配任务。

### 2.3 配置文件方式

模板支持从JSON配置文件读取参数：

```bash
python training_template.py --config my_config.json
```

配置文件格式见 `config_template.json`。

## 三、数据格式要求

### 3.1 张量输入

所有核心函数接受以下张量格式：
- 嵌套列表：`[[1.0, 2.0], [3.0, 4.0]]`
- `vector_map_as_tensor` 对象
- 任何实现 `__shape__()` / `__getitem__()` 协议的对象（鸭子类型）
- 一维序列：`[1.0, 2.0, 3.0]`

### 3.2 训练数据目录结构

```
DATA_DIR/
├── class_A/
│   ├── sample_001.txt   # 每行一个数值，或逗号分隔
│   ├── sample_002.txt
│   └── ...
├── class_B/
│   ├── sample_001.txt
│   └── ...
└── ...
```

或统一格式：
```
DATA_DIR/
├── train.csv            # 首列标签，其余列为特征
└── test.csv
```

## 四、回调机制

被动/主动模式支持回调函数，用于进度监控和错误处理：

```python
def on_start(ctx):
    print(f"Start: output shape={ctx.output.shape}")

def on_end(ctx):
    print("Done")

def on_local_error(ctx, error):
    print(f"Local error at {ctx.position}: {error}")

result = cos_comparison_passive(
    data, window_size=(3,3),
    start_callback=on_start,
    end_callback=on_end,
    local_error_callback=on_local_error,
)
```

## 五、注意事项

1. **形状一致性**：`cos(a, b)` 要求 a 和 b 形状完全相同，否则抛 ValueError
2. **窗口有效性**：被动/主动模式要求 `end - start - window_size >= 0`，否则抛 "effectless args"
3. **后端差异**：pydll 后端最快但需编译；ctypes 次之；纯 Python 最兼容
4. **零维张量**：空张量（含零维度）的 mean/variance 返回 None
5. **输出复用**：可传入 `output` 参数复用预分配张量，避免重复分配
