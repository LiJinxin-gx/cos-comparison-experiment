# Call Template Guide

> This document explains how to use the direct call templates for the
> cos_comparison project. All templates support setting training data location
> via config file or code parameters.

## 1. Core API Overview

### 1.1 Backend Selection

```python
from cos_comparison.core import set_mode, get_mode, get_available_backends

# List available backends
print(get_available_backends())  # ('.cos_comparison_pydll', '.cos_comparison_c', '.cos_comparison')

# Switch backend (priority high to low)
set_mode(['cos_comparison_c', 'cos_comparison'])  # ctypes preferred, fallback pure Python
set_mode('cos_comparison')  # force pure Python
```

### 1.2 Core Functions

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `cos(a, b, algorithm=...)` | Full-tensor cosine similarity | a,b: same-shape tensors |
| `cos_comparison_passive(data, ...)` | Passive mode (sliding window self-similarity) | window_size, w1,w2,b1,b2, start/end/step |
| `cos_comparison_active(data, kernel, ...)` | Active mode (template matching) | kernel: template tensor, w1,w2,b1,b2 |
| `mean_local(data, local_size, ...)` | Local mean | local_size: window size, step |
| `local_variance(data, local_size, ...)` | Local variance | local_size: window size, step |
| `infer_shape(data)` | Infer tensor shape | data: any tensor |
| `create_void_list(shape, default)` | Create empty tensor | shape: dimension tuple |
| `load_as_default_data(data, ...)` | Load to standard format | data, start, shape, step |
| `threshold_filter(data, low, high, ...)` | Threshold filter | low/high: threshold range |
| `threshold_map(data, pairs, ...)` | Threshold mapping | pairs: list of (low,high,value) |
| `data_filter(data, callback, ...)` | Custom filter | callback: filter function |
| `data_mapping(data, callback, ...)` | Custom mapping | callback: mapping function |
| `elementwise(*tensors, func, ...)` | Element-wise operation | func: operation function |

### 1.3 Tensor Class `vector_map_as_tensor`

```python
from cos_comparison.core import vector_map_as_tensor

t = vector_map_as_tensor(vector=[1,2,3,4], shape=(2,2), strides=(2,1))
t.mean()       # mean
t.variance()   # variance
t[0, 1]        # index access
t[0, 1] = 5.0  # index assignment
len(t)         # total element count
for x in t: ... # iteration
```

### 1.4 Algorithm Selection

The `algorithm` parameter for `cos()` and `cos_comparison_passive/active`:

- `_cos` — standard cosine similarity (default)
- `_mod` — magnitude ratio
- `_cosmod` — cosine x magnitude ratio

## 2. Template Usage

### 2.1 Training Template `training_template.py`

Copy the template and modify the top config section:

```python
# ===== CONFIG =====
DATA_DIR = r"C:\path\to\training\data"   # training data location
BACKEND = 'cos_comparison_c'              # backend selection
WINDOW_SIZE = (3, 3)                      # passive mode window
KERNEL = None                             # active mode template (None=auto-generate)
OUTPUT_DIR = r"C:\path\to\output"         # output location
# ==================
```

Run:
```bash
python training_template.py
```

### 2.2 Matching Template `matching_template.py`

For query-reference matching tasks.

### 2.3 Config File Mode

Templates support reading parameters from a JSON config file:

```bash
python training_template.py --config my_config.json
```

Config file format see `config_template.json`.

## 3. Data Format Requirements

### 3.1 Tensor Input

All core functions accept these tensor formats:
- Nested list: `[[1.0, 2.0], [3.0, 4.0]]`
- `vector_map_as_tensor` object
- Any object implementing `__shape__()` / `__getitem__()` protocol (duck typing)
- 1D sequence: `[1.0, 2.0, 3.0]`

### 3.2 Training Data Directory Structure

```
DATA_DIR/
├── class_A/
│   ├── sample_001.txt   # one value per line, or comma-separated
│   ├── sample_002.txt
│   └── ...
├── class_B/
│   ├── sample_001.txt
│   └── ...
└── ...
```

Or unified format:
```
DATA_DIR/
├── train.csv            # first column label, rest features
└── test.csv
```

## 4. Callback Mechanism

Passive/active modes support callback functions for progress monitoring and error handling:

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

## 5. Notes

1. **Shape consistency**: `cos(a, b)` requires a and b to have identical shape, otherwise ValueError
2. **Window validity**: passive/active modes require `end - start - window_size >= 0`, otherwise "effectless args"
3. **Backend differences**: pydll fastest but requires compilation; ctypes medium; pure Python most compatible
4. **Zero-dimension tensors**: mean/variance of empty tensors (with zero dimension) returns None
5. **Output reuse**: pass `output` parameter to reuse pre-allocated tensor, avoid repeated allocation
