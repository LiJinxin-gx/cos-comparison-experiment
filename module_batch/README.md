# module_batch / Batch Instruction Files

> Command-line batch mode instruction file collection. Extension `.module_bat`,
> executed via `python -m cos_comparison batch <file.module_bat>`.

## Instruction Format

```
func arg1 arg2 -kw value          # execute function call
func arg1 arg2 -> pos             # store result at data[pos]
dataN                             # reference value at data[N] (N is a digit, no angle brackets)
IF <cond-call> ... ELSE ... END   # conditional block
WHILE <cond-call> ... END         # loop block
# comment                         # comments (blank lines skipped)
```

### Literal Rules

| Type | Syntax | Example |
|------|--------|---------|
| Integer | direct | `42` |
| Float | with decimal point | `3.14` |
| String | quoted | `"hello"` |
| Tuple | `(a b c)` space-separated | `(2 2)` |
| List | `[a b c]` space-separated | `[1.0 2.0 3.0]` |
| Nested list | `[[a b] [c d]]` | `[[1.0 2.0] [3.0 4.0]]` |
| Boolean | `True` / `False` | `True` |
| Null | `None` | `None` |
| Variable ref | `dataN` | `data0` |

### Namespace Operations

```
let name value        # define variable (stored in shell namespace, NOT data dict)
get name              # get variable value
delete name1 name2    # delete variables
import_module mod     # import module into namespace
list_modules          # list available modules
```

**Important**: `let` stores in the shell namespace (`_ns["vars"]`), while `dataN`
references the batch executor's data dict. These are separate namespaces. To reuse
a value across instructions, store it via `-> pos` and reference via `dataN`.

## Available Functions

### Core Algorithms

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `cos` | Full-tensor cosine similarity | `a b -algorithm cos/mod/cosmod` |
| `cos_comparison_passive` | Passive mode (sliding window self-similarity) | `-window_size -step -d -w1 -w2 -b1 -b2` |
| `cos_comparison_active` | Active mode (template matching) | `-kernel -step -w1 -w2 -b1 -b2` |
| `mean_local` | Local mean | `-local_size -step` |
| `local_variance` | Local variance | `-local_size -step` |
| `threshold_map` | Threshold mapping | `-pairs -default_value` |
| `data_filter` | Custom filter | `-callback` |
| `data_mapping` | Custom mapping | `-callback` |
| `infer_shape` | Infer tensor shape | `data` |
| `load_as_default_data` | Load standard format | `data -start -shape -step` |

### Fourier Analysis

| Function | Purpose |
|----------|---------|
| `dft` | Discrete Fourier Transform |
| `idft` | Inverse DFT |
| `power_spectrum` | Power spectrum |
| `dft_kernel_real` | DFT real kernel |
| `dft_kernel_imag` | DFT imaginary kernel |

### Topology Analysis

| Function | Purpose |
|----------|---------|
| `shortest_path_between` | Graph shortest path |
| `Euler_characteristic_compute_by_cell` | Euler characteristic |

### Built-in Functions

`len`, `sum`, `int`, `str`, `list`, `print`, `max`, `min`, `abs`, `range`,
`sorted`, `reversed`, `enumerate`, `zip`, `map`, `filter` and all Python
built-in callables.

## File Listing

| File | Content |
|------|---------|
| `basic_demo.module_bat` | Basic operations: cos/passive/active/statistics/threshold |
| `feature_extraction.module_bat` | Multi-level feature extraction pipeline (6 levels) |
| `matching_comparison.module_bat` | Multi-tensor matching comparison and ranking |
| `control_flow.module_bat` | Control flow and data chaining demo |
| `fourier_analysis.module_bat` | Fourier transform and power spectrum analysis |
| `topology_analysis.module_bat` | Graph shortest path and Euler characteristic |
| `full_pipeline.module_bat` | End-to-end full pipeline (6 stages) |

## Execution

```bash
# Execute a single batch file
python -m cos_comparison batch basic_demo.module_bat

# Execute via stdin
echo "cos [1 2 3] [1 2 3]" | python -m cos_comparison batch

# In interactive shell
python -m cos_comparison
>>> shell
... run basic_demo.module_bat
```

## Notes

1. **Active mode kernel must use keyword argument**: `cos_comparison_active data -kernel [...]`,
   positional passing is captured by `*arg`.
2. **Tuples/lists use space separation**, not commas: `(2 2)` not `(2,2)`.
3. **dataN indexing starts at 0**, `-> pos` also uses 0-based index.
4. **Strings must be quoted**, otherwise treated as function name or variable.
5. **Nested lists written directly**: `[[1.0 2.0] [3.0 4.0]]`.
6. **Each instruction result is automatically printed** by the batch executor.
