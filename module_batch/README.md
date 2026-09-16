# module_batch / Batch Instruction Files

Command-line batch mode instruction files.  Extension `.module_bat`,
executed via `python -m cos_comparison batch <file.module_bat>`.
The file's directory joins `sys.path`, so helper modules next to the
file (e.g. `input`) are importable.

## Instruction Format

```
func arg1 arg2 -kw value          function call (one instruction)
func arg1 arg2 -> pos             store the result at data[pos]
data[index]                       data region reference (dataN legacy ok)
%expr                             unpack a sequence into arguments
%%expr                            unpack a mapping into keyword arguments
let name value                    variable assignment (function style)
IF <cond-call> ... ELSE ... END   conditional block
WHILE <cond-call> ... END         loop block
# comment / blank line            skipped
```

## Literal Rules

| Type | Syntax | Example |
|------|--------|---------|
| Integer | direct | `42` |
| Float | decimal point | `3.14` |
| String | quoted | `"hello"` |
| Tuple | `(a b c)` | `(2 2)` |
| List | `[a b c]` | `[1.0 2.0 3.0]` |
| Nested list | `[[a b] [c d]]` | `[[1.0 2.0] [3.0 4.0]]` |
| Boolean | `True` / `False` | `True` |
| Null | `None` | `None` |
| Variable ref | `data[index]` | `data[0]` |

## Namespace

- Built-in functions (`len`, `sum`, `int`, `str`, ...) are pre-imported.
- Project modules by dotted names (`core.add_chain`).
- `import_module <mod> [-name alias]` — import a module.
- `import_all_module <mod> [-namespace target]` — attach ALL public
  objects of a module (like `from module import *`); the target
  namespace defaults to the current shell function table.
- `list_modules` — list project / imported modules.
- `let` / `get` / `delete` — variables.
- `&name` / `*index` — variable index / dereference (pointer style).

## Available Templates

| File | Purpose | Data location |
|------|---------|---------------|
| `quick_call.module_bat` | direct core calls | inline data |
| `training.module_bat` | training pipeline | `input.set_data_dir "path"` |
| `matching.module_bat` | matching pipeline | `input.set_query_dir` / `set_reference_dir` |
| `basic_demo.module_bat` | basic operations | inline data |
| `feature_extraction.module_bat` | 6-level feature pipeline | inline data |
| `matching_comparison.module_bat` | multi-tensor matching | inline data |
| `control_flow.module_bat` | control flow / data chains | inline data |
| `fourier_analysis.module_bat` | Fourier analysis | inline data |
| `topology_analysis.module_bat` | graph / Euler analysis | inline data |
| `full_pipeline.module_bat` | end-to-end pipeline | inline data |

The pre-call templates import the `input` helper (`input.py` beside
them) to set the training / query / reference data locations.

## Core API Summary

| Function | Purpose | Key params |
|----------|---------|-----------|
| `cos` | tensor cosine similarity | `a b -algorithm cos/mod/cosmod` |
| `cos_comparison_passive` | sliding-window self-similarity | `-window_size -step -d -w1 -w2 -b1 -b2` |
| `cos_comparison_active` | template matching | `-kernel -step -w1 -w2 -b1 -b2` |
| `mean_local` / `local_variance` | local statistics | `-local_size -step` |
| `threshold_filter` / `threshold_map` | thresholding | `-low -high` / `-pairs -default_value` |
| `infer_shape` | shape inference | `data` |
| `linear_algebra.*` | dot / norm / normalize / scale / add / multiply / power / clip / flatten / tensor_sum / tensor_mean | duck typed, `-output`, integer status |

## Execution

```bash
python -m cos_comparison batch basic_demo.module_bat
echo "cos [1 2 3] [1 2 3]" | python -m cos_comparison batch
python -m cos_comparison      # then: ... run basic_demo.module_bat
```

## Notes

1. `active` mode needs the kernel as a keyword: `cos_comparison_active data -kernel [...]`.
2. Groups use spaces, not commas: `(2 2)`.
3. Strings need quotes; `data[index]` uses brackets.
