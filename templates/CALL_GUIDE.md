# Pre-Call Template Instruction Files

Pre-call templates translated from `templates/` into instruction files
(`module_batch/*.module_bat`).  Run them with the batch executor and
set the data locations by importing the `input` helper next to the
templates.

## Templates

| Template | Purpose | Data location |
|----------|---------|---------------|
| `quick_call.module_bat` | direct core calls (cos / passive / active / stats / tensor / threshold) | inline data |
| `training.module_bat` | training pipeline (load / features / similarity) | `input.set_data_dir "path"` |
| `matching.module_bat` | matching pipeline (query / reference / top-k) | `input.set_query_dir` / `input.set_reference_dir` |

## Usage

```bash
python -m cos_comparison batch training.module_bat
```

Edit the data-location line before running.  The `input` helper
(`input.py` beside the templates) provides:

- `set_data_dir` / `set_query_dir` / `set_reference_dir` — data locations
- `load_tensor` / `load_all` / `load_training_data` / `load_queries` /
  `load_references` — tensor loading (txt / csv / nested JSON)
- `count` / `key_at` / `value_at` / `sample_at` — iteration helpers
- `top_matches(query, references, k)` — cosine top-k ranking

## Instruction Format (shared with shell / app)

```
func arg1 arg2 -kw value        one instruction (function call)
func arg1 arg2 -> pos           store the result at data[pos]
data[index]                     data region reference
%expr / %%expr                  sequence / mapping unpack
IF ... ELSE ... END             conditional block
WHILE ... END                   loop block
# comment                       skipped
```

## Data Formats

- `txt`: one float per line; `csv`: comma-separated floats; `nested`:
  JSON nested list.
- Training data uses class directories:
  `data_dir/class_A/sample_001.txt` ...

## Core API Summary

| Function | Purpose |
|----------|---------|
| `cos(a, b, algorithm=...)` | full-tensor cosine similarity |
| `cos_comparison_passive(data, window_size, ...)` | sliding-window self-similarity |
| `cos_comparison_active(data, kernel, ...)` | template matching |
| `mean_local` / `local_variance` | local statistics |
| `infer_shape` | shape inference |
| `threshold_filter` / `threshold_map` | thresholding |
| `linear_algebra.*` | dot / norm / normalize / scale / add / multiply / power / clip / flatten / tensor_sum / tensor_mean (duck typed, `output` keyword, integer status) |

## Notes

1. `active` mode requires the kernel as a keyword: `cos_comparison_active data -kernel [...]`.
2. Groups use spaces, not commas: `(2 2)`.
3. Strings need quotes; `data[index]` uses brackets.
