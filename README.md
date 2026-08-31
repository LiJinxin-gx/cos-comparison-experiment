# cos-comparison-experience

> Algorithm exploration repository for the [cos_comparison](https://github.com/LiJinxin-gx/cos-comparison) project.
> Zero dependencies. Tensor-in, tensor-out. Ideas over implementations.

## What is this?

This repository is the **exploration wing** of the cos_comparison project — a
place where new algorithms, training paradigms, and architectural ideas are
prototyped, tested, and refined before being considered for the main library.

The main `cos-comparison` repository is a production library with C extensions,
multiple computation backends, and a full API surface. This repository exists to
share the **ideas** cleanly: every algorithm here is stripped of engineering
overhead, written in pure Python with no external dependencies, so the core
mechanisms are visible and adaptable.

## Purpose

- **Explore freely**: test new training methods, representation schemes, and
  cross-modal approaches without affecting the production codebase.
- **Share ideas**: each algorithm is self-contained and readable — the goal is
  to communicate *how* something works, not to ship optimized code.
- **Validate principles**: the core design principles of cos_comparison
  (continuous mapping, hierarchical isolation, on-demand extraction,
  interpretability) are tested across domains and modalities here.

## Core Principles

These principles guide all exploration in this repository and remain stable
regardless of which specific algorithm is being tested:

| Principle | Meaning |
|-----------|---------|
| **Continuous Mapping Hierarchical Isolation** | Multi-level representation; irrelevant detail is isolated at low levels, high-level context can recover it on demand |
| **On-Demand Extraction** | Information is extracted when needed, not compressed wholesale — unlike black-box weight learning |
| **Interpretability First** | Every match traces to explicit templates; every generation decomposes to a traceable path |
| **Zero Dependencies** | Pure Python standard library. Tensors are nested lists. No numpy, no PyTorch, no PIL |
| **No Recursion** | All algorithms use iterative traversal — guaranteed stack safety |
| **Cross-Domain by Design** | Algorithms operate on tensors, so the same mechanism applies to images, text, audio, video, and structured data |

## Repository Structure

```
explore/
├── core/            # Foundational primitives (similarity, multi-scale, hierarchy)
├── algorithms/      # Complete algorithm implementations (each file = one idea)
├── examples/        # Runnable demonstrations of each algorithm
├── templates/       # Call templates for the cos_comparison production library
├── module_batch/    # Batch instruction files for command-line execution
└── docs/            # Design notes and exploration logs
```

The structure is organic — new directories and files appear as new directions
are explored. The `core/` layer provides shared primitives; everything above it
is a self-contained experiment.

## How to Use

### Run an example

```bash
python examples/<example_name>.py
```

Every example is standalone and requires no installation beyond Python 3.8+.

### Use an algorithm in your own code

```python
import sys
sys.path.insert(0, "/path/to/explore")
from algorithms.<name> import <AlgorithmClass>
```

All algorithms accept tensors (nested lists of numbers) and return tensors or
explicit structured results.

### Run batch instructions

```bash
python -m cos_comparison batch module_batch/<file>.module_bat
```

Batch files use the cos_comparison command-line protocol to chain multiple
operations in sequence.

## Relationship to cos_comparison

| Aspect | cos-comparison (main) | cos-comparison-experience (this repo) |
|--------|----------------------|---------------------------------------|
| Focus | Production library | Algorithm exploration |
| Code | C + Python, 3 backends | Pure Python, zero dependencies |
| Performance | Optimized (SIMD, PyBuffer, free-thread) | Readable, not optimized |
| Stability | Versioned releases | Living experiments |
| Input | Full API (duck typing, protocols) | Tensors only (nested lists) |

Ideas validated here may eventually be reimplemented in the main repository
with full optimization and API support.

## License

MIT — share and adapt freely.
