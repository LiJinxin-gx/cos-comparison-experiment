# cos-comparison-experience

> Algorithm exploration repository for the [cos_comparison](https://github.com/LiJinxin-gx/cos-comparison) project.
> **Recommended surface: [`platform/`](docs_platform/PLATFORM_GUIDE.md)** — a mature open-integration platform (NumPy allowed, formal backend auto-selected, I/O decoupled from processing).
> Legacy folders (`core/`, `algorithms/`, `examples/`) stay zero-dependency idea sketches.

## Latest platform (v0.5.3)

The **`platform/` package** deeply integrates the latest core identity:
extraction and generation are the forward / reverse of **one local comparison
relation**, and **closure** is the universal criterion.

- open integration: measurement auto-selects the installed cos_comparison
  (C/python) -> NumPy -> pure Python;
- matched I/O pairs (`save_field`/`load_field`, `fit_symbols`/`decode_symbols`)
  decoupled from processing (`run_continuous`, `run_discrete`);
- runs standalone (`python -m platform`) or imports as a module;
- verified: I/O symmetry True, continuous closure mean|dE|~0 / max~2e-6,
  discrete grounded connectors 100%.

Start with [docs_platform/PLATFORM_GUIDE.md](docs_platform/PLATFORM_GUIDE.md)
and [examples_platform/demo_platform.py](examples_platform/demo_platform.py).

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
├── platform/          # Recommended: open-integration platform (backends, io_pairs, relation_core, heatmap, depscan)
├── candidates/        # Immature but promising results, tracked publicly (each with DEPENDENCIES)
├── examples_platform/ # Mature standalone demo (out_platform/ heatmaps)
├── docs_platform/     # Platform guide
├── core/              # [legacy] Foundational primitives (zero dependency)
├── algorithms/        # [legacy] One file = one idea (zero dependency)
├── examples/          # [legacy] Runnable demos of legacy algorithms
├── templates/         # [legacy] Call templates for the production library
├── module_batch/      # [legacy] Batch instruction files
└── docs/              # [legacy] Design notes and exploration logs
```

The `platform/` package is the recommended surface for mature work; `candidates/`
tracks promising immature results. Both use **distributed dependency
annotation** — each file declares its own `DEPENDENCIES` (Python version,
required/optional distributions, backends, status, target); scan with
`python -m platform.depscan`. The legacy folders are retained unchanged.

## License

MIT — share and adapt freely.
