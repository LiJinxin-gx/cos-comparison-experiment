# cos-comparison-experience

> Algorithm exploration repository for the [cos_comparison](https://github.com/LiJinxin-gx/cos-comparison) project.
> **Recommended surface: [`platform/`](docs_platform/PLATFORM_GUIDE.md)** — a mature open-integration platform (NumPy allowed, formal backend auto-selected, I/O decoupled from processing).
> Legacy folders (`core/`, `algorithms/`, `examples/`) stay zero-dependency idea sketches.

## Latest platform (v1.0.0)

The **`platform/` package** integrates the v1.0.0 core identity:
extraction and generation are the forward / reverse of **one local comparison
relation**, and **closure** is the universal criterion.

- open integration: measurement auto-selects the installed cos_comparison
  (C/python) -> NumPy -> pure Python;
- matched I/O pairs (`save_field`/`load_field`, `fit_symbols`/`decode_symbols`)
  decoupled from processing (`run_continuous`, `run_discrete`);
- runs standalone (`python -m platform`) or imports as a module;
- verified: I/O symmetry True, continuous closure mean|dE|~0 / max~2e-6,
  discrete grounded connectors 100%.

Root-level exploration scripts (v1.0.0):

| File | Purpose |
|------|---------|
| `companion_demo.py` | Demonstrates the installed v1.0.0 engine (cos/mod/cosmod, passive, active, pooling, backend parity) |
| `minimal_principle.py` | Minimum readable self-contained principle walkthrough |
| `atomic_contrast.py` | Atomic contrast matching |
| `group_hierarchical.py` | Group / hierarchical isolation experiments |
| `heatmap_analysis.py` | Large-scale heatmap analysis |
| `local_knn.py` | Local k-NN on-demand learner |
| `memory.py` | Open memory system exploration |
| `demo_image.py` / `demo_text.py` / `demo_speech.py` / `demo_reasoning.py` / `demo_generation.py` | Domain demos |
| `test_mnist.py` / `test_multi_domain.py` | Multi-domain recognition tests |
| `train_llm.py` / `train_real.py` | Training paradigms |

Start with [companion_demo.py](companion_demo.py) for the installed-engine
walkthrough, and [platform/](platform/) for the mature integration surface.

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
├── companion_demo.py    # v1.0.0: installed-engine walkthrough
├── minimal_principle.py # v1.0.0: minimum principle
├── atomic_contrast.py  # v1.0.0: atomic contrast matching
├── group_hierarchical.py
├── heatmap_analysis.py
├── local_knn.py
├── memory.py
├── demo_image.py / demo_text.py / demo_speech.py / demo_reasoning.py / demo_generation.py
├── test_mnist.py / test_multi_domain.py
├── train_llm.py / train_real.py
├── platform/            # Recommended: open-integration platform (backends, io_pairs, relation_core, heatmap, depscan)
├── candidates/          # Immature but promising results, tracked publicly
├── core/                # [legacy] Foundational primitives (zero dependency)
├── algorithms/          # [legacy] One file = one idea (zero dependency)
├── examples/            # [legacy] Runnable demos of legacy algorithms
├── templates/           # [legacy] Call templates for the production library
├── module_batch/        # [legacy] Batch instruction files
└── docs/                # [legacy] Design notes and exploration logs
```

The `platform/` package is the recommended surface for mature work; `candidates/`
tracks promising immature results. Both use **distributed dependency
annotation** — each file declares its own `DEPENDENCIES` (Python version,
required/optional distributions, backends, status, target); scan with
`python -m platform.depscan`. The legacy folders are retained unchanged.

## License

MIT — share and adapt freely.
