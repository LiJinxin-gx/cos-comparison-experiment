# cos-comparison-experience

> Algorithm exploration repository for the cos_comparison project.
> Zero dependencies. Tensor-in, tensor-out. Ideas over implementations.

## What is this?

This repository shares the **algorithms** discovered and validated during
the cos_comparison project — not the production code. Every algorithm here
is:

- **Zero-dependency**: pure Python standard library. No numpy, no PIL, no
  PyTorch. Tensors are nested lists.
- **Self-contained**: no data loading, no preprocessing, no I/O. Inputs are
  tensors (lists) or strings that can be processed directly.
- **Interpretable**: every match traces to explicit templates; every
  generation decomposes to a template path. No black-box weights.

## Why a separate repository?

The main `cos-comparison` repository is a production library with C
extensions, three backends, GUI, and full API surface. This repository
exists to share the **ideas** cleanly — the algorithms that make the
project work, stripped of engineering overhead.

## Core Principles

| Principle | Meaning |
|-----------|---------|
| **Continuous Mapping Hierarchical Isolation** | Multi-level representation; irrelevant detail is isolated at low levels, high-level context can recover it |
| **Structure First** | Represent structural patterns, not semantic labels. "Understanding precedes describing." |
| **Information Arises from Difference** | Passive mode finds boundaries, active mode matches templates — complementary |
| **No Residuals** | Variation captured by multiple templates, not residual compensation |
| **Instructions as Data** | Fixed interpreter executes data-stored behavior programs |
| **Zero Recursion** | All algorithms iterative (odometer-style carry loops) |

See [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md) for full explanation.

## Repository Structure

```
explore/
├── core/                    # Core algorithmic primitives
│   ├── cosine.py           # Cosine similarity, tensor utilities
│   ├── multiscale.py       # Multi-scale matching (most broadly applicable)
│   ├── passive_active.py   # Passive boundary + active template joint extraction
│   └── hierarchical.py     # Hierarchical representation with isolation + drive
├── algorithms/              # Full algorithm implementations
│   ├── bhsm.py             # Bidirectional Hierarchical Similarity Matching
│   ├── thcn.py             # Pure Template Network (no residual)
│   ├── cross_modal.py      # Cross-modal mapping via paired bridging
│   ├── martian_encoding.py # Language-agnostic internal representation
│   ├── memory.py           # Hierarchical memory with level driving
│   └── von_neumann_agent.py# Instructions-as-data agent engine
├── examples/                # Runnable demonstrations (zero deps)
│   ├── example_cosine.py
│   ├── example_passive_active.py
│   ├── example_hierarchical.py
│   ├── example_thcn.py
│   ├── example_cross_modal.py
│   ├── example_martian.py
│   ├── example_memory.py
│   └── example_agent.py
└── docs/
    ├── PRINCIPLES.md       # Core design principles
    └── ALGORITHMS.md       # API reference
```

## Quick Start

```bash
# No installation needed. Just run any example:
python examples/example_cosine.py
python examples/example_thcn.py
python examples/example_cross_modal.py
```

Or use as a library:

```python
import sys
sys.path.insert(0, "/path/to/explore")

from core import multiscale_cosine
from algorithms import PureTemplateNetwork, martian_similarity

# Multi-scale matching (verified across 5 domains)
sim = multiscale_cosine([1,2,3,4,5], [1,2,3,4,6])

# Pure template classification + generation
net = PureTemplateNetwork(levels=(64, 32, 16, 8), k_per_level=3)
net.train({"A": [...], "B": [...]})
label, score = net.classify(new_sample)
generated = net.generate([("A",0), ("A",0), ("B",0), ("B",0)])

# Language-agnostic text similarity
sim = martian_similarity("hello world", "你好世界")
```

## Algorithms Overview

### Multi-Scale Cosine Matching (`core/multiscale.py`)
The most broadly applicable algorithm. Represent data at multiple scales
(original + downsampled), match independently, ensemble with finer scales
weighted higher. Verified across: face (77.5%), captcha (100%), text
(66.7%), video frames (100%), audio (100%).

### BHSM (`algorithms/bhsm.py`)
Bidirectional Hierarchical Similarity Matching. 11-stage pipeline:
boundary extraction → environment suppression → multi-threshold response →
keypoint selection → geometric signature → template matching → coarse rank
→ fine re-rank. Current best: 73.3% on standard face benchmark (local optimum).

### Pure Template Network (`algorithms/thcn.py`)
Hierarchical template network WITHOUT residual connections. Each data point
= template path (nearest template per level). Classification via weighted
voting. Generation via cross-class template path selection. Verified: 95.8%
face classification, 100% generation consistency.

### Cross-Modal Paired Bridging (`algorithms/cross_modal.py`)
Two-stage retrieval solving the cross-modal distance problem (direct
image-text similarity ~0.088). Stage 1: quantized attribute coarse screen.
Stage 2: paired bridge — match against candidate's paired sample in the
SAME modality. Result: image→text 100% retrieval.

### Martian Encoding (`algorithms/martian_encoding.py`)
Language-agnostic internal representation. Symbol anonymization (structure
preserved, identity forgotten) + structural features + normalized
compression distance. "Understanding the world precedes describing it."

### Hierarchical Memory (`algorithms/memory.py`)
Multi-level memory with bidirectional level driving. Search and
organization decoupled (parallel search across levels). Top-down driven
search narrows low-level scope using high-level context.

### Von Neumann Agent (`algorithms/von_neumann_agent.py`)
Fixed-code interpreter for data-stored behavior programs. Instructions as
data: behavior changes by changing the program, not the code. External
config generates programs. Few-shot learning via memory retrieval.

## Verified Results

| Domain | Task | Algorithm | Result |
|--------|------|-----------|--------|
| Face | Recognition | BHSM v12 | 73.3% (standard face benchmark, local optimum) |
| Face | Classification | Pure Template (k=5) | 95.8% |
| Face | Generation consistency | Pure Template | 100% |
| Captcha | Recognition | Multi-scale | 100% |
| Text | Topic classification | Multi-scale | 66.7% |
| Video | Frame retrieval | Multi-scale | 100% |
| Audio | Classification | Multi-scale | 100% |
| Cross-modal | Image→Text | Paired Bridge v4 | 100% |
| Cross-modal | Text→Image | Paired Bridge v4 | 35% (Top10=100%, info ceiling) |
| Cross-domain | Joint training | Unified vector | 100% within-domain |
| General AI | Instruction following | Structure matching | 100% |

## Call Templates (`templates/`)

Ready-to-use templates for direct calls to the `cos_comparison` production
library. Each template supports config-file or inline parameter setup,
including training data location.

| Template | Purpose | Key Features |
|----------|---------|-------------|
| `quick_call_template.py` | Minimal direct API calls | 6 demos: cos / passive / active / statistics / tensor / threshold |
| `training_template.py` | Full training pipeline | Multi-class data loading, feature extraction, similarity matrix, JSON output |
| `matching_template.py` | Query-reference matching | cos / passive / active modes, top-K results, JSON output |
| `config_template.json` | JSON configuration | All parameters in one file for `--config` usage |
| `CALL_GUIDE.md` | API reference & instructions | Full function signatures, parameter docs, callback guide, data formats |

```bash
# Quick start (requires cos_comparison installed)
python templates/quick_call_template.py

# Training with custom data
python templates/training_template.py --data-dir /path/to/data --output-dir /path/to/out

# Matching with config file
python templates/matching_template.py --config my_config.json
```

## License

MIT — share and adapt freely.
