# Core Principles

This repository shares algorithms, not implementations. Every algorithm here
is built on a small set of interlocking principles.

## 1. Continuous Mapping Hierarchical Isolation

The foundational mechanism. Data is represented at multiple abstraction
levels (fine → coarse).

- **Isolation (bottom-up)**: Low-level detail deemed irrelevant is masked
  and does NOT propagate to higher levels. This is "on-demand extraction" —
  we keep only what matters at each level.
- **Drive (top-down)**: High-level context can unmask low-level positions,
  recovering detail when needed. No information is permanently lost; it is
  merely gated.

This is NOT dimensionality reduction. It is selective attention across
scales.

## 2. Structure First

Representation is built from structural patterns, not semantic labels.
The model does not need to know what a "nose" or a "noun" is — it finds
recurring structural configurations and stores them as templates.

- Training uses structure as the core criterion.
- Martian encoding is the extreme expression: symbols are anonymized, only
  their structural pattern remains.
- "Understanding the world precedes describing it."

## 3. Information Arises from Difference

Passive mode extracts boundaries — positions where data changes. These are
structural anchors. Active mode matches templates — these are semantic
matches. The two are complementary:

- Passive answers WHERE information density is highest.
- Active answers WHAT pattern matches there.
- Joint extraction uses passive points to guide active matching, avoiding
  wasted computation on uniform regions.

## 4. No Residuals

Residual connections are a deep-learning black-box compensation for
information loss in opaque transformations. This project extracts information
on demand — adding residuals contradicts the design and introduces
interference (verified: removing residuals improved generation consistency
from 70-92% to 100%).

Instead, variation is captured by **multiple templates per class per level**.
Each template represents a discrete variation mode. A data point is its
**template path**: the sequence of nearest template indices at each level.

## 5. Instructions as Data (Von Neumann)

Behavior is not hardcoded. Instructions are stored as data in a database;
a fixed interpreter reads and executes them at runtime. This achieves:

- Fixed code implements plural logic.
- Execution logic can be generated and modified without changing code.
- Behavior is controlled by external config / prompt.

## 6. Zero Recursion, Zero External Dependencies

Every algorithm is implemented iteratively (odometer-style carry loops).
No numpy, no PIL, no PyTorch — pure Python standard library. Tensors are
nested lists. This makes the algorithms portable, auditable, and focused
on the idea rather than the framework.

## 7. Interpretability by Design

- Every match can be traced to specific templates at specific levels.
- Every generation can be decomposed into its template path.
- No black-box weight matrices — knowledge is stored as explicit,
  inspectable templates.
