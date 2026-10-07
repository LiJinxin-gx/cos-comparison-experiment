# Candidate results registry

Immature but principle-conforming results, tracked publicly here so they are
decoupled from the main library's release cadence. Each module runs
independently and carries a distributed `DEPENDENCIES` block; scan the whole
repository with `python -m platform.depscan`.

Status legend: **candidate** = promising, open bottleneck; may move to
`platform` once the criterion is met.

| Module | Targets bottleneck | Evidence (self-contained data) | Open question |
|--------|--------------------|--------------------------------|---------------|
| `cand_context_states.py` | coherent generation without learned weights | 7 self-emergent context states; unsupervised purity 100%; constraint-satisfaction generation closure 100% but coverage 4/16 | recorded neighbor transitions alone under-determine growth; how to supply enough target constraints without guessing |
| `cand_feedback.py` | in-application adaptation without polluting the main DB | coarse-to-fine cascade: confident accuracy 80%, 1 abstained; fine profiles held in inner DB; controlled merge into a main fine section | derive the ambiguity gap from the score distribution; validate fine-stage gains at scale |
| `cand_template_search.py` | scaling comparison to large libraries | shortlist recall 100%, routed answer == brute 100%, 5.8x fewer elementwise ops (N=36, K=3) | keep recall at 100% as the library and signature change; choose K from data |

## How these relate to the core identity

All three are refinements around the same identity (extraction / generation are
forward / reverse of one local comparison; closure is the criterion):

- context states attack the **generation** side: content is an arrangement that
  satisfies local contextual expectations;
- feedback attacks the **application / learning** side: ambiguous points get
  finer extraction in an isolated inner DB until a reviewed merge;
- template search attacks the **complexity** side: a cheap local signature routes
  to a shortlist so per-query cost stays linear.

## Distributed dependency annotation

Each file declares its own dependencies inline (Python version, required and
optional distributions, measurement backends, status, target). The scanner parses
these without importing and checks them against the running interpreter:

```bash
python -m platform.depscan
```

This is deliberate: exploration modules may track different versions of the
formal library, so dependencies are stated next to the code rather than assumed
from a single central manifest.
