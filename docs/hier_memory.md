# Continuous Mapping Hierarchical Isolation Memory

Script: test/hierarchical_memory.py (v1-v4) | Formalized: tests/memory_hier.py
Data flow: L1 entry layer -> L2 topic prototype layer -> L3 viewpoint layer

## Core Mechanism

```
Continuous mapping: entry feature vector <-> topic prototype cos continuous score
Threshold isolation: score > threshold -> assigned to prototype layer, else isolated as new topic
On-demand extraction: only demand(needs)-relevant entries abstract into L2,
  irrelevant data isolated at L1 and does not rise
  (NOT full compression! high level stays pure)
Detail retention: abstracted entries retain difference vectors + detail words (no omission)
Hierarchical mutual driving: bottom-up aggregation (L1->L2->L3);
  top-down demand signals (L3 gaps + detail words)
  -> drives L2 mapping and L1 search; isolation is not fixed (re-evaluated as demand evolves)
Search/organization decoupled: producers (parallel search) only write L1;
  consumers read L1 write L2/L3
```

## Results

- On-demand extraction stats: abstraction/isolation counts (irrelevant data does not rise) verified
- Demand evolution: covered classes exit demand, gap classes enter demand -> drives targeted search verified
- Co-training (training + validation both participate in stats): frequency distribution
  becomes more accurate with data accumulation verified

## Formalization

tests/memory_hier.py: pure standard library + cos HierMemory (on-demand extraction +
inter-layer driving), test_explore_smoke coverage (demand hit/isolation/gap driving) verified

## Conclusion

Continuous mapping hierarchical isolation applied to memory = on-demand extraction +
bidirectional interaction + accumulability (database thinking), fundamentally different
from deep learning full compression.
