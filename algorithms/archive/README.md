# Archived Trial Set: `archived_pgd_principle_spatial_trials_v045.zip`

This folder holds one compressed archive of early, superseded exploration
trials. It is kept for provenance only and is **not** part of the active
exploration code.

## Set name

`archived_pgd_principle_spatial_trials_v045`

## Contents (59 files, ~187 KB uncompressed)

- `pgd_framework.py` and `pgd_framework_v2` ... `v9` (and `v7_1`, `v8_1/2/3`):
  an iterative "policy gradient / depth" framework series.
- `principle_driven.py`, `principle_driven_r2.py`, `principle_enhancement.py`.
- `spatial_bottleneck_resolution.py`, `spatial_bottleneck_v2.py`.
- `deep_breakthrough.py`, `generality_test.py`, `multitask_validation.py`,
  `retrospective_analysis.py`, `universal_enhancements.py`.
- Their paired `*_REPORT.md` / `*_BOTTLENECK*.md` / `PGD_*.md` /
  `PRINCIPLE_*.md` notes and `*_results.json` result dumps.

## Why archived

These trials mixed in gradient/weight-style mechanisms and global operations
that conflict with the project's core principles (strictly local comparison,
continuous-mapping hierarchical isolation, unsupervised structure-first
learning, no deep-learning machinery). They were intermediate, semantically
confused iterations and have been superseded by the minimal local k-NN learner
(`core/localknn.py`, PCML v53).

## Restore

Unzip the archive to recover any file:

```powershell
Expand-Archive algorithms\archive\archived_pgd_principle_spatial_trials_v045.zip `
  -DestinationPath algorithms\archive\_restored
```

The active, supported exploration code lives in `core/`, `algorithms/` (top
level), `examples/`, `templates/` and `module_batch/`.
