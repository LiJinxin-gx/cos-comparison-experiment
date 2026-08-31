# Face Recognition Exploration Log — Statistical Local Attention Series

> Environment: free-threaded Python 3.14, cp314t core/numpy
> Datasets: (1) 67 ID photos (testdata/face) + face1/face2 validation
>           (2) Standard face benchmark 40x10 (testdata/orl/faces, 400 PGM)
> Plan: exploration data and scripts to be organized into GitHub repository

## v1 — Feedback Dual-Channel (face_feedback.py)

- Passive + active dual channel -> label statistics -> intersection key regions ->
  position capture -> feedback parameter tuning
- Results: face2 high purity verified; face1 failed (environment interference);
  variant feedback tuning 0.724-0.754

## v2 — Statistical Local Attention (face_attention.py)

- Multi-parameter contrast points (passive x3 + active x4 = 7 parameters) ->
  frequency statistics -> key structural points
- Co-training (training + validation), dynamic target region locking,
  out-of-region masking, set coverage comparison
- C acceleration: core.threshold_map + numpy vectorization + free-threaded 18 workers
- Results: **face1 subject A verified / face2 subject B verified (2/2)**;
  standard benchmark 40x10 average 0.328 (std 0.0225, 3 rounds, random baseline 0.025)

## v3 — Structural Descriptor + Multi-Level Fusion + Region Multi-Scale + Self-Correction

- Keypoint structural descriptor: neighborhood (5x5) response histogram (4 bins) +
  mean/variance (6 dims), anti-interference matching
- Multi-level frequency threshold fusion: 95/98 two-level key structures ->
  each combination independently locked/scored -> voting + score fusion
- Self-correction: multi-tolerance (3/2/1) x multi-level keypoint iterative locking,
  highest confidence combination wins
- Region multi-scale: designed for locked region magnification comparison
  (not enabled in this run, recorded as TODO)

### v3 Results

- face1 -> subject A verified / face2 -> subject B verified (2/2, fusion score 0.545/1.000 maintained)
- Standard benchmark 40x10 (first 2 rounds): 0.300 / 0.330 — on par with v2 (no improvement)
- Performance: self-correction (2 levels x 3 tolerances = 6 combinations) makes
  benchmark each round ~3x slower, full evaluation incomplete

## v4 — Discriminative Key Structure + Descriptor Matching + Intensity Weighting + High-Low Bidirectional

- Low level: coarse-fine dual-grade contrast points (coarse 60% + fine 75%,
  same response reuse); mid level: frequency + intra-class concentration
- High level: discriminative keypoints = high frequency AND intra-class concentrated;
  high level drives low level: key structures -> locked region + fine grade
- Comparison: descriptor cosine filtering + intensity weighted set coverage;
  self-correction (multi-level x multi-tolerance)

### v4 Results and Conclusion

| Variant | face1 | face2 | Benchmark(1 round) | Conclusion |
|---------|-------|-------|---------------------|------------|
| Descriptor matching included | subject C mismatch | subject B verified | - | cross-domain descriptor mismatch, harmful |
| Concentration keypoints (single sample) | 0 points -> fallback | - | 0 points -> fallback | single sample per class cannot compute concentration |
| Concentration keypoints (benchmark 5 samples) | - | - | 0 points -> fallback | position-level features have no class-exclusive points, too strict |
| Coarse-fine dual-grade vector | subject C mismatch | subject B verified | - | fine-grade count breaks matching |
| Intensity weighting (bound to desc) | - | - | - | not independently verified |
| Final retained state | subject A verified | subject B verified | 0.30 | fallback to pure position coverage + coarse grade = v3 baseline |

## v5-v12 Evolution Summary

Subsequent versions (v5-v12) explored: multi-template clustering, cross-modal
paired bridging, pure template network (no residual), hierarchical memory integration,
and cross-domain joint training. Best result on standard face benchmark: **73.3%**
(local optimum) with the BHSM v12 pipeline (11-stage: boundary extraction ->
environment suppression -> multi-threshold response -> keypoint selection ->
geometric signature -> template matching -> coarse rank -> fine re-rank).

## Key Findings

1. **Position coverage + coarse grade** is the most robust baseline; fine-grade
   and descriptor matching introduce cross-domain mismatch risk
2. **Single-sample concentration** is unreliable; need >=5 samples per class
3. **Self-correction** improves verification confidence but costs 3x runtime
4. **Multi-template clustering** at different abstraction levels improves
   discriminability without residual connections
5. **Passive + active joint extraction** is the key success factor — passive
   captures boundaries, active captures structure, their intersection is stable
