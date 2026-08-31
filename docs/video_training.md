# Video Hierarchical Abstraction Generation Training

Script: test/video_agent_gen_v3.py | Output: test/video_output_v3/

## Method

```
L0 frames -> L1 cos features (passive/active + downsampling) -> L2 scene segment prototypes (mean = common points)
     -> L3 video summary (segment sequence)
Prototypes + contrast points (segment differences) stored in SQLite;
generation = high-level concretization (contrast point weighted aggregation, learned weights)
Training: joint across 3 videos (36 frame pairs), coordinate descent + parallel evaluation,
free-threaded 18 workers
```

## Results (vs v2 baseline)

| Metric | v2 (fixed weight guidance) | v3 (learned weights) |
|--------|---------------------------|----------------------|
| Generation improvement | -1.6% | **+20.2%** (training MAE +47.1%) |
| Cross-video generalization | none | **+6.4%** (unseen video) |
| Data | 1 video | 10 videos / 22545 frames in DB |

Weights: struct=-0.600 edge=-0.500 motion=-0.125 (long-interval interpolation
subtraction of motion interference optimal)

## Conclusion

Hierarchical abstraction + contrast point aggregation generation task pattern effective;
more training data -> more accurate contrast point statistics (database thinking).
