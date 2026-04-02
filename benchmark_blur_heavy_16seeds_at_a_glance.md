# At-a-Glance Comparison (blur_heavy, 16 seeds)

| metric | baseline | zero_relationship | delta | relative_delta | verdict |
|---|---:|---:|---:|---:|---|
| zero_nonzero_auc | 0.791785 | 0.723908 | -0.067877 | -8.57% | improved |
| zero_nonzero_score_gap | 1.245884 | 0.734360 | -0.511524 | -41.06% | improved |
| avg_abs_dim_corr | 0.075622 | 0.074575 | -0.001047 | -1.38% | improved |
| lag1_corr | 0.000973 | 0.000148 | -0.000825 | -84.79% | improved |
| entropy | 3.061518 | 2.414487 | -0.647031 | -21.13% | worse |
| zero_ratio | 0.197262 | 0.258807 | 0.061546 | 31.20% | neutral |
| std | 2.020559 | 1.319911 | -0.700648 | -34.68% | neutral |
| mean | -0.003280 | 0.000385 | 0.003664 | 111.72% | neutral |

## Quick Read
- AUC (lower is better for zero/non-zero confusion): 0.791785 -> 0.723908 (-0.067877)
- Score gap (lower is better): 1.245884 -> 0.734360 (-0.511524)
- Avg abs dim corr (lower is better): 0.075622 -> 0.074575 (-0.001047)
- Entropy (higher is better): 3.061518 -> 2.414487 (-0.647031)
