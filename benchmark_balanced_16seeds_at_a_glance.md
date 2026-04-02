# At-a-Glance Comparison (balanced, 16 seeds)

| metric | baseline | zero_relationship | delta | relative_delta | verdict |
|---|---:|---:|---:|---:|---|
| zero_nonzero_auc | 0.791785 | 0.739567 | -0.052218 | -6.59% | improved |
| zero_nonzero_score_gap | 1.245884 | 0.841594 | -0.404290 | -32.45% | improved |
| avg_abs_dim_corr | 0.075622 | 0.074283 | -0.001340 | -1.77% | improved |
| lag1_corr | 0.000973 | -0.000812 | -0.000161 | -16.55% | improved |
| entropy | 3.061518 | 2.604004 | -0.457514 | -14.94% | worse |
| zero_ratio | 0.197262 | 0.236108 | 0.038847 | 19.69% | neutral |
| std | 2.020559 | 1.493161 | -0.527398 | -26.10% | neutral |
| mean | -0.003280 | -0.001316 | 0.001963 | 59.86% | neutral |

## Quick Read
- AUC (lower is better for zero/non-zero confusion): 0.791785 -> 0.739567 (-0.052218)
- Score gap (lower is better): 1.245884 -> 0.841594 (-0.404290)
- Avg abs dim corr (lower is better): 0.075622 -> 0.074283 (-0.001340)
- Entropy (higher is better): 3.061518 -> 2.604004 (-0.457514)
