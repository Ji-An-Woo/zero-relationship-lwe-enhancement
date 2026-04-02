# At-a-Glance Comparison (default, 16 seeds)

| metric | baseline | zero_relationship | delta | relative_delta | verdict |
|---|---:|---:|---:|---:|---|
| zero_nonzero_auc | 0.791785 | 0.749280 | -0.042506 | -5.37% | improved |
| zero_nonzero_score_gap | 1.245884 | 0.915329 | -0.330556 | -26.53% | improved |
| avg_abs_dim_corr | 0.075622 | 0.073385 | -0.002237 | -2.96% | improved |
| lag1_corr | 0.000973 | -0.000058 | -0.000915 | -94.08% | improved |
| entropy | 3.061518 | 2.695204 | -0.366314 | -11.97% | worse |
| zero_ratio | 0.197262 | 0.240074 | 0.042812 | 21.70% | neutral |
| std | 2.020559 | 1.583325 | -0.437234 | -21.64% | neutral |
| mean | -0.003280 | 0.000527 | 0.003807 | 116.07% | neutral |

## Quick Read
- AUC (lower is better for zero/non-zero confusion): 0.791785 -> 0.749280 (-0.042506)
- Score gap (lower is better): 1.245884 -> 0.915329 (-0.330556)
- Avg abs dim corr (lower is better): 0.075622 -> 0.073385 (-0.002237)
- Entropy (higher is better): 3.061518 -> 2.695204 (-0.366314)
