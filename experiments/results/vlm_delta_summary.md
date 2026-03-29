# VLM Delta Summary Report

## Source
- Parsed from: `..\results\vlm_delta_analysis.txt`
- Counterfactual pairs analyzed: 3
- Total layers summarized: 11

## Ranked Layer Table
| Layer | Avg Var (s^2) | Avg Hoeffding eps | Avg Bernstein eps | H/B Ratio |
|---|---:|---:|---:|---:|
| final_layer_norm | 0.002166 | 0.015031 | 0.002062 | 7.29 |
| attn_layer_14 | 0.001880 | 0.009049 | 0.001901 | 4.76 |
| attn_layer_15 | 0.000626 | 0.003976 | 0.001084 | 3.67 |
| attn_layer_17 | 0.000388 | 0.003438 | 0.000862 | 3.99 |
| mlp_layer_27 | 0.000322 | 0.005692 | 0.000774 | 7.35 |
| mlp_layer_8 | 0.000023 | 0.001113 | 0.000217 | 5.13 |
| mlp_layer_6 | 0.000014 | 0.001174 | 0.000173 | 6.79 |
| attn_layer_7 | 0.000014 | 0.000940 | 0.000169 | 5.56 |
| mlp_layer_5 | 0.000010 | 0.001064 | 0.000149 | 7.14 |
| attn_layer_1 | 0.000004 | 0.001038 | 0.000099 | 10.48 |
| mlp_layer_2 | 0.000004 | 0.000490 | 0.000088 | 5.57 |

## Key Findings
### Most Sensitive Layers (largest Bernstein width)
- final_layer_norm: bernstein=0.002062, variance=0.002166, hoeffding=0.015031
- attn_layer_14: bernstein=0.001901, variance=0.001880, hoeffding=0.009049
- attn_layer_15: bernstein=0.001084, variance=0.000626, hoeffding=0.003976

### Most Stable Layers (smallest Bernstein width)
- mlp_layer_2: bernstein=0.000088, variance=0.000004, hoeffding=0.000490
- attn_layer_1: bernstein=0.000099, variance=0.000004, hoeffding=0.001038
- mlp_layer_5: bernstein=0.000149, variance=0.000010, hoeffding=0.001064

### Highest Variance Layers
- final_layer_norm: variance=0.002166, bernstein=0.002062, hoeffding=0.015031
- attn_layer_14: variance=0.001880, bernstein=0.001901, hoeffding=0.009049
- attn_layer_15: variance=0.000626, bernstein=0.001084, hoeffding=0.003976

## Interpretation
- Mean Hoeffding/Bernstein ratio across layers: 6.16x
- Bernstein is consistently tighter than Hoeffding, indicating the observed deltas are structured and variance-aware bounds are informative.
- Sensitivity is concentrated in a subset of layers rather than uniformly distributed across all monitored circuits.

## Suggested Next Steps
1. Increase image-pair count and regenerate this report for more stable rankings.
2. Track per-example distributions (not only averages) to identify outlier pairs.
3. Evaluate a fixed global bound per layer for stricter cross-run Hoeffding comparability.
