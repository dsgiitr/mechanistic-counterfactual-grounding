# Behavior-Layer Correlation Report

- Generated: 2026-03-29T07:57:06.789704Z
- Matched pairs: 3
- Layers analyzed: 11
- Target behavior metric: |margin_delta_cf_minus_orig| for yes/no logits

## Pair-Level Behavior Deltas
- example_1_skiing_cf_0_deltas.pt: margin_delta_cf_minus_orig=0.140625, abs=0.140625
- example_1_skiing_cf_1_deltas.pt: margin_delta_cf_minus_orig=-0.140625, abs=0.140625
- example_2_cat_cf_0_deltas.pt: margin_delta_cf_minus_orig=-0.718750, abs=0.718750

## Top Layers by |Correlation|
- mlp_layer_27: corr=0.9999
- mlp_layer_8: corr=0.9919
- final_layer_norm: corr=0.9881
- mlp_layer_5: corr=0.9862
- attn_layer_7: corr=0.9857

## Full Layer Correlations
- mlp_layer_27: corr=0.9999
- mlp_layer_8: corr=0.9919
- final_layer_norm: corr=0.9881
- mlp_layer_5: corr=0.9862
- attn_layer_7: corr=0.9857
- attn_layer_15: corr=0.9798
- mlp_layer_6: corr=0.9753
- attn_layer_1: corr=0.9739
- attn_layer_17: corr=0.9706
- mlp_layer_2: corr=0.8795
- attn_layer_14: corr=0.8636

## Caveat
- Correlations are exploratory because sample size is very small. Use larger pair counts before drawing strong causal conclusions.

## Generated Plots
- layer_behavior_correlation_bar.png
- pair_layer_effect_size_heatmap.png
