# Behavior-Layer Correlation Report

- Generated: 2026-04-15T08:12:08.189748Z
- Matched pairs: 3
- Layers analyzed: 51
- Target behavior metric: |margin_delta_cf_minus_orig| for yes/no logits

## Pair-Level Behavior Deltas
- example_1_skiing_cf_0_deltas.pt: margin_delta_cf_minus_orig=0.140625, abs=0.140625
- example_1_skiing_cf_1_deltas.pt: margin_delta_cf_minus_orig=-0.140625, abs=0.140625
- example_2_cat_cf_0_deltas.pt: margin_delta_cf_minus_orig=-0.718750, abs=0.718750

## Top Layers by |Correlation|
- layer_8_head_25: corr=-0.9999
- layer_14_head_30: corr=0.9998
- layer_21_head_2: corr=0.9998
- layer_27_head_0: corr=0.9995
- layer_9_head_10: corr=0.9992

## Full Layer Correlations
- layer_8_head_25: corr=-0.9999
- layer_14_head_30: corr=0.9998
- layer_21_head_2: corr=0.9998
- layer_27_head_0: corr=0.9995
- layer_9_head_10: corr=0.9992
- layer_9_head_18: corr=0.9962
- layer_16_head_7: corr=0.9962
- layer_28_head_26: corr=0.9944
- layer_15_head_31: corr=0.9919
- layer_8_head_10: corr=-0.9904
- layer_10_head_1: corr=0.9891
- final_layer_norm: corr=0.9881
- layer_11_head_26: corr=0.9868
- layer_18_head_5: corr=0.9846
- layer_19_head_18: corr=0.9817
- layer_16_head_14: corr=0.9754
- layer_24_head_7: corr=0.9735
- layer_11_head_25: corr=0.9591
- layer_13_head_16: corr=0.9585
- layer_14_head_6: corr=-0.9384
- layer_13_head_8: corr=0.9302
- layer_17_head_29: corr=0.9157
- layer_18_head_28: corr=0.9084
- layer_8_head_11: corr=-0.9074
- layer_9_head_6: corr=0.9035
- layer_11_head_0: corr=0.8895
- layer_18_head_21: corr=0.8891
- layer_17_head_20: corr=0.8861
- layer_17_head_13: corr=0.8693
- layer_6_head_6: corr=-0.8387
- layer_22_head_6: corr=0.8177
- layer_15_head_30: corr=-0.7983
- layer_31_head_10: corr=0.7922
- layer_13_head_17: corr=0.7782
- layer_12_head_3: corr=-0.7317
- layer_12_head_19: corr=0.7100
- layer_13_head_21: corr=-0.7006
- layer_12_head_1: corr=-0.6945
- layer_23_head_20: corr=-0.6610
- layer_2_head_11: corr=-0.6471
- layer_9_head_30: corr=0.6012
- layer_12_head_9: corr=-0.5402
- layer_11_head_31: corr=0.4350
- layer_18_head_29: corr=-0.4272
- layer_30_head_25: corr=0.4067
- layer_4_head_17: corr=-0.3261
- layer_6_head_21: corr=-0.1676
- layer_12_head_0: corr=-0.1317
- layer_15_head_3: corr=-0.1245
- layer_11_head_27: corr=-0.0397
- layer_15_head_7: corr=0.0294

## Caveat
- Correlations are exploratory because sample size is very small. Use larger pair counts before drawing strong causal conclusions.

## Generated Plots
- layer_behavior_correlation_bar.png
- pair_layer_effect_size_heatmap.png
