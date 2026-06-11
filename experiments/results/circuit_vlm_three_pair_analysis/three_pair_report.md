# Three-Pair VLM Delta Deep Analysis

- Generated: 2026-04-15T07:44:47.883115Z
- Input delta dir: /teamspace/studios/this_studio/mcg/experiments/results/circuit_vlm_deltas
- Files analyzed (3): example_1_skiing_cf_0_deltas.pt, example_1_skiing_cf_1_deltas.pt, example_2_cat_cf_0_deltas.pt
- Common layers: 51
- Confidence delta: 0.05

## Top Sensitive Layers (Avg Bernstein)

- layer_13_head_16: avg_bernstein=0.033036, avg_variance=0.013073, avg_hoeffding=0.077380, avg_mean=-0.005044
- layer_12_head_1: avg_bernstein=0.028501, avg_variance=0.009519, avg_hoeffding=0.067493, avg_mean=-0.001685
- layer_13_head_8: avg_bernstein=0.020139, avg_variance=0.004879, avg_hoeffding=0.044185, avg_mean=0.004375

## Top Stable Layers (Avg Bernstein)

- layer_28_head_26: avg_bernstein=0.000293, avg_variance=0.000001, avg_hoeffding=0.000713, avg_mean=-0.000052
- layer_2_head_11: avg_bernstein=0.000424, avg_variance=0.000002, avg_hoeffding=0.001096, avg_mean=-0.000048
- layer_27_head_0: avg_bernstein=0.000451, avg_variance=0.000003, avg_hoeffding=0.001086, avg_mean=-0.000115

## Pairwise Rank Consistency (Spearman)

- example_1_skiing_cf_0_deltas vs example_1_skiing_cf_1_deltas: rho=0.9676
- example_1_skiing_cf_0_deltas vs example_2_cat_cf_0_deltas: rho=0.8455
- example_1_skiing_cf_1_deltas vs example_2_cat_cf_0_deltas: rho=0.8644

## Leave-One-Out Robustness (Top-3)

- left_out=example_1_skiing_cf_0_deltas: overlap=1.000, spearman_full_vs_loo=0.9950, loo_top=layer_13_head_16, layer_12_head_1, layer_13_head_8
- left_out=example_1_skiing_cf_1_deltas: overlap=1.000, spearman_full_vs_loo=0.9900, loo_top=layer_13_head_16, layer_12_head_1, layer_13_head_8
- left_out=example_2_cat_cf_0_deltas: overlap=1.000, spearman_full_vs_loo=0.9654, loo_top=layer_12_head_1, layer_13_head_16, layer_13_head_8

## Family Aggregates

- circuit_head: avg_bernstein=0.004963, avg_variance=0.000821, layer_count=50
- final: avg_bernstein=0.002062, avg_variance=0.002166, layer_count=1

## Sign Consistency Across Pairs

- layer_18_head_21: soft=0.7214, strict=0.5781
- layer_24_head_7: soft=0.7005, strict=0.5469
- layer_12_head_3: soft=0.6406, strict=0.4531
- layer_18_head_29: soft=0.6380, strict=0.4453
- layer_11_head_0: soft=0.6302, strict=0.4453
- layer_13_head_17: soft=0.6276, strict=0.4375
- layer_17_head_20: soft=0.6276, strict=0.4375
- layer_17_head_13: soft=0.6250, strict=0.4375
- layer_15_head_31: soft=0.6198, strict=0.4297
- layer_23_head_20: soft=0.6198, strict=0.4219
- layer_27_head_0: soft=0.6172, strict=0.4297
- layer_18_head_5: soft=0.5990, strict=0.3984
- layer_28_head_26: soft=0.5885, strict=0.3828
- layer_16_head_7: soft=0.5833, strict=0.3750
- layer_12_head_19: soft=0.5833, strict=0.3750
- layer_13_head_8: soft=0.5781, strict=0.3672
- layer_9_head_18: soft=0.5781, strict=0.3672
- layer_12_head_0: soft=0.5677, strict=0.3516
- layer_14_head_6: soft=0.5677, strict=0.3516
- layer_14_head_30: soft=0.5677, strict=0.3516
- layer_11_head_26: soft=0.5625, strict=0.3438
- layer_31_head_10: soft=0.5573, strict=0.3359
- layer_30_head_25: soft=0.5521, strict=0.3281
- layer_12_head_1: soft=0.5469, strict=0.3203
- layer_15_head_30: soft=0.5469, strict=0.3203
- layer_15_head_7: soft=0.5417, strict=0.3125
- layer_6_head_6: soft=0.5365, strict=0.2969
- layer_9_head_6: soft=0.5365, strict=0.3047
- layer_11_head_25: soft=0.5286, strict=0.2969
- layer_12_head_9: soft=0.5260, strict=0.2891
- layer_8_head_10: soft=0.5234, strict=0.2812
- layer_8_head_11: soft=0.5156, strict=0.2656
- layer_4_head_17: soft=0.5104, strict=0.2656
- layer_6_head_21: soft=0.5104, strict=0.2656
- layer_9_head_10: soft=0.5052, strict=0.2578
- layer_10_head_1: soft=0.5026, strict=0.2578
- layer_11_head_27: soft=0.5000, strict=0.2422
- layer_19_head_18: soft=0.5000, strict=0.2500
- layer_2_head_11: soft=0.4948, strict=0.2422
- layer_8_head_25: soft=0.4948, strict=0.2422
- layer_15_head_3: soft=0.4922, strict=0.2344
- layer_9_head_30: soft=0.4896, strict=0.2344
- layer_13_head_21: soft=0.4870, strict=0.2266
- final_layer_norm: soft=0.4850, strict=0.2266
- layer_13_head_16: soft=0.4844, strict=0.2266
- layer_16_head_14: soft=0.4792, strict=0.2188
- layer_21_head_2: soft=0.4792, strict=0.2188
- layer_11_head_31: soft=0.4635, strict=0.1953
- layer_18_head_28: soft=0.4557, strict=0.1797
- layer_22_head_6: soft=0.4479, strict=0.1719
- layer_17_head_29: soft=0.4141, strict=0.1172

## Generated Plots

- plots/pair_layer_bernstein_heatmap.png
- plots/avg_bernstein_by_layer.png
- plots/pairwise_spearman_heatmap.png
- plots/sign_consistency_by_layer.png
- plots/family_avg_bernstein.png
