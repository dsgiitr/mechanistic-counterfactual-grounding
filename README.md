# Mechanistic Counterfactual Grounding

This repository studies how internal Vision-Language Model (VLM) representations change under image counterfactuals, and whether those changes can be bounded using concentration inequalities.

## Project Status (Updated: 2026-03-29)

The initial simulation-only phase has been extended to a real-model phase using LLaVA-1.5 7B.

## Data and Prompt Assets

- Image pairs:
	- [images/example_1_skiing](images/example_1_skiing)
	- [images/example_2_cat](images/example_2_cat)
- Counterfactual question file:
	- [images/counterfactual_questions.json](images/counterfactual_questions.json)

## Main Scripts Added/Used

- VLM extraction and delta generation:
	- [experiments/scripts/run_vlm_extraction.py](experiments/scripts/run_vlm_extraction.py)
- Baseline delta-bound analysis:
	- [experiments/scripts/analyze_vlm_deltas.py](experiments/scripts/analyze_vlm_deltas.py)
- Markdown summary generation from analysis output:
	- [experiments/scripts/generate_vlm_report.py](experiments/scripts/generate_vlm_report.py)
- Three-pair deep analysis (robustness + plots):
	- [experiments/scripts/analyze_vlm_three_pairs.py](experiments/scripts/analyze_vlm_three_pairs.py)
- Behavior-link analysis (layer effect size vs yes/no logit-margin shift):
	- [experiments/scripts/analyze_layer_behaviour_correlation.py](experiments/scripts/analyze_layer_behaviour_correlation.py)

## Experiments Completed Since Last README

### 1) Real VLM Delta Extraction

Ran LLaVA on original and counterfactual image pairs and saved layer-wise deltas:

$$
\Delta = R_{original} - R_{counterfactual}
$$

Artifacts:

- Delta tensors:
	- [experiments/results/vlm_deltas](experiments/results/vlm_deltas)

### 2) Concentration-Inequality Summary on Real Deltas

Computed Hoeffding and Empirical Bernstein widths using layer dimensions as sample size $m$.

Artifacts:

- Raw text output:
	- [experiments/results/vlm_delta_analysis.txt](experiments/results/vlm_delta_analysis.txt)
- Ranked markdown summary:
	- [experiments/results/vlm_delta_summary.md](experiments/results/vlm_delta_summary.md)

Key findings from this phase:

- Most sensitive layers (largest average Bernstein width):
	- `final_layer_norm`
	- `attn_layer_14`
	- `attn_layer_15`
- Most stable layers (smallest average Bernstein width):
	- `mlp_layer_2`
	- `attn_layer_1`
	- `mlp_layer_5`
- Empirical Bernstein bounds were consistently tighter than Hoeffding.

### 3) Three-Pair Deep Analysis + Visualizations

Using the 3 available counterfactual pairs, added:

- Pairwise rank consistency (Spearman)
- Leave-one-out top-k robustness
- Family-level aggregation (`final`, `attn`, `mlp`)
- Sign-consistency metrics across dimensions

Artifacts:

- Plot folder:
	- [experiments/results/vlm_three_pair_analysis/plots](experiments/results/vlm_three_pair_analysis/plots)
- Generated plots:
	- [experiments/results/vlm_three_pair_analysis/plots/pair_layer_bernstein_heatmap.png](experiments/results/vlm_three_pair_analysis/plots/pair_layer_bernstein_heatmap.png)
	- [experiments/results/vlm_three_pair_analysis/plots/avg_bernstein_by_layer.png](experiments/results/vlm_three_pair_analysis/plots/avg_bernstein_by_layer.png)
	- [experiments/results/vlm_three_pair_analysis/plots/pairwise_spearman_heatmap.png](experiments/results/vlm_three_pair_analysis/plots/pairwise_spearman_heatmap.png)
	- [experiments/results/vlm_three_pair_analysis/plots/sign_consistency_by_layer.png](experiments/results/vlm_three_pair_analysis/plots/sign_consistency_by_layer.png)
	- [experiments/results/vlm_three_pair_analysis/plots/family_avg_bernstein.png](experiments/results/vlm_three_pair_analysis/plots/family_avg_bernstein.png)

Reported findings from this run:

- Pairwise rank consistency was very high (roughly 0.96 to 0.98).
- Leave-one-out top-3 overlap was 1.0.
- Family aggregate sensitivity ordering: `final > attn > mlp`.

### 4) Behavior-Link Analysis (Internal Shift vs Output Margin Shift)

Linked layer delta effect sizes to answer-margin changes (yes/no logits).

Artifacts:

- Behavior-link report:
	- [experiments/results/vlm_behavior/vlm_behavior_link/behavior_layer_correlation_report.md](experiments/results/vlm_behavior/vlm_behavior_link/behavior_layer_correlation_report.md)
- Behavior-link plots:
	- [experiments/results/vlm_behavior/vlm_behavior_link/layer_behavior_correlation_bar.png](experiments/results/vlm_behavior/vlm_behavior_link/layer_behavior_correlation_bar.png)
	- [experiments/results/vlm_behavior/vlm_behavior_link/pair_layer_effect_size_heatmap.png](experiments/results/vlm_behavior/vlm_behavior_link/pair_layer_effect_size_heatmap.png)

Current behavior-link note:

- Correlations are exploratory because only 3 matched pairs are available.
- Treat rankings as hypothesis-generating, not final causal claims.

## Repro Commands

From repository root:

```bash
cd experiments/scripts
python run_vlm_extraction.py
python analyze_vlm_deltas.py
python generate_vlm_report.py
python analyze_vlm_three_pairs.py
python analyze_layer_behaviour_correlation.py
```

## Next Steps

1. Add more counterfactual pairs (target: 30+).
2. Re-run the same pipeline to test ranking stability at larger sample sizes.
3. Compare signed and absolute margin-delta correlations in behavior-link analysis.
4. Add per-example dispersion stats (median/IQR/outlier flags) to strengthen reporting.
