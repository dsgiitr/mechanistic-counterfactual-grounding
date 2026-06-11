<<<<<<< HEAD
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
---

# CD-T: Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition
This is a mechanistic interpretation tool to build circuits of internal components (e.g. attention heads) in transformers. Our code supports both BERT-like models and GPT-like models.

## Organization
- `greater_than_task`: utilities for the Greater-than task.
- `im_utils`: utilities for the docstring task.
- `methods`: utilities for the pathology report classification of the BERT model that we tested. (the pathology report eperiment on BERT is not included in the paper and the data/model is also not uploaded to this repo because of protected patient information. however, the provided CD-T implementation for BERT is general. readers should feel free to plug in their own BERTs and play with the code.)
- `notebooks/correctness_tests`: sanity checks of our contextual decomposition implementation.
- `notebooks`: major circuit experiments demonstrated on GPT2 a shown in the paper. we additionally include an `local_importance.ipynb` to compare and contrast CD-T with other modern local importance tools, such as LIME, SHAP, and layer intergrated gradients.
- `pyfunctions`: general utilities (users need to create their own config.py).

## Set up environment
---

# Counterfactual Vision and Language Learning (CVL) for LLaVA

Implementation of **"Counterfactual Vision and Language Learning"** (Abbasnejad et al., CVPR 2020) applied to **LLaVA** on Visual Question Answering (VQA).

## Overview

This codebase adapts the CVL framework to LLaVA by introducing:
- **Counterfactual generation** via exogenous variable interpolation in the embedding space
- **Beta-distributed exogenous variables** over vision and language embeddings
- **Counterfactual Risk Minimization (CRM)** as an alternative to standard ERM
- Full training loop alternating between observational and counterfactual updates

## Architecture

```
cvl_llava/
├── configs/
│   └── cvl_llava_vqa.yaml        # All hyperparameters
├── data/
│   ├── vqa_dataset.py            # VQA v2 / VQA-CP dataset loaders
│   └── collator.py               # Batch collation for LLaVA
├── models/
│   ├── llava_cvl.py              # LLaVA + CVL wrapper
│   ├── exogenous_module.py       # Exogenous variable module (Beta posterior)
│   └── counterfactual_generator.py  # Counterfactual sample generation
├── training/
│   ├── trainer.py                # CVL training loop
│   ├── crm_loss.py               # Counterfactual Risk Minimization loss
│   └── erm_loss.py               # Standard ERM loss
├── evaluation/
│   └── evaluator.py              # VQA evaluation metrics
├── utils/
│   ├── logging_utils.py
│   └── checkpoint_utils.py
└── scripts/
    ├── train.py                  # Main training script
    └── evaluate.py               # Evaluation script
```

## Setup

```bash
pip install -r requirements.txt
```

## Citation
If you use any of our code in your work, please cite:
```bash
@misc{hsu2024efficientautomatedcircuitdiscovery,
      title={Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition}, 
      author={Aliyah R. Hsu and Georgia Zhou and Yeshwanth Cherapanamjeri and Yaxuan Huang and Anobel Y. Odisho and Peter R. Carroll and Bin Yu},
      year={2024},
      eprint={2407.00886},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.00886}, 
}
```
---

## Training on VQA-CP v2 (recommended for bias evaluation)

```bash
python scripts/train.py --config configs/cvl_llava_vqa.yaml \
    --dataset vqa_cp \
    --data_dir /path/to/vqa_cp \
    --output_dir ./outputs/cvl_llava_vqacp
```

## Training on VQA v2

```bash
python scripts/train.py --config configs/cvl_llava_vqa.yaml \
    --dataset vqa_v2 \
    --data_dir /path/to/vqa_v2 \
    --output_dir ./outputs/cvl_llava_vqa2
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/cvl_llava_vqacp/best_model \
    --dataset vqa_cp \
    --data_dir /path/to/vqa_cp
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_cf` | 0.5 | Counterfactual loss weight λ |
| `epsilon` | 0.1 | Interpolation threshold ε |
| `clip_M` | 10.0 | Importance weight clipping M |
| `beta_alpha0` | 0.1 | Beta prior α₀ |
| `beta_beta0` | 0.1 | Beta prior β₀ |
| `cf_steps` | 5 | Gradient ascent steps for u optimization |
| `cf_lr` | 0.01 | Learning rate for u optimization |
| `alternate_every` | 1 | Alternate ERM/CRM every N steps |

