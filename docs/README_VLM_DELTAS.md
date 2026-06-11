# VLM Counterfactual Delta Distribution Analysis

This module replaces the completely simulated distributions in earlier scripts with **real empirical distributions** of mechanistic differences ($\Delta$) extracted directly from the internal representations of a Vision-Language Model (`llava-hf/llava-1.5-7b-hf`).

## Plan: VLM Counterfactual Delta Distributions

**What, Why, and How**
We replace the previous simulated scripts with real empirical computations of differences extracted from vision-language representations on our dataset of original vs. counterfactual pairs. For each representation layer, we treat the neurons/feature dimensions as our sample size ($m$), and mathematically bound their deviations using Hoeffding/Empirical Bernstein inequalities, then finally average the measurements over the image pairs.

**Steps**

1. **Initialize LLaVA and Data Loader**: Create `experiments/scripts/run_vlm_extraction.py` to load `llava-hf/llava-1.5-7b-hf` via Hugging Face `transformers` and iterate through `images/counterfactual_questions.json` to load the paired images.
2. **Representation Hooks**: Add forward hooks dynamically to the LLaVA model to extract:
   - Final layer hidden states.
   - 5 randomly selected specific attention head outputs.
   - 5 randomly selected MLP layer outputs.
3. **Compute Real Deltas**: Run inference on the original image + prompt, then the counterfactual image + prompt. Compute the true delta matrix ($\Delta = R_{org} - R_{cf}$) for each hooked representation.
4. **Apply Statistical Bounds**: Create an analysis script `experiments/scripts/analyze_vlm_deltas.py` that computes the empirical mean ($\hat{\mu}$) and sample variance ($s^2$) of these real $\Delta$ tensors across the feature dimension $D$ serving as the sample size $m$. Calculate the Hoeffding and Empirical Bernstein CI bounds using logic from `calc_width_CI.py`.
5. **Averaging Across Images**: Aggregate the calculated bounding inequalities for the $\Delta$ distributions by averaging the results across the different image pairs for the same layer to summarize global limits.

**Verification**

1. Run `python experiments/scripts/run_vlm_extraction.py` and verify that $\Delta$ tensors are saved correctly for both `skiing` and `cat` image examples on all selected layers/circuits.
2. Run `python experiments/scripts/analyze_vlm_deltas.py` and verify that empirical confidence interval widths are logged for each layer and aggregated correctly.

**Decisions**

- **Model Framework**: LLaVA-1.5 7B (`llava-hf/llava-1.5-7b-hf`).
- **Sample Strategy**: The feature dimension ($D$) / neurons of a single layer serves as the sample size ($m$) for evaluating the statistical bounds on a _single_ input pair. The overall representation difference limits for that layer are then reported as an average across all image pairs.

## Concept and Reasoning

To mechanically ground counterfactual reasoning, we need to understand how the model's internal representations shift when a key contextual entity (e.g., a snowboard or a bed) is removed from the image, but the question remains the same.

Let the representation of the original image + prompt be $R_{original}$ and the counterfactual be $R_{counterfactual}$.
The difference is defined as:
$$ \Delta = R*{original} - R*{counterfactual} $$

Instead of randomly generating these differences using NumPy/PyTorch normal distributions, we run real forward passes on LLaVA to extract them.

### Why Treat the Feature Dimension as Sample Size ($m$)?

When evaluating a single image pair's shift at a specific layer, we treat the $D$ independent dimensions (or neurons) of that layer as our sample size ($m$).

- _Reasoning:_ We want to mathematically bound the maximum expected perturbation (shift) an image manipulation causes on the layer as a whole. A high sample variance across the $D$ dimensions means the counterfactual introduced unstable, widespread changes.
- By applying **Hoeffding** and **Empirical Bernstein** bounds to these $D$ dimensions, we can calculate a 95% Confidence Interval width ($\epsilon$) that guarantees the true mean behavioral shift of that circuit is mathematically bounded.
- **Empirical Bernstein** generally yields a tighter bound than Hoeffding because it factors in the layer's actual sample variance ($s^2$).

### Why these Specific Layers?

1. **Final Layer Norm (CLS equivalent proxy):** In causal LMs, the representation of the final token aggregates all previous context (image pixels + prompt instructions) right before generating the "Yes/No" logit. This represents the global contextual shift.
2. **5 Random Attention Heads & 5 Random MLPs:** To sample "random circuits", validating if the concentration inequalities behave consistently at both local and global scales within the VLM.

## Workflow

### 1. Extraction: `run_vlm_extraction.py`

This script uses PyTorch forward hooks to intercept the hidden states of LLaVA 1.5.

- It loads pairs of images (`original` and `counterfactual`) from `images/counterfactual_questions.json`.
- It prompts LLaVA specifying a strict Yes/No requirement.
- It subtracts the counterfactual tensors from the original tensors.
- It saves these true $\Delta$ tensors locally in `experiments/results/vlm_deltas/`.

_Note for Lightning Adaptation: The script currently handles a small batch serially. For lightning modules, `register_forward_hook` should be placed inside the `configure_model` or equivalent setup phase, and extraction can be done simultaneously across distributed batches._

### 2. Analysis: `analyze_vlm_deltas.py`

This script applies the concentration inequalities (from our earlier mathematical verification scripts) directly to the real VLM data.

- Loads the saved $\Delta$ tensors.
- Calculates empirical bounds (Hoeffding Width, Empirical Bernstein Width) for each layer mathematically.
- Computes this independently per image pair, then **averages** the confidence interval widths to provide a holistic view of how bounded the VLM's counterfactual representations are across the dataset.
