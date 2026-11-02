# Corrected implementation and supported scope

Native LLaVA-1.5 and Qwen3-VL Instruct/Thinking are supported. Qwen3 uses
its own model class, chat template, multimodal rotary positions, query/key
normalization and DeepStack features. SmolVLM is not implemented.

The learned visual-latent generator has a training/checkpoint/sampling path.
See [native models and latent counterfactuals](latent_counterfactuals.md) for
commands, the explicit training objective, and validation scope.

## Restoration effects

Run commands from the repository root. Discover a circuit using the commands in
[pipeline_commands.md](pipeline_commands.md), then extract node-wise effects:

```bash
python vlm_experiments/LLaVA3.5-7B/run_circuit_extraction.py \
  --circuit_path assets/circuit_iterative_decay_k50.json \
  --image_dir images \
  --output output/circuit_behavior_metrics.jsonl
```

By default this command consumes **supplied counterfactual images**. Pass
`--counterfactual_mode latent --generator_checkpoint PATH` to generate visual
latent alternatives with a trained generator. Neither mode certifies visual
plausibility. Keep the factual question
and token layout fixed. Each example directory contains one `original*` image
and one or more `counterfactual*` images. The questions JSON is keyed by directory:

```json
{
  "example_1_skiing": {
    "question": "Is there a snowboard present in the image?",
    "prompt": "Please answer strictly with 'yes' or 'no'.",
    "target_answer": "yes"
  }
}
```

Use `--questions_path` for a different JSON file. `target_token_ids` overrides
`target_answer`; specify IDs when the exact target tokenization matters. Without
either field, the target is the factual model's first greedy token, recorded as
`target_scope: first_generated_token`. An explicit answer can contain multiple
tokens: the extractor sums their teacher-forced log probabilities. This is not
free-running generation of a new answer for each counterfactual.

For each circuit node and counterfactual, the extractor replaces the input slice
to that head's output projection with its factual activation and reruns the
native model. A whole-head node (`seq_pos=-1`, the default) restores all sequence
positions; a positional node restores only its specified position. The saved
`node_patch_deltas` are `log p(target | patched CF) - log p(target | CF)`.
Hooks are removed even when a forward pass fails.

## Empirical Bernstein estimation

Supply a **justified fixed support interval** chosen before examining the effects.
Do not use the observed minimum or maximum as a bound on the distribution. For
example, if an independently established effect interval is [-1, 1]:

```bash
python vlm_experiments/LLaVA3.5-7B/eval_hallucination_bernstein.py \
  --circuit_path assets/circuit_iterative_decay_k50.json \
  --behavior_jsonl output/circuit_behavior_metrics.jsonl \
  --effect_min -1 --effect_max 1 --epsilon 0.1 --delta 0.05
```

The interval above is illustrative, not a claimed bound for LLaVA. Raw
log-probability differences have no universal [-1, 1] support guarantee. The
estimator rejects observations outside the supplied interval and never clips or
invents observations. Its confidence interpretation assumes iid interventions
and a maximum sample budget fixed before sampling. It allocates the total failure
probability across supplied examples, nodes, and all tested sample-size prefixes.
This is more conservative than the paper's unadjusted per-node, fixed-size formula.
The radius uses `log(4/delta_per_check)` to cover both tails of the
[one-sided Maurer-Pontil bound (Theorem 4)](https://arxiv.org/pdf/0907.3740).

Results stay separate by example and target. Each node reports the first
**observed** prefix meeting the empirical Bernstein radius criterion. If any
node remains unresolved, the example's `m_bar` is null and its status is
`insufficient_counterfactuals`. This is a stability estimate, not a calibrated
hallucination classifier. The former `--tau`, `--model_id`, and `--device` flags
are unnecessary and removed from this offline estimator.

Old behavior logs contain activation differences and margins, not restoration
effects. They must be regenerated; there is no random-data fallback. Historical
neuron-coordinate analyses and saved plots do not establish counterfactual sample
complexity: feature coordinates are not iid counterfactual observations.

## Circuit faithfulness

Faithfulness now uses native model inference with all non-circuit head positions
replaced by background means. It reports baseline-corrected logit-margin recovery:

`(circuit margin - all-mean margin) / (full margin - all-mean margin)`.

Undefined denominators produce null values. This definition differs from the old
CD-contribution ratios, so old saved results are not directly comparable. Native
ablation and CD attribution are separate operations. Circuit discovery still uses
the repository's CD relevance/sign-ranking heuristic; this correction does not
add the paper's unspecified greedy-pruning protocol.

## CVL training

The training path now samples the Beta posterior, optimizes its interpolation
variables using the existing CVL objective, and applies interpolation at the
projected visual/token embeddings. A donor input supplies embedding references;
the CF forward keeps the factual input IDs and image layout. Language intervention
is optional and defaults to false to preserve the factual question. Inner
optimization computes gradients only for intervention variables and restores
checkpointing flags without leaving replacement hooks behind.

This repairs the existing CVL utility; it does **not** establish the paper's
unspecified realism-preserving generator training objective or reproduce the
reported benchmark tables. A separate trained visual-latent generator now connects directly to the
restoration experiment via `training/train_generator.py`; see
[its objective and commands](latent_counterfactuals.md). The CVL interpolation
utility remains a separate training approach.

## Verification

```bash
python -m pip install torch transformers peft pillow numpy tqdm pytest
python -m pytest -q tests
```

Tests instantiate tiny random Llama and LLaVA models locally. They require no
model weights or network downloads. They check native-forward equivalence,
causality, exact restoration and hook cleanup, multi-token scoring, mean
ablation, intervention gradients, posterior sampling, confidence-budget
allocation, missing/invalid observations, and benchmark parsing/alignment.
Full-size GPU inference and benchmark reproduction must be run separately.
