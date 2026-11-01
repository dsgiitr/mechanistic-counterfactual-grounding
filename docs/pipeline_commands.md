# LLaVA circuit and restoration pipeline

Run from the repository root. See [implementation_status.md](implementation_status.md)
for supported models, statistical assumptions, and remaining method gaps.

## 1. Background mean activations

```bash
python vlm_experiments/LLaVA3.5-7B/store_mean_activations.py \
  --dataset_name lmms-lab/POPE --dataset_split test --max_samples 3000 \
  --output_path data/mean_activations.pt
```

## 2. Hallucinated examples

```bash
python vlm_experiments/LLaVA3.5-7B/extract_hallucinations.py \
  --max_samples 50 --batch_size 16 --output_dir data/hallucinations \
  --dataset_name lmms-lab/POPE
```

## 3. Adjacency and circuit discovery

```bash
python vlm_experiments/LLaVA3.5-7B/run_adjacency_matrix.py \
  --dataset_path data/hallucinations/hallucinations.json \
  --mean_activations_path data/mean_activations.pt \
  --batch_size 32 --yes_no --output adjacency_matrix.pt

python vlm_experiments/LLaVA3.5-7B/find_circuit_from_matrix.py \
  --matrix_path adjacency_matrix.pt --num_outliers 2 --output circuit_result.json
```

## 4. Native mean-ablation faithfulness

```bash
python vlm_experiments/LLaVA3.5-7B/evaluate_circuit.py \
  --circuit_path circuit_result.json \
  --eval_dataset data/hallucinations/hallucinations.json \
  --mean_activations_path data/mean_activations.pt --yes_no
```

## 5. Exact restoration effects

Provide factual and counterfactual images plus questions as described in
[implementation_status.md](implementation_status.md). For per-example circuits,
run this step separately for each example/circuit rather than pooling circuits.

```bash
python vlm_experiments/LLaVA3.5-7B/run_circuit_extraction.py \
  --circuit_path circuit_result.json --image_dir images \
  --output output/circuit_behavior_metrics.jsonl
```

## 6. Empirical Bernstein estimation

A justified, pre-specified support interval is required. Replace LOWER and UPPER
with those numeric endpoints; they are not estimated from observed effects.

```bash
python vlm_experiments/LLaVA3.5-7B/eval_hallucination_bernstein.py \
  --circuit_path circuit_result.json \
  --behavior_jsonl output/circuit_behavior_metrics.jsonl \
  --effect_min LOWER --effect_max UPPER --epsilon 0.1 --delta 0.05
```

The result reports insufficient data if no observed prefix meets the precision
criterion. It does not classify an example as hallucinated.

## Native Qwen3 and learned latent generation

The Qwen3 Instruct and Thinking directories expose the same circuit/restoration
commands with their native checkpoint IDs. For actual generator training and
latent counterfactual sampling, follow [latent_counterfactuals.md](latent_counterfactuals.md).
`run_circuit_extraction.py` accepts `--counterfactual_mode latent`,
`--generator_checkpoint`, `--num_counterfactuals` and `--seed`.
