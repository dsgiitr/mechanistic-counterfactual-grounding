# Native Qwen3-VL and learned latent counterfactual experiments

Run commands from the repository root. Install `requirements.txt` in your GPU
Python environment. Qwen3-VL requires Transformers 4.57 or newer; the local
mechanical tests use Transformers 5.22.1 and PyTorch 2.15.0. No full-size model
weights are downloaded by the tests.

## Native model support

Both `models/qwen3_instruct_wrapper.py` and `models/qwen3_thinking_wrapper.py`
load `Qwen3VLForConditionalGeneration` with their actual checkpoint IDs. Prompts
use the processor's chat template and preserve `image_grid_thw` and multimodal
token types, following the [native Transformers interface](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).

Circuit decomposition captures the factual decoder's actual rotary embeddings,
attention mask and visual injections. It includes Qwen3's per-head query/key
normalization, grouped-query attention and DeepStack features. DeepStack
injections are treated as fixed background inputs when tracing decoder heads.
The native forward remains the reference for restoration and ablation effects.

The Instruct and Thinking experiment directories support adjacency extraction,
circuit selection, background means, faithfulness, generation and restoration
extraction. `run_vlm_extraction.py` exports descriptive head-activation deltas
and requires `--output_dir`; it does not compute restoration effects.

## 1. Train the visual-latent generator

Create a JSON **training** manifest. Image paths are relative to its directory:

```json
[
  {
    "image": "images/training_example.png",
    "question": "Is there a cat in the image?",
    "correct_answer": "yes"
  }
]
```

Use `correct_token_ids` instead of `correct_answer` to control tokenization
exactly. Use held-out examples for the later stability and benchmark evaluation.

```bash
python training/train_generator.py \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --dataset_path data/generator_train.json \
  --output output/qwen3_generator.pt \
  --epochs 3 --learning_rate 0.001 --noise_dim 16 --rank 64 \
  --max_relative_shift 0.1 --seed 42 --device cuda
```

The same trainer supports `llava-hf/llava-1.5-7b-hf` and
`Qwen/Qwen3-VL-8B-Thinking`; train a separate checkpoint for each model.
For a custom local checkpoint directory, pass `--family llava` or
`--family qwen3_vl` explicitly.

The generator is an MLP conditioned on normalized visual features and an iid
Gaussian latent vector. It modifies only visual slots at the language-decoder
input and, for Qwen3, every DeepStack visual stream. Questions, image-grid
positions, and other text embeddings stay fixed. Each coordinate's change is
bounded by `max_relative_shift * RMS(original visual token)`.

The explicit training objective is:

`mean grounded-answer NLL + distance_weight * relative_squared_distance
 + diversity_weight * max(0, min_diversity - pairwise_relative_squared_distance)`.

Two independently sampled latent vectors supply each update. Only generator
parameters are optimized; VLM weights stay frozen. The checkpoint contains the
generator configuration, parameters, model ID, objective settings, seed and
training-manifest hash. Training logs contain actual losses. Sampling requires a
checkpoint; it never silently substitutes an untrained generator.

This is an implementable, supervised latent-intervention objective. It is a
specified experimental choice for the manuscript's otherwise underspecified
`G_phi`, and does not claim to recover its unknown training recipe. A bound on
latent change does not establish decoded-image realism or semantic validity.
Validate intervention quality separately for the intended dataset.

## 2. Discover a circuit for the target model

Prepare JSON examples with `image`, `question`, `hallucinated_answer` and
`correct_answer`, or explicit corresponding token IDs. The generation collector
can create such data from actual POPE runs:

```bash
python vlm_experiments/Qwen3-VL-8B-Instruct/extract_hallucinations.py \
  --dataset_name lmms-lab/POPE --dataset_split test --max_samples 50 \
  --output_dir data/qwen_hallucinations --correct_dir data/qwen_grounded
```

Use `--dataset_config` when needed. A local JSON list with `image`, `question`
and a yes/no `answer` can be supplied with `--dataset_path`. Collection uses
native autoregressive generation; no output or correction rate is prescribed.

```bash
python vlm_experiments/Qwen3-VL-8B-Instruct/run_adjacency_matrix.py \
  --dataset_path data/qwen_hallucinations/hallucinations.json \
  --correct_path data/qwen_grounded/correct_examples.json \
  --batch_size 1 --output output/qwen_adjacency.pt

python vlm_experiments/Qwen3-VL-8B-Instruct/find_circuit_from_matrix.py \
  --matrix_path output/qwen_adjacency.pt --num_outliers 2 \
  --output output/qwen_circuit.json
```

Adjacency files record the model's actual head/layer counts; offline selection
reads them instead of assuming a fixed architecture. Full adjacency extraction
is expensive: it propagates each candidate source through each example.
Choose the example and source batching budget according to available hardware.
The circuit-search ranking heuristic is inherited from this repository; this
addition does not introduce an unspecified greedy-pruning procedure.

For faithfulness, supply the generated circuit and a background mean cache via
`evaluate_circuit.py`. Its definition is documented in
[implementation_status.md](implementation_status.md).

## Thinking checkpoints and answer context

A Thinking model can emit `<think>` before its final answer. Its first generated
token is not necessarily the answer being studied. The collector saves the
**exact generated token prefix** before the final yes/no answer in
`answer_prefix_token_ids`; dataset preprocessing and circuit discovery replay
these IDs. Unfinished reasoning and unparseable final answers are skipped.

For hand-built manifests/questions, include `answer_prefix_token_ids` (preferred)
or `answer_prefix` when measuring a final answer after reasoning. An explicit
`target_token_ids` or `target_answer` fixes the answer to score. Without one, the
restoration extractor studies the first factual greedy token and labels its
scope accordingly. Prefix tokens can include factual reasoning; retaining this
context defines the effect being measured and should be reported in experiments.

## 3. Sample latent counterfactuals and measure restoration

Use the existing keyed questions format under `images/`. Each example directory
needs an `original*` image; latent mode needs no edited images. For example:

```json
{
  "example_1_skiing": {
    "question": "Is there a snowboard present in the image?",
    "prompt": "Please answer with yes or no.",
    "target_answer": "yes"
  }
}
```

```bash
python vlm_experiments/Qwen3-VL-8B-Instruct/run_circuit_extraction.py \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --circuit_path output/qwen_circuit.json --image_dir images \
  --counterfactual_mode latent --generator_checkpoint output/qwen3_generator.pt \
  --num_counterfactuals 50 --seed 123 \
  --output output/qwen_latent_effects.jsonl
```

Each draw uses independent Gaussian noise with the trained generator held fixed.
The same noise is replayed for that draw's CF baseline and every node-restoration
run. Records include the noise vector, target IDs, factual/CF log probabilities,
actual node-patch effects, model ID, circuit hash and generator-checkpoint hash.
A fixed seed permits replay; no sample is altered to obtain a desired result.

Use `--counterfactual_mode images` to consume supplied edited images instead.
Train on independent data and avoid selecting/retuning the generator or circuit
using the same effects on which confidence bounds will be computed.

## 4. Estimate stability

```bash
python vlm_experiments/Qwen3-VL-8B-Instruct/eval_hallucination_bernstein.py \
  --circuit_path output/qwen_circuit.json \
  --behavior_jsonl output/qwen_latent_effects.jsonl \
  --effect_min LOWER --effect_max UPPER --epsilon 0.1 --delta 0.05
```

Replace LOWER and UPPER with a justified support interval fixed before sampling.
They are not the observed extrema. The estimator reports insufficient data when
the available sample budget cannot meet the precision criterion. It neither
forces the paper's reported budgets nor classifies hallucinations using an
uncalibrated threshold. See [implementation_status.md](implementation_status.md)
for confidence allocation and statistical assumptions.

## Verification and compute scope

```bash
python -m pip install torch transformers peft pillow numpy tqdm pytest
python -m pytest -q tests
```

Native tiny Qwen3-VL tests include actual image patch processing, multimodal RoPE,
query/key normalization, grouped-query attention and DeepStack. Tests also verify
native-decoder reconstruction, actual generator parameter updates, frozen VLM
weights, bounded visual edits, unchanged text, checkpoint round trips, deterministic
latent replay, exact patch scoring, Thinking-prefix handling, and the full
training/checkpoint/extraction/statistics flow. Full 8B inference, data quality and
benchmark performance need to be evaluated on the intended datasets/hardware.
