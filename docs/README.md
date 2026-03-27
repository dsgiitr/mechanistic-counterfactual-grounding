# LLaVA Hallucination Circuit Discovery

This directory outlines the setup for tracing hallucination circuits in the LLaVA-1.5 Vision-Language Model using Contextual Decomposition (CD-T). Specifically, the method is adapted to target the VLM's generated **residual stream** at the answer position.

## Overview

Unlike standard CD-T tracing, which typically uses pre-defined downstream attention heads as a target layer, this implementation traces the hallucination directly by computing its effect on the final target **scalar direction** responsible for hallucinating a specific incorrect answer token.

The sequence positions (`visual`, `question`, and `answer`) dynamically vary. We handle this dynamically by matching visual and question token positions offset bounds. We explicitly evaluate how these three regions drive the resulting error.

## Dataset Format

Your datasets must be presented as a JSON array of dicts. Example:

```json
[
  {
    "image": "images/test_img_1.jpg",
    "question": "What color is the stop sign?",
    "correct_answer": "red",
    "hallucinated_answer": "green",
    "correct_token_id": 3521,
    "hallucinated_token_id": 4832
  }
]
```

`correct_token_id` and `hallucinated_token_id` are optional; if excluded, the framework will infer them from the tokenizer.

## How to Run

Use the overarching wrapper `run_hallucination_circuit.py`:

```bash
python run_hallucination_circuit.py \
    --dataset_path data/hallucination_examples.json \
    --correct_path data/correct_examples.json \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --top_percentile 95.0 \
    --device "cuda"
```

## Output

The script generates `hallucination_circuit_results.json` containing:
1. **`ranked_heads_top_100`**: The highest contributing heads ranked by their combined CD-T metric.
2. **`circuit`**: Details on each circuit component.
    - `label`: `"hallucination promoter"` or `"grounding suppressor"`.
    - `r_avg_visual / r_avg_question`: Subdivided propagation metrics tracking relevance of the heads relative to token location.
3. **`faithfulness`**: Metric confirming if the traced computational circuit matches the full model's capability to hallucinate.
