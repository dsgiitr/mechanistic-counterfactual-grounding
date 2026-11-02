# How Many Counterfactuals Does It Take? Probing VLM Hallucinations Through Circuits and Causal Effects

[![arXiv](https://img.shields.io/badge/arXiv-2606.08777-b31b1b.svg)](https://arxiv.org/abs/2606.08777)

This repository contains the official implementation of the paper **"How Many Counterfactuals Does It Take? Probing VLM Hallucinations Through Circuits and Causal Effects"**. We study the sample complexity of counterfactual robustness for hallucinated outputs in Vision-Language Models (VLMs) by linking mechanistic interpretability (circuit discovery) to causal effect statistics.

## Implementation status

Native **LLaVA-1.5 and Qwen3-VL Instruct/Thinking** are supported. A trainable,
bounded visual-latent generator supplies repeated counterfactuals directly to
exact restoration-effect extraction and empirical Bernstein estimation.

See [native models and latent generator commands](docs/latent_counterfactuals.md)
and [implementation scope](docs/implementation_status.md). The generator's
training objective is explicit; results come from actual model runs. Historical
assets have not been recomputed as part of these implementation changes.

## Abstract

Visual Language Models (VLMs) are known to produce hallucinated predictions that are not grounded in visual evidence, yet existing approaches lack a principled understanding of how robust such predictions are under counterfactual perturbations. In this work, we study the sample complexity of counterfactual robustness for hallucinated outputs in VLMs. We define a causal influence metric based on log-probability differences between factual, counterfactual, and activation-patched runs, and use it to characterize the stability of hallucinated predictions. By leveraging circuit discovery techniques (CD-T), we identify model components responsible for these predictions and track their activation differences across counterfactual samples. We then derive empirical bounds on the minimum number of counterfactual samples $m$ required to reliably detect instability in hallucinated outputs, using concentration inequalities and variance estimates of the causal influence distribution.

## Repository Structure

The repository is modularized to cleanly separate the VLM mechanistic experiments, evaluation metrics, statistical baselines, and CVL training utilities.

```text
mechanistic-counterfactual-grounding/
├── assets/                       # Model data, learned circuits, and notebook results
├── docs/                         # Additional documentation and pipeline commands
├── eval_metrics/                 # Scripts for computing evaluation metrics
│   ├── eval_coco_recall.py       # COCO Context Recall evaluation
│   ├── eval_hallusionbench.py    # HallusionBench evaluation
│   ├── eval_llm_judge.py         # LLM-as-a-judge scoring
│   └── eval_pope.py              # POPE Accuracy evaluation
├── methods/                      # Core mechanistic interpretability algorithms
│   ├── counterfactual_gen.py     # Counterfactual intervention logic
│   └── hallucination_circuit.py  # CD-T based circuit discovery logic
├── models/                       # Model-specific wrappers for consistent API
│   ├── llava_wrapper.py          # LLaVA-1.5-7B Wrapper
│   ├── qwen3_instruct_wrapper.py # Qwen3-VL-8B-Instruct Wrapper
│   └── qwen3_thinking_wrapper.py # Qwen3-VL-8B-Thinking Wrapper
├── statistical_experiments/      # Theoretical bounding & baseline statistics simulations
├── training/                     # Counterfactual Vision & Language (CVL) framework
└── vlm_experiments/              # Core VLM experimental pipelines per model
    ├── LLaVA3.5-7B/
    ├── Qwen3-VL-8B-Instruct/
    └── Qwen3-VL-8B-Thinking/
        ├── eval_hallucination_bernstein.py  # Computes Bernstein sample complexity (Algorithm 1)
        ├── run_circuit_extraction.py        # Extracts sparse hallucination circuits
        └── run_vlm_extraction.py            # Generates latent counterfactuals and logs logits
```

## Getting Started

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/dsgiitr/mechanistic-counterfactual-grounding.git
cd mechanistic-counterfactual-grounding
pip install -r requirements.txt
```

### Running the Pipeline

Follow the [pipeline commands](docs/pipeline_commands.md) from the repository
root. After discovering a circuit, extract **actual activation-restoration
effects** with `run_circuit_extraction.py`, then evaluate its JSONL output with
`eval_hallucination_bernstein.py`. The latter requires `--effect_min` and
`--effect_max`: a justified support interval fixed before sampling. It uses no
model weights and never substitutes random effects for missing observations.

The extraction command accepts supplied images or latent counterfactuals from a
trained generator checkpoint. Train it with `training/train_generator.py`; its
supervised locality/diversity objective is documented explicitly. The older CVL
utility remains a separate training approach.
`run_vlm_extraction.py` remains an optional activation-difference diagnostic;
its output is not an input to the restoration-effect estimator.

### Evaluation

Answer-level scoring and concept-recall scripts are available in the `eval_metrics/` directory to reproduce performance benchmarks including POPE, COCO, and HallusionBench. See the scripts directly for input arguments.

## Citation

If you find this code or our paper useful in your research, please cite:

```bibtex
@misc{gupta2026counterfactualsdoestakeprobing,
      title={How Many Counterfactuals Does It Take? Probing VLM Hallucinations Through Circuits and Causal Effects}, 
      author={Abhivansh Gupta and Simardeep Singh and Advika Sinha and Shreyansh Modi and Akshat Tomar},
      year={2026},
      eprint={2606.08777},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.08777}, 
}
```
