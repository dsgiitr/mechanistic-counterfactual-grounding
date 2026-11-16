# Mechanistic Signals of Unintended Misinformation: Counterfactual Circuit Stability in Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2606.08777-b31b1b.svg)](https://arxiv.org/abs/2606.08777)

Official implementation. For every prediction of a vision-language model (VLM) we

1. discover a sparse circuit of decoder attention heads with contextual decomposition for transformers (CD-T),
2. perturb the visual representation with repeated latent counterfactuals
   $x_{\mathrm{cf}}^{(k)} = G_\phi(E_v(I) + \alpha z_k)$, $z_k \sim \mathcal N(0, I)$,
3. measure, for every circuit node $v$, the restoration effect
   $\Delta_v^{(k)} = \log p_\theta(y \mid \mathrm{Restore}(v, x_{\mathrm{cf}}^{(k)})) - \log p_\theta(y \mid x_{\mathrm{cf}}^{(k)})$,
4. turn the spread of these effects into a projected counterfactual budget through Bernstein's inequality,
   $\hat K_v = \lceil (2\hat\sigma_v^2 + \tfrac23 B_v\epsilon)\,\epsilon^{-2}\log(2/\delta) \rceil$, $\bar K = |S^\star|^{-1}\sum_v \hat K_v$,
5. test causal involvement with an oracle-assisted patch of the circuit.

Hallucinated and grounded predictions are compared on POPE, COCO and HallusionBench for LLaVA-1.5-7B, Qwen3-VL-8B (Thinking and Instruct) and SmolVLM-500M/256M.

## Supported models

| Key (`--model`) | Checkpoint | Studies |
|---|---|---|
| `llava-1.5-7b` | `llava-hf/llava-1.5-7b-hf` | stability, patching, circuit-size sensitivity |
| `qwen3-vl-8b-thinking` | `Qwen/Qwen3-VL-8B-Thinking` | stability, patching |
| `qwen3-vl-8b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` | patching |
| `smolvlm-500m` | `HuggingFaceTB/SmolVLM-500M-Instruct` | stability, patching |
| `smolvlm-256m` | `HuggingFaceTB/SmolVLM-256M-Instruct` | stability, patching |

All models run natively through Hugging Face Transformers (LLaVA, Qwen3-VL with mRoPE and DeepStack,
Idefics3/SmolVLM). Another checkpoint of one of these families can be used by passing its Hub ID as `--model`.

## Installation

```bash
git clone https://github.com/dsgiitr/mechanistic-counterfactual-grounding.git
cd mechanistic-counterfactual-grounding
pip install -r requirements.txt        # transformers >= 4.57 is required for Qwen3-VL
bash scripts/download_data.sh          # COCO val2014, instance annotations, CHAIR synonym list
```

POPE (`lmms-lab/POPE`) and HallusionBench (`lmms-lab/HallusionBench`) are downloaded from the Hugging Face Hub on first use.

## Reproducing the paper

```bash
bash scripts/reproduce_paper.sh
```

runs every step for every model and writes all tables and figures to `results/paper/`.
Use `MODELS="llava-1.5-7b"` to restrict models, `STEPS="3 6"` to rerun steps, `DEVICE=cuda:1` to pick a GPU and
`SHARDS=4` to split the benchmark-patching step over four GPUs. Every step resumes where it stopped.

The steps can also be run individually:

```bash
# 1. base predictions on the benchmark; select 50 hallucinated + 50 grounded examples
python experiments/01_select_examples.py   --model llava-1.5-7b --dataset pope
# 2. background mean activation of every head (CD-T / ablation reference)
python experiments/02_background_means.py  --model llava-1.5-7b
# 3. circuits, K latent counterfactuals, retention, restoration effects, oracle patch
python experiments/03_stability.py         --model llava-1.5-7b --dataset pope --K 50
# 4. benchmark scores before / after patching every item
python experiments/04_benchmark_patching.py --model llava-1.5-7b --dataset pope
# 5. circuit-size sensitivity of node-selection strategies (LLaVA, POPE)
python experiments/05_budget_sensitivity.py
# 6. all tables, figures and statistics
python experiments/06_analyze.py
```

| Paper | Produced by | Output in `results/paper/` |
|---|---|---|
| Table 1 — generated / retained counterfactuals | steps 3, 6 | `table1_retention.md` |
| Figure 2 — $V_i$ and $\bar K$, grounded vs hallucinated | steps 3, 6 | `figure2_stability_<dataset>.pdf` |
| Section 4.2 — variance and budget ratios, $\bar K$ vs. $8 + 216\,V_i$ check | steps 3, 6 | `stability_summary.md` |
| Table 2 — corrections by oracle-assisted patching (Wilson 95% CI) | steps 3, 6 | `table2_correction.md` |
| Table 3 — POPE / COCO $R_{\mathrm{ctx}}$ / HallusionBench, base vs. patch | steps 1, 4, 6 | `table3_benchmark.md` |
| Figure 3 — circuit-size sensitivity | steps 5, 6 | `figure3_budget_sensitivity.pdf` |
| Section 3.4, Appendix B — bound constants (floor 7.98, slope 216, empirical-Bernstein $K\approx 29$) | step 6 | `bound_constants.md` |
| Section 6 — per-example distributions, bootstrap CIs, Mann–Whitney test, AUROC, confidence-matched comparison, label-free comparison, corrected vs. broken counts | steps 3, 4, 6 | `stability_summary.{md,json}`, `figure_stability_distributions_<dataset>.pdf`, `table3_benchmark.md` |

Per-example records (circuits, every counterfactual's retention votes and restoration effects, oracle-patch
outcomes) are kept in `results/<model>/<dataset>/`, so every statistic can be recomputed without rerunning models.
The label-free comparison requires `--set counterfactuals.effects_on=all` in step 3, which records restoration
effects for every counterfactual, not only the retained ones.

## Configuration

All settings are in [`configs/paper.yaml`](configs/paper.yaml); any entry can be overridden from the command line,
e.g. `--set counterfactuals.alpha=0.2 bounds.epsilon=0.1`.

| Setting | Value |
|---|---|
| Counterfactuals $K$ | 50 / 100 / 150 (LLaVA, Qwen3-VL-Thinking); 50 (others) |
| Intervention strength $\alpha$; generator $G_\phi$ | 0.1; the model's frozen multimodal projector |
| Sampled decodings per counterfactual; temperature | 10 ($K\le 100$), 15 ($K=150$); 0.7 |
| Retention | correct in $\ge 9/10$ (or $14/15$) decodings, same rule for both groups |
| Bounds $\epsilon$, $\delta$, $B_v$ | 0.185, 0.05, 0.6 (common to all nodes) |
| CD-T schedule | $K_0 = K_i = 4$ heads per iteration |
| Examples | 50 hallucinated (first in benchmark order) + 50 grounded (random), distinct images |

[docs/method.md](docs/method.md) specifies each computation: how $h(I)$ and $G_\phi$ are defined per architecture,
how the answer $y$ is chosen for yes/no questions and captions, how the reasoning trace of the Thinking model is
held fixed, the CD-T scoring and search rules, background means, patching and the circuit-size metrics.

## Compute

Per example, circuit discovery costs one CD-T propagation per decoder layer and batch of source heads; each of the
$K$ counterfactuals costs one sampled-decoding call (`decodings` sequences) and, if retained, $|S^\star| + 1$
teacher-forced forward passes (batched over nodes, `restoration.batch_size`). The benchmark-patching step repeats
circuit discovery and counterfactual sampling for every benchmark item and is the most expensive step; shard it
with `SHARDS`. The default batch sizes target an 80 GB GPU; reduce `cdt.batch_size`, `cdt.max_attention_elements`
and `restoration.batch_size` on smaller GPUs.

## Tests

```bash
python -m pytest -q tests
```

The tests use tiny randomly initialised LLaVA, Idefics3 (SmolVLM) and Qwen3-VL models and need no downloads. They
check that the CD-T decomposition reproduces the native forward pass of each architecture, that restoration effects
equal an explicit single-head patch and do not depend on batching, that counterfactual draws are deterministic and
change only the visual input, that patching affects only the prefill during generation, the bound constants used in
the paper, and the COCO object extraction.

## Repository structure

```text
mechanistic-counterfactual-grounding/
├── mcg/                          # method implementation
│   ├── models.py                 # native VLM adapter: LLaVA, Qwen3-VL, SmolVLM
│   ├── counterfactuals.py        # latent counterfactuals  G_phi(E_v(I) + alpha z)      (Eq. 1)
│   ├── tasks.py                  # POPE / HallusionBench / COCO, answers y, retention rule
│   ├── background.py             # background mean activations
│   ├── cdt.py                    # CD-T decomposition and per-head scores              (App. C)
│   ├── circuits.py               # iterative circuit search and selection strategies
│   ├── restoration.py            # restoration effects                                 (Eq. 2)
│   ├── patching.py               # oracle-assisted patching, mean ablation
│   ├── bounds.py                 # Bernstein, plug-in, empirical-Bernstein budgets      (Eqs. 3-5)
│   ├── stats.py                  # Wilson intervals, bootstrap, rank test, AUROC, matching
│   └── pipeline.py               # Algorithm 1 for one example
├── experiments/                  # steps 01-06 (see above)
├── configs/paper.yaml            # all experimental settings
├── scripts/                      # reproduce_paper.sh, download_data.sh
├── docs/method.md                # implementation details
├── tests/                        # tests on tiny models
├── training/, methods/           # counterfactual vision-and-language (CVL) training utility
├── statistical_experiments/      # simulations of concentration-bound behaviour
├── pyfunctions/, im_utils/       # reference CD-T implementation for non-VLM tasks
└── assets/                       # earlier exploratory circuit analyses
```

## Citation

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
