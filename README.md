# What Does It Cost to Audit a Hallucination? Counterfactual Circuit Stability as an Offline Signal for VLM Review

[![arXiv](https://img.shields.io/badge/arXiv-2606.08777-b31b1b.svg)](https://arxiv.org/abs/2606.08777)

Official implementation. For an answer $y$ of a vision-language model (VLM) to a question about an image $I$, the audit

1. discovers a sparse decoder circuit $S^\star$ of attention heads with contextual decomposition for transformers (CD-T);
2. generates counterfactuals by counterfactual vision-and-language (CVL) donor mixing of the projected visual tokens,
   $\tilde h = u\,h(I) + (1-u)\,h(I')$ if $u < 1-\epsilon_g$ (else $h(I)$), with $u \sim \mathrm{Beta}$ and a donor image $I'$;
3. keeps a counterfactual if at least 9 of 10 sampled decodings are correct (14 of 15 for $K=150$);
4. measures the restoration effect of each node $v \in S^\star$,
   $\Delta_v^{(k)} = \log p_\theta(y \mid \mathrm{Patch}(v, x_{\mathrm{cf}}^{(k)}; A_v^f)) - \log p_\theta(y \mid x_{\mathrm{cf}}^{(k)})$;
5. summarises an example by its mean node variance $V$ and converts it into a plug-in Bernstein budget
   $\hat K_v = \lceil (2\hat\sigma_v^2 + \tfrac23 B\epsilon)\,\epsilon^{-2}\log(2/\delta) \rceil$, $\bar K = |S^\star|^{-1}\sum_v \hat K_v$;
6. reports what the audit costs (model calls, GPU-seconds), where ground-truth labels enter, and an oracle upper bound
   on what restoring the circuit recovers.

Steps 3 and 6 use ground-truth labels (retention and the oracle patch); the label-free variants of Section 6 are
available as options.

## Supported models

| Key (`--model`) | Checkpoint |
|---|---|
| `llava-1.5-7b` | `llava-hf/llava-1.5-7b-hf` |
| `qwen3-vl-8b-thinking` | `Qwen/Qwen3-VL-8B-Thinking` |
| `smolvlm-500m` | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| `smolvlm-256m` | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| `qwen3-vl-8b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` |

All models run natively through Hugging Face Transformers. Donor mixing is applied to each model's projector output:
LLaVA's multimodal projector, SmolVLM's connector, and Qwen3-VL's patch merger together with its DeepStack mergers.

## Installation

```bash
git clone https://github.com/dsgiitr/mechanistic-counterfactual-grounding.git
cd mechanistic-counterfactual-grounding
pip install -r requirements.txt        # transformers >= 4.57 is required for Qwen3-VL
bash scripts/download_data.sh          # COCO val2014, instance annotations, CHAIR synonym list
```

POPE (`lmms-lab/POPE`) and HallusionBench (`lmms-lab/HallusionBench`) are downloaded from the Hugging Face Hub on first use.

## Counterfactual generator

$u$ is drawn i.i.d. from a Beta distribution. Pass a trained CVL checkpoint (`training/`, LLaVA-1.5 configuration:
LoRA rank 16, VQA-CP, prior Beta(0.1, 0.1), five inner Adam steps, weight 0.5) to use its learned posterior:

```bash
python training/train.py --config training/configs/cvl_llava_vqa.yaml        # trains CVL for LLaVA-1.5
BETA_CHECKPOINT=path/to/checkpoint.pt bash scripts/reproduce_paper.sh
```

Without a checkpoint, $u$ follows `counterfactuals.beta_alpha/beta_beta` in [`configs/paper.yaml`](configs/paper.yaml)
(default: the prior Beta(0.1, 0.1)). Donors are POPE images whose COCO annotations contain neither the queried object
nor its five most frequently co-occurring categories; each donor is resized to the factual image so both yield the
same visual tokens.

## Reproducing the paper

```bash
bash scripts/reproduce_paper.sh
```

runs every step for the four studied models and writes all tables and figures to `results/paper/`. `MODELS=...`
restricts models, `STEPS="3 8"` reruns steps, `DEVICE=cuda:1` selects a GPU; every step resumes where it stopped.

| Step | Script | Produces |
|---|---|---|
| 1 | `experiments/01_select_examples.py --model M --dataset D` | base predictions; 50 hallucinated + 50 grounded examples |
| 2 | `experiments/02_background_means.py --model M` | head background means (CD-T reference) |
| 3 | `experiments/03_stability.py --model M --dataset D --K K` | circuits, counterfactuals, retention, $\Delta_v^{(k)}$, oracle patch |
| 4 | `experiments/04_confidence_control.py` | confidence-matched output-level control on LLaVA / POPE |
| 5 | `experiments/05_budget_sensitivity.py` | circuit-construction strategies |
| 6 | `experiments/06_timing.py` | timing of the audit primitives |
| 7 | `experiments/07_coverage_simulation.py` | coverage simulations of the budgets (no model) |
| 8 | `experiments/08_analyze.py` | all tables and figures |

| Paper | Output in `results/` |
|---|---|
| Figure 1 — donor mixing vs. confidence-matched controls | `llava-1.5-7b/pope/figure1_confidence_control.pdf`, `confidence_control.json` |
| Table 1 — retained counterfactuals | `paper/table1_retention.md` |
| Table 2 — group-mean $V$ and plug-in $\bar K$ | `paper/table2_variance.md` |
| Table 3 — budget ratio over $(\epsilon, B)$ | `paper/table3_budget_sensitivity.md` |
| Table 4 — timings | `timing.json` |
| Table 5 — budgets under plug-in, empirical-Bernstein and Hoeffding rules | `paper/table5_rules.md` |
| Table 6 — cost of auditing one answer | `paper/table6_audit_cost.md` |
| Table 7 — oracle restoration (Wilson 95% CI) | `paper/table7_oracle_restoration.md` |
| Figures 4–6 — Hoeffding vs. Bernstein, budget rules, ratios | `paper/figure4_*`, `figure5_*`, `figure6_*` |
| Figure 7 — circuit-construction budget | `paper/figure7_circuit_construction.pdf` |
| Appendix B — coverage simulations | `paper/coverage_simulation.md` |
| Section 6, step 1 — per-example $V_i$, AUROC, Mann–Whitney test, confidence-matched comparison | `paper/stability_summary.md` |

Tables 2, 3, 5, 6 and Figures 4–6 are functions of the group means, retention rates and timings.
`08_analyze.py --source reported` recomputes them from the values reported in the paper
([`configs/reported.yaml`](configs/reported.yaml)); the default uses fresh runs.

**Label-free variants (Section 6).** `--set counterfactuals.effects_on=all` in step 3 records restoration effects for
every counterfactual, so step 8 also reports $V$ on unfiltered counterfactuals (step 2 of Section 6).

## Configuration

All settings are in [`configs/paper.yaml`](configs/paper.yaml); any entry can be overridden with `--set key=value`.

| Setting | Value |
|---|---|
| Generator | CVL donor mixing, gate $\epsilon_g = 0.1$, $u \sim$ Beta (trained posterior or configured) |
| Counterfactuals $K$ | 50 / 100 / 150 (LLaVA, Qwen3-VL-Thinking); 50 (SmolVLM) |
| Retention | $\ge 9/10$ ($14/15$ for $K=150$) sampled decodings correct, temperature 0.7 |
| Bounds $\epsilon$, $\delta$, $B$ | 0.185, 0.05, 0.6 |
| CD-T schedule | $m_0 = m_i = 4$ heads per iteration |

[docs/method.md](docs/method.md) specifies each computation.

## Tests

```bash
python -m pytest -q tests
```

The tests use tiny randomly initialised LLaVA, Idefics3 (SmolVLM) and Qwen3-VL models and need no downloads. They
check that the CD-T decomposition reproduces the native forward pass, that donor mixing interpolates the projected
tokens (and DeepStack features) exactly and is batch-consistent, the donor rule and $u$ draws, that restoration effects
equal an explicit single-head patch, that patching touches only the prefill, and the bound constants.

## Repository structure

```text
mechanistic-counterfactual-grounding/
├── mcg/                          # method implementation
│   ├── models.py                 # native VLM adapter: LLaVA, Qwen3-VL, SmolVLM
│   ├── counterfactuals.py        # CVL donor mixing (Eq. 1), donor pool, Beta draws
│   ├── tasks.py                  # POPE / HallusionBench / COCO, answers y, retention rule
│   ├── background.py             # background mean activations
│   ├── cdt.py, circuits.py       # CD-T decomposition and circuit search (App. D, E)
│   ├── restoration.py            # restoration effects (Eq. 3)
│   ├── patching.py               # oracle patch, mean ablation
│   ├── bounds.py, report.py      # budgets (Eq. 4, App. A), Tables 2-6
│   ├── stats.py                  # Wilson, bootstrap (clustered), Mann-Whitney, AUROC, matching
│   └── pipeline.py               # Algorithm 1 for one answer
├── experiments/                  # steps 01-08
├── configs/                      # paper.yaml (settings), reported.yaml (reported quantities)
├── scripts/                      # reproduce_paper.sh, download_data.sh
├── training/, methods/           # CVL training (LLaVA-1.5)
├── docs/method.md
├── tests/
├── statistical_experiments/      # earlier concentration-bound simulations
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
