# Implementation details

This page states, for every quantity in the paper, exactly how the code computes it.
Settings live in [`configs/paper.yaml`](../configs/paper.yaml).

## Models (`mcg/models.py`)

| Registry key | Checkpoint | Decoder (circuit nodes) | Projector $G_\phi$ |
|---|---|---|---|
| `llava-1.5-7b` | `llava-hf/llava-1.5-7b-hf` | Llama-2, 32 × 32 heads | `multi_modal_projector` (2-layer MLP) |
| `qwen3-vl-8b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` | Qwen3, 36 × 32 heads | `visual.merger` + DeepStack mergers |
| `qwen3-vl-8b-thinking` | `Qwen/Qwen3-VL-8B-Thinking` | Qwen3, 36 × 32 heads | `visual.merger` + DeepStack mergers |
| `smolvlm-500m` | `HuggingFaceTB/SmolVLM-500M-Instruct` | SmolLM2, 32 × 15 heads | `connector` (pixel shuffle + linear) |
| `smolvlm-256m` | `HuggingFaceTB/SmolVLM-256M-Instruct` | SmolLM2, 30 × 9 heads | `connector` (pixel shuffle + linear) |

Prompts use each processor's chat template. A circuit node is one decoder attention
head; its activation is the head's slice of the input to the layer's output projection
(`o_proj`), at all sequence positions.

## Latent counterfactuals (Eq. 1, `mcg/counterfactuals.py`)

$h(I)$ is the input of the projector, i.e. the vision-encoder output after the model's own
feature selection (LLaVA: penultimate CLIP layer without CLS). Each draw adds
$\alpha z_k$, $z_k\sim\mathcal N(0,I)$, to that input; the frozen projector maps it to the visual
tokens. For Qwen3-VL the DeepStack merger inputs receive independent noise in the same way.
$\alpha = 0.1$ is an absolute scale on $h(I)$. The noise of draw $k$ of example $e$ is seeded by
`(seed, e, k, stream)`, so the counterfactual baseline, every restoration run and every
sampled decoding of draw $k$ see the same visual tokens.

## Answers and labels (`mcg/tasks.py`)

| Dataset | Prompt | Hallucinated if | $y$ (scored by Eq. 2) |
|---|---|---|---|
| POPE (`Full/random`, 3,000) | `{question} Please answer yes or no.` | parsed yes/no ≠ label | the yes/no token(s) |
| HallusionBench (`image` + `non_image`) | same | parsed yes/no ≠ `gt_answer` | the yes/no token(s) |
| COCO val2014 (500 images, seeded) | `Please describe this image in detail.` | caption mentions a category absent from the instance annotations | first mention of an absent object (hallucinated) or of an annotated object (grounded), teacher-forced after the caption text that precedes it |

Object mentions are extracted CHAIR-style (Rohrbach et al., 2018): words are singularised,
two-word phrases are matched before single words, and synonyms map to COCO categories using
the CHAIR synonym list (`scripts/download_data.sh`). Context recall is
$|\mathcal C_{\text{pred}}\cap\mathcal C_{\text{gold}}|/|\mathcal C_{\text{gold}}|$ per image.

For Qwen3-VL-8B-Thinking the reasoning trace is generated greedily on the factual input
(up to `max_new_tokens_reasoning`), ends at `</think>`, and is held fixed (teacher-forced) in
every counterfactual, restoration and patched run; examples whose trace does not close are
not scorable.

**Example selection.** `num_hallucinated` = the first hallucinated items in benchmark order;
`num_grounded` = a seeded uniform draw from the correctly answered items; every example uses
a distinct image and must have a visual input.

## Retention rule

For each counterfactual, `decodings` answers (10 for $K\in\{50,100\}$, 15 for $K=150$) are
sampled from the fixed prefix at temperature 0.7 with top-$k$/top-$p$ filtering disabled. The
counterfactual is retained if at least `decodings − 1` of them are correct (binary: equals the
label; caption: mentions no absent object). The rule is identical for hallucinated and
grounded examples. With `counterfactuals.effects_on: all`, restoration effects are also
recorded for non-retained counterfactuals, which enables the label-free comparison.

## CD-T circuit discovery (Appendix C, `mcg/cdt.py`, `mcg/circuits.py`)

Decomposition rules are listed at the top of `mcg/cdt.py`; with no source the decomposition
reproduces the native forward pass (tested for all three architectures). The source head's
relevant part is its activation minus the background mean. Background means
(`experiments/02_background_means.py`) are computed over 500 POPE prompts and aligned to each
example: visual tokens by offset, the answer position by the final-position mean, other text
by the text mean.

Scores of source head $j$ are read at the position that predicts the first answer token:
$\beta_{\text{logit}}(j) = \langle\beta_{\text{final}},u\rangle$ with $u=W_U[y]-W_U[y_{\text{alt}}]$
for yes/no answers ($y_{\text{alt}}$: highest-scoring token of the opposite answer) and $u=W_U[y]$
for captions; $S_{\text{logit}}=|\beta_{\text{logit}}|/|\gamma_{\text{logit}}|$;
$S_{\text{head}}(j,t)=\|\beta_t\|_1/\|\gamma_t\|_1$ and $\beta_{\text{head}}(j,t)=\langle\beta_t, a_t-\bar a_t\rangle$
for every head $t$ in a later layer. The search keeps the top $K_0=4$ positive logit-facing
heads and then adds $K_i=4$ heads per iteration until no positive upstream head remains or
all targets are in layer 0. Circuits are cached per example and reused across $K$.

## Restoration effects and budgets (Eqs. 2–5)

`mcg/restoration.py` computes $\Delta_v^{(k)}$ exactly as Eq. 2, batching nodes; row 0 is the
unpatched counterfactual. `mcg/bounds.py` computes, per node, the unbiased variance,
$\hat K_v$ (Eq. 4), the empirical-Bernstein radius (Eq. 5, also with $\delta/|S^\star|$) and the
observed effect range (flagged when it exceeds $B_v$). Per example, $V_i$ and $\bar K_i$ are
the circuit means. Examples with fewer than two retained counterfactuals have no variance and
are excluded from the group means; their number is reported.

## Oracle-assisted patch (Section 4.3, `mcg/pipeline.py`)

Each circuit node is replaced by its mean activation over the retained counterfactuals in a
forward pass on the factual input. Binary answers: the patched decision compares the best
yes-token and best no-token logits at the answer position. Captions: the patch is applied to
the prefill and the caption is regenerated greedily; it is corrected if it mentions no absent
object. Examples with no retained counterfactual count as not corrected. For the benchmark
table every item is patched; items without a visual input (HallusionBench `non_image`), with
an unscorable answer, an empty circuit or no retained counterfactual keep their base
prediction, and the reason is stored per item.

## Circuit-size sensitivity (Appendix D)

Per-example CD-T scores of the hallucinated LLaVA/POPE examples are combined as
$\overline{S}\cdot|\overline{\beta}|/\overline{|\beta|}$. Strategies: **Global** (top-$n$ positive
logit-facing heads), **Layer Norm** (scores divided by the best score of the same layer),
**Iter Decay** (iterative search with $K_i=\max(1,\lfloor K_0\,0.5^i\rfloor)$, $K_0=\lceil n/2\rceil$,
union of targets) and **Random Noise** (uniform heads, mean over 5 draws). With background
mean ablation and $m=\log p(y)$: faithfulness $=\sum_i(m_i(C)-m_i(\varnothing))/\sum_i(m_i(M)-m_i(\varnothing))$,
hallucination remaining uses $m_i(M\setminus C)$ in the numerator, and accuracy recovered is
the fraction of examples answered correctly with $C$ ablated.
