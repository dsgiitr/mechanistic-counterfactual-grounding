"""Counterfactual circuit stability for vision-language models.

Implementation of "What Does It Cost to Audit a Hallucination? Counterfactual
Circuit Stability as an Offline Signal for VLM Review".

Modules
-------
models          native VLM adapter (LLaVA-1.5, Qwen3-VL Instruct/Thinking, SmolVLM)
counterfactuals CVL donor mixing  u h(I) + (1 - u) h(I'), u ~ Beta, gate eps_g      (Eq. 1)
tasks           benchmarks, answer extraction, hallucination labels, retention rule
background      background mean activations used as the CD-T / ablation reference
cdt             contextual decomposition for transformers (Appendix D)
circuits        circuit search schedules (iterative, global, layer-norm, iter-decay, random)
restoration     restoration effects Delta_v^(k)                                   (Eq. 3)
patching        oracle-assisted circuit patching and mean ablation
bounds          Bernstein / plug-in / empirical-Bernstein budgets                 (Eq. 4, App. A)
report          budget tables and audit cost from group means                      (Tables 2-6)
stats           group comparisons, Wilson intervals, rank tests, AUROC
"""

__version__ = "1.0.0"
