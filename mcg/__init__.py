"""Counterfactual circuit stability for vision-language models.

Implementation of "Mechanistic Signals of Unintended Misinformation:
Counterfactual Circuit Stability in Vision-Language Models".

Modules
-------
models          native VLM adapter (LLaVA-1.5, Qwen3-VL Instruct/Thinking, SmolVLM)
counterfactuals latent counterfactuals  x_cf = G_phi(E_v(I) + alpha z)          (Eq. 1)
tasks           benchmarks, answer extraction, hallucination labels, retention rule
background      background mean activations used as the CD-T / ablation reference
cdt             contextual decomposition for transformers (Appendix C)
circuits        circuit search schedules (iterative, global, layer-norm, iter-decay, random)
restoration     restoration effects Delta_v^(k)                                   (Eq. 2)
patching        oracle-assisted circuit patching and mean ablation
bounds          Bernstein / plug-in / empirical-Bernstein budgets                 (Eqs. 3-5)
stats           group comparisons, Wilson intervals, rank tests, AUROC
"""

__version__ = "1.0.0"
