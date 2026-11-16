"""Head-level interventions on the native forward pass.

* `patch_heads`  oracle-assisted circuit patch (Section 4.3): each circuit node's
  output is replaced by its mean activation over the retained counterfactuals,
  in a forward pass on the factual input. During generation only the prefill
  (prompt and fixed prefix) is patched; newly generated positions are not.
* `mean_ablation` background-mean ablation used for circuit faithfulness and
  the circuit-size sensitivity analysis (Appendix D).
"""
from contextlib import contextmanager
from collections import defaultdict

import torch

from mcg.models import HookSet


@contextmanager
def patch_heads(vlm, replacements):
    """replacements: {(layer, head): tensor [S, D]} applied to positions [:seq_len]."""
    D = vlm.head_dim
    by_layer = defaultdict(list)
    for (layer, head), value in replacements.items():
        by_layer[layer].append((head, value))

    def hook_for(layer):
        def hook(module, args):
            x = args[0]
            length = x.shape[1]
            values = by_layer[layer]
            if length == 1 and values[0][1].shape[0] > 1:  # cached decoding step: leave unpatched
                return None
            x = x.clone()
            for head, value in values:
                if length > value.shape[0]:
                    raise ValueError("Patch is shorter than the patched sequence")
                x[:, :, head * D:(head + 1) * D] = value[:length].to(x.dtype)
            return (x,) + tuple(args[1:])
        return hook

    with HookSet() as hooks:
        for layer in by_layer:
            hooks.pre(vlm.o_proj(layer), hook_for(layer))
        yield


@contextmanager
def mean_ablation(vlm, means, keep=None, ablate=None):
    """Replace head outputs by background means.

    keep   : set of (layer, head); every *other* head is mean-ablated (circuit only)
    ablate : set of (layer, head); only these heads are mean-ablated (circuit removed)
    Exactly one of keep / ablate must be given (keep=set() ablates every head).
    """
    if (keep is None) == (ablate is None):
        raise ValueError("Pass exactly one of keep / ablate")
    H, D = vlm.num_heads, vlm.head_dim

    def hook_for(layer):
        def hook(module, args):
            x = args[0]
            B, S, _ = x.shape
            mean = means[layer][:S].to(x.dtype)  # [S, H, D]
            view = x.view(B, S, H, D)
            if keep is not None:
                out = mean.unsqueeze(0).expand(B, -1, -1, -1).clone()
                for l, h in keep:
                    if l == layer:
                        out[:, :, h] = view[:, :, h]
            else:
                out = view.clone()
                for l, h in ablate:
                    if l == layer:
                        out[:, :, h] = mean[:, h]
            return (out.reshape(B, S, H * D),) + tuple(args[1:])
        return hook

    with HookSet() as hooks:
        layers = range(vlm.num_layers) if keep is not None else sorted({l for l, _ in ablate})
        for layer in layers:
            hooks.pre(vlm.o_proj(layer), hook_for(layer))
        yield


@torch.no_grad()
def last_logits(vlm, inputs):
    return vlm.forward_logits(inputs, keep=1)[0, -1]
