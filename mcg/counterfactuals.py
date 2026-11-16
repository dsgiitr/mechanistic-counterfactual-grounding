"""Latent counterfactuals (Section 3.1, Eq. 1).

    h(I) = E_v(I),   h~(k) = h(I) + alpha * z_k,   z_k ~ N(0, I_d) i.i.d.,
    x_cf(k) = G_phi(h~(k))

E_v is the frozen vision encoder (including the model's own feature selection)
and G_phi is the model's frozen multimodal projector, so the intervention
distribution is fully specified by alpha and the Gaussian. The noise is added
to the input of every projector module returned by `VLM.projector_modules()`
(for Qwen3-VL this includes the DeepStack mergers, each with its own z_k).
The question and the token layout are unchanged.

Draw k of example e always uses the same z_k (seeded by (seed, e, k, stream)),
so the counterfactual baseline, every node-restoration run and every sampled
decoding of draw k see exactly the same visual tokens.
"""
from contextlib import contextmanager

import torch

from mcg.models import HookSet
from mcg.utils import derive_seed


class LatentCounterfactuals:
    def __init__(self, vlm, alpha, seed, example_key):
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.vlm = vlm
        self.alpha = float(alpha)
        self.seed = seed
        self.example_key = str(example_key)
        self.unit_shapes = None
        self._cache = {}

    @torch.no_grad()
    def observe(self, inputs):
        """Record the per-example shape of h(I) at each projector input."""
        shapes = {}

        def recorder(index):
            def hook(module, args):
                shapes[index] = tuple(args[0].shape)
            return hook

        with HookSet() as hooks:
            for index, module in enumerate(self.vlm.projector_modules()):
                hooks.pre(module, recorder(index))
            self.vlm.forward_logits(inputs, keep=1)
        if not shapes:
            raise ValueError("The input contains no image: a visual counterfactual is undefined")
        self.unit_shapes = shapes
        return shapes

    def noise(self, k, stream):
        key = (k, stream)
        if key not in self._cache:
            generator = torch.Generator(device="cpu").manual_seed(
                derive_seed(self.seed, self.example_key, k, stream, "z"))
            self._cache[key] = torch.randn(self.unit_shapes[stream], generator=generator)
        return self._cache[key]

    def clear(self):
        self._cache.clear()

    @contextmanager
    def draw(self, k):
        """Context in which every forward pass sees counterfactual k."""
        if self.unit_shapes is None:
            raise RuntimeError("Call observe(factual_inputs) first")

        def perturb(stream):
            def hook(module, args):
                h = args[0]
                z = self.noise(k, stream).to(h.device, h.dtype)
                if h.shape != z.shape:
                    repeats = h.shape[0] // z.shape[0]
                    if repeats * z.shape[0] != h.shape[0] or h.shape[1:] != z.shape[1:]:
                        raise ValueError(f"Visual feature shape {tuple(h.shape)} does not tile {tuple(z.shape)}")
                    z = z.repeat(repeats, *([1] * (z.dim() - 1)))
                return (h + self.alpha * z,) + tuple(args[1:])
            return hook

        with HookSet() as hooks:
            for index, module in enumerate(self.vlm.projector_modules()):
                if index in self.unit_shapes:
                    hooks.pre(module, perturb(index))
            yield
