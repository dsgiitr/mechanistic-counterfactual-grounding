"""Restoration effects (Section 3.3, Eq. 3).

    Delta_v^(k) = log p(y | Restore(v, x_cf^(k))) - log p(y | x_cf^(k))

Restore(v, .) replaces the input slice of head v to its layer's output projection
(the head's output z_v, all sequence positions) by the factual value, in a
forward pass on counterfactual k. Factual and counterfactual runs share the
token layout because the counterfactual only changes visual embeddings.

Restorations of different nodes are batched: row 0 of the first batch is the
unpatched counterfactual baseline (whose node activations are also recorded for
the oracle-assisted patch of Section 4.3), every other row restores one node.
"""
from collections import defaultdict

import torch

from mcg.models import HookSet


class Restorer:
    def __init__(self, vlm, nodes, factual_acts):
        self.vlm = vlm
        self.nodes = [(int(n["layer"]), int(n["head"])) for n in nodes]
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Circuit nodes must be unique")
        D = vlm.head_dim
        self.factual = {(l, h): factual_acts[l][:, h].clone() for l, h in self.nodes}  # [S, D]
        self.D = D

    @torch.no_grad()
    def run(self, inputs, target_ids, batch_size=16):
        """Return (cf_logprob, {node: Delta}, {node: cf activation [S, D]})."""
        D = self.D
        rows = [None] + self.nodes
        logp = {}
        cf_acts = {}
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            replace = defaultdict(list)
            for r, node in enumerate(chunk):
                if node is not None:
                    replace[node[0]].append((r, node[1]))
            capture_row = chunk.index(None) if None in chunk else None
            capture_layers = defaultdict(list)
            if capture_row is not None:
                for l, h in self.nodes:
                    capture_layers[l].append(h)

            def hook_for(layer):
                def hook(module, args):
                    x = args[0]
                    for h in capture_layers.get(layer, ()):
                        cf_acts[(layer, h)] = x[capture_row, :, h * D:(h + 1) * D].clone()
                    if layer in replace:
                        x = x.clone()
                        for r, h in replace[layer]:
                            x[r, :, h * D:(h + 1) * D] = self.factual[(layer, h)].to(x.dtype)
                    return (x,) + tuple(args[1:])
                return hook

            with HookSet() as hooks:
                for layer in set(replace) | set(capture_layers):
                    hooks.pre(self.vlm.o_proj(layer), hook_for(layer))
                scores = self.vlm.target_logprob(self.vlm.expand(inputs, len(chunk)), target_ids).tolist()
            for node, score in zip(chunk, scores):
                logp[node] = score
        baseline = logp[None]
        deltas = {node: logp[node] - baseline for node in self.nodes}
        return baseline, deltas, cf_acts


def node_key(node):
    layer, head = node
    return f"{layer}_{head}"
