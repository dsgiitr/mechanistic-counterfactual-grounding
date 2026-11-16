"""Circuit search on CD-T scores.

`iterative_search` is the per-example procedure of Appendix C:
  (1) keep heads with beta_logit > 0, take the top K_0 by S_logit as T_0 and C = T_0;
  (2) score j not in C by S_agg(j) = sum_{t in T_i} S_head(j,t); discard j if
      beta_agg(j) = sum_{t in T_i} beta_head(j,t) <= 0;
  (3) add the top K_i remaining heads S_i to C, T_{i+1} = S_i;
  (4) stop when no valid upstream head remains or all targets are in layer 0.
The schedule {K_i} (default K_0 = K_i = 4) counts heads, not counterfactuals.

The other selectors are the strategies compared in Appendix D (Figure 3):
global, layer-normalised, iterative-decay and a size-matched random control.
They operate on dataset-level scores from `aggregate_scores`.
"""
import math
import random

import torch


def _node(index, H):
    return {"layer": int(index // H), "head": int(index % H)}


def iterative_search(scores, k0=4, k_schedule=None, max_nodes=None, update="replace"):
    H = scores["num_heads"]
    s_logit, b_logit = scores["s_logit"], scores["beta_logit"]
    s_head, b_head = scores["s_head"], scores["beta_head"]
    k_schedule = k_schedule or (lambda i: k0)
    valid = (b_logit > 0) & torch.isfinite(s_logit) & (s_logit > 0)
    if not valid.any():
        return []
    ranked = torch.nonzero(valid, as_tuple=True)[0]
    targets = ranked[torch.argsort(s_logit[ranked], descending=True)][:k0].tolist()
    circuit = [dict(_node(t, H), iteration=0, target="logits", score=float(s_logit[t]),
                    beta=float(b_logit[t])) for t in targets]
    chosen = set(targets)
    iteration = 1
    while max_nodes is None or len(circuit) < max_nodes:
        targets = [t for t in targets if t // H > 0]
        if not targets:  # every target is in layer 0: nothing upstream remains
            break
        agg_s = s_head[:, targets].sum(1)
        agg_b = b_head[:, targets].sum(1)
        candidate = (agg_s > 0) & (agg_b > 0) & torch.isfinite(agg_s)
        if chosen:
            candidate[list(chosen)] = False
        if not candidate.any():
            break
        idx = torch.nonzero(candidate, as_tuple=True)[0]
        count = max(1, int(k_schedule(iteration)))
        if max_nodes is not None:
            count = min(count, max_nodes - len(circuit))
        picked = idx[torch.argsort(agg_s[idx], descending=True)][:count].tolist()
        for j in picked:
            circuit.append(dict(_node(j, H), iteration=iteration, score=float(agg_s[j]),
                                beta=float(agg_b[j]), target=[_node(t, H) for t in targets]))
            chosen.add(j)
        targets = picked if update == "replace" else sorted(set(targets) | set(picked))
        iteration += 1
    return circuit


# ------------------------------------------------------------------- dataset-level strategies
def aggregate_scores(per_example):
    """Combine per-example scores: combined = mean(S) * |mean(beta)| / mean(|beta|)."""
    def combine(key_s, key_b):
        s = torch.stack([e[key_s] for e in per_example]).float()
        b = torch.stack([e[key_b] for e in per_example]).float()
        consistency = b.mean(0).abs() / b.abs().mean(0).clamp_min(1e-8)
        return s.mean(0) * consistency, b.mean(0)
    s_logit, b_logit = combine("s_logit", "beta_logit")
    s_head, b_head = combine("s_head", "beta_head")
    first = per_example[0]
    return {"s_logit": s_logit, "beta_logit": b_logit, "s_head": s_head, "beta_head": b_head,
            "num_layers": first["num_layers"], "num_heads": first["num_heads"]}


def global_topk(scores, n):
    H = scores["num_heads"]
    s = torch.where(scores["beta_logit"] > 0, scores["s_logit"], torch.full_like(scores["s_logit"], -1.0))
    order = torch.argsort(s, descending=True)[:n]
    return [dict(_node(j, H), score=float(s[j])) for j in order.tolist() if s[j] > 0]


def layer_normalized_topk(scores, n):
    """S_logit divided by the maximum S_logit of the same layer, then global top-n."""
    H = scores["num_heads"]
    s = torch.where(scores["beta_logit"] > 0, scores["s_logit"], torch.zeros_like(scores["s_logit"]))
    per_layer = s.view(-1, H)
    normalized = (per_layer / per_layer.max(1, keepdim=True).values.clamp_min(1e-12)).reshape(-1)
    order = torch.argsort(normalized, descending=True)[:n]
    return [dict(_node(j, H), score=float(normalized[j])) for j in order.tolist() if normalized[j] > 0]


def iterative_decay(scores, n, decay=0.5):
    """Iterative search with K_i = max(1, floor(K_0 decay^i)), K_0 = ceil(n (1-decay)), union targets."""
    k0 = max(1, math.ceil(n * (1 - decay)))
    return iterative_search(scores, k0=k0, k_schedule=lambda i: max(1, int(k0 * decay ** i)),
                            max_nodes=n, update="union")


def random_circuit(num_layers, num_heads, n, seed):
    rng = random.Random(seed)
    picked = rng.sample(range(num_layers * num_heads), n)
    return [_node(j, num_heads) for j in picked]
