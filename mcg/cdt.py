"""Contextual decomposition for transformers (CD-T), Appendix C.

Every activation is split as x = beta + gamma (relevant / irrelevant) and both
parts are propagated through the native decoder geometry captured on the
factual run (attention mask, rotary/mRoPE embeddings, Qwen3 DeepStack
injections, which are treated as fixed irrelevant inputs).

Rules
  linear     beta' = W beta + |W beta|/(|W beta|+|W gamma|) * b      (bias shared by magnitude)
  RMSNorm    both parts scaled by the scale of the total
  attention  beta_att = beta_q beta_k^T / sqrt(d);  P_beta = softmax(M(beta_att));
             P_gamma = softmax(M(total)) - P_beta;  beta_z = P_beta beta_v;  gamma_z = z - beta_z
  gated MLP  gamma_mid = f(gamma);  beta_mid = f(total) - gamma_mid
  source     at the source head s: beta = a_s - mean_s, gamma = mean_s

Scores for a source head j (whole head, all positions)
  logit target   beta_logit(j) = <beta_final[ans], u>, S_logit(j) = |beta_logit| / |gamma_logit|
  head target t  S_head(j,t) = ||beta_t[ans]||_1 / ||gamma_t[ans]||_1,
                 beta_head(j,t) = <beta_t[ans], a_t[ans] - mean_t[ans]>
where u = W_U[y] - W_U[y_alt] (binary answers) or W_U[y] (captions) and ans is the
position that predicts the first answer token. R(s,T) = sum_t S_head(s,t).
"""
import sys

import torch
from torch import nn

from mcg.models import HookSet

EPS = 1e-8


def normalize_rel_irrel(rel, irrel):
    """Sign-consistency step of the reference CD-T implementation (in place)."""
    total = rel + irrel
    opposite = (rel * irrel) < 0
    relevant = opposite & (rel.abs() >= irrel.abs())
    irrelevant = opposite & ~relevant
    rel[relevant] = total[relevant]
    rel[irrelevant] = 0
    irrel[irrelevant] = total[irrelevant]
    irrel[relevant] = 0


def causal_mask(length, device, dtype):
    return torch.triu(torch.full((length, length), torch.finfo(dtype).min, device=device, dtype=dtype),
                      diagonal=1)[None, None]


@torch.no_grad()
def capture_context(vlm, inputs, keep=1):
    """Run the factual forward once and record everything CD-T needs.

    ctx["logits"] holds fp32 logits of the last `keep` positions, [keep, V].
    """
    ctx = {"layer_inputs": [None] * vlm.num_layers, "acts": [None] * vlm.num_layers}

    def decoder_hook(module, args, kwargs):
        ctx["deepstack"] = kwargs.get("deepstack_visual_embeds")
        ctx["visual_pos_masks"] = kwargs.get("visual_pos_masks")

    def layer_hook(index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            ctx["layer_inputs"][index] = hidden.detach()
            if index == 0:
                ctx["attention_mask"] = kwargs.get("attention_mask")
                ctx["position_embeddings"] = kwargs.get("position_embeddings")
        return hook

    def act_hook(index):
        def hook(module, args):
            x = args[0][0]
            ctx["acts"][index] = x.view(x.shape[0], vlm.num_heads, vlm.head_dim).detach()
        return hook

    with HookSet() as hooks:
        hooks.pre(vlm.decoder, decoder_hook, with_kwargs=True)
        for index, layer in enumerate(vlm.decoder.layers):
            hooks.pre(layer, layer_hook(index), with_kwargs=True)
            hooks.pre(layer.self_attn.o_proj, act_hook(index))
        ctx["logits"] = vlm.forward_logits(inputs, keep=keep)[0]
    first = ctx["layer_inputs"][0]
    length, dtype = first.shape[1], first.dtype
    mask = ctx.get("attention_mask")
    if isinstance(mask, dict):
        mask = mask.get("full_attention")
    if mask is None:
        mask = causal_mask(length, first.device, dtype)
    elif mask.dtype == torch.bool:
        mask = torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, torch.finfo(dtype).min)
    ctx["mask"] = mask.to(dtype)
    return ctx


# ------------------------------------------------------------------------------ rules
def _linear(rel, irrel, layer):
    r = rel @ layer.weight.T
    i = irrel @ layer.weight.T
    if layer.bias is not None:
        share = r.abs() / (r.abs() + i.abs() + EPS)
        r = r + share * layer.bias
        i = i + (1 - share) * layer.bias
    return r, i


def _rmsnorm(rel, irrel, norm):
    total = rel + irrel
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    scale = torch.rsqrt(total.float().pow(2).mean(-1, keepdim=True) + eps)
    return ((rel.float() * scale).to(total.dtype) * norm.weight,
            (irrel.float() * scale).to(total.dtype) * norm.weight)


def _mlp(rel, irrel, mlp):
    rel_gate, irrel_gate = _linear(rel, irrel, mlp.gate_proj)
    rel_up, irrel_up = _linear(rel, irrel, mlp.up_proj)
    irrel_mid = mlp.act_fn(irrel_gate) * irrel_up
    rel_mid = mlp.act_fn(rel_gate + irrel_gate) * (rel_up + irrel_up) - irrel_mid
    return _linear(rel_mid, irrel_mid, mlp.down_proj)


def _attention(rel, irrel, attn, mask, position_embeddings, sources, mean):
    B, S, _ = rel.shape
    cfg = attn.config
    H = cfg.num_attention_heads
    D = getattr(attn, "head_dim", None) or cfg.hidden_size // H
    KV = getattr(cfg, "num_key_value_heads", H)

    def heads(x, n):
        return x.view(B, S, n, D).transpose(1, 2)

    rq, iq = (heads(x, H) for x in _linear(rel, irrel, attn.q_proj))
    rk, ik = (heads(x, KV) for x in _linear(rel, irrel, attn.k_proj))
    rv, iv = (heads(x, KV) for x in _linear(rel, irrel, attn.v_proj))
    if hasattr(attn, "q_norm"):
        rq, iq = _rmsnorm(rq, iq, attn.q_norm)
        rk, ik = _rmsnorm(rk, ik, attn.k_norm)
    rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb
    cos, sin = position_embeddings
    rq, rk = rope(rq, rk, cos, sin)
    iq, ik = rope(iq, ik, cos, sin)
    if KV != H:
        rk, ik, rv, iv = (x.repeat_interleave(H // KV, dim=1) for x in (rk, ik, rv, iv))
    scaling = getattr(attn, "scaling", D ** -0.5)
    tq, tk, tv = rq + iq, rk + ik, rv + iv
    total_scores = (tq @ tk.transpose(2, 3)) * scaling + mask
    rel_scores = (rq @ rk.transpose(2, 3)) * scaling + mask
    p_total = nn.functional.softmax(total_scores, dim=-1, dtype=torch.float32).to(rel.dtype)
    p_rel = nn.functional.softmax(rel_scores, dim=-1, dtype=torch.float32).to(rel.dtype)
    total_ctx = p_total @ tv
    rel_ctx = p_rel @ rv
    irrel_ctx = total_ctx - rel_ctx
    if sources is not None:
        mean_hsd = mean.transpose(0, 1).to(rel.dtype)  # [H, S, D]
        for row, row_heads in enumerate(sources):
            for h in row_heads:
                rel_ctx[row, h] = total_ctx[row, h] - mean_hsd[h]
                irrel_ctx[row, h] = mean_hsd[h]
    rel_out, irrel_out = _linear(rel_ctx.transpose(1, 2).reshape(B, S, H * D),
                                 irrel_ctx.transpose(1, 2).reshape(B, S, H * D), attn.o_proj)
    return rel_out, irrel_out, rel_ctx, irrel_ctx


def _layer(rel, irrel, layer, ctx, sources, mean):
    r, i = _rmsnorm(rel, irrel, layer.input_layernorm)
    r, i, rel_ctx, irrel_ctx = _attention(r, i, layer.self_attn, ctx["mask"], ctx["position_embeddings"],
                                          sources, mean)
    rel, irrel = rel + r, irrel + i
    normalize_rel_irrel(rel, irrel)
    r, i = _mlp(*_rmsnorm(rel, irrel, layer.post_attention_layernorm), layer.mlp)
    rel, irrel = rel + r, irrel + i
    normalize_rel_irrel(rel, irrel)
    return rel, irrel, rel_ctx, irrel_ctx


@torch.no_grad()
def propagate(vlm, ctx, start_layer, sources, means, on_layer=None):
    """Propagate a decomposition whose sources are heads of `start_layer`.

    sources[b] lists the source heads of batch row b. Before `start_layer` the
    relevant part is exactly zero, so propagation starts from the captured
    factual input of that layer. `on_layer(layer, rel_ctx, irrel_ctx)` is called
    for each later layer. Returns the decomposition after the final norm.
    """
    B = len(sources)
    irrel = ctx["layer_inputs"][start_layer].expand(B, -1, -1).clone()
    rel = torch.zeros_like(irrel)
    deepstack = ctx.get("deepstack")
    visual = ctx.get("visual_pos_masks")
    for index in range(start_layer, vlm.num_layers):
        layer = vlm.decoder.layers[index]
        is_source = index == start_layer
        rel, irrel, rel_ctx, irrel_ctx = _layer(rel, irrel, layer, ctx, sources if is_source else None,
                                                means[index] if is_source else None)
        if on_layer is not None and not is_source:
            on_layer(index, rel_ctx, irrel_ctx)
        if deepstack is not None and index < len(deepstack):
            injection = torch.zeros_like(irrel[:1])
            injection[visual] = deepstack[index].to(injection)
            irrel = irrel + injection
    return _rmsnorm(rel, irrel, vlm.decoder.norm)


def answer_direction(vlm, target_id, contrast_id=None):
    u = vlm.lm_head.weight[target_id].float()
    if contrast_id is not None:
        u = u - vlm.lm_head.weight[contrast_id].float()
    return u


@torch.no_grad()
def sweep(vlm, ctx, means, answer_pos, direction, batch_size=16, progress=None, max_attention_elements=1e9):
    """CD-T scores of every head as a source (Appendix C, step 1-2 inputs).

    The source batch shrinks for long sequences so that one attention-score
    tensor (batch x heads x S x S) stays below `max_attention_elements`.
    """
    L, H = vlm.num_layers, vlm.num_heads
    N = L * H
    S = ctx["layer_inputs"][0].shape[1]
    batch_size = max(1, min(batch_size, int(max_attention_elements // (H * S * S))))
    s_logit, b_logit = torch.zeros(N), torch.zeros(N)
    s_head, b_head = torch.zeros(N, N), torch.zeros(N, N)
    d_target = [(ctx["acts"][t][answer_pos] - means[t][answer_pos]).float() for t in range(L)]  # [H, D]
    u = direction.to(ctx["layer_inputs"][0].device)
    for layer in range(L):
        for start in range(0, H, batch_size):
            heads = list(range(start, min(H, start + batch_size)))
            rows = torch.tensor([layer * H + h for h in heads])

            def on_layer(t, rel_ctx, irrel_ctx):
                r = rel_ctx[:, :, answer_pos].float()   # [B, H, D]
                i = irrel_ctx[:, :, answer_pos].float()
                s_head[rows, t * H:(t + 1) * H] = (r.abs().sum(-1) / i.abs().sum(-1).clamp_min(EPS)).cpu()
                b_head[rows, t * H:(t + 1) * H] = (r * d_target[t]).sum(-1).cpu()

            rel, irrel = propagate(vlm, ctx, layer, [[h] for h in heads], means, on_layer)
            beta = rel[:, answer_pos].float() @ u
            gamma = irrel[:, answer_pos].float() @ u
            b_logit[rows] = beta.cpu()
            s_logit[rows] = (beta.abs() / gamma.abs().clamp_min(EPS)).cpu()
            if progress is not None:
                progress.update(len(heads))
    return {"s_logit": s_logit, "beta_logit": b_logit, "s_head": s_head, "beta_head": b_head,
            "num_layers": L, "num_heads": H}
