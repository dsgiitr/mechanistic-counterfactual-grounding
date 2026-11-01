import torch
import torch.nn as nn
import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from pyfunctions.decomposition import normalize_rel_irrel
from pyfunctions.nodes import Node
from tqdm import tqdm
from models.native_utils import factual_context


def causal_attention_mask(seq_len, device, dtype):
    """Additive decoder mask: past/current tokens are zero, future tokens blocked."""
    return torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype),
        diagonal=1,
    )[None, None]

def prop_rmsnorm(rel, irrel, rmsnorm_module):
    tot = rel + irrel
    variance = tot.float().pow(2).mean(-1, keepdim=True)
    epsilon = getattr(rmsnorm_module, "variance_epsilon", getattr(rmsnorm_module, "eps", None))
    scale = torch.rsqrt(variance + epsilon)
    # Match native Llama's fp32 accumulation and cast before learned scaling.
    return ((rel.float() * scale).to(tot.dtype) * rmsnorm_module.weight,
            (irrel.float() * scale).to(tot.dtype) * rmsnorm_module.weight)

def prop_llama_mlp(rel, irrel, mlp_module):
    rel_gate = torch.matmul(rel, mlp_module.gate_proj.weight.T)
    irrel_gate = torch.matmul(irrel, mlp_module.gate_proj.weight.T)
    
    rel_up = torch.matmul(rel, mlp_module.up_proj.weight.T)
    irrel_up = torch.matmul(irrel, mlp_module.up_proj.weight.T)
    
    def F(gate, up):
        return mlp_module.act_fn(gate) * up
        
    irrel_mid = F(irrel_gate, irrel_up)
    tot_mid = F(rel_gate + irrel_gate, rel_up + irrel_up)
    rel_mid = tot_mid - irrel_mid
    
    rel_out = torch.matmul(rel_mid, mlp_module.down_proj.weight.T)
    irrel_out = torch.matmul(irrel_mid, mlp_module.down_proj.weight.T)
    return rel_out, irrel_out

def prop_llama_attention(rel, irrel, attention_mask, position_ids, attn_module, 
                         layer_idx, ablated_nodes, mean_c_layer, position_embeddings=None):
    bsz, q_len, _ = rel.shape
    # Newer transformers moved these to config
    cfg = getattr(attn_module, 'config', None)
    num_heads = getattr(attn_module, 'num_heads', cfg.num_attention_heads if cfg else None)
    head_dim = getattr(attn_module, 'head_dim', cfg.hidden_size // num_heads if cfg else None)
    hidden_size = num_heads * head_dim
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    if num_heads % num_kv_heads:
        raise ValueError("Attention query heads must be divisible by KV heads")

    # Q, K, V Projections
    rel_q = torch.matmul(rel, attn_module.q_proj.weight.T)
    irrel_q = torch.matmul(irrel, attn_module.q_proj.weight.T)
    rel_k = torch.matmul(rel, attn_module.k_proj.weight.T)
    irrel_k = torch.matmul(irrel, attn_module.k_proj.weight.T)
    rel_v = torch.matmul(rel, attn_module.v_proj.weight.T)
    irrel_v = torch.matmul(irrel, attn_module.v_proj.weight.T)
    for name, tensor in (("q_proj", irrel_q), ("k_proj", irrel_k), ("v_proj", irrel_v)):
        bias = getattr(attn_module, name).bias
        if bias is not None:
            tensor.add_(bias)
    
    # Reshape for heads
    rel_q = rel_q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    irrel_q = irrel_q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    rel_k = rel_k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    irrel_k = irrel_k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    rel_v = rel_v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    irrel_v = irrel_v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    # Qwen3 normalizes each query/key head before applying its native mRoPE.
    if hasattr(attn_module, "q_norm"):
        rel_q, irrel_q = prop_rmsnorm(rel_q, irrel_q, attn_module.q_norm)
        rel_k, irrel_k = prop_rmsnorm(rel_k, irrel_k, attn_module.k_norm)
    
    # RoPE
    if hasattr(attn_module, "rotary_emb"):
        # Older transformers calculate RoPE at attention level
        cos, sin = attn_module.rotary_emb(rel_v, position_ids)
    else:
        # Newer transformers calculate RoPE at model level
        cos, sin = position_embeddings
        
    rel_q, rel_k = apply_rotary_pos_emb(rel_q, rel_k, cos, sin)
    irrel_q, irrel_k = apply_rotary_pos_emb(irrel_q, irrel_k, cos, sin)
    repeats = num_heads // num_kv_heads
    rel_k, irrel_k, rel_v, irrel_v = [
        value.repeat_interleave(repeats, dim=1)
        for value in (rel_k, irrel_k, rel_v, irrel_v)
    ]
    
    # Dot product
    tot_q = rel_q + irrel_q
    tot_k = rel_k + irrel_k
    scaling = getattr(attn_module, "scaling", head_dim ** -0.5)
    tot_scores = torch.matmul(tot_q, tot_k.transpose(2, 3)) * scaling
    rel_scores = torch.matmul(rel_q, rel_k.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        tot_scores += attention_mask
        rel_scores += attention_mask
        
    tot_probs = nn.functional.softmax(tot_scores, dim=-1, dtype=torch.float32).to(rel.dtype)
    rel_probs = nn.functional.softmax(rel_scores, dim=-1, dtype=torch.float32).to(rel.dtype)
    irrel_probs = tot_probs - rel_probs
    
    # Multiply by V
    tot_v = rel_v + irrel_v
    tot_context = torch.matmul(tot_probs, tot_v)
    rel_context = torch.matmul(rel_probs, rel_v)
    irrel_context = tot_context - rel_context
    
    # Step 4: Patch rel/irrel for ablated nodes in this layer
    for b in range(bsz):
        # We now process ablated_nodes as a list of lists, one for each batch element
        for n in ablated_nodes[b]:
            if n.layer_idx == layer_idx:
                n_seq = n.sequence_idx
                n_head = n.attn_head_idx
                
                if n_seq == -1:
                    # Ablate the entire attention head across all sequence positions
                    current_act = tot_context[b, n_head, :q_len, :]
                    
                    # Check bounds for mean_act just in case mean_c_layer is shorter than q_len
                    valid_len = min(q_len, mean_c_layer.shape[1])
                    mean_act = mean_c_layer[0, :valid_len, n_head, :]
                    
                    irrel_context[b, n_head, :valid_len, :] = mean_act
                    rel_context[b, n_head, :valid_len, :] = current_act[:valid_len, :] - mean_act
                    continue
                
                # Prevent IndexError: dynamically sized examples might be shorter than statically generated max_seq nodes
                if n_seq >= q_len:
                    continue
                    
                # The current output of the head is tot_context
                current_act = tot_context[b, n_head, n_seq, :]
                
                # Also safety check for mean_act bounds, just in case
                if n_seq >= mean_c_layer.shape[1]:
                    continue
                    
                mean_act = mean_c_layer[0, n_seq, n_head, :]
                
                irrel_context[b, n_head, n_seq, :] = mean_act
                rel_context[b, n_head, n_seq, :] = current_act - mean_act

    # Save per-head context BEFORE reshape for intermediate extraction
    # Shape: (bsz, num_heads, q_len, head_dim)
    rel_context_heads = rel_context.clone()
    irrel_context_heads = irrel_context.clone()
    
    # Reshape back
    rel_context = rel_context.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
    irrel_context = irrel_context.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
    
    # Output projection
    rel_out = torch.matmul(rel_context, attn_module.o_proj.weight.T)
    irrel_out = torch.matmul(irrel_context, attn_module.o_proj.weight.T)
    if attn_module.o_proj.bias is not None:
        irrel_out = irrel_out + attn_module.o_proj.bias
    return rel_out, irrel_out, rel_context_heads, irrel_context_heads

def prop_llama_layer(rel, irrel, layer_module, attention_mask, position_ids, 
                     layer_idx, ablated_nodes, mean_c_layer, position_embeddings=None):
    rel_norm, irrel_norm = prop_rmsnorm(rel, irrel, layer_module.input_layernorm)
    rel_attn, irrel_attn, rel_ctx_heads, irrel_ctx_heads = prop_llama_attention(
        rel_norm, irrel_norm, attention_mask, position_ids, layer_module.self_attn, 
        layer_idx, ablated_nodes, mean_c_layer, position_embeddings=position_embeddings
    )
    
    rel_mid = rel + rel_attn
    irrel_mid = irrel + irrel_attn
    normalize_rel_irrel(rel_mid, irrel_mid)
    
    rel_norm2, irrel_norm2 = prop_rmsnorm(rel_mid, irrel_mid, layer_module.post_attention_layernorm)
    rel_mlp, irrel_mlp = prop_llama_mlp(rel_norm2, irrel_norm2, layer_module.mlp)
    
    rel_out = rel_mid + rel_mlp
    irrel_out = irrel_mid + irrel_mlp
    normalize_rel_irrel(rel_out, irrel_out)
    return rel_out, irrel_out, rel_ctx_heads, irrel_ctx_heads

def prop_decoder(dataset, ablated_batch, base_embeds, mean_c, return_intermediates=False):
    """CD through native Llama/Qwen3 decoder geometry captured on the factual run."""
    wrapper = dataset.wrapper
    model = wrapper.language_model
    if not hasattr(model, "layers"):
        model = model.model
    batch_size = len(ablated_batch)
    length = base_embeds.shape[1]
    irrel = base_embeds.expand(batch_size, -1, -1).clone()
    rel = torch.zeros_like(irrel)
    context = getattr(wrapper, "propagation_context", None)
    if context is not None:
        if context["base_embeds"].shape != base_embeds.shape:
            raise ValueError("Decoder context does not match the current factual input")
        position_ids = context["position_ids"]
        mask = context["attention_mask"]
        # SDPA can omit a causal mask on a prefill; this manual attention path cannot.
        if mask is None:
            mask = causal_attention_mask(length, base_embeds.device, base_embeds.dtype)
        elif mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=base_embeds.dtype).masked_fill(~mask, torch.finfo(base_embeds.dtype).min)
        position_embeddings = context["position_embeddings"]
    else:
        if getattr(model.config, "model_type", "") == "qwen3_vl_text":
            raise ValueError("Capture the native Qwen3 decoder context before propagation")
        position_ids = torch.arange(length, device=base_embeds.device).unsqueeze(0)
        mask = causal_attention_mask(length, base_embeds.device, base_embeds.dtype)
        position_embeddings = model.rotary_emb(base_embeds, position_ids) if hasattr(model, "rotary_emb") else None
    deepstack = context.get("deepstack_visual_embeds") if context else None
    visual_mask = context.get("visual_pos_masks") if context else None
    intermediates = {}
    for layer_idx, layer in enumerate(model.layers):
        rel, irrel, rel_heads, irrel_heads = prop_llama_layer(
            rel, irrel, layer, mask, position_ids, layer_idx, ablated_batch,
            mean_c[layer_idx], position_embeddings=position_embeddings)
        if return_intermediates:
            intermediates[layer_idx] = (rel_heads, irrel_heads)
        # Native Qwen3 adds fixed vision-encoder features after early decoder layers.
        # These are background inputs, not contributions from a decoder source head.
        if deepstack is not None and layer_idx < len(deepstack):
            injection = torch.zeros_like(base_embeds)
            injection[visual_mask] = deepstack[layer_idx].to(injection)
            irrel = irrel + injection.expand_as(irrel)
    rel, irrel = prop_rmsnorm(rel, irrel, model.norm)
    return (rel, irrel, intermediates) if return_intermediates else (rel, irrel)


def prop_LLaVA(dataset, ablated_batch, base_embeds, mean_c):
    # Compatibility name retained for existing experiment scripts.
    return prop_decoder(dataset, ablated_batch, base_embeds, mean_c)


def prop_LLaVA_with_intermediates(dataset, ablated_batch, base_embeds, mean_c):
    return prop_decoder(dataset, ablated_batch, base_embeds, mean_c, True)

def build_adjacency_matrix(dataset, source_nodes, yes_no=False, batch_size=32):
    """
    Builds the full 1024x1024 adjacency matrix and 1024-dim logit vector.
    
    For each source node S (ablated), propagates through the entire network
    and extracts:
      - For every intermediate target head T at layer > S.layer:
          beta_T = dot(rel_at_T, d_T) where d_T = clean_act_T - mean_act_T
      - For the output logits:
          beta_logit = dot(rel_final, d_logit) where d_logit = W_U[hall] - W_U[corr]
    
    Aggregates beta values across all dataset examples into:
      r_avg, r_consistency, combined_score = r_avg * r_consistency
    
    Returns:
        adj_combined: Tensor (N, N) of combined_score for head-to-head edges
        adj_r_avg: Tensor (N, N) of r_avg
        adj_r_consistency: Tensor (N, N) of r_consistency
        adj_mean_beta: Tensor (N, N) of mean_beta (for sign labeling)
        logit_scores: dict with keys 'combined', 'r_avg', 'r_consistency', 'mean_beta'
            each a Tensor of shape (N,)
    where N = num_layers * num_heads (1024 for LLaVA-7B).
    """
    wrapper = dataset.wrapper
    num_layers = wrapper.num_layers
    num_heads = wrapper.num_heads
    N = num_layers * num_heads  # 1024
    eps = 1e-8
    
    def node_to_idx(layer, head):
        return layer * num_heads + head
    
    num_examples = len(dataset.hallucination_examples)
    
    # Accumulators: store per-example beta values for every (source, target) pair
    # For head-to-head edges: (N_source, N_target, num_examples)
    # For logit edges: (N_source, num_examples)
    # We accumulate incrementally to avoid a massive 1024*1024*num_examples tensor
    head_beta_sum = torch.zeros((N, N), dtype=torch.float64)
    head_abs_beta_sum = torch.zeros((N, N), dtype=torch.float64)
    head_r_sum = torch.zeros((N, N), dtype=torch.float64)
    
    logit_beta_sum = torch.zeros((N,), dtype=torch.float64)
    logit_abs_beta_sum = torch.zeros((N,), dtype=torch.float64)
    logit_r_sum = torch.zeros((N,), dtype=torch.float64)
    
    # Precompute fixed direction for Yes/No tasks
    if yes_no:
        yes_id = wrapper.get_correct_token_id("Yes")
        no_id = wrapper.get_correct_token_id("No")
        raw_d = wrapper.W_U[yes_id] - wrapper.W_U[no_id]
        fixed_d_logit = raw_d / torch.norm(raw_d, p=2)
        print(f"Yes/No mode: yes_id={yes_id}, no_id={no_id}")
    
    for ex_idx, example in tqdm(enumerate(dataset.hallucination_examples), 
                                 total=num_examples, desc="Building adjacency matrix"):
        # Logit direction for this example
        if yes_no:
            d_logit = fixed_d_logit
        else:
            hall_id = example["hallucinated_token_id"]
            corr_id = example["correct_token_id"]
            raw_d = wrapper.W_U[hall_id] - wrapper.W_U[corr_id]
            d_logit = raw_d / torch.norm(raw_d, p=2)
        
        ans_pos = example["answer_position"]
        vis_pos = example["visual_positions"]
        q_pos = example["question_positions"]
        
        # Get mean activations and base embeddings
        mean_c = dataset.get_mean_correct(vis_pos, q_pos, ans_pos)
        
        inputs, outputs, base_embeds = factual_context(
            wrapper, example["image_loaded"], example["question"], example.get("answer_prefix", ""), example.get("answer_prefix_token_ids"))
        
        # Also run a clean forward pass to get the CLEAN activations for d_T directions.
        # The wrapper hooks already captured them during the forward pass above.
        clean_acts = {l: wrapper.activations[l].clone() for l in range(num_layers)}
        # clean_acts[l] shape: (1, seq_len, num_heads, head_dim)
        
        # Compute d_T for every target head: d_T = clean_act - mean_act
        # Shape per layer: (1, seq_len, num_heads, head_dim)
        d_T_per_layer = {}
        for l in range(num_layers):
            # mean_c[l] shape: (1, seq_len, num_heads, head_dim)
            d_T_per_layer[l] = clean_acts[l] - mean_c[l]
        
        # Batch over source nodes
        for b_start in tqdm(range(0, len(source_nodes), batch_size), 
                            desc=f"Batches (Ex {ex_idx})", leave=False):
            batch_nodes = source_nodes[b_start : b_start + batch_size]
            ablated_batch = [[node] for node in batch_nodes]
            
            # Run forward pass with intermediate extraction
            rel_final, irrel_final, intermediates = prop_LLaVA_with_intermediates(
                dataset, ablated_batch, base_embeds, mean_c
            )
            
            cur_bsz = len(batch_nodes)
            
            for b_idx, src_node in enumerate(batch_nodes):
                src_idx = node_to_idx(src_node.layer_idx, src_node.attn_head_idx)
                
                # --- Logit edge ---
                rel_at_ans = rel_final[b_idx, ans_pos, :]
                irrel_at_ans = irrel_final[b_idx, ans_pos, :]
                beta_logit = torch.dot(d_logit, rel_at_ans).item()
                gamma_logit = torch.dot(d_logit, irrel_at_ans).item()
                r_logit = abs(beta_logit) / (abs(gamma_logit) + eps)
                
                logit_beta_sum[src_idx] += beta_logit
                logit_abs_beta_sum[src_idx] += abs(beta_logit)
                logit_r_sum[src_idx] += r_logit
                
                # --- Head-to-head edges ---
                for tgt_layer in range(num_layers):
                    # Enforce causality: target must be STRICTLY after source
                    if tgt_layer <= src_node.layer_idx:
                        continue
                    
                    # rel_ctx_heads shape: (bsz, num_heads, seq_len, head_dim)
                    rel_ctx, irrel_ctx = intermediates[tgt_layer]
                    # d_T shape: (1, seq_len, num_heads, head_dim)
                    # Transpose d_T to (1, num_heads, seq_len, head_dim) to match
                    d_T = d_T_per_layer[tgt_layer].permute(0, 2, 1, 3)
                    
                    for tgt_head in range(num_heads):
                        tgt_idx = node_to_idx(tgt_layer, tgt_head)
                        
                        # Extract at answer position for this head
                        rel_vec = rel_ctx[b_idx, tgt_head, ans_pos, :]
                        irrel_vec = irrel_ctx[b_idx, tgt_head, ans_pos, :]
                        d_vec = d_T[0, tgt_head, ans_pos, :]
                        
                        beta_h = torch.dot(d_vec, rel_vec).item()
                        gamma_h = torch.dot(d_vec, irrel_vec).item()
                        r_h = abs(beta_h) / (abs(gamma_h) + eps)
                        
                        head_beta_sum[src_idx, tgt_idx] += beta_h
                        head_abs_beta_sum[src_idx, tgt_idx] += abs(beta_h)
                        head_r_sum[src_idx, tgt_idx] += r_h
    
    # --- Aggregate across examples ---
    # Head-to-head
    head_r_avg = head_r_sum / max(num_examples, 1)
    head_mean_beta = head_beta_sum / max(num_examples, 1)
    head_mean_abs_beta = head_abs_beta_sum / max(num_examples, 1)
    head_r_consistency = torch.abs(head_mean_beta) / (head_mean_abs_beta + eps)
    head_combined = head_r_avg * head_r_consistency
    
    # Enforce causality mask: zero out invalid edges (L_target <= L_source)
    for src in range(N):
        src_layer = src // num_heads
        for tgt in range(N):
            tgt_layer = tgt // num_heads
            if tgt_layer <= src_layer:
                head_combined[src, tgt] = 0.0
                head_r_avg[src, tgt] = 0.0
                head_r_consistency[src, tgt] = 0.0
                head_mean_beta[src, tgt] = 0.0
    
    # Logit scores
    logit_r_avg = logit_r_sum / max(num_examples, 1)
    logit_mean_beta = logit_beta_sum / max(num_examples, 1)
    logit_mean_abs_beta = logit_abs_beta_sum / max(num_examples, 1)
    logit_r_consistency = torch.abs(logit_mean_beta) / (logit_mean_abs_beta + eps)
    logit_combined = logit_r_avg * logit_r_consistency
    
    return {
        "head_combined": head_combined.float(),
        "head_r_avg": head_r_avg.float(),
        "head_r_consistency": head_r_consistency.float(),
        "head_mean_beta": head_mean_beta.float(),
        "logit_combined": logit_combined.float(),
        "logit_r_avg": logit_r_avg.float(),
        "logit_r_consistency": logit_r_consistency.float(),
        "logit_mean_beta": logit_mean_beta.float(),
    }

def score_nodes(dataset, source_nodes, current_circuit, yes_no=False, batch_size=32):
    """
    Step 5 & 6: Pre-computes R_i(s) across all examples for targeted source_nodes, 
    and aggregates them into combined_score, R_avg, and R_consistency.
    """
    wrapper = dataset.wrapper
    eps = 1e-8
    
    node_to_r_i = {n: [] for n in source_nodes}
    node_to_beta_T = {n: [] for n in source_nodes}
    
    # Track split positions for logging
    node_to_r_vis = {n: [] for n in source_nodes}
    node_to_r_q = {n: [] for n in source_nodes}

    # Precompute fixed direction for Yes/No tasks
    if yes_no:
        yes_id = wrapper.get_correct_token_id("Yes")
        no_id = wrapper.get_correct_token_id("No")
        raw_d = wrapper.W_U[yes_id] - wrapper.W_U[no_id]
        fixed_d_i = raw_d / torch.norm(raw_d, p=2)
        print(f"Yes/No mode: yes_id={yes_id}, no_id={no_id}, d_i norm={torch.norm(raw_d).item():.4f}")

    for idx, example in tqdm(enumerate(dataset.hallucination_examples), total=len(dataset.hallucination_examples), desc="Scoring examples"):
        if yes_no:
            d_i = fixed_d_i
        else:
            hall_id = example["hallucinated_token_id"]
            corr_id = example["correct_token_id"]
            raw_d_i = wrapper.W_U[hall_id] - wrapper.W_U[corr_id]
            d_i = raw_d_i / torch.norm(raw_d_i, p=2)
            
        ans_pos = example["answer_position"]
        vis_pos = example["visual_positions"]
        q_pos = example["question_positions"]
        
        # Calculate mean context and get base embeddings ONCE for the whole sample
        mean_c = dataset.get_mean_correct(vis_pos, q_pos, ans_pos)
        
        inputs, outputs, base_embeds = factual_context(
            wrapper, example["image_loaded"], example["question"], example.get("answer_prefix", ""), example.get("answer_prefix_token_ids"))
            
        # Batch evaluation
        for b_start in tqdm(range(0, len(source_nodes), batch_size), desc=f"Batches (Example {idx})", leave=False):
            batch_nodes = source_nodes[b_start : b_start + batch_size]
            ablated_batch = [current_circuit + [node] for node in batch_nodes]
            
            # Run one massively parallel CD propagation
            rel_h, irrel_h = prop_LLaVA(dataset, ablated_batch, base_embeds, mean_c)
            
            for b_idx, node in enumerate(batch_nodes):
                rel_final_i = rel_h[b_idx, ans_pos, :]
                irrel_final_i = irrel_h[b_idx, ans_pos, :]
                
                beta_T_i = torch.dot(d_i, rel_final_i).item()
                gamma_T_i = torch.dot(d_i, irrel_final_i).item()
            
                R_i = abs(beta_T_i) / (abs(gamma_T_i) + eps)
                node_to_r_i[node].append(R_i)
                node_to_beta_T[node].append(beta_T_i)
                
                if node.sequence_idx in vis_pos:
                    node_to_r_vis[node].append(R_i)
                elif node.sequence_idx in q_pos:
                    node_to_r_q[node].append(R_i)
    
    # Step 6
    scores = {}
    for node in source_nodes:
        r_avg = sum(node_to_r_i[node]) / len(node_to_r_i[node])
        
        avg_beta = sum(node_to_beta_T[node]) / len(node_to_beta_T[node])
        avg_abs_beta = sum(abs(b) for b in node_to_beta_T[node]) / len(node_to_beta_T[node])
        r_consistency = abs(avg_beta) / (avg_abs_beta + eps)
        
        combined = r_avg * r_consistency
        
        r_vis = sum(node_to_r_vis[node]) / len(node_to_r_vis[node]) if node_to_r_vis[node] else 0.0
        r_q = sum(node_to_r_q[node]) / len(node_to_r_q[node]) if node_to_r_q[node] else 0.0
        
        scores[node] = {
            "combined": combined,
            "r_avg": r_avg,
            "r_consistency": r_consistency,
            "mean_beta_T": avg_beta,
            "r_avg_visual": r_vis,
            "r_avg_question": r_q
        }
        
    return scores

def compute_faithfulness(dataset, circuit, yes_no=False):
    from methods.faithfulness import evaluate_faithfulness
    return evaluate_faithfulness(dataset, circuit, yes_no)["overall_faithfulness"]
