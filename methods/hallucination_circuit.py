import torch
import torch.nn as nn
import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from pyfunctions.cdt_core import normalize_rel_irrel
from pyfunctions.wrappers import Node
from tqdm import tqdm

def prop_rmsnorm(rel, irrel, rmsnorm_module):
    tot = rel + irrel
    variance = tot.pow(2).mean(-1, keepdim=True)
    scale = torch.rsqrt(variance + rmsnorm_module.variance_epsilon) * rmsnorm_module.weight
    return rel * scale, irrel * scale

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
                         layer_idx, ablated_nodes, mean_c_layer):
    bsz, q_len, _ = rel.shape
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    hidden_size = attn_module.hidden_size

    # Q, K, V Projections
    rel_q = torch.matmul(rel, attn_module.q_proj.weight.T)
    irrel_q = torch.matmul(irrel, attn_module.q_proj.weight.T)
    rel_k = torch.matmul(rel, attn_module.k_proj.weight.T)
    irrel_k = torch.matmul(irrel, attn_module.k_proj.weight.T)
    rel_v = torch.matmul(rel, attn_module.v_proj.weight.T)
    irrel_v = torch.matmul(irrel, attn_module.v_proj.weight.T)
    
    # Reshape for heads
    rel_q = rel_q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    irrel_q = irrel_q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    rel_k = rel_k.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    irrel_k = irrel_k.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    rel_v = rel_v.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    irrel_v = irrel_v.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    
    # RoPE
    cos, sin = attn_module.rotary_emb(rel_v, position_ids)
    rel_q, rel_k = apply_rotary_pos_emb(rel_q, rel_k, cos, sin)
    irrel_q, irrel_k = apply_rotary_pos_emb(irrel_q, irrel_k, cos, sin)
    
    # Dot product
    tot_q = rel_q + irrel_q
    tot_k = rel_k + irrel_k
    tot_scores = torch.matmul(tot_q, tot_k.transpose(2, 3)) / math.sqrt(head_dim)
    rel_scores = torch.matmul(rel_q, rel_k.transpose(2, 3)) / math.sqrt(head_dim)
    
    if attention_mask is not None:
        tot_scores += attention_mask
        rel_scores += attention_mask
        
    tot_probs = nn.functional.softmax(tot_scores, dim=-1)
    rel_probs = nn.functional.softmax(rel_scores, dim=-1)
    irrel_probs = tot_probs - rel_probs
    
    # Multiply by V
    tot_v = rel_v + irrel_v
    tot_context = torch.matmul(tot_probs, tot_v)
    rel_context = torch.matmul(rel_probs, rel_v)
    irrel_context = tot_context - rel_context
    
    # Step 4: Patch rel/irrel for ablated nodes in this layer
    # If a node is in the circuit, we mean-ablate it:
    # rel = act - mean, irrel = mean
    # Because we are propagating the "hallucination component" relative to the "mean correct",
    # the node's contribution to `rel` is its deviation from the mean, and its `irrel` is the mean.
    for n in ablated_nodes:
        if n.layer_idx == layer_idx:
            n_seq = n.sequence_idx
            n_head = n.attn_head_idx
            # The current output of the head is tot_context
            current_act = tot_context[0, n_head, n_seq, :]
            mean_act = mean_c_layer[0, n_seq, n_head, :]
            
            irrel_context[0, n_head, n_seq, :] = mean_act
            rel_context[0, n_head, n_seq, :] = current_act - mean_act

    # Reshape back
    rel_context = rel_context.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
    irrel_context = irrel_context.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
    
    # Output projection
    rel_out = torch.matmul(rel_context, attn_module.o_proj.weight.T)
    irrel_out = torch.matmul(irrel_context, attn_module.o_proj.weight.T)
    return rel_out, irrel_out

def prop_llama_layer(rel, irrel, layer_module, attention_mask, position_ids, 
                     layer_idx, ablated_nodes, mean_c_layer):
    rel_norm, irrel_norm = prop_rmsnorm(rel, irrel, layer_module.input_layernorm)
    rel_attn, irrel_attn = prop_llama_attention(
        rel_norm, irrel_norm, attention_mask, position_ids, layer_module.self_attn, 
        layer_idx, ablated_nodes, mean_c_layer
    )
    
    rel_mid = rel + rel_attn
    irrel_mid = irrel + irrel_attn
    normalize_rel_irrel(rel_mid, irrel_mid)
    
    rel_norm2, irrel_norm2 = prop_rmsnorm(rel_mid, irrel_mid, layer_module.post_attention_layernorm)
    rel_mlp, irrel_mlp = prop_llama_mlp(rel_norm2, irrel_norm2, layer_module.mlp)
    
    rel_out = rel_mid + rel_mlp
    irrel_out = irrel_mid + irrel_mlp
    normalize_rel_irrel(rel_out, irrel_out)
    return rel_out, irrel_out

def prop_LLaVA(dataset, example, ablated_nodes):
    """
    Runs a single full forward pass propagating rel and irrel from embedding to final layer norm.
    Returns (rel_final, irrel_final) representing the hidden state just before lm_head projection.
    """
    wrapper = dataset.wrapper
    device = wrapper.device
    model = wrapper.language_model.model
    
    prompt = wrapper.get_prompt(example["question"])
    inputs = wrapper.processor(text=prompt, images=example["image_loaded"], return_tensors="pt")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        # Get input embeddings for the joint vision-text seq
        inputs_embeds = wrapper.model.get_input_embeddings()(inputs["input_ids"])
        if "pixel_values" in inputs:
            # LLaVA merges vision tokens into inputs_embeds
            # For exact reproduction of embeddings without rewriting LLaVA's embedding logic:
            # We can run a dummy forward and capture the embeddings entering the first layer.
            pass
            
    # Capture the exact hidden states entering Layer 0
    # To be extremely precise, we can use the model's actual hidden states from a normal forward pass.
    # LLaVA's `prepare_inputs_labels_for_multimodal` does the multimodal merging.
    # It's publicly exposed or we can extract it via the model output.
    with torch.no_grad():
        outputs = wrapper.language_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    # hidden_states[0] is the output of the embedding layer (input to layer 0)
    embeds = outputs.hidden_states[0].to(torch.float16)
    
    # Initialize rel=0, irrel=embeds
    irrel = embeds
    rel = torch.zeros_like(embeds)
    
    # Setup masks
    bsz, seq_len, _ = embeds.shape
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    
    attention_mask = inputs.get("attention_mask", torch.ones((bsz, seq_len), device=device))
    mask = attention_mask[:, None, None, :].to(torch.float16)
    mask = (1.0 - mask) * torch.finfo(torch.float16).min
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)
    causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float16).min
    full_mask = torch.maximum(mask, causal_mask.to(torch.float16))
    
    vis_pos = example["visual_positions"]
    q_pos = example["question_positions"]
    ans_pos = example["answer_position"]
    mean_c = dataset.get_mean_correct(vis_pos, q_pos, ans_pos)
    
    for i, layer_module in enumerate(tqdm(model.layers, desc="Propagating layers", leave=False)):
        mean_c_layer = mean_c[i]
        rel, irrel = prop_llama_layer(
            rel, irrel, layer_module, full_mask, position_ids, i, ablated_nodes, mean_c_layer
        )
        
    rel_final, irrel_final = prop_rmsnorm(rel, irrel, model.norm)
    return rel_final, irrel_final

def score_nodes(dataset, source_nodes, current_circuit):
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

    for idx, example in tqdm(enumerate(dataset.hallucination_examples), total=len(dataset.hallucination_examples), desc="Scoring examples"):
        # Step 3
        hall_id = example["hallucinated_token_id"]
        corr_id = example["correct_token_id"]
        
        raw_d_i = wrapper.W_U[hall_id] - wrapper.W_U[corr_id]
        d_i = raw_d_i / torch.norm(raw_d_i, p=2) 
        ans_pos = example["answer_position"]
        vis_pos = example["visual_positions"]
        q_pos = example["question_positions"]
        
        for node in tqdm(source_nodes, desc=f"Nodes (example {idx})", leave=False):
            # Ablate current_circuit + this candidate node
            ablated = current_circuit + [node]
            rel_h, irrel_h = prop_LLaVA(dataset, example, ablated)
            
            # Extract final residual stream exactly at answer token
            rel_final_i = rel_h[0, ans_pos, :]
            irrel_final_i = irrel_h[0, ans_pos, :]
            
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

def compute_faithfulness(dataset, circuit):
    """
    Step 9: Computes the faithfulness of the discovered circuit.
    m(C) = mean over examples i [d_i . h_final_C_i]
    """
    wrapper = dataset.wrapper
    m_c_sum = 0
    m_c_correct_sum = 0
    
    for i, example in tqdm(enumerate(dataset.hallucination_examples), total=len(dataset.hallucination_examples), desc="Computing faithfulness"):
        hall_id = example["hallucinated_token_id"]
        corr_id = example["correct_token_id"]
        
        d_i_hall = wrapper.W_U[hall_id] - wrapper.W_U[corr_id]
        d_i_hall = d_i_hall / torch.norm(d_i_hall, p=2) 
        
        d_i_corr = wrapper.W_U[corr_id] - wrapper.W_U[hall_id]
        d_i_corr = d_i_corr / torch.norm(d_i_corr, p=2) 
        
        # When only heads in circuit C are active (all others mean-ablated)
        # Wait, the prompt says: "h_final_C_i is the final residual stream when only heads 
        # in circuit C are active (all others mean-ablated)."
        # So we construct an ablation list containing ALL heads NOT in C.
        all_heads = [] # TODO: implement all heads 
        
        # Then run prop_LLaVA using all_heads_except_C as ablated_nodes
        # Since this involves all heads not in C, we need to gather them.
        pass
    
    return m_c_sum / max(1, len(dataset.hallucination_examples))
