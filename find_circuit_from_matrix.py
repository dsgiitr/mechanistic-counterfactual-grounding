"""
Offline circuit discovery from a precomputed adjacency matrix.
No GPU or model loading required — runs instantly on CPU.

Usage:
    python find_circuit_from_matrix.py --matrix_path adjacency_matrix.pt \\
        --num_outliers 2 --output circuit_result.json
"""

import argparse
import json
import torch


def find_circuit(adj, num_layers=32, num_heads=32, num_outliers=2):
    """
    Performs iterative backward search on the precomputed adjacency matrix.
    
    Algorithm:
        1. Find the top K logit-targeting heads.
        2. Set them as target_nodes.
        3. While any target_node is not in layer 0:
            a. For each source node, sum its edge scores to all current target_nodes.
            b. Select the top K source nodes.
            c. Add them to the circuit.
            d. Replace target_nodes with the newly selected nodes.
    
    Returns:
        circuit: list of dicts with layer, head, iteration, label
        iterations: list of lists showing what was selected per iteration
    """
    N = num_layers * num_heads
    head_combined = adj["head_combined"]  # (N, N)
    logit_combined = adj["logit_combined"]  # (N,)
    logit_mean_beta = adj["logit_mean_beta"]  # (N,)
    head_mean_beta = adj["head_mean_beta"]  # (N, N)
    
    def idx_to_node(idx):
        return (idx // num_heads, idx % num_heads)
    
    def node_to_idx(layer, head):
        return layer * num_heads + head
    
    # --- Iteration 0: Find top K heads targeting logits ---
    # Only consider nodes that are hallucination promoters (beta > 0)
    logit_scores_masked = logit_combined.clone()
    logit_scores_masked[logit_mean_beta <= 0] = -1.0
    
    top_logit_indices = torch.argsort(logit_scores_masked, descending=True)[:num_outliers]
    
    circuit = []
    iterations = []
    seen = set()
    
    iter_0_nodes = []
    for idx in top_logit_indices:
        layer, head = idx_to_node(idx.item())
        node_key = (layer, head)
        if node_key not in seen:
            seen.add(node_key)
            beta = logit_mean_beta[idx].item()
            label = "hallucination promoter" if beta > 0 else "grounding suppressor"
            circuit.append({
                "layer": layer,
                "head": head,
                "iteration": 0,
                "target": "logits",
                "combined_score": logit_combined[idx].item(),
                "mean_beta": beta,
                "label": label,
            })
            iter_0_nodes.append(node_key)
    
    iterations.append(iter_0_nodes)
    current_target_indices = [idx.item() for idx in top_logit_indices]
    
    print(f"Iteration 0 (target=logits): {[f'L{l}H{h}' for l,h in iter_0_nodes]}")
    
    # --- Iterative backward search ---
    iteration = 1
    while True:
        # Stopping condition: all current targets are at layer 0
        all_at_layer_0 = all(idx // num_heads == 0 for idx in current_target_indices)
        if all_at_layer_0:
            print(f"All targets at layer 0. Stopping.")
            break
        
        # For each candidate source node, sum its edge scores to ALL current target nodes
        aggregated_scores = torch.zeros(N)
        aggregated_betas = torch.zeros(N)
        for tgt_idx in current_target_indices:
            aggregated_scores += head_combined[:, tgt_idx]
            aggregated_betas += head_mean_beta[:, tgt_idx]
            
        # Mask out non-promoter nodes
        aggregated_scores[aggregated_betas <= 0] = -1.0
        
        # Exclude nodes already in the circuit
        for node_key in seen:
            aggregated_scores[node_to_idx(*node_key)] = -1.0
        
        # Select top K
        top_indices = torch.argsort(aggregated_scores, descending=True)[:num_outliers]
        
        iter_nodes = []
        new_target_indices = []
        for idx in top_indices:
            score = aggregated_scores[idx].item()
            if score <= 0:
                continue  # No valid edges left
            layer, head = idx_to_node(idx.item())
            node_key = (layer, head)
            if node_key not in seen:
                seen.add(node_key)
                # Compute aggregate beta sign across all targets
                total_beta = sum(head_mean_beta[idx.item(), tgt_idx].item() for tgt_idx in current_target_indices)
                label = "hallucination promoter" if total_beta > 0 else "grounding suppressor"
                target_str = ", ".join(f"L{t // num_heads}H{t % num_heads}" for t in current_target_indices)
                circuit.append({
                    "layer": layer,
                    "head": head,
                    "iteration": iteration,
                    "target": target_str,
                    "combined_score": score,
                    "mean_beta": total_beta,
                    "label": label,
                })
                iter_nodes.append(node_key)
                new_target_indices.append(idx.item())
        
        if not iter_nodes:
            print(f"Iteration {iteration}: No valid nodes found. Stopping.")
            break
            
        iterations.append(iter_nodes)
        print(f"Iteration {iteration} (target={[f'L{t // num_heads}H{t % num_heads}' for t in current_target_indices]}): "
              f"Selected {[f'L{l}H{h}' for l,h in iter_nodes]}")
        
        # REPLACE targets with newly found nodes
        current_target_indices = new_target_indices
        iteration += 1
    
    return circuit, iterations


def main(args):
    print(f"Loading adjacency matrix from {args.matrix_path}...")
    adj = torch.load(args.matrix_path, map_location="cpu")
    
    # Infer dimensions from the matrix
    N = adj["head_combined"].shape[0]
    num_heads = args.num_heads
    num_layers = N // num_heads
    
    print(f"Matrix size: {N} nodes ({num_layers} layers x {num_heads} heads)")
    print(f"Outliers per iteration: {args.num_outliers}")
    print()
    
    circuit, iterations = find_circuit(
        adj, 
        num_layers=num_layers, 
        num_heads=num_heads,
        num_outliers=args.num_outliers,
    )
    
    print(f"\n--- Final Circuit ({len(circuit)} nodes) ---")
    for node in circuit:
        print(f"  L{node['layer']}H{node['head']} iter={node['iteration']} "
              f"score={node['combined_score']:.4f} {node['label']} -> {node['target']}")
    
    result = {
        "circuit": circuit,
        "iterations": [[{"layer": l, "head": h} for l, h in it] for it in iterations],
        "params": {
            "num_outliers": args.num_outliers,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nCircuit saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find circuit from precomputed adjacency matrix (no GPU needed)")
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to adjacency_matrix.pt")
    parser.add_argument("--num_outliers", type=int, default=2, help="Number of top nodes to keep per iteration")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads per layer")
    parser.add_argument("--output", type=str, default="circuit_result.json", help="Output JSON file")
    
    args = parser.parse_args()
    main(args)
