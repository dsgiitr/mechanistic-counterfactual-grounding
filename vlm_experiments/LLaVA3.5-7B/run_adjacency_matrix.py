import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import torch
from pyfunctions.nodes import Node
from models.llava_wrapper import LlavaWrapper
from tasks.hallucination_vlm import HallucinationDataset
from methods.hallucination_circuit import build_adjacency_matrix

def run_adjacency_matrix(args):
    """
    Builds the native decoder head adjacency matrix and saves it to disk.
    This is the heavy GPU pass; circuit discovery happens offline afterward.
    """
    print(f"Loading wrapper for {args.model_id} on {args.device}...")
    wrapper = LlavaWrapper(model_id=args.model_id, device=args.device)
    
    print(f"Loading datasets...")
    with open(args.dataset_path, "r") as f:
        hall_data = json.load(f)
        
    corr_data = None
    if args.correct_path:
        with open(args.correct_path, "r") as f:
            corr_data = json.load(f)
            
    dataset = HallucinationDataset(hall_data, wrapper, corr_data, mean_cache_path=args.mean_activations_path)
    
    # Generate all 1024 source nodes (32 layers x 32 heads, sequence_idx=-1 for full head)
    print("Preparing source nodes...")
    num_layers = wrapper.num_layers
    num_heads = wrapper.num_heads
    
    source_nodes = []
    for l in range(num_layers):
        for h in range(num_heads):
            source_nodes.append(Node(layer_idx=l, sequence_idx=-1, attn_head_idx=h))
                
    print(f"Generated {len(source_nodes)} source nodes. Building {len(source_nodes)}x{len(source_nodes)} adjacency matrix...")
    
    adj = build_adjacency_matrix(dataset, source_nodes, yes_no=args.yes_no, batch_size=args.batch_size)
    
    # Save entire adjacency data
    output_path = args.output
    adj.update({"num_heads": wrapper.num_heads, "num_layers": wrapper.num_layers, "model_id": args.model_id})
    torch.save(adj, output_path)
    print(f"Adjacency matrix saved to {output_path}")
    
    # Print summary statistics
    N = num_layers * num_heads
    head_combined = adj["head_combined"]
    logit_combined = adj["logit_combined"]
    
    nonzero_edges = (head_combined > 0).sum().item()
    total_possible = N * N
    print(f"\n--- Summary ---")
    print(f"Matrix size: {N}x{N} = {total_possible} entries")
    print(f"Non-zero edges (causal): {nonzero_edges}")
    print(f"Top 10 logit-targeting heads:")
    top_logit = torch.argsort(logit_combined, descending=True)[:10]
    for rank, idx in enumerate(top_logit):
        layer = idx.item() // num_heads
        head = idx.item() % num_heads
        score = logit_combined[idx].item()
        beta = adj["logit_mean_beta"][idx].item()
        label = "promoter" if beta > 0 else "suppressor"
        print(f"  #{rank+1}: L{layer}H{head} combined={score:.4f} ({label})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the native decoder CD-T adjacency matrix")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to JSON file of hallucination examples")
    parser.add_argument("--correct_path", type=str, default=None, help="Path to JSON file of correct examples")
    parser.add_argument("--mean_activations_path", type=str, default=None, help="Path to precomputed mean activations .pt file")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="adjacency_matrix.pt", help="Output file for adjacency matrix")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for source node evaluation")
    parser.add_argument("--yes_no", action="store_true", help="Use hardcoded Yes/No target direction")
    
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    run_adjacency_matrix(args)
