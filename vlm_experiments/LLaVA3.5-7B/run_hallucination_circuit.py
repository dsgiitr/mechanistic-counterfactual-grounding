import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import torch
from pyfunctions.nodes import Node
from models.llava_wrapper import LlavaWrapper
from tasks.hallucination_vlm import HallucinationDataset
from methods.hallucination_circuit import score_nodes, compute_faithfulness

def run_circuit_discovery(args):
    """
    Step 11: Main entry point for LLaVA Hallucination circuit discovery.
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
    
    # We will score ALL heads over ALL positions natively
    # To reduce computational burden if necessary, we can sample.
    print("Preparing source nodes...")
    num_layers = wrapper.num_layers
    num_heads = wrapper.num_heads
    
    # Here we define the search space. 
    # The CD-T framework usually ablated by (layer, head, pos).
    # Since we aggregate over sequence positions based on their segment,
    # we can construct nodes for every combination of layer and head, across all positions.
    source_nodes = []
    
    # HEAD-LEVEL DISCOVERY: Generate 1 node per attention head (32 x 32 = 1024 nodes)
    # We use sequence_idx=-1 to indicate the entire sequence should be considered.
    for l in range(num_layers):
        for h in range(num_heads):
            source_nodes.append(Node(layer_idx=l, sequence_idx=-1, attn_head_idx=h))
                
    print(f"Generated {len(source_nodes)} source nodes to evaluate. Beginning scoring...")
    
    scores = score_nodes(dataset, source_nodes, current_circuit=[], yes_no=args.yes_no, batch_size=args.batch_size)
    
    # Sort by combined score
    ranked_nodes = sorted(scores.items(), key=lambda x: x[1]["combined"], reverse=True)
    
    # Threshold top percentile
    if not 0 <= args.top_percentile < 100:
        raise ValueError("top_percentile must be in [0, 100)")
    num_to_keep = max(1, int(len(ranked_nodes) * (1 - args.top_percentile / 100.0)))
    circuit_nodes = [n for n, s in ranked_nodes[:num_to_keep]]
    
    print(f"Selected {len(circuit_nodes)} nodes for the circuit.")
    
    # Step 7: Sign labeling
    circuit_details = []
    for node in circuit_nodes:
        s = scores[node]
        mean_beta = s["mean_beta_T"]
        label = "hallucination promoter" if mean_beta > 0 else "grounding suppressor"
        circuit_details.append({
            "layer": node.layer_idx,
            "head": node.attn_head_idx,
            "seq_pos": node.sequence_idx,
            "label": label,
            "combined_score": s["combined"],
            "r_avg": s["r_avg"],
            "r_consistency": s["r_consistency"],
            "r_avg_visual": s["r_avg_visual"],
            "r_avg_question": s["r_avg_question"]
        })
        
    print("Computing faithfulness...")
    faithfulness = compute_faithfulness(dataset, circuit_nodes, yes_no=args.yes_no)
    
    result = {
        "ranked_heads_top_100": [{"layer": n.layer_idx, "head": n.attn_head_idx, "seq": n.sequence_idx, "score": s["combined"], "r_avg": s["r_avg"]} for n, s in ranked_nodes[:100]],
        "circuit": circuit_details,
        "faithfulness": faithfulness
    }
    
    output_path = getattr(args, "output", "hallucination_circuit_results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
        
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to JSON file of hallucination examples")
    parser.add_argument("--correct_path", type=str, default=None, help="Path to JSON file of correct examples")
    parser.add_argument("--mean_activations_path", type=str, default=None, help="Path to precomputed mean activations .pt file")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--top_percentile", type=float, default=95.0, help="Percentile threshold")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="hallucination_circuit_results.json", help="Output file for circuit results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluating source nodes")
    parser.add_argument("--yes_no", action="store_true", help="Use hardcoded Yes/No target direction d_i = W_U(Yes) - W_U(No)")
    
    args = parser.parse_args()
    
    # Ensure gradients are off globally as requested
    torch.set_grad_enabled(False)
    
    run_circuit_discovery(args)
