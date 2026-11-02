import argparse
import json
import torch
import os
from tqdm import tqdm

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from pyfunctions.nodes import Node
from models.llava_wrapper import LlavaWrapper
from tasks.hallucination_vlm import HallucinationDataset

def evaluate_circuit(args):
    print(f"Loading wrapper for {args.model_id} on {args.device}...")
    wrapper = LlavaWrapper(model_id=args.model_id, device=args.device)
    
    print(f"Loading evaluation dataset {args.eval_dataset}...")
    with open(args.eval_dataset, "r") as f:
        eval_data = json.load(f)
        
    dataset = HallucinationDataset(eval_data, wrapper, None, mean_cache_path=args.mean_activations_path)
    
    print(f"Loading circuit from {args.circuit_path}...")
    with open(args.circuit_path, "r") as f:
        circuit_results = json.load(f)
        
    from methods.faithfulness import evaluate_faithfulness
    circuit = [Node(layer_idx=int(n["layer"]), sequence_idx=int(n.get("seq_pos", -1)),
                    attn_head_idx=int(n["head"])) for n in circuit_results.get("circuit", [])]
    if not circuit:
        raise ValueError("Circuit must not be empty")
    result = evaluate_faithfulness(dataset, circuit, yes_no=args.yes_no)
    from pathlib import Path
    path = Path(args.circuit_path)
    output = path.with_name(path.stem + "_eval.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit_path", type=str, required=True, help="Path to hallucination_circuit_results.json")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Path to dataset JSON for evaluation")
    parser.add_argument("--mean_activations_path", type=str, required=True, help="Path to precomputed mean activations .pt file")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--yes_no", action="store_true", help="Use hardcoded Yes/No target direction")
    
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    evaluate_circuit(args)
