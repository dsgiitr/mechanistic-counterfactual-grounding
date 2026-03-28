import argparse
import json
import torch
import os
from tqdm import tqdm

import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from pyfunctions.wrappers import Node
from models.llava_wrapper import LlavaWrapper
from tasks.hallucination_vlm import HallucinationDataset
from methods.hallucination_circuit import prop_LLaVA

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
        
    # Extract circuit nodes
    circuit_nodes = set()
    for item in circuit_results.get("circuit", []):
        circuit_nodes.add((item["layer"], item["head"], item["seq_pos"]))
        
    print(f"Loaded circuit with {len(circuit_nodes)} nodes.")
    
    # Generate list of all possible nodes to figure out non-circuit nodes
    num_layers = wrapper.num_layers
    num_heads = wrapper.num_heads
    
    # We dynamically figure out max seq length per example during the loop
    # or roughly generate all possible up to a large number.
    # To be safe, let's create non_circuit_nodes on the fly per example.
    
    faithfulness_ratios = []
    full_model_scores = []
    circuit_only_scores = []
    
    # Precompute fixed direction for Yes/No tasks
    if args.yes_no:
        yes_id = wrapper.get_correct_token_id("Yes")
        no_id = wrapper.get_correct_token_id("No")
        raw_d = wrapper.W_U[yes_id] - wrapper.W_U[no_id]
        fixed_d_i = raw_d / torch.norm(raw_d, p=2)
        print(f"Yes/No mode: yes_id={yes_id}, no_id={no_id}")
    
    for i, example in tqdm(enumerate(dataset.hallucination_examples), total=len(dataset.hallucination_examples), desc="Evaluating circuit"):
        ans_pos = example["answer_position"]
        vis_pos = example["visual_positions"]
        q_pos = example["question_positions"]
        
        if args.yes_no:
            d_i = fixed_d_i
        else:
            hall_id = example["hallucinated_token_id"]
            corr_id = example["correct_token_id"]
            raw_d_i = wrapper.W_U[hall_id] - wrapper.W_U[corr_id]
            d_i = raw_d_i / torch.norm(raw_d_i, p=2)
            
        # Get embeddings ONCE for the example
        mean_c = dataset.get_mean_correct(vis_pos, q_pos, ans_pos)
        
        prompt = wrapper.get_prompt(example["question"])
        inputs = wrapper.processor(text=prompt, images=example["image_loaded"], return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        inputs = {k: v.to(wrapper.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = wrapper.model(**inputs, output_hidden_states=True, return_dict=True)
            base_embeds = outputs.hidden_states[0].to(torch.float16)
        
        # 1. Run full model (0 ablated nodes, batch size 1)
        rel_full, _ = prop_LLaVA(dataset, [[]], base_embeds, mean_c)
        full_beta = torch.dot(d_i, rel_full[0, ans_pos, :]).item()
        full_model_scores.append(full_beta)
        
        # 2. Run circuit only (ablate everything NOT in circuit)
        # In head-level tracing, seq_pos is -1.
        non_circuit_nodes = []
        for l in range(num_layers):
            for h in range(num_heads):
                if (l, h, -1) not in circuit_nodes:
                    non_circuit_nodes.append(Node(layer_idx=l, sequence_idx=-1, attn_head_idx=h))
                        
        rel_C, _ = prop_LLaVA(dataset, [non_circuit_nodes], base_embeds, mean_c)
        C_beta = torch.dot(d_i, rel_C[0, ans_pos, :]).item()
        circuit_only_scores.append(C_beta)
        
        ratio = C_beta / (full_beta + 1e-8)
        faithfulness_ratios.append(ratio)
        
        print(f"Example {i}: Full Model Beta = {full_beta:.4f}, Circuit Beta = {C_beta:.4f}, Faithfulness = {ratio:.2%}")
        
    mean_full = sum(full_model_scores) / max(1, len(full_model_scores))
    mean_C = sum(circuit_only_scores) / max(1, len(circuit_only_scores))
    overall_faithfulness = mean_C / (mean_full + 1e-8)
    
    print("-" * 50)
    print(f"Overall Faithfulness (Mean C / Mean Full): {overall_faithfulness:.2%}")
    print(f"Average of Ratios: {sum(faithfulness_ratios)/max(1, len(faithfulness_ratios)):.2%}")
    
    # Save the evaluation
    out_dict = {
        "overall_faithfulness": overall_faithfulness,
        "average_of_ratios": sum(faithfulness_ratios)/max(1, len(faithfulness_ratios)),
        "metrics_per_example": faithfulness_ratios
    }
    
    out_file = args.circuit_path.replace(".json", "_eval.json")
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"Saved evaluation results to {out_file}")


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
