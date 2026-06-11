import argparse
import torch
import json
import os
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.qwen3_thinking_wrapper import Qwen3ThinkingWrapper

def store_mean_activations(args):
    """
    Step 1: Compute and store mean activations over all attention heads and MLPs
    for the entire (or subset) background dataset without encountering OOM issues.
    """
    print(f"Loading wrapper for {args.model_id} on {args.device}...")
    wrapper = Qwen3ThinkingWrapper(model_id=args.model_id, device=args.device)
    
    print(f"Loading dataset {args.dataset_name} ({args.dataset_split})...")
    ds = load_dataset(args.dataset_name, "default")
    split_ds = ds[args.dataset_split]
    
    num_layers = wrapper.num_layers
    num_heads = wrapper.num_heads
    head_dim = wrapper.head_dim
    
    # Track sum of activations and count of samples per offset
    # Shapes will be: (num_layers, num_heads, head_dim)
    # We use fp32 for accumulation to avoid precision loss / overflow over thousands of samples
    visual_sums = defaultdict(lambda: torch.zeros((num_layers, num_heads, head_dim), dtype=torch.float32, device=args.device))
    question_sums = defaultdict(lambda: torch.zeros((num_layers, num_heads, head_dim), dtype=torch.float32, device=args.device))
    answer_sum = torch.zeros((num_layers, num_heads, head_dim), dtype=torch.float32, device=args.device)
    
    visual_counts = defaultdict(int)
    question_counts = defaultdict(int)
    answer_count = 0
    
    total_samples = min(args.max_samples, len(split_ds)) if args.max_samples > 0 else len(split_ds)
    
    print(f"Processing {total_samples} samples to compute mean activations...")
    
    for i in tqdm(range(total_samples), desc="Computing activations"):
        item = split_ds[i]
        
        img = item["image"]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            continue 
            
        question = item["question"]
            
        with torch.no_grad():
            gen_token_id, logits, _ = wrapper.run_greedy(img, question)
            acts = wrapper.get_activations() # Dict: layer -> shape (batch=1, seq, heads, d_head)
            
        vis_pos = wrapper.visual_positions
        q_pos = wrapper.question_positions
        ans_pos = wrapper.answer_position
        
        # Accumulate answer
        # acts[l] shape: (1, seq, n_heads, d_head) -> acts[l][0, ans_pos] shape: (n_heads, d_head)
        for l in range(num_layers):
            answer_sum[l] += acts[l][0, ans_pos].to(torch.float32).to(args.device)
        answer_count += 1
        
        # Accumulate visual
        for idx_v in range(len(vis_pos)):
            seq_idx = vis_pos[idx_v]
            for l in range(num_layers):
                visual_sums[idx_v][l] += acts[l][0, seq_idx].to(torch.float32).to(args.device)
            visual_counts[idx_v] += 1
            
        # Accumulate question
        for idx_q in range(len(q_pos)):
            seq_idx = q_pos[idx_q]
            for l in range(num_layers):
                question_sums[idx_q][l] += acts[l][0, seq_idx].to(torch.float32).to(args.device)
            question_counts[idx_q] += 1
            

            
    print("Computing final means...")
    # Convert sums back to fp16 and divide by count
    visual_means = {}
    for idx_v, s_tensor in visual_sums.items():
        visual_means[idx_v] = (s_tensor / visual_counts[idx_v]).to(torch.float16).cpu()
        
    question_means = {}
    for idx_q, s_tensor in question_sums.items():
        question_means[idx_q] = (s_tensor / question_counts[idx_q]).to(torch.float16).cpu()
        
    avg_answer = (answer_sum / max(1, answer_count)).to(torch.float16).cpu()
    
    # Store everything together
    mean_cache = {
        "avg_visual": visual_means,
        "avg_question": question_means,
        "avg_answer": avg_answer
    }
    
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    torch.save(mean_cache, args.output_path)
    print(f"Successfully saved mean activations to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/POPE")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to process. -1 for all.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3Thinking")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the resulting .pt file")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    store_mean_activations(args)
