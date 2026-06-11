import os
import json
import torch
import warnings
from transformers import pipeline
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

def extract_hallucination_data(
    max_samples=20,
    batch_size=16,
    output_dir="data/hallucinations", 
    correct_dir="data/correct",
    dataset_name="lmms-lab/POPE", 
    dataset_split="test",
    model_id="llava-hf/llava-1.5-7b-hf"
):
    """
    Finds hallucination samples (where the model predicts Yes but GT is No) 
    and correct samples (where model predicts No and GT is No) 
    to use in the CD pipeline.
    
    Saves images and schema-compatible JSON metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)
    
    print(f"Loading pipeline for {model_id}...")
    pipe = pipeline(
        "image-text-to-text", 
        model=model_id, 
        device=0,                
        torch_dtype=torch.float16,
        model_kwargs={"low_cpu_mem_usage": True}
    )

    print(f"Loading dataset {dataset_name}...")
    ds = load_dataset(dataset_name, "default")
    split_ds = ds[dataset_split]

    hallucinations = []
    correct_samples = []

    print(f"Scanning for {max_samples} hallucinations with batch_size={batch_size}")
    
    batch_messages = []
    batch_indices = []
    
    for i in tqdm(range(len(split_ds)), desc="Scanning dataset"):
        if len(hallucinations) >= max_samples:
            break
            
        example = split_ds[i]
        image = example['image'] # PIL Image
        question = example['question']
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        batch_messages.append(messages)
        batch_indices.append(i)
        
        # When batch is full or we reached the end of the dataset
        if len(batch_messages) == batch_size or i == len(split_ds) - 1:
            
            # Generate answers from model (automatically parallelized inference over the batch)
            outs = pipe(text=batch_messages, batch_size=len(batch_messages), max_new_tokens=10, return_full_text=False)
            
            for b_idx, out in zip(batch_indices, outs):
                if len(hallucinations) >= max_samples:
                    break
                    
                example_b = split_ds[b_idx]
                image_b = example_b['image']
                question_b = example_b['question']
                gt_answer_b = example_b['answer']
                gt_bool = 1 if 'yes' in str(gt_answer_b).lower() else 0
                
                if isinstance(out, list):
                    model_outp = out[0]['generated_text'].strip()
                else:
                    model_outp = out['generated_text'].strip()
                
                model_outp_lower = model_outp.lower()
                if model_outp_lower.startswith('y'):
                    pred_bool = 1
                elif model_outp_lower.startswith('n'):
                    pred_bool = 0
                else:
                    continue
        
                # Condition 1: Hallucination -> Model says Yes (1), but GT is No (0)
                if pred_bool == 1 and gt_bool == 0 and len(hallucinations) < max_samples:
                    img_filename = f"hal_{len(hallucinations)}.jpg"
                    img_path = os.path.join(output_dir, img_filename)
                    if isinstance(image_b, str):
                        image_b = Image.open(image_b)
                    image_b.convert('RGB').save(img_path)
                    
                    sample_info = {
                        "id": b_idx,
                        "image": img_path,
                        "question": question_b,
                        "model_output": model_outp,
                        "gt_output": str(gt_answer_b)
                    }
                    hallucinations.append(sample_info)
                    print(f"Found Hallucination {len(hallucinations)}/{max_samples} at Index {b_idx}")
                    print(f"  Q: {question_b} | Model: {model_outp}")
            
            # Reset batch buffers
            batch_messages = []
            batch_indices = []
    
    # Save to JSONs
    hallucination_json_path = os.path.join(output_dir, "hallucinations.json")
    with open(hallucination_json_path, "w") as f:
        json.dump(hallucinations, f, indent=4)
        
    print(f"\nExtraction complete!")
    print(f"Saved {len(hallucinations)} hallucinations to {hallucination_json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="data/hallucinations")
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/POPE")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    
    args = parser.parse_args()
    extract_hallucination_data(
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model_id=args.model_id
    )
