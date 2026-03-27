import os
import json
import torch
import warnings
from transformers import pipeline
from datasets import load_dataset

warnings.filterwarnings("ignore")

def extract_hallucination_data(
    max_samples=20, 
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

    print(f"Scanning for {max_samples} hallucinations")
    for i in range(len(split_ds)):
        if len(hallucinations) >= max_samples:
            break
            
        example = split_ds[i]
        image = example['image'] # PIL Image
        question = example['question']
        gt_answer = example['answer'] # 'yes' or 'no'
        
        gt_bool = 1 if 'yes' in str(gt_answer).lower() else 0
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        # Generate answer from model
        out = pipe(text=messages, max_new_tokens=10, return_full_text=False)
        model_outp = out[0]['generated_text'].strip()
        
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
            image.convert('RGB').save(img_path)
            
            # Format compatible with HallucinationDataset
            sample_info = {
                "id": i,
                "image": img_path,
                "question": question,
                "model_output": model_outp,
                "gt_output": str(gt_answer)
            }
            hallucinations.append(sample_info)
            print(f"Found Hallucination {len(hallucinations)}/{max_samples} at Index {i}")
            print(f"  Q: {question} | Model: {model_outp}")
    
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
    parser.add_argument("--output_dir", type=str, default="data/hallucinations")
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/POPE")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    
    args = parser.parse_args()
    extract_hallucination_data(
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model_id=args.model_id
    )
