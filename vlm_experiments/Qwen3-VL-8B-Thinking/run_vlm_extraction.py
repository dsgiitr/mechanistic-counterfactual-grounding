import os
import json
import torch
import random
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def get_llm_backbone(model):
    """Return the LLaMA backbone across Transformers layout variants."""
    language_model = model.language_model
    if hasattr(language_model, "model"):
        return language_model.model
    return language_model


def get_input_device(model):
    """Pick a valid input device when model is sharded/offloaded by Accelerate."""
    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
        for dev in model.hf_device_map.values():
            if isinstance(dev, str):
                return torch.device(dev)
    return next(model.parameters()).device


def move_inputs_to_device(inputs, device, model_dtype):
    """Move processor outputs to the model input device without breaking integer tensors."""
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if torch.is_floating_point(v):
                moved[k] = v.to(device=device, dtype=model_dtype)
            else:
                moved[k] = v.to(device=device)
        else:
            moved[k] = v
    return moved


def first_token_id(tokenizer, candidates):
    """Return the first valid single-token id from candidate strings."""
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], text
    raise RuntimeError(
        "Could not find single-token encoding for candidates: " + ", ".join(candidates)
    )


def get_yes_no_token_ids(processor):
    tokenizer = processor.tokenizer
    yes_id, yes_text = first_token_id(tokenizer, [" yes", "Yes", "yes"])
    no_id, no_text = first_token_id(tokenizer, [" no", "No", "no"])
    print(
        f"Using yes/no tokens: '{yes_text}' -> {yes_id}, '{no_text}' -> {no_id}"
    )
    return yes_id, no_id


def forward_with_features_and_margin(model, inputs, features, yes_token_id, no_token_id):
    with torch.no_grad():
        features.clear()
        outputs = model(**inputs)
        feat_dict = {k: v.clone() for k, v in features.items()}

        # Next-token logits at the assistant step.
        last_logits = outputs.logits[:, -1, :]
        yes_logit = float(last_logits[0, yes_token_id].detach().cpu().item())
        no_logit = float(last_logits[0, no_token_id].detach().cpu().item())
        margin = yes_logit - no_logit

    return feat_dict, yes_logit, no_logit, margin

def get_image_paths(base_dir, example_folder):
    """Dynamically finds the original and all counterfactual images in a folder."""
    folder_path = os.path.join(base_dir, example_folder)
    files = os.listdir(folder_path)
    
    original = None
    counterfactuals = []
    
    for f in files:
        if f.startswith("original"):
            original = os.path.join(folder_path, f)
        elif f.startswith("counterfactual"):
            counterfactuals.append(os.path.join(folder_path, f))
    counterfactuals.sort()
    return original, counterfactuals

def register_hooks(backbone, num_layers=32):
    """
    Registers forward hooks to extract specific representations:
    1. Final layer
    2. 5 random Attention heads (outputs)
    3. 5 random MLPs
    """
    features = {}
    
    def get_hook(name):
        def hook(model, input, output):
            # output is typically a tuple for some layers, tensor for others.
            # We want the main output tensor. Usually it's output[0] if it's a tuple.
            tensor_out = output[0] if isinstance(output, tuple) else output
            
            # We are interested in the final token's representation for autoregressive 
            # models since it aggregates the image + prompt context for the next generation.
            # shape: (batch, seq_len, hidden_dim). Select seq_len = -1
            features[name] = tensor_out[:, -1, :].detach().cpu()
        return hook

    handles = []
    
    # 1. Final hidden state (from the final layer norm before LM head)
    # Different versions of LLaVA place the backbone at either
    # model.language_model.model or model.language_model.
    handles.append(
        backbone.norm.register_forward_hook(get_hook("final_layer_norm"))
    )
    
    # Randomly select 5 layers for Attn and MLP
    random.seed(42) # for reproducibility
    n_select = min(5, num_layers)
    attn_layers = random.sample(range(num_layers), n_select)
    mlp_layers = random.sample(range(num_layers), n_select)
    
    print(f"Hooking Attention layers: {attn_layers}")
    print(f"Hooking MLP layers: {mlp_layers}")
    
    for i in attn_layers:
        handles.append(
            backbone.layers[i].self_attn.register_forward_hook(get_hook(f"attn_layer_{i}"))
        )
        
    for i in mlp_layers:
        handles.append(
            backbone.layers[i].mlp.register_forward_hook(get_hook(f"mlp_layer_{i}"))
        )
        
    return features, handles

def main():
    model_id = "Qwen/Qwen3Thinking"
    print(f"Loading {model_id}...")
    
    # Using bitsandbytes for 8-bit loading could save memory, but we keep it standard here.
    # User will run this over lightning later, potentially on multi-GPU.
    processor = AutoProcessor.from_pretrained(model_id)
    yes_token_id, no_token_id = get_yes_no_token_ids(processor)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        device_map="auto", 
        dtype=torch.float16
    )
    model.eval()
    model_dtype = next(model.parameters()).dtype
    input_device = get_input_device(model)
    
    # Discover number of layers dynamically
    backbone = get_llm_backbone(model)
    num_layers = len(backbone.layers)
    features, handles = register_hooks(backbone, num_layers)
    
    # Load JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_image_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "images"))
    json_path = os.path.join(base_image_dir, "counterfactual_questions.json")
    with open(json_path, 'r') as f:
        questions_data = json.load(f)

    # Output dir
    out_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "vlm_deltas"))
    behavior_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "vlm_behavior"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(behavior_dir, exist_ok=True)
    behavior_records = []
    
    for example_folder, data in questions_data.items():
        print(f"\nProcessing {example_folder}...")
        original_img_path, cf_img_paths = get_image_paths(base_image_dir, example_folder)
        
        prompt = f"USER: <image>\n{data['question']} {data['prompt']}\nASSISTANT:"
        
        # 1. Process Original Image
        orig_image = Image.open(original_img_path).convert("RGB")
        inputs_orig = processor(text=prompt, images=orig_image, return_tensors="pt")
        inputs_orig = move_inputs_to_device(inputs_orig, input_device, model_dtype)
        orig_features, orig_yes_logit, orig_no_logit, orig_margin = forward_with_features_and_margin(
            model,
            inputs_orig,
            features,
            yes_token_id,
            no_token_id,
        )
            
        # 2. Process Counterfactuals and compute Deltas
        for cf_idx, cf_path in enumerate(cf_img_paths):
            cf_image = Image.open(cf_path).convert("RGB")
            inputs_cf = processor(text=prompt, images=cf_image, return_tensors="pt")
            inputs_cf = move_inputs_to_device(inputs_cf, input_device, model_dtype)
            cf_features, cf_yes_logit, cf_no_logit, cf_margin = forward_with_features_and_margin(
                model,
                inputs_cf,
                features,
                yes_token_id,
                no_token_id,
            )
                
            # Compute Deltas
            deltas = {}
            for layer_name in orig_features.keys():
                delta_tensor = orig_features[layer_name] - cf_features[layer_name]
                deltas[layer_name] = delta_tensor
                
            # Save Deltas
            delta_file_name = f"{example_folder}_cf_{cf_idx}_deltas.pt"
            out_file = os.path.join(out_dir, delta_file_name)
            torch.save(deltas, out_file)
            print(f"Saved deltas to {out_file}")

            # Save behavior record aligned with this delta file.
            behavior_records.append(
                {
                    "delta_file": delta_file_name,
                    "example": example_folder,
                    "counterfactual_index": cf_idx,
                    "question": data["question"],
                    "prompt_constraint": data["prompt"],
                    "orig_yes_logit": orig_yes_logit,
                    "orig_no_logit": orig_no_logit,
                    "orig_margin_yes_minus_no": orig_margin,
                    "cf_yes_logit": cf_yes_logit,
                    "cf_no_logit": cf_no_logit,
                    "cf_margin_yes_minus_no": cf_margin,
                    "margin_delta_cf_minus_orig": cf_margin - orig_margin,
                }
            )

    behavior_jsonl = os.path.join(behavior_dir, "behavior_metrics.jsonl")
    with open(behavior_jsonl, "w", encoding="utf-8") as f:
        for rec in behavior_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved behavior metrics to {behavior_jsonl}")

    # Remove hooks
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()
