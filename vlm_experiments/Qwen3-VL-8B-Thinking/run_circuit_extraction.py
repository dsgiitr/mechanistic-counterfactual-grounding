import os
import json
import torch
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
    print(f"Using yes/no tokens: '{yes_text}' -> {yes_id}, '{no_text}' -> {no_id}")
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

def register_circuit_hooks(backbone, circuit_nodes, head_dim=128):
    """
    Registers forward hooks to extract specific representations:
    1. Final layer norm
    2. Specific attention heads dynamically parsed from circuit_iterative_decay_k50.json
    """
    features = {}
    handles = []
    
    # 1. Final hidden state (from the final layer norm before LM head)
    def get_final_hook(name):
        def hook(model, input, output):
            tensor_out = output[0] if isinstance(output, tuple) else output
            features[name] = tensor_out[:, -1, :].detach().cpu()
        return hook

    handles.append(
        backbone.norm.register_forward_hook(get_final_hook("final_layer_norm"))
    )
    
    # 2. Extract Heads specific to the circuit nodes
    layer_to_heads = {}
    for node in circuit_nodes:
        l, h = node["layer"], node["head"]
        if l not in layer_to_heads:
            layer_to_heads[l] = []
        layer_to_heads[l].append(h)
        
    num_targeted_heads = sum(len(hs) for hs in layer_to_heads.values())
    print(f"Hooking {num_targeted_heads} attention heads across {len(layer_to_heads)} unique layers.")
    
    def get_o_proj_pre_hook(layer_idx, head_indices):
        def pre_hook(module, args):
            # args[0] is the input to o_proj, i.e., concatenated head outputs.
            # shape: (batch_size, seq_len, hidden_size) where hidden_size = num_heads * head_dim
            tensor_in = args[0]
            
            # We want the last token representation: shape becomes (batch_size, hidden_size)
            last_token_states = tensor_in[:, -1, :]
            
            for h in head_indices:
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                # Slice the exact head's dimensions
                features[f"layer_{layer_idx}_head_{h}"] = last_token_states[:, start_idx:end_idx].detach().cpu()
        return pre_hook
        
    for layer_idx, head_indices in layer_to_heads.items():
        handles.append(
            backbone.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                get_o_proj_pre_hook(layer_idx, head_indices)
            )
        )
        
    return features, handles

def main():
    model_id = "Qwen/Qwen3Thinking"
    print(f"Loading {model_id}...")
    
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
    backbone = get_llm_backbone(model)
    
    # Path Resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Load Circuit
    circuit_json_path = os.path.join(base_dir, "circuit_iterative_decay_k50.json")
    with open(circuit_json_path, 'r') as f:
        circuit_data = json.load(f)
        circuit_nodes = circuit_data.get("circuit", [])
        
    # We assume LLaVA head_dim = 4096 / 32 = 128
    features, handles = register_circuit_hooks(backbone, circuit_nodes, head_dim=128)
    
    # Load Questions
    base_image_dir = os.path.join(base_dir, "images")
    json_path = os.path.join(base_image_dir, "counterfactual_questions.json")
    with open(json_path, 'r') as f:
        questions_data = json.load(f)

    # Output Directories
    out_dir = os.path.join(base_dir, "experiments", "results", "circuit_vlm_deltas")
    behavior_dir = os.path.join(base_dir, "experiments", "results", "circuit_vlm_behavior")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(behavior_dir, exist_ok=True)
    behavior_records = []
    
    for example_folder, data in questions_data.items():
        # Check if corresponding images exist, if not gracefully skip
        try:
            original_img_path, cf_img_paths = get_image_paths(base_image_dir, example_folder)
        except FileNotFoundError:
            print(f"Skipping {example_folder} (folder missing)")
            continue
            
        print(f"\nProcessing {example_folder}...")
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

    behavior_jsonl = os.path.join(behavior_dir, "circuit_behavior_metrics.jsonl")
    with open(behavior_jsonl, "w", encoding="utf-8") as f:
        for rec in behavior_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nCompleted! Saved all behavioral metrics to: {behavior_jsonl}")

    # Clean up hooks to avoid memory leaks
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()
