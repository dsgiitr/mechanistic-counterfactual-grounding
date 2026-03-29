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
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading {model_id}...")
    
    # Using bitsandbytes for 8-bit loading could save memory, but we keep it standard here.
    # User will run this over lightning later, potentially on multi-GPU.
    processor = AutoProcessor.from_pretrained(model_id)
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
    os.makedirs(out_dir, exist_ok=True)
    
    for example_folder, data in questions_data.items():
        print(f"\nProcessing {example_folder}...")
        original_img_path, cf_img_paths = get_image_paths(base_image_dir, example_folder)
        
        prompt = f"USER: <image>\n{data['question']} {data['prompt']}\nASSISTANT:"
        
        # 1. Process Original Image
        orig_image = Image.open(original_img_path).convert("RGB")
        inputs_orig = processor(text=prompt, images=orig_image, return_tensors="pt")
        inputs_orig = move_inputs_to_device(inputs_orig, input_device, model_dtype)
        
        with torch.no_grad():
            features.clear() # clear previous hook outputs
            _ = model(**inputs_orig)
            orig_features = {k: v.clone() for k, v in features.items()}
            
        # 2. Process Counterfactuals and compute Deltas
        for cf_idx, cf_path in enumerate(cf_img_paths):
            cf_image = Image.open(cf_path).convert("RGB")
            inputs_cf = processor(text=prompt, images=cf_image, return_tensors="pt")
            inputs_cf = move_inputs_to_device(inputs_cf, input_device, model_dtype)
            
            with torch.no_grad():
                features.clear()
                _ = model(**inputs_cf)
                cf_features = {k: v.clone() for k, v in features.items()}
                
            # Compute Deltas
            deltas = {}
            for layer_name in orig_features.keys():
                delta_tensor = orig_features[layer_name] - cf_features[layer_name]
                deltas[layer_name] = delta_tensor
                
            # Save Deltas
            out_file = os.path.join(out_dir, f"{example_folder}_cf_{cf_idx}_deltas.pt")
            torch.save(deltas, out_file)
            print(f"Saved deltas to {out_file}")

    # Remove hooks
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()
