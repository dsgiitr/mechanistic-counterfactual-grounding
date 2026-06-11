import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import gc

class Qwen3InstructWrapper:
    """
    Wrapper for Qwen3-VL-8B-Instruct (Mocked with Qwen2VL architecture).
    """
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda"):
        self.device = device
        self.model_id = model_id
        
        print(f"Loading {model_id} on {device} in fp16...")
        # Since Qwen3-VL is a placeholder, we use Qwen2VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        
        self.tokenizer = self.processor.tokenizer
        
        self.language_model = self.model.model
        self.config = self.language_model.config
        
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        
        self.W_U = self.model.lm_head.weight.detach().clone()
        
        self.activations = {}
        self.hooks = []
        self._register_hooks()
        
        self.N_vis = 0
        self.N_q = 0
        self.visual_positions = []
        self.question_positions = []
        self.answer_position = -1

    def _register_hooks(self):
        def get_hook(layer_idx):
            def hook(module, args):
                hidden_states = args[0].detach()
                batch_size, seq_len, _ = hidden_states.shape
                reshaped = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
                self.activations[layer_idx] = reshaped
            return hook

        layers = self.language_model.layers
        for i, layer in enumerate(layers):
            h = layer.self_attn.o_proj.register_forward_pre_hook(get_hook(i))
            self.hooks.append(h)
            
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_correct_token_id(self, answer_str):
        tokens = self.tokenizer(" " + answer_str, add_special_tokens=False).input_ids
        if len(tokens) == 0:
            tokens = self.tokenizer(answer_str, add_special_tokens=False).input_ids
        return tokens[-1] if len(tokens) > 0 else None

    def clear_activations(self):
        self.activations.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def get_prompt(self, question):
        return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"

    def run_greedy(self, image, question):
        self.clear_activations()
        prompt = self.get_prompt(question)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(
                **inputs, 
                output_hidden_states=False, 
                output_attentions=False
            )
            
        logits = outputs.logits
        expanded_seq_len = logits.shape[1]
        logits_at_answer_pos = logits[0, -1, :]
        generated_token_id = logits_at_answer_pos.argmax().item()
        
        # Simplified position assignments for the mock
        self.visual_positions = []
        self.question_positions = list(range(0, expanded_seq_len - 1))
        self.N_vis = 0
            
        self.N_q = expanded_seq_len - 1
        self.answer_position = expanded_seq_len - 1
        
        return generated_token_id, logits_at_answer_pos, outputs.logits
        
    def get_activations(self):
        return self.activations
