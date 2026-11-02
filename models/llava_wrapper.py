import torch
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration
from collections import defaultdict
import gc

class LlavaWrapper:
    """
    Step 10: LLaVA-1.5 Wrapper for Hallucination Circuit Discovery
    """
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        self.device = device
        self.model_id = model_id
        self.dtype = torch.float32 if torch.device(device).type == "cpu" else torch.float16
        
        print(f"Loading {model_id} on {device} with dtype {self.dtype}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        
        self.tokenizer = self.processor.tokenizer
        # New transformers: LlavaForConditionalGeneration.model -> LlavaModel
        # LlavaModel.language_model -> LlamaModel (the decoder)
        # LlavaForConditionalGeneration.lm_head -> nn.Linear (moved out)
        if hasattr(self.model, 'language_model'):
            # Older transformers layout
            self.language_model = self.model.language_model
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            # Newer transformers layout
            self.language_model = self.model.model.language_model
        else:
            raise AttributeError(
                "Cannot locate language_model in LlavaForConditionalGeneration. "
                f"Top-level children: {[n for n,_ in self.model.named_children()]}"
            )
        self.config = self.language_model.config
        
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        
        # W_U matrix for target node projection
        # In newer transformers, lm_head lives on the top-level model
        if hasattr(self.model, 'lm_head'):
            self.W_U = self.model.lm_head.weight.detach().clone()
        else:
            self.W_U = self.language_model.lm_head.weight.detach().clone()
        
        self.activations = {}
        self.hooks = []
        self._register_hooks()
        
        self.N_vis = 0
        self.N_q = 0
        self.visual_positions = []
        self.question_positions = []
        self.answer_position = -1

    def _register_hooks(self):
        """
        Registers forward pre-hooks on `o_proj` to capture head outputs.
        """
        def get_hook(layer_idx):
            def hook(module, args):
                hidden_states = args[0].detach()
                batch_size, seq_len, _ = hidden_states.shape
                reshaped = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
                self.activations[layer_idx] = reshaped
            return hook

        # In newer transformers, language_model IS the inner model (LlamaModel),
        # so layers are at language_model.layers, not language_model.model.layers
        layers = getattr(self.language_model, 'layers', None)
        if layers is None:
            layers = self.language_model.model.layers
        for i, layer in enumerate(layers):
            h = layer.self_attn.o_proj.register_forward_pre_hook(get_hook(i))
            self.hooks.append(h)
            
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_correct_token_id(self, answer_str):
        """
        Returns the FIRST token id of the answer string.
        """
        # Prefixing with a space because generations typically output ' word'
        # if the prompt ends with 'ASSISTANT:'
        tokens = self.tokenizer(" " + answer_str, add_special_tokens=False).input_ids
        if len(tokens) == 0:
            tokens = self.tokenizer(answer_str, add_special_tokens=False).input_ids
        return tokens[0] if len(tokens) > 0 else None

    def clear_activations(self):
        self.activations.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def get_prompt(self, question):
        return f"USER: <image>\n{question}\nASSISTANT:"

    def prepare_inputs(self, image, question, answer_prefix=""):
        inputs = self.processor(text=self.get_prompt(question) + answer_prefix,
                                images=image, return_tensors="pt")
        return {key: value.to(device=self.device, dtype=self.dtype)
                if torch.is_tensor(value) and torch.is_floating_point(value)
                else value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()}

    def run_greedy(self, image, question, answer_prefix="", answer_prefix_token_ids=None):
        """
        Runs a forward pass and generates 1 token greedily to verify hallucination.
        Returns (generated_token_id, logits_at_answer_pos).
        """
        self.clear_activations()
        prompt = self.get_prompt(question) + answer_prefix
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        if answer_prefix_token_ids is not None:
            # Exact generated token context takes precedence over decoded text.
            from models.native_utils import prepare_inputs
            inputs = prepare_inputs(self, image, question, answer_prefix_token_ids=answer_prefix_token_ids)
        
        # Float16 for image pixels
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
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
        
        text_input_ids = inputs["input_ids"][0]
        image_token_id = getattr(self.model.config, 'image_token_index', 32000)
        
        img_idx_tensor = torch.nonzero(text_input_ids == image_token_id, as_tuple=True)[0]
        if len(img_idx_tensor) > 0:
            image_token_start = img_idx_tensor[0].item()
            num_image_tokens_in_input = len(img_idx_tensor)
            
            if num_image_tokens_in_input == 1:
                # Old layout: single <image> placeholder expanded internally
                self.N_vis = expanded_seq_len - len(text_input_ids) + 1
            else:
                # New layout: all image tokens already in input_ids
                self.N_vis = num_image_tokens_in_input
            
            visual_start = image_token_start
            visual_end = visual_start + self.N_vis
            self.visual_positions = list(range(visual_start, visual_end))
            self.question_positions = list(range(0, visual_start)) + list(range(visual_end, expanded_seq_len - 1))
        else:
            self.visual_positions = []
            self.question_positions = list(range(0, expanded_seq_len - 1))
            self.N_vis = 0
            
        self.N_q = expanded_seq_len - 1
        self.answer_position = expanded_seq_len - 1
        
        return generated_token_id, logits_at_answer_pos, outputs.logits
        
    def get_activations(self):
        """
        Returns the captured attention head activations.
        """
        return self.activations
