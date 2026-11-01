"""Native Qwen3-VL Instruct/Thinking wrapper; no architecture substitution."""
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from methods.restoration_effects import decoder_backbone
from models.native_utils import capture_decoder_context, prepare_inputs


class Qwen3VLWrapper:
    def __init__(self, model_id, device="cuda", *, model=None, processor=None,
                 min_pixels=256 * 32 * 32, max_pixels=1280 * 32 * 32):
        self.device = device
        self.model_id = model_id
        self.dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(model_id)
        self.model = model if model is not None else Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype, attn_implementation="eager").to(device)
        if self.model.config.model_type != "qwen3_vl":
            raise ValueError("Qwen3VLWrapper requires a native Qwen3-VL model")
        self.model.eval()
        self.dtype = next(self.model.parameters()).dtype
        self.tokenizer = self.processor.tokenizer
        self.language_model = decoder_backbone(self.model)
        self.config = self.language_model.config
        self.num_layers = len(self.language_model.layers)
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.W_U = self.model.lm_head.weight.detach()
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.activations, self.hooks = {}, []
        self.visual_positions, self.question_positions = [], []
        self.answer_position, self.N_vis, self.N_q = -1, 0, 0
        self._register_hooks()

    def get_prompt(self, question):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def prepare_inputs(self, image, question, answer_prefix=""):
        inputs = self.processor(text=[self.get_prompt(question) + answer_prefix], images=[image],
                                return_tensors="pt", min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        return {key: value.to(device=self.device, dtype=self.dtype)
                if torch.is_tensor(value) and torch.is_floating_point(value)
                else value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()}

    def _register_hooks(self):
        def capture(layer):
            def hook(module, args):
                tensor = args[0].detach()
                self.activations[layer] = tensor.reshape(
                    *tensor.shape[:2], self.num_heads, self.head_dim)
            return hook
        for index, layer in enumerate(self.language_model.layers):
            self.hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture(index)))

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def clear_activations(self):
        self.activations.clear()

    def get_activations(self):
        return self.activations

    def get_correct_token_id(self, answer_str):
        tokens = self.tokenizer.encode(answer_str, add_special_tokens=False)
        if not tokens:
            raise ValueError("Answer has no tokens")
        return tokens[0]

    def run_greedy(self, image, question, answer_prefix="", answer_prefix_token_ids=None):
        """Score the next token at an explicit assistant/answer prefix.

        Thinking checkpoints may first emit a reasoning delimiter; callers
        analyzing a final answer must supply its generated reasoning prefix.
        """
        self.clear_activations()
        inputs = prepare_inputs(self, image, question, answer_prefix, answer_prefix_token_ids)
        output, self.propagation_context = capture_decoder_context(self.model, inputs)
        ids = inputs["input_ids"][0]
        visual = ids == self.model.config.image_token_id
        self.visual_positions = visual.nonzero(as_tuple=True)[0].tolist()
        self.answer_position = ids.numel() - 1
        self.question_positions = [i for i in range(self.answer_position) if not visual[i]]
        self.N_vis, self.N_q = len(self.visual_positions), len(self.question_positions)
        logits = output.logits[0, -1]
        return logits.argmax().item(), logits, output.logits
