import os
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration

from methods.counterfactual_gen import CounterFactualGenerator
from training.exogenous_module import ExogenousVariableModule


class VQAHead(nn.Module):
    def __init__(self, hidden_size: int, num_answers: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_answers),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Keep activations aligned with classifier weights to avoid Half/Float
        # mismatches when mixed precision and quantized backbones are combined.
        ref = self.classifier[0].weight
        if hidden.device != ref.device or hidden.dtype != ref.dtype:
            hidden = hidden.to(device=ref.device, dtype=ref.dtype)
        return self.classifier(hidden)


class LLaVACVL(nn.Module):
    def __init__(
        self,
        llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
        num_answers: int = 3129,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
        freeze_vision_encoder: bool = True,
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_use_double_quant: bool = True,
        device_map: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        epsilon: float = 0.1,
        cf_grad_steps: int = 5,
        cf_lr: float = 0.01,
        lambda_cf: float = 0.5,
        clip_M: float = 10.0,
        beta_alpha0: float = 0.1,
        beta_beta0: float = 0.1,
        intervene_vision: bool = True,
        intervene_language: bool = False,
    ):
        super().__init__()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        compute_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(str(bnb_4bit_compute_dtype).lower(), torch.bfloat16)

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.llava = LlavaForConditionalGeneration.from_pretrained(
            llava_model_name,
            **model_kwargs,
        )

        # Reduce activation memory during backward.
        self.llava.config.use_cache = False
        self.llava.gradient_checkpointing_enable()
        self.llava.enable_input_require_grads()

        if freeze_vision_encoder:
            for param in self.llava.model.vision_tower.parameters():
                param.requires_grad_(False)

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.llava.model.language_model = get_peft_model(
                self.llava.model.language_model,
                lora_config,
            )

        hidden_size = self.llava.config.text_config.hidden_size
        self.vqa_head = VQAHead(hidden_size, num_answers)
        self.exogenous_module = ExogenousVariableModule(
            alpha0=beta_alpha0,
            beta0=beta_beta0,
            epsilon=epsilon,
        )
        self.cf_generator = CounterFactualGenerator(
            epsilon=epsilon,
            cf_grad_steps=cf_grad_steps,
            cf_lr=cf_lr,
            lambda_cf=lambda_cf,
            clip_M=clip_M,
            intervene_vision=intervene_vision,
            intervene_language=intervene_language,
        )
        self.clip_M = clip_M

    def _encode_and_pool(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.llava(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        # Last non-padding position works for both left and right padding.
        positions = torch.arange(attention_mask.size(1), device=hidden_states.device)
        seq_lens = positions.expand_as(attention_mask).masked_fill(
            attention_mask.to(hidden_states.device) == 0, -1
        ).max(dim=1).values
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        seq_lens = seq_lens.clamp(min=0, max=hidden_states.size(1) - 1)
        pooled = hidden_states[batch_indices, seq_lens]
        return pooled

    def _counterfactual_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        rand_input_ids: Optional[torch.Tensor],
        rand_attention_mask: Optional[torch.Tensor],
        rand_pixel_values: Optional[torch.Tensor],
    ):
        if rand_input_ids is None:
            rand_input_ids = torch.roll(input_ids, shifts=1, dims=0)
        if rand_attention_mask is None:
            rand_attention_mask = torch.roll(attention_mask, shifts=1, dims=0)
        if rand_pixel_values is None:
            rand_pixel_values = torch.roll(pixel_values, shifts=1, dims=0)
        return rand_input_ids, rand_attention_mask, rand_pixel_values

    @contextmanager
    def _embedding_hooks(self, vision=None, language=None, capture=None):
        """Replace/capture projected visual and token embeddings before fusion."""
        handles = []
        def hook_for(name, replacement):
            def hook(module, args, output):
                if capture is not None:
                    capture[name] = output
                return replacement if replacement is not None else output
            return hook
        try:
            handles.append(self.llava.model.multi_modal_projector.register_forward_hook(hook_for("vision", vision)))
            handles.append(self.llava.get_input_embeddings().register_forward_hook(hook_for("language", language)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _aligned_language_donor(self, factual, donor, input_ids, donor_ids, attention_mask, donor_mask):
        """Align donor text tokens, preserving factual padding and image slots."""
        aligned = factual.detach().clone()
        image_id = self.llava.config.image_token_index
        protected_ids = [image_id, self.llava.config.pad_token_id,
                         self.llava.config.text_config.bos_token_id,
                         self.llava.config.text_config.eos_token_id]
        def text_mask(ids, mask):
            valid = mask.bool().clone()
            for token in protected_ids:
                if token is not None:
                    valid &= ids != token
            return valid
        factual_mask = text_mask(input_ids, attention_mask)
        random_mask = text_mask(donor_ids, donor_mask)
        for batch in range(factual.shape[0]):
            positions = factual_mask[batch].nonzero(as_tuple=True)[0]
            source = donor[batch, random_mask[batch]]
            if positions.numel() and not source.shape[0]:
                raise ValueError("Language donor contains no usable text tokens")
            if positions.numel():
                indices = torch.arange(positions.numel(), device=source.device) % source.shape[0]
                aligned[batch, positions] = source[indices].to(aligned)
        return aligned

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        answer_idx: Optional[torch.Tensor] = None,
        run_counterfactual: bool = False,
        rand_input_ids: Optional[torch.Tensor] = None,
        rand_attention_mask: Optional[torch.Tensor] = None,
        rand_pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        factual_embeddings = {}
        if run_counterfactual:
            with self._embedding_hooks(capture=factual_embeddings):
                pooled = self._encode_and_pool(input_ids, attention_mask, pixel_values)
        else:
            pooled = self._encode_and_pool(input_ids, attention_mask, pixel_values)
        logits = self.vqa_head(pooled)

        outputs: Dict[str, torch.Tensor] = {"logits": logits}

        if not run_counterfactual:
            return outputs

        cf_input_ids, cf_attention_mask, cf_pixel_values = self._counterfactual_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            rand_input_ids=rand_input_ids,
            rand_attention_mask=rand_attention_mask,
            rand_pixel_values=rand_pixel_values,
        )
        if cf_attention_mask.shape != cf_input_ids.shape:
            pad_token_id = self.llava.config.pad_token_id
            if pad_token_id is None:
                cf_attention_mask = torch.ones_like(cf_input_ids)
            else:
                cf_attention_mask = (cf_input_ids != pad_token_id).long()
        donor_embeddings = {}
        with torch.no_grad(), self._embedding_hooks(capture=donor_embeddings):
            self._encode_and_pool(cf_input_ids, cf_attention_mask, cf_pixel_values)
        vision = factual_embeddings["vision"]
        language = factual_embeddings["language"]
        donor_vision = donor_embeddings["vision"].detach()
        if donor_vision.shape != vision.shape:
            raise ValueError("Visual donor and factual projected embeddings must match in shape")
        donor_language = self._aligned_language_donor(
            language, donor_embeddings["language"], input_ids, cf_input_ids,
            attention_mask, cf_attention_mask,
        )
        u_v, u_q = self.exogenous_module(input_ids.size(0), vision.device)

        # Replacement hooks must remain effective through the inner gradient
        # computation. Checkpoint recomputation after hook removal would use
        # factual embeddings, so these CF forwards run without checkpointing.
        checkpointing_flags = [(module, module.gradient_checkpointing)
                               for module in self.llava.modules()
                               if hasattr(module, "gradient_checkpointing")]
        for module, _ in checkpointing_flags:
            module.gradient_checkpointing = False
        try:
            def fusion(v, q):
                with self._embedding_hooks(vision=v, language=q):
                    return self.vqa_head(self._encode_and_pool(input_ids, attention_mask, pixel_values))
            if answer_idx is not None and self.training:
                with torch.enable_grad():
                    u_v, u_q = self.cf_generator.optimize_u_for_intervention(
                        vision, language, donor_vision, donor_language, u_v, u_q,
                        answer_idx, fusion,
                    )
            cf_vision, cf_language = self.cf_generator.generate(
                vision, language, donor_vision, donor_language, u_v, u_q,
            )
            with self._embedding_hooks(vision=cf_vision, language=cf_language):
                cf_pooled = self._encode_and_pool(input_ids, attention_mask, pixel_values)
        finally:
            # Restore flags directly: gradient_checkpointing_enable() can add
            # input-gradient hooks on every call in some Transformers versions.
            for module, enabled in checkpointing_flags:
                module.gradient_checkpointing = enabled
        logits_cf = self.vqa_head(cf_pooled)

        if answer_idx is None:
            omega = torch.ones(logits.size(0), device=logits.device, dtype=logits.dtype)
        else:
            log_p = torch.log_softmax(logits, dim=-1)
            log_p_cf = torch.log_softmax(logits_cf, dim=-1)
            log_pa = log_p.gather(1, answer_idx.unsqueeze(1)).squeeze(1)
            log_pa_cf = log_p_cf.gather(1, answer_idx.unsqueeze(1)).squeeze(1)
            omega = torch.exp(log_pa - log_pa_cf).clamp(max=self.clip_M)

        outputs["logits_cf"] = logits_cf
        outputs["omega"] = omega
        return outputs
