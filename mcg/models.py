"""Native VLM adapter with one interface for all models studied in the paper.

Supported families
------------------
llava     LLaVA-1.5            (CLIP ViT-L/14 -> MLP projector -> Llama-2 decoder)
qwen3_vl  Qwen3-VL Instruct/Thinking (ViT -> patch merger + DeepStack mergers -> Qwen3 decoder)
idefics3  SmolVLM-256M/500M    (SigLIP -> pixel-shuffle connector -> SmolLM2/Llama decoder)

For every family the adapter exposes
  * the decoder stack whose attention heads are circuit nodes,
  * the projector modules G_phi that map vision-encoder features h(I) to the
    visual tokens consumed by the decoder (the latent counterfactual is applied
    to their inputs, Eq. 1),
  * chat-template prompting, teacher-forced scoring, greedy and sampled decoding.
"""
import re
from contextlib import ExitStack

import torch

MODEL_REGISTRY = {
    "llava-1.5-7b": {"model_id": "llava-hf/llava-1.5-7b-hf", "family": "llava",
                     "label": "LLaVA-v1.5-7B"},
    "qwen3-vl-8b-instruct": {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "family": "qwen3_vl",
                             "label": "Qwen3-VL-8B-Instruct"},
    "qwen3-vl-8b-thinking": {"model_id": "Qwen/Qwen3-VL-8B-Thinking", "family": "qwen3_vl",
                             "thinking": True, "label": "Qwen3-VL-8B-Thinking"},
    "smolvlm-500m": {"model_id": "HuggingFaceTB/SmolVLM-500M-Instruct", "family": "idefics3",
                     "label": "SmolVLM-500M"},
    "smolvlm-256m": {"model_id": "HuggingFaceTB/SmolVLM-256M-Instruct", "family": "idefics3",
                     "label": "SmolVLM-256M"},
}

_FAMILY_BY_MODEL_TYPE = {"llava": "llava", "qwen3_vl": "qwen3_vl",
                         "idefics3": "idefics3", "smolvlm": "idefics3"}

_WORD = re.compile(r"[^a-z]")


def model_label(name):
    return MODEL_REGISTRY.get(name, {}).get("label", name)


def _default_dtype(family, device):
    if torch.device(device).type == "cpu":
        return torch.float32
    return torch.float16 if family == "llava" else torch.bfloat16


class VLM:
    """Thin adapter around a Hugging Face image-text-to-text model."""

    def __init__(self, name, device="cuda", dtype=None, *, model=None, processor=None,
                 family=None, thinking=None, attn_implementation="sdpa", processor_kwargs=None):
        spec = MODEL_REGISTRY.get(name, {"model_id": name})
        self.name = name
        self.model_id = spec["model_id"]
        self.device = torch.device(device)
        self.thinking = spec.get("thinking", False) if thinking is None else thinking
        if processor is None:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        if model is None:
            from transformers import AutoConfig, AutoModelForImageTextToText
            config = AutoConfig.from_pretrained(self.model_id)
            fam = family or spec.get("family") or _FAMILY_BY_MODEL_TYPE[config.model_type]
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=dtype or _default_dtype(fam, device),
                attn_implementation=attn_implementation, low_cpu_mem_usage=True)
            model.to(self.device)
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.family = family or spec.get("family") or _FAMILY_BY_MODEL_TYPE[self.model.config.model_type]
        self.dtype = next(self.model.parameters()).dtype
        self.decoder = self._decoder()
        cfg = self.decoder.config
        self.num_layers = len(self.decoder.layers)
        self.num_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.num_heads
        self.lm_head = self.model.lm_head
        config = self.model.config
        self.image_token_id = getattr(config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = getattr(config, "image_token_index")
        self.think_end_id = None
        if self.thinking:
            ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            self.think_end_id = ids[0] if len(ids) == 1 else None
        self._binary_sets = None
        self.processor_kwargs = processor_kwargs or {}

    # ------------------------------------------------------------------ modules
    def _decoder(self):
        inner = getattr(self.model, "model", self.model)
        for attribute in ("language_model", "text_model"):
            decoder = getattr(inner, attribute, None)
            if decoder is not None:
                decoder = getattr(decoder, "model", decoder)
                if hasattr(decoder, "layers"):
                    return decoder
        raise ValueError("Cannot locate the language decoder of this model")

    def projector_modules(self):
        """G_phi: modules whose *input* is the vision-encoder embedding h(I)."""
        inner = getattr(self.model, "model", self.model)
        if self.family == "llava":
            return [inner.multi_modal_projector]
        if self.family == "idefics3":
            return [inner.connector]
        if self.family == "qwen3_vl":
            visual = inner.visual
            return [visual.merger, *list(getattr(visual, "deepstack_merger_list", []))]
        raise ValueError(self.family)

    def o_proj(self, layer):
        return self.decoder.layers[layer].self_attn.o_proj

    # ------------------------------------------------------------------ inputs
    def prompt_text(self, question, has_image=True):
        content = ([{"type": "image"}] if has_image else []) + [{"type": "text", "text": question}]
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def prepare(self, image, question, extra_ids=()):
        """Model inputs for prompt (+ image) followed by exact extra token IDs."""
        text = self.prompt_text(question, image is not None)
        kwargs = {"text": [text], "return_tensors": "pt", **self.processor_kwargs}
        if image is not None:
            kwargs["images"] = [[image]] if self.family == "idefics3" else [image]
        inputs = dict(self.processor(**kwargs))
        inputs = {key: value.to(self.device, self.dtype) if torch.is_floating_point(value)
                  else value.to(self.device) for key, value in inputs.items() if torch.is_tensor(value)}
        return append_tokens(inputs, list(extra_ids))

    @staticmethod
    def expand(inputs, batch):
        """Tile a single-example input `batch` times along dimension 0.

        Works for every family: batch-first tensors (ids, masks, LLaVA/Idefics3
        pixels) and Qwen3's flattened patch tensor/grid both tile correctly.
        """
        if batch == 1:
            return inputs
        return {key: value.repeat(batch, *([1] * (value.dim() - 1))) for key, value in inputs.items()}

    def has_image(self, inputs):
        return bool((inputs["input_ids"] == self.image_token_id).any())

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def forward_logits(self, inputs, keep=1):
        """Logits for the last `keep` positions, shape [B, keep, V] (fp32)."""
        out = self.model(**inputs, use_cache=False, logits_to_keep=keep)
        return out.logits[:, -keep:].float()

    @torch.no_grad()
    def target_logprob(self, inputs, target_ids):
        """sum_t log p(y_t | x, y_<t); inputs must end with target_ids[:-1]. Returns [B]."""
        logits = self.forward_logits(inputs, keep=len(target_ids))
        target = torch.tensor(target_ids, device=logits.device)
        logp = logits.log_softmax(-1).gather(-1, target.expand(logits.shape[0], -1)[..., None])
        return logp[..., 0].sum(-1)

    # ------------------------------------------------------------------ decoding
    def _eos_ids(self):
        ids = self.model.generation_config.eos_token_id
        ids = [ids] if isinstance(ids, int) else list(ids or [])
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        return sorted(set(ids))

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens, *, sample=False, temperature=1.0, n=1, seed=None):
        """Return a list of n generated token-ID lists (new tokens only, EOS removed)."""
        eos = self._eos_ids()
        pad = self.tokenizer.pad_token_id
        options = dict(max_new_tokens=max_new_tokens, eos_token_id=eos or None,
                       pad_token_id=pad if pad is not None else (eos[0] if eos else 0))
        if sample:
            # Pure temperature sampling: disable model-default top-k / top-p filters.
            options.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                           num_return_sequences=n)
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
        else:
            options.update(do_sample=False, temperature=None, top_k=None, top_p=None)
        output = self.model.generate(**inputs, **options)
        start = inputs["input_ids"].shape[1]
        sequences = []
        for row in output[:, start:].tolist():
            for index, token in enumerate(row):
                if token in eos:
                    row = row[:index]
                    break
            sequences.append(row)
        return sequences

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def split_reasoning(self, generated):
        """(reasoning_ids, answer_ids) for Thinking models; ([], ids) otherwise.

        Returns None when a Thinking model did not close its reasoning trace.
        """
        if not self.thinking:
            return [], list(generated)
        if self.think_end_id is not None and self.think_end_id in generated:
            index = generated.index(self.think_end_id) + 1
            return list(generated[:index]), list(generated[index:])
        return None

    def char_offsets(self, ids):
        """End character offset of each token in decode(ids) (incremental decoding)."""
        return [len(self.decode(ids[: i + 1])) for i in range(len(ids))]

    def span_tokens(self, ids, start, end, offsets=None):
        """Token index range [i, j) covering characters [start, end) of decode(ids)."""
        offsets = offsets or self.char_offsets(ids)
        first = next(i for i, off in enumerate(offsets) if off > start)
        last = next(j for j, off in enumerate(offsets) if off >= end)
        return first, last + 1

    # ------------------------------------------------------------------ binary answers
    def binary_token_sets(self):
        """All vocabulary entries that decode to 'yes' or 'no' (any case/spacing)."""
        if self._binary_sets is None:
            sets = {"yes": [], "no": []}
            vocab = self.tokenizer.get_vocab()
            for token_id in sorted(set(vocab.values())):
                word = _WORD.sub("", self.tokenizer.decode([token_id]).lower())
                if word in sets and len(self.tokenizer.decode([token_id]).strip()) <= 5:
                    sets[word].append(token_id)
            if not sets["yes"] or not sets["no"]:
                raise ValueError("Tokenizer has no single-token yes/no answers")
            self._binary_sets = {k: torch.tensor(v, device=self.device) for k, v in sets.items()}
        return self._binary_sets

    def binary_decision(self, last_logits):
        """'yes'/'no' by the larger maximal logit over each answer's token set."""
        sets = self.binary_token_sets()
        yes = last_logits[..., sets["yes"]].max(-1).values
        no = last_logits[..., sets["no"]].max(-1).values
        return ["yes" if y >= n else "no" for y, n in zip(yes.reshape(-1).tolist(), no.reshape(-1).tolist())]

    def opposite_token(self, token_id, answer, last_logits):
        """Highest-scoring token of the opposite binary answer (contrast direction)."""
        other = self.binary_token_sets()["no" if answer == "yes" else "yes"]
        return int(other[last_logits[other].argmax()].item())


def append_tokens(inputs, tokens):
    """Append exact token IDs to a single prompt; multimodal masks are extended."""
    if not tokens:
        return dict(inputs)
    prepared = dict(inputs)
    ids = inputs["input_ids"]
    extra = torch.tensor([tokens], device=ids.device, dtype=ids.dtype)
    prepared["input_ids"] = torch.cat((ids, extra), dim=1)
    prepared["attention_mask"] = torch.cat(
        (inputs.get("attention_mask", torch.ones_like(ids)), torch.ones_like(extra)), dim=1)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in inputs:
            prepared[key] = torch.cat((inputs[key], torch.zeros_like(extra).to(inputs[key])), dim=1)
    prepared.pop("position_ids", None)
    return prepared


class HookSet(ExitStack):
    """Register forward(-pre) hooks and remove all of them on exit, even on error."""

    def pre(self, module, hook, **kwargs):
        self.callback(module.register_forward_pre_hook(hook, **kwargs).remove)

    def post(self, module, hook, **kwargs):
        self.callback(module.register_forward_hook(hook, **kwargs).remove)
