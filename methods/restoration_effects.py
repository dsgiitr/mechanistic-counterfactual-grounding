"""Exact factual-activation restoration on the model's native forward pass."""
import torch
from contextlib import nullcontext
from methods.bernstein import node_key


def decoder_backbone(model):
    for container in (model, getattr(model, "model", None)):
        language = getattr(container, "language_model", None)
        if language is not None:
            decoder = getattr(language, "model", language)
            if hasattr(decoder, "layers"):
                return decoder
    raise ValueError("Cannot locate a supported decoder with layers")


def sequence_log_probability(model, inputs, target_token_ids):
    """Differentiable sum log p(y|x); inputs contain prompt followed by y[:-1]."""
    logits = model(**inputs, use_cache=False).logits
    count = len(target_token_ids)
    if logits.shape[0] != 1 or not count or logits.shape[1] < count:
        raise ValueError("Expected one prompt and a nonempty target fitting the sequence")
    target = torch.tensor(target_token_ids, device=logits.device, dtype=torch.long)
    log_probs = logits[0, -count:, :].float().log_softmax(dim=-1)
    return log_probs.gather(1, target[:, None]).sum()


def target_log_probability(model, inputs, target_token_ids):
    with torch.no_grad():
        return float(sequence_log_probability(model, inputs, target_token_ids).item())


def append_target_prefix(inputs, target_token_ids):
    """Prepare teacher forcing without changing the visual input or question."""
    if not target_token_ids:
        raise ValueError("A nonempty target token sequence is required")
    return append_prompt_tokens(inputs, target_token_ids[:-1])


def append_prompt_tokens(inputs, tokens):
    """Append exact context tokens and extend text/multimodal masks consistently."""
    prepared = dict(inputs)
    if tokens:
        ids = inputs["input_ids"]
        prefix = torch.tensor([tokens], device=ids.device, dtype=ids.dtype)
        prepared["input_ids"] = torch.cat((ids, prefix), dim=1)
        prepared["attention_mask"] = torch.cat((
            inputs.get("attention_mask", torch.ones_like(ids)), torch.ones_like(prefix)), dim=1)
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in inputs:
                prepared[key] = torch.cat((inputs[key], torch.zeros_like(prefix).to(inputs[key])), dim=1)
        prepared.pop("position_ids", None)
    return prepared


class FactualRestoration:
    """Restore one factual head at a time; require matching factual/CF layouts.

    Whole-head nodes replace all sequence positions, including teacher-forced
    answer positions. Position-specific nodes replace exactly their seq_pos.
    """

    def __init__(self, model, nodes):
        self.model = model
        self.decoder = decoder_backbone(model)
        self.nodes = list(nodes)
        keys = [node_key(n) for n in self.nodes]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Circuit must contain nonempty, unique nodes")
        cfg = self.decoder.config
        self.num_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.num_heads
        self.factual = {}
        for node in self.nodes:
            if int(node["layer"]) >= len(self.decoder.layers) or int(node["head"]) >= self.num_heads:
                raise ValueError(f"Node outside model dimensions: {node}")

    def _projection(self, node):
        return self.decoder.layers[int(node["layer"])].self_attn.o_proj

    def _slice(self, tensor, node):
        start = int(node["head"]) * self.head_dim
        position = int(node.get("seq_pos", -1))
        if position >= tensor.shape[1]:
            raise ValueError("Circuit sequence position exceeds input length")
        positions = slice(None) if position == -1 else slice(position, position + 1)
        return tensor[:, positions, start:start + self.head_dim]

    def capture(self, factual_inputs, target_token_ids):
        self.factual.clear()
        self.target_token_ids = tuple(target_token_ids)
        handles = []
        def capture_node(node):
            def hook(module, args):
                self.factual[node_key(node)] = (
                    args[0].shape, self._slice(args[0], node).detach().clone())
            return hook
        try:
            for node in self.nodes:
                handles.append(self._projection(node).register_forward_pre_hook(capture_node(node)))
            score = target_log_probability(self.model, factual_inputs, target_token_ids)
        except Exception:
            self.factual.clear()
            raise
        finally:
            for handle in handles:
                handle.remove()
        if len(self.factual) != len(self.nodes):
            raise RuntimeError("Some circuit hooks did not run")
        return score

    def effects(self, cf_inputs, target_token_ids, forward_context=nullcontext):
        if not self.factual:
            raise RuntimeError("Capture the factual activations first")
        if tuple(target_token_ids) != self.target_token_ids:
            raise ValueError("Factual and counterfactual runs must score the same target")
        with forward_context():
            baseline = target_log_probability(self.model, cf_inputs, target_token_ids)
        deltas = {}
        for node in self.nodes:
            key = node_key(node)
            shape, reference = self.factual[key]
            def restore(module, args):
                current = args[0]
                if current.shape != shape:
                    raise ValueError("Factual and counterfactual token layouts must match")
                patched = current.clone()
                self._slice(patched, node).copy_(reference.to(patched))
                return (patched,) + args[1:]
            handle = self._projection(node).register_forward_pre_hook(restore)
            try:
                with forward_context():
                    patched_score = target_log_probability(self.model, cf_inputs, target_token_ids)
            finally:
                handle.remove()
            deltas[key] = patched_score - baseline
        return baseline, deltas
