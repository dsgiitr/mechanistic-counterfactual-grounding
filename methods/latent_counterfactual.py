"""Trainable, bounded visual-latent interventions for frozen native VLMs.

The objective is explicitly grounded-answer NLL plus distance/diversity penalties.
Latent locality is enforced; decoded-image realism is not assumed or claimed.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import torch
from torch import nn
from methods.restoration_effects import decoder_backbone, sequence_log_probability


@dataclass
class GeneratorConfig:
    hidden_size: int
    noise_dim: int = 16
    rank: int = 64
    max_relative_shift: float = 0.1


class LatentCounterfactualGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        if min(config.hidden_size, config.noise_dim, config.rank) <= 0:
            raise ValueError("Generator dimensions must be positive")
        if not math.isfinite(config.max_relative_shift) or config.max_relative_shift <= 0:
            raise ValueError("max_relative_shift must be finite and positive")
        self.config = config
        self.network = nn.Sequential(nn.Linear(config.hidden_size + config.noise_dim, config.rank),
                                     nn.SiLU(), nn.Linear(config.rank, config.hidden_size))

    def forward(self, features, noise):
        if features.ndim != 3 or noise.shape != (features.shape[0], self.config.noise_dim):
            raise ValueError("Expected visual features [B,V,H] and noise [B,Z]")
        if features.shape[-1] != self.config.hidden_size:
            raise ValueError("Generator and model hidden sizes differ")
        parameter = next(self.parameters())
        original = features.to(device=parameter.device, dtype=parameter.dtype)
        noise = noise.to(original)
        rms = original.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized = original / rms
        latent = noise[:, None].expand(-1, features.shape[1], -1)
        delta = self.config.max_relative_shift * rms * self.network(torch.cat((normalized, latent), -1)).tanh()
        return (original + delta).to(features)

    def save(self, path, model_id, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "config": asdict(self.config),
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                    "model_id": model_id, "metadata": metadata}, path)

    @classmethod
    def load(cls, path, model_id, hidden_size, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1 or checkpoint["model_id"] != model_id:
            raise ValueError("Generator checkpoint belongs to a different model or format")
        config = GeneratorConfig(**checkpoint["config"])
        if config.hidden_size != hidden_size:
            raise ValueError("Generator checkpoint hidden size differs from the VLM")
        generator = cls(config).to(device)
        generator.load_state_dict(checkpoint["state_dict"], strict=True)
        return generator.eval()


@contextmanager
def visual_intervention(model, inputs, generator, noise):
    """Transform visual slots and Qwen3 DeepStack features; keep all text fixed.

    Re-enter this context with the same noise for every baseline/patched run of
    a counterfactual. The frozen trained generator then defines a fixed mapping.
    """
    decoder = decoder_backbone(model)
    config = model.config
    image_id = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
    mask = inputs["input_ids"] == image_id
    if mask.shape[0] != 1 or not mask.any():
        raise ValueError("Latent intervention currently requires one image/question at a time")
    state = {"original": [], "generated": []}
    def transform(original):
        generated = generator(original.unsqueeze(0), noise).squeeze(0)
        state["original"].append(original)
        state["generated"].append(generated)
        return generated
    def hook(module, args, kwargs):
        kwargs = dict(kwargs)
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None or embeddings.shape[:2] != mask.shape:
            raise ValueError("Processor must expand image slots to match native decoder embeddings")
        visual_mask = mask.to(embeddings.device)
        embeddings = embeddings.clone()
        embeddings[visual_mask] = transform(embeddings[visual_mask])
        kwargs["inputs_embeds"] = embeddings
        deepstack = kwargs.get("deepstack_visual_embeds")
        if deepstack is not None:
            kwargs["deepstack_visual_embeds"] = [transform(feature) for feature in deepstack]
        return args, kwargs
    handle = decoder.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield state
    finally:
        handle.remove()


@contextmanager
def frozen_vlm(model):
    """Freeze VLM weights/dropout/checkpointing while keeping gradients to inputs."""
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    flags = [(module, module.gradient_checkpointing) for module in model.modules()
             if hasattr(module, "gradient_checkpointing")]
    try:
        model.eval()
        for p, _ in parameters:
            p.requires_grad_(False)
        for module, _ in flags:
            module.gradient_checkpointing = False
        yield
    finally:
        for p, requires_grad in parameters:
            p.requires_grad_(requires_grad)
        for module, mode in modes:
            module.training = mode
        for module, enabled in flags:
            module.gradient_checkpointing = enabled


def generator_training_step(model, inputs, target_ids, generator, optimizer,
                            noise_a, noise_b, distance_weight=1., diversity_weight=.1,
                            min_diversity=.01):
    """One actual supervised update; no gradients accumulate on VLM parameters."""
    if min(distance_weight, diversity_weight, min_diversity) < 0:
        raise ValueError("Objective weights and diversity threshold must be nonnegative")
    optimizer.zero_grad(set_to_none=True)
    with frozen_vlm(model):
        states, negative_log_probs = [], []
        for noise in (noise_a, noise_b):
            with visual_intervention(model, inputs, generator, noise) as state:
                negative_log_probs.append(-sequence_log_probability(model, inputs, target_ids) / len(target_ids))
            states.append(state)
        if not states[0]["original"]:
            raise RuntimeError("Visual intervention hook did not execute")
        distances, diversities = [], []
        for original, first, second in zip(states[0]["original"], states[0]["generated"], states[1]["generated"]):
            scale = original.float().square().mean(-1, keepdim=True).clamp_min(1e-12)
            distances.append(((first.float()-original.float()).square()/scale).mean())
            distances.append(((second.float()-original.float()).square()/scale).mean())
            diversities.append(((first.float()-second.float()).square()/scale).mean())
        nll = torch.stack(negative_log_probs).mean()
        distance = torch.stack(distances).mean()
        diversity = torch.stack(diversities).mean()
        loss = nll + distance_weight * distance + diversity_weight * torch.relu(min_diversity-diversity)
        parameters = list(generator.parameters())
        gradients = torch.autograd.grad(loss, parameters)
        if not torch.isfinite(loss) or any(not torch.isfinite(grad).all() for grad in gradients):
            raise FloatingPointError("Nonfinite generator objective/gradient")
        for p, gradient in zip(parameters, gradients):
            p.grad = gradient
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
    return {"loss": loss.item(), "answer_nll": nll.item(),
            "relative_distance": distance.item(), "diversity": diversity.item()}
