"""Circuit faithfulness via native mean-ablation, separate from CD attribution."""
from contextlib import contextmanager
import torch
from methods.restoration_effects import decoder_backbone
from models.native_utils import prepare_inputs


@contextmanager
def mean_ablation(model, means, circuit):
    """Keep circuit positions factual and replace every other head by its mean."""
    decoder = decoder_backbone(model)
    num_heads = decoder.config.num_attention_heads
    head_dim = getattr(decoder.config, "head_dim", None) or decoder.config.hidden_size // num_heads
    keep = {(n.layer_idx, n.attn_head_idx, n.sequence_idx) for n in circuit}
    handles = []

    def hook_for(layer):
        def hook(module, args):
            current = args[0]
            batch, length, _ = current.shape
            mean = means[layer].to(current)
            if mean.shape[1:] != (length, num_heads, head_dim):
                raise ValueError("Mean-activation layout does not match model activations")
            original = current.view(batch, length, num_heads, head_dim)
            patched = mean.expand(batch, -1, -1, -1).clone()
            for head in range(num_heads):
                if (layer, head, -1) in keep:
                    patched[:, :, head] = original[:, :, head]
                else:
                    for l, h, pos in keep:
                        if l == layer and h == head:
                            if not 0 <= pos < length:
                                raise ValueError("Circuit position outside sequence")
                            patched[:, pos, head] = original[:, pos, head]
            return (patched.reshape_as(current),) + args[1:]
        return hook

    try:
        for layer, module in enumerate(decoder.layers):
            handles.append(module.self_attn.o_proj.register_forward_pre_hook(hook_for(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def evaluate_faithfulness(dataset, circuit, yes_no=False):
    """Baseline-corrected logit-margin recovery; undefined denominators stay null."""
    wrapper = dataset.wrapper
    margins = []
    for example in dataset.hallucination_examples:
        hall = wrapper.get_correct_token_id("Yes") if yes_no else example["hallucinated_token_id"]
        correct = wrapper.get_correct_token_id("No") if yes_no else example["correct_token_id"]
        means = dataset.get_mean_correct(example["visual_positions"], example["question_positions"], example["answer_position"])
        inputs = prepare_inputs(wrapper, example["image_loaded"], example["question"], example.get("answer_prefix", ""), example.get("answer_prefix_token_ids"))
        def margin():
            with torch.no_grad():
                logits = wrapper.model(**inputs, use_cache=False).logits[0, -1].float()
            return (logits[hall] - logits[correct]).item()
        full = margin()
        with mean_ablation(wrapper.model, means, []):
            baseline = margin()
        with mean_ablation(wrapper.model, means, circuit):
            recovered = margin()
        denominator = full - baseline
        margins.append({"full_margin": full, "baseline_margin": baseline,
                        "circuit_margin": recovered,
                        "faithfulness": (recovered-baseline)/denominator if abs(denominator) > 1e-8 else None})
    if not margins:
        raise ValueError("Cannot evaluate an empty dataset")
    full_difference = sum(m["full_margin"]-m["baseline_margin"] for m in margins)
    circuit_difference = sum(m["circuit_margin"]-m["baseline_margin"] for m in margins)
    return {"overall_faithfulness": circuit_difference/full_difference if abs(full_difference) > 1e-8 else None,
            "definition": "(circuit_margin - all_mean_margin) / (full_margin - all_mean_margin)",
            "metrics_per_example": margins}
