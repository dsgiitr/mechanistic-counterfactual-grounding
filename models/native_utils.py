"""Shared native multimodal input preparation and decoder context capture."""
import torch
from methods.restoration_effects import decoder_backbone, append_prompt_tokens


def prepare_inputs(wrapper, image, question, answer_prefix="", answer_prefix_token_ids=None):
    if answer_prefix_token_ids is not None:
        inputs = prepare_inputs(wrapper, image, question)
        return append_prompt_tokens(inputs, answer_prefix_token_ids)
    if hasattr(wrapper, "prepare_inputs"):
        return wrapper.prepare_inputs(image, question, answer_prefix=answer_prefix)
    inputs = wrapper.processor(text=wrapper.get_prompt(question) + answer_prefix,
                               images=image, return_tensors="pt")
    return {key: value.to(device=wrapper.device, dtype=wrapper.W_U.dtype)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value.to(wrapper.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()}


def capture_decoder_context(model, inputs):
    """Capture the native mask, RoPE and visual injections rather than recreate them."""
    decoder = decoder_backbone(model)
    context = {}
    def decoder_input(module, args, kwargs):
        context["deepstack_visual_embeds"] = kwargs.get("deepstack_visual_embeds")
        context["visual_pos_masks"] = kwargs.get("visual_pos_masks")
    def layer_input(module, args, kwargs):
        context["base_embeds"] = (args[0] if args else kwargs["hidden_states"]).detach()
        context["attention_mask"] = kwargs.get("attention_mask")
        context["position_ids"] = kwargs.get("position_ids")
        context["position_embeddings"] = kwargs.get("position_embeddings")
    handles = [decoder.register_forward_pre_hook(decoder_input, with_kwargs=True),
               decoder.layers[0].register_forward_pre_hook(layer_input, with_kwargs=True)]
    try:
        with torch.no_grad():
            output = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if "base_embeds" not in context:
        raise RuntimeError("Native decoder context hooks did not execute")
    return output, context


def factual_context(wrapper, image, question, answer_prefix="", answer_prefix_token_ids=None):
    inputs = prepare_inputs(wrapper, image, question, answer_prefix, answer_prefix_token_ids)
    output, context = capture_decoder_context(wrapper.model, inputs)
    wrapper.propagation_context = context
    return inputs, output, context["base_embeds"]
