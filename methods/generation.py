"""Native autoregressive generation and explicit final-answer prefix extraction."""
import re
import torch
from models.native_utils import prepare_inputs


def generate_binary_answer(wrapper, image, question, max_new_tokens=512):
    inputs = prepare_inputs(wrapper, image, question)
    with torch.no_grad():
        output = wrapper.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False, use_cache=True)
    tokens = output[0, inputs["input_ids"].shape[1]:].tolist()
    decoded = wrapper.tokenizer.decode(tokens, skip_special_tokens=False)
    # An unfinished reasoning sequence is not a binary answer.
    opening = decoded.find("<think>")
    closing = decoded.rfind("</think>")
    prompt = wrapper.tokenizer.decode(inputs["input_ids"][0].tolist(), skip_special_tokens=False)
    prefilled_reasoning = prompt.rfind("<think>") > prompt.rfind("</think>")
    if (opening >= 0 and closing < opening) or (prefilled_reasoning and closing < 0):
        return None
    start = closing + len("</think>") if closing >= 0 else 0
    match = re.match(r'''\s*["']?(yes|no)\b''', decoded[start:], flags=re.IGNORECASE)
    if match is None:
        return None
    answer_start = start + match.start(1)
    # Keep the exact generated token prefix (including reasoning) for subsequent
    # factual/CF runs. Never mistake the initial <think> token for the answer.
    prefix_length = 0
    for index in range(len(tokens)):
        prefix = wrapper.tokenizer.decode(tokens[:index], skip_special_tokens=False)
        if len(prefix) <= answer_start:
            prefix_length = index
        else:
            break
    return {"answer": match.group(1).lower(), "generated_text": decoded,
            "answer_prefix": wrapper.tokenizer.decode(tokens[:prefix_length], skip_special_tokens=False),
            "answer_prefix_token_ids": tokens[:prefix_length],
            "answer_token_id": tokens[prefix_length]}
