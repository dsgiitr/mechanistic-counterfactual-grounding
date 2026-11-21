"""Algorithm 1 for one example, shared by the stability and benchmark experiments."""
import time

import torch

from mcg.cdt import answer_direction, capture_context, sweep
from mcg.circuits import iterative_search
from mcg.counterfactuals import DonorMixingCounterfactuals, load_beta
from mcg.models import HookSet
from mcg.patching import patch_heads
from mcg.restoration import Restorer
from mcg.tasks import analyse_generation, sample_is_correct
from mcg.utils import derive_seed


def max_new_tokens(cfg, vlm, task, factual):
    gen = cfg["generation"]
    if factual and vlm.thinking:
        return gen["max_new_tokens_reasoning"]
    return gen["max_new_tokens_binary"] if task == "binary" else gen["max_new_tokens_caption"]


def base_prediction(vlm, bench, item, cfg):
    """Greedy factual generation, labelled and with y defined (or None if unscorable)."""
    inputs = vlm.prepare(item["image"], bench.prompt(item))
    generated = vlm.generate(inputs, max_new_tokens(cfg, vlm, bench.task, factual=True))[0]
    return analyse_generation(vlm, bench, item, generated)


class ExampleState:
    """Factual inputs, decoder context and answer definition of one example."""

    def __init__(self, vlm, bench, item, record):
        self.item = item
        self.prompt = bench.prompt(item)
        self.task = bench.task
        self.target = list(record["target_ids"])
        self.gen_prefix = list(record["gen_prefix_ids"])
        self.score_prefix = list(record["score_prefix_ids"])
        self.score_inputs = vlm.prepare(item["image"], self.prompt, self.score_prefix + self.target[:-1])
        self.gen_inputs = vlm.prepare(item["image"], self.prompt, self.gen_prefix)
        self.answer_pos = self.score_inputs["input_ids"].shape[1] - len(self.target)
        self.ctx = capture_context(vlm, self.score_inputs, keep=len(self.target))
        logits = self.ctx["logits"]
        target = torch.tensor(self.target, device=logits.device)
        self.factual_logprob = float(logits.log_softmax(-1).gather(-1, target[:, None]).sum())
        self.contrast = None
        if self.task == "binary":
            self.contrast = vlm.opposite_token(self.target[0], record["answer"], logits[0])


def discover_circuit(vlm, state, means, cfg, progress=None):
    """Per-example CD-T circuit S* (Appendix D)."""
    c = cfg["cdt"]
    direction = answer_direction(vlm, state.target[0], state.contrast)
    scores = sweep(vlm, state.ctx, means, state.answer_pos, direction, c["batch_size"], progress,
                   c.get("max_attention_elements", 1e9))
    circuit = iterative_search(scores, k0=c["k0"], k_schedule=lambda i: c["k_i"], max_nodes=c["max_nodes"])
    return circuit, scores


@torch.no_grad()
def node_activations(vlm, inputs, nodes):
    """Activations [S, D] of `nodes` in one (counterfactual) forward pass."""
    D = vlm.head_dim
    out = {}

    def hook_for(layer, heads):
        def hook(module, args):
            for h in heads:
                out[(layer, h)] = args[0][0, :, h * D:(h + 1) * D].clone()
        return hook

    by_layer = {}
    for l, h in nodes:
        by_layer.setdefault(l, []).append(h)
    with HookSet() as hooks:
        for layer, heads in by_layer.items():
            hooks.pre(vlm.o_proj(layer), hook_for(layer, heads))
        vlm.forward_logits(inputs, keep=1)
    return out


def make_generator(vlm, pool, state, record, cfg, seed, example_id):
    """CVL donor-mixing generator for one example (Eq. 1)."""
    c = cfg["counterfactuals"]
    return DonorMixingCounterfactuals(
        vlm, pool, load_beta(cfg), c["gate"], seed, example_id, state.item["image"], state.prompt,
        queried=pool.queried_object(state.item, record), factual_key=state.item.get("image_key"))


def run_counterfactuals(vlm, bench, state, circuit, cfg, K, seed, example_id, generator, effects=True,
                        effects_on="retained"):
    """Generate K counterfactuals, apply the retention rule, and measure.

    Returns per-counterfactual records (u, donor, retention votes, restoration
    effects) and the mean circuit activation over the retained counterfactuals
    (used by the oracle-assisted patch).
    """
    cf_cfg = cfg["counterfactuals"]
    decodings = int(cf_cfg["decodings"][K] if K in cf_cfg["decodings"] else cf_cfg["decodings"][str(K)])
    threshold = decodings - int(cf_cfg["retention_margin"])
    nodes = [(n["layer"], n["head"]) for n in circuit]
    restorer = Restorer(vlm, circuit, state.ctx["acts"]) if effects else None
    sums, retained = {}, 0
    records = []
    budget = max_new_tokens(cfg, vlm, state.task, factual=False)
    for k in range(K):
        with generator.draw(k):
            samples = vlm.generate(state.gen_inputs, budget, sample=True, temperature=cf_cfg["temperature"],
                                   n=decodings, seed=derive_seed(seed, example_id, k, "decode"))
            n_correct = sum(sample_is_correct(vlm, bench, state.item, s) for s in samples)
            keep = n_correct >= threshold
            draw = generator.sample(k)
            record = {"k": k, "u": draw["u"], "donor": draw["donor"], "factual_gate": draw["factual_gate"],
                      "n_correct": int(n_correct), "retained": bool(keep)}
            acts = None
            if effects and (keep or effects_on == "all"):
                baseline, deltas, acts = restorer.run(state.score_inputs, state.target,
                                                      cfg["restoration"]["batch_size"])
                record["cf_logprob"] = baseline
                record["deltas"] = [deltas[node] for node in nodes]
            elif keep:
                acts = node_activations(vlm, state.score_inputs, nodes)
            if keep:
                retained += 1
                for node in nodes:
                    value = acts[node].float()
                    sums[node] = sums[node] + value if node in sums else value
        generator.clear()
        records.append(record)
    mean_acts = {node: (value / retained) for node, value in sums.items()} if retained else None
    return records, mean_acts, {"decodings": decodings, "threshold": threshold, "num_retained": retained}


@torch.no_grad()
def oracle_patch(vlm, bench, state, mean_acts, cfg):
    """Factual forward with every circuit node replaced by its retained-CF mean."""
    with patch_heads(vlm, mean_acts):
        if state.task == "binary":
            logits = vlm.forward_logits(state.score_inputs, keep=len(state.target))[0, 0]
            answer = vlm.binary_decision(logits)[0]
            return {"answer": answer, "correct": answer == state.item["gt"]}
        generated = vlm.generate(state.gen_inputs, max_new_tokens(cfg, vlm, "caption", factual=False))[0]
    record = analyse_generation(vlm, bench, state.item, state.gen_prefix + generated) if vlm.thinking \
        else analyse_generation(vlm, bench, state.item, generated)
    if record is None:
        return {"answer": None, "correct": False}
    return {"answer": record["answer"], "correct": record["correct"],
            "hallucinated_objects": record["hallucinated_objects"],
            "mentioned_objects": record["mentioned_objects"], "recall": record["recall"]}


class Timer:
    def __init__(self):
        self.start = time.time()

    def __call__(self):
        return round(time.time() - self.start, 2)
