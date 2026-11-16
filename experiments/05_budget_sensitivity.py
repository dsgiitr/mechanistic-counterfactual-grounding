"""Step 5: circuit-size sensitivity of four node-selection strategies (Appendix D, Figure 3).

Dataset-level CD-T scores are aggregated over the hallucinated examples
(combined = mean S x |mean beta| / mean |beta|). For each circuit size n and
strategy (Global, Layer Norm, Iter Decay, size-matched Random) we report, with
background-mean ablation on the native forward pass and m = log p(y):
  faithfulness            sum_i (m_i(C) - m_i(empty)) / sum_i (m_i(M) - m_i(empty))
                          (only C active; Hanna et al. 2024 normalisation)
  hallucination remaining sum_i (m_i(M without C) - m_i(empty)) / sum_i (m_i(M) - m_i(empty))
  accuracy recovered      fraction of examples answered correctly with C ablated

Output  results/<model>/<dataset>/budget_sensitivity.json
"""
import json

import torch
from tqdm import tqdm

from common import config, load_model, parser, run_dir

from mcg.background import MeanCache
from mcg.cdt import answer_direction, sweep
from mcg.circuits import aggregate_scores, global_topk, iterative_decay, layer_normalized_topk, random_circuit
from mcg.patching import mean_ablation
from mcg.pipeline import ExampleState
from mcg.tasks import load_benchmark
from mcg.utils import write_json


def main():
    p = parser(__doc__, model=False, dataset=False)
    args = p.parse_args()
    cfg = config(args)
    bs = cfg["budget_sensitivity"]
    args.model, args.dataset = bs["model"], bs["dataset"]
    out = run_dir(args)
    examples = [e for e in json.loads((out / "examples.json").read_text())
                if e["group"] == "hallucinated"][:bs["num_examples"]]
    bench = load_benchmark(args.dataset, cfg)
    vlm = load_model(args)
    means = MeanCache(out.parent / "background_means.pt", vlm.device)
    prepared, per_example = [], []
    for ex in tqdm(examples, desc="CD-T scores"):
        state = ExampleState(vlm, bench, bench.item(ex["index"]), ex)
        m = means.for_inputs(vlm, state.score_inputs["input_ids"], state.answer_pos)
        per_example.append(sweep(vlm, state.ctx, m, state.answer_pos,
                                 answer_direction(vlm, state.target[0], state.contrast), cfg["cdt"]["batch_size"]))
        with mean_ablation(vlm, m, keep=set()):
            empty = float(vlm.target_logprob(state.score_inputs, state.target)[0])
        prepared.append((state.score_inputs, state.target, state.answer_pos, m, ex["gt"],
                         state.factual_logprob, empty))
        state.ctx = None
    scores = aggregate_scores(per_example)
    torch.save(scores, out / "aggregated_cdt_scores.pt")

    def evaluate(circuit):
        nodes = {(n["layer"], n["head"]) for n in circuit}
        num_c = num_rem = den = 0.0
        correct = 0
        for inputs, target, pos, m, gt, full, empty in prepared:
            with mean_ablation(vlm, m, keep=nodes):
                only = float(vlm.target_logprob(inputs, target)[0])
            with mean_ablation(vlm, m, ablate=nodes):
                logits = vlm.forward_logits(inputs, keep=len(target))[0]
            target_t = torch.tensor(target, device=logits.device)
            without = float(logits.log_softmax(-1).gather(-1, target_t[:, None]).sum())
            correct += vlm.binary_decision(logits[0])[0] == gt
            num_c += only - empty
            num_rem += without - empty
            den += full - empty
        return {"size": len(nodes), "faithfulness": num_c / den, "hallucination_remaining": num_rem / den,
                "accuracy_recovered": correct / len(prepared)}

    results = {"sizes": bs["sizes"], "strategies": {}, "num_examples": len(prepared), "model": args.model,
               "dataset": args.dataset}
    strategies = {"Global": lambda n: global_topk(scores, n),
                  "Layer Norm": lambda n: layer_normalized_topk(scores, n),
                  "Iter Decay": lambda n: iterative_decay(scores, n, bs["decay"])}
    for name, select in strategies.items():
        results["strategies"][name] = []
        for n in bs["sizes"]:
            circuit = select(n)
            results["strategies"][name].append(dict(evaluate(circuit), requested=n,
                                                    circuit=[{"layer": c["layer"], "head": c["head"]} for c in circuit]))
            print(name, results["strategies"][name][-1]["size"], {k: round(v, 4) for k, v in
                  results["strategies"][name][-1].items() if isinstance(v, float)})
    results["strategies"]["Random Noise"] = []
    for n in bs["sizes"]:
        draws = [evaluate(random_circuit(vlm.num_layers, vlm.num_heads, n, seed=cfg["seed"] * 1000 + r))
                 for r in range(bs["random_draws"])]
        merged = {k: sum(d[k] for d in draws) / len(draws) for k in draws[0]}
        merged.update(requested=n, draws=draws)
        results["strategies"]["Random Noise"].append(merged)
    write_json(out / "budget_sensitivity.json", results)
    print(f"Saved {out / 'budget_sensitivity.json'}")


if __name__ == "__main__":
    main()
