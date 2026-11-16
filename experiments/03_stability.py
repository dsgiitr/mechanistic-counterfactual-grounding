"""Step 3: counterfactual circuit stability (Algorithm 1) and oracle-assisted patching.

For every selected example (hallucinated and grounded):
  1. discover its CD-T circuit S* (cached in circuits.jsonl and reused across K),
  2. draw K latent counterfactuals, sample `decodings` answers per counterfactual
     at the configured temperature and apply the retention rule,
  3. for every retained counterfactual and node, record the restoration effect
     Delta_v^(k) (Eq. 2),
  4. patch every node with its mean activation over the retained counterfactuals
     on the factual input and check whether the answer becomes correct (Table 2).

Output  results/<model>/<dataset>/stability_K<K>.jsonl   one record per example
"""
import json

import torch
from tqdm import tqdm

from common import config, load_model, parser, run_dir

from mcg.background import MeanCache
from mcg.bounds import example_statistics
from mcg.pipeline import ExampleState, Timer, discover_circuit, oracle_patch, run_counterfactuals
from mcg.tasks import load_benchmark
from mcg.utils import append_jsonl, done_ids, read_jsonl


def main():
    p = parser(__doc__)
    p.add_argument("--K", type=int, default=50, help="counterfactuals generated per example")
    p.add_argument("--limit", type=int, default=None, help="process only the first N examples")
    p.add_argument("--group", choices=["hallucinated", "grounded"], default=None)
    args = p.parse_args()
    cfg = config(args)
    out = run_dir(args)
    examples = json.loads((out / "examples.json").read_text())
    if args.group:
        examples = [e for e in examples if e["group"] == args.group]
    examples = examples[:args.limit] if args.limit else examples
    bench = load_benchmark(args.dataset, cfg)
    vlm = load_model(args)
    means = MeanCache(out.parent / "background_means.pt", vlm.device)
    circuit_path = out / "circuits.jsonl"
    circuits = {r["example_id"]: r for r in read_jsonl(circuit_path)}
    result_path = out / f"stability_K{args.K}.jsonl"
    finished = done_ids(result_path)
    b = cfg["bounds"]
    for ex in tqdm(examples, desc=f"{args.model}/{args.dataset} K={args.K}"):
        if ex["example_id"] in finished:
            continue
        timer = Timer()
        item = bench.item(ex["index"])
        state = ExampleState(vlm, bench, item, ex)
        example_means = means.for_inputs(vlm, state.score_inputs["input_ids"], state.answer_pos)
        if ex["example_id"] not in circuits:
            circuit, _ = discover_circuit(vlm, state, example_means, cfg)
            circuits[ex["example_id"]] = {"example_id": ex["example_id"], "circuit": circuit,
                                          "contrast_id": state.contrast, "cdt": cfg["cdt"]}
            append_jsonl(circuit_path, circuits[ex["example_id"]])
        circuit = circuits[ex["example_id"]]["circuit"]
        record = {"example_id": ex["example_id"], "group": ex["group"], "model": args.model,
                  "dataset": args.dataset, "K": args.K, "gt": ex["gt"], "answer": ex["answer"][-500:],
                  "target_ids": state.target, "contrast_id": state.contrast,
                  "factual_logprob": state.factual_logprob,
                  "factual_prob": float(torch.tensor(state.factual_logprob).exp()),
                  "circuit": [{"layer": n["layer"], "head": n["head"]} for n in circuit],
                  "alpha": cfg["counterfactuals"]["alpha"], "seed": cfg["seed"]}
        if not circuit:
            record.update(status="empty_circuit", counterfactuals=[], num_retained=0)
            append_jsonl(result_path, record)
            continue
        cfs, mean_acts, info = run_counterfactuals(
            vlm, bench, state, circuit, cfg, args.K, cfg["seed"], ex["example_id"],
            effects=True, effects_on=cfg["counterfactuals"]["effects_on"])
        record.update(info, counterfactuals=cfs, status="ok")
        retained = [c for c in cfs if c["retained"]]
        effects = {f"{n['layer']}_{n['head']}": [c["deltas"][i] for c in retained]
                   for i, n in enumerate(circuit)}
        summary = example_statistics(effects, b["epsilon"], b["delta"], b["B"])
        record["summary"] = None if summary is None else {k: v for k, v in summary.items() if k != "nodes"}
        record["oracle_patch"] = (oracle_patch(vlm, bench, state, mean_acts, cfg)
                                  if mean_acts is not None else {"answer": None, "correct": False,
                                                                 "status": "no_retained_counterfactual"})
        record["seconds"] = timer()
        append_jsonl(result_path, record)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
