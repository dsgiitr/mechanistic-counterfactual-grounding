"""Step 4: benchmark scores before and after oracle-assisted circuit patching (Table 3).

Every item is patched: its circuit is discovered for the model's own answer,
K latent counterfactuals are drawn and filtered by the retention rule, and each
circuit node is replaced by its mean activation over the retained
counterfactuals in a forward pass on the factual input. Items with no visual
input, no scorable answer, an empty circuit or no retained counterfactual keep
their base prediction (the reason is recorded).

Shard with --num_shards/--shard to spread the benchmark over several GPUs.
Output  results/<model>/<dataset>/benchmark[_shard<i>of<n>].jsonl
"""
import torch
from tqdm import tqdm

from common import config, load_model, parser, run_dir

from mcg.background import MeanCache
from mcg.pipeline import ExampleState, base_prediction, discover_circuit, oracle_patch, run_counterfactuals
from mcg.tasks import load_benchmark
from mcg.utils import append_jsonl, done_ids, read_jsonl


def summary(record):
    if record is None:
        return {"answer": None, "correct": False}
    keys = ("answer", "correct", "hallucinated_objects", "mentioned_objects", "recall")
    out = {k: record[k] for k in keys if k in record}
    if isinstance(out.get("answer"), str):
        out["answer"] = out["answer"][-1000:]
    return out


def main():
    p = parser(__doc__)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--max_items", type=int, default=None)
    args = p.parse_args()
    cfg = config(args)
    out = run_dir(args)
    bench = load_benchmark(args.dataset, cfg)
    vlm = load_model(args)
    means = MeanCache(out.parent / "background_means.pt", vlm.device)
    cached = {r["example_id"]: r for r in read_jsonl(out / "predictions.jsonl")}
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard}of{args.num_shards}"
    path = out / f"benchmark{suffix}.jsonl"
    finished = done_ids(path)
    count = len(bench) if args.max_items is None else min(args.max_items, len(bench))
    for index in tqdm(range(args.shard, count, args.num_shards), desc=f"{args.model}/{args.dataset}"):
        item = bench.item(index)
        if item["example_id"] in finished:
            continue
        row = cached.get(item["example_id"])
        base = (row if row.get("parsed") else None) if row is not None else base_prediction(vlm, bench, item, cfg)
        record = {"example_id": item["example_id"], "index": index, "gt": item["gt"], "base": summary(base)}
        status = None
        if item["image"] is None:
            status = "no_visual_input"
        elif base is None:
            status = "unscorable_base_answer"
        elif not base.get("target_ids"):
            status = "no_target_mention"
        if status is None:
            state = ExampleState(vlm, bench, item, base)
            ex_means = means.for_inputs(vlm, state.score_inputs["input_ids"], state.answer_pos)
            circuit, _ = discover_circuit(vlm, state, ex_means, cfg)
            if not circuit:
                status = "empty_circuit"
            else:
                _, mean_acts, info = run_counterfactuals(vlm, bench, state, circuit, cfg, args.K, cfg["seed"],
                                                         item["example_id"], effects=False)
                record.update(info, circuit_size=len(circuit))
                if mean_acts is None:
                    status = "no_retained_counterfactual"
                else:
                    record["patched"] = oracle_patch(vlm, bench, state, mean_acts, cfg)
                    status = "patched"
            del state
        if "patched" not in record:
            record["patched"] = dict(record["base"])
        record["status"] = status
        append_jsonl(path, record)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
