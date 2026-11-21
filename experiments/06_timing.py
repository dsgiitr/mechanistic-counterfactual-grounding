"""Step 6: timing of the audit primitives (Section 5.2, Table 4; Appendix C).

For each model and the first `num_items` POPE (random split) items, times on the
factual input (no counterfactual hooks):
  TF b=1   one teacher-forced forward pass                       (20 timed runs)
  TF b=16  one batched pass of 16 copies, reported per copy      (10 timed runs)
  10 dec.  one sampled-decoding call: 10 sequences of at most 8 new tokens at
           temperature 0.7                                       (5 timed runs)
after warm-up, and reports medians over items. For a Thinking model it also
times generating the reasoning trace once on the factual input.

Output  results/timing.json
"""
import platform
import statistics
import time
from pathlib import Path

import torch

from common import config, load_model, parser

from mcg.tasks import load_benchmark
from mcg.utils import read_jsonl, write_json


def timed(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def main():
    p = parser(__doc__, model=False, dataset=False)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--num_items", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()
    cfg = config(args)
    bench = load_benchmark("pope", cfg)
    path = Path(args.results) / "timing.json"
    result = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else platform.processor(),
              "torch": torch.__version__, "dtype": args.dtype, "models": {}}
    import transformers
    result["transformers"] = transformers.__version__
    for model in args.models or cfg["models"]["stability"]:
        args.model = model
        vlm = load_model(args, dtype=getattr(torch, args.dtype))
        rows = {r["index"]: r for r in read_jsonl(Path(args.results) / model / "pope" / "predictions.jsonl")}
        per_item = []
        for index in range(args.num_items):
            item = bench.item(index)
            prompt = bench.prompt(item)
            entry = {}
            prefix = []
            if vlm.thinking:
                inputs = vlm.prepare(item["image"], prompt)
                start = time.perf_counter()
                generated = vlm.generate(inputs, cfg["generation"]["max_new_tokens_reasoning"])[0]
                entry["trace_seconds"] = time.perf_counter() - start
                split = vlm.split_reasoning(generated)
                prefix = split[0] if split else generated
                entry["trace_tokens"] = len(prefix)
            elif index in rows and rows[index].get("score_prefix_ids") is not None:
                prefix = rows[index]["score_prefix_ids"]
            inputs = vlm.prepare(item["image"], prompt, prefix)
            batch = vlm.expand(inputs, 16)
            entry["tokens"] = int(inputs["input_ids"].shape[1])
            entry["tf_b1"] = timed(lambda: vlm.forward_logits(inputs), 20)
            entry["tf_b16_per_pass"] = timed(lambda: vlm.forward_logits(batch), 10) / 16
            entry["decode10"] = timed(lambda: vlm.generate(inputs, 8, sample=True, temperature=0.7, n=10), 5)
            per_item.append(entry)
        summary = {key: statistics.median(e[key] for e in per_item) for key in per_item[0] if key != "tokens"}
        summary["tokens"] = [min(e["tokens"] for e in per_item), max(e["tokens"] for e in per_item)]
        summary["items"] = per_item
        result["models"][model] = summary
        print(model, {k: v for k, v in summary.items() if k != "items"})
        vlm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(path, result)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
