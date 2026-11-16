"""Step 1: base predictions on a benchmark and selection of the studied examples.

Runs the model greedily on every benchmark item (the Thinking model generates
its reasoning trace here; that trace is later held fixed), labels each answer as
hallucinated or grounded, and selects (Section 4.1)
  * the first `num_hallucinated` hallucinated items in benchmark order,
  * `num_grounded` items drawn uniformly at random from the correctly answered ones,
each with a distinct image and a scorable answer y.

Outputs  results/<model>/<dataset>/predictions.jsonl   one row per benchmark item
         results/<model>/<dataset>/examples.json        the selected examples
"""
import random

from common import config, load_model, parser, run_dir

from mcg.pipeline import base_prediction
from mcg.tasks import load_benchmark
from mcg.utils import append_jsonl, done_ids, read_jsonl, write_json


def select(rows, cfg, bench):
    ex_cfg = cfg["examples"]
    usable = [r for r in rows if r.get("parsed") and r.get("target_ids") and r.get("has_image")]
    used, hallucinated = set(), []
    for r in usable:
        if r["hallucinated"] and len(hallucinated) < ex_cfg["num_hallucinated"]:
            if ex_cfg["distinct_images"] and r["image_key"] in used:
                continue
            used.add(r["image_key"])
            hallucinated.append(dict(r, group="hallucinated"))
    pool = [r for r in usable if r["correct"]]
    random.Random(cfg["seed"]).shuffle(pool)
    grounded = []
    for r in pool:
        if len(grounded) >= ex_cfg["num_grounded"]:
            break
        if ex_cfg["distinct_images"] and r["image_key"] in used:
            continue
        used.add(r["image_key"])
        grounded.append(dict(r, group="grounded"))
    return hallucinated, grounded


def main():
    p = parser(__doc__)
    p.add_argument("--max_items", type=int, default=None, help="scan only the first N items")
    args = p.parse_args()
    cfg = config(args)
    out = run_dir(args)
    bench = load_benchmark(args.dataset, cfg)
    vlm = load_model(args)
    pred_path = out / "predictions.jsonl"
    finished = done_ids(pred_path)
    count = len(bench) if args.max_items is None else min(args.max_items, len(bench))
    for index in range(count):
        item = bench.item(index)
        if item["example_id"] in finished:
            continue
        row = {"example_id": item["example_id"], "index": index, "image_key": item["image_key"],
               "gt": item["gt"], "has_image": item["image"] is not None}
        record = base_prediction(vlm, bench, item, cfg)
        row["parsed"] = record is not None
        if record is not None:
            row.update(record)
            row["answer_text"] = row["answer_text"][-2000:]
        append_jsonl(pred_path, row)
        if index % 50 == 0:
            print(f"[{args.model}/{args.dataset}] {index + 1}/{count}", flush=True)
    rows = sorted(read_jsonl(pred_path), key=lambda r: r["index"])
    hallucinated, grounded = select(rows, cfg, bench)
    write_json(out / "examples.json", hallucinated + grounded)
    parsed = [r for r in rows if r.get("parsed")]
    print(f"{len(rows)} items, {len(parsed)} parsed, "
          f"{sum(r['hallucinated'] for r in parsed)} hallucinated; selected "
          f"{len(hallucinated)} hallucinated + {len(grounded)} grounded examples -> {out / 'examples.json'}")


if __name__ == "__main__":
    main()
