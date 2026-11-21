"""Step 4: output-level confidence control of donor mixing (Section 5.1, Figure 1).

On POPE (random split) the model's hallucinated "yes" answers (object absent)
are compared with (i) randomly drawn grounded "yes" answers and (ii) grounded
"yes" answers matched one-to-one on factual p(yes). Each example is mixed with
`donors` donor images (Fig. 1 rule: no queried object and none of its five most
frequently co-occurring COCO categories) at each u of a fixed grid,
h~ = u h(I) + (1 - u) h(I').

p(yes) = P(yes tokens) / (P(yes tokens) + P(no tokens)) at the answer position.
Signals (hallucinated = positive class):
  confidence   -p(yes) on the factual input (low confidence flags)
  u50          largest u at which the donor-averaged p(yes) falls below 0.5
               (linear interpolation on the grid; 0 if it never does)
  donor var.   variance across donors of logit p(yes) at u = 0.7
AUROCs and the curve bands use 95% bootstrap intervals clustered by image.

Requires step 1 (predictions.jsonl) for the same model on POPE.
Output  results/<model>/pope/confidence_control.json and figure1_confidence_control.pdf
"""
import json
import random

import numpy as np
import torch
from tqdm import tqdm

from common import config, load_model, parser, run_dir

from mcg.counterfactuals import donor_inputs, mixing, projector_outputs
from mcg.stats import cluster_bootstrap_auroc, cluster_bootstrap_mean, match_on_confidence
from mcg.tasks import load_benchmark, load_donor_pool
from mcg.utils import derive_seed, read_jsonl, write_json


def p_yes(vlm, logits):
    sets = vlm.binary_token_sets()
    probs = logits.float().softmax(-1)
    yes, no = probs[sets["yes"]].sum(), probs[sets["no"]].sum()
    return float(yes / (yes + no))


def crossing(us, curve):
    """Largest u (scanning from u = 1 downwards) where the curve drops below 0.5."""
    for i in range(1, len(us)):
        if curve[i] < 0.5 <= curve[i - 1]:
            frac = (curve[i - 1] - 0.5) / (curve[i - 1] - curve[i])
            return us[i - 1] + frac * (us[i] - us[i - 1])
    return 1.0 if curve[0] < 0.5 else 0.0


def main():
    p = parser(__doc__, model="llava-1.5-7b", dataset=False)
    p.add_argument("--num_random", type=int, default=120)
    p.add_argument("--donors", type=int, default=4)
    p.add_argument("--u_grid", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    p.add_argument("--variance_u", type=float, default=0.7)
    args = p.parse_args()
    args.dataset = "pope"
    cfg = config(args)
    out = run_dir(args)
    rows = [r for r in read_jsonl(out / "predictions.jsonl") if r.get("parsed") and r.get("target_ids")]
    hallucinated = [r for r in rows if r["answer"] == "yes" and r["gt"] == "no"]
    grounded = [r for r in rows if r["answer"] == "yes" and r["gt"] == "yes"]
    bench = load_benchmark("pope", cfg)
    pool = load_donor_pool(cfg)
    vlm = load_model(args)

    def factual(r):
        item = bench.item(r["index"])
        inputs = vlm.prepare(item["image"], bench.prompt(item), r["score_prefix_ids"])
        return item, inputs, p_yes(vlm, vlm.forward_logits(inputs)[0, -1])

    for r in tqdm(hallucinated + grounded, desc="factual p(yes)"):
        r["factual_p_yes"] = factual(r)[2]
    rng = random.Random(cfg["seed"])
    random_grounded = rng.sample(grounded, min(args.num_random, len(grounded)))
    matched = [g for _, g in match_on_confidence(hallucinated, grounded, key="factual_p_yes", caliper=1.0)]
    groups = {"hallucinated": hallucinated, "grounded_random": random_grounded, "grounded_matched": matched}

    curves = {}
    for r in tqdm({id(x): x for g in groups.values() for x in g}.values(), desc="donor mixing"):
        item, inputs, base = factual(r)
        queried = pool.queried_object(item)
        candidates = pool.candidates(queried, item["image_key"])
        picker = random.Random(derive_seed(cfg["seed"], r["example_id"], "fig1-donors"))
        donors = picker.sample(candidates, min(args.donors, len(candidates)))
        grid = np.zeros((len(donors), len(args.u_grid)))
        for d, index in enumerate(donors):
            feats = projector_outputs(vlm, donor_inputs(vlm, pool.images[index][1](), item["image"], bench.prompt(item)))
            for j, u in enumerate(args.u_grid):
                if u >= 1.0:
                    grid[d, j] = base
                    continue
                with mixing(vlm, feats, u):
                    grid[d, j] = p_yes(vlm, vlm.forward_logits(inputs)[0, -1])
        curves[r["example_id"]] = {"image": item["image_key"], "p_yes": grid.tolist(), "factual": base,
                                   "donors": [pool.images[i][0] for i in donors]}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def signals(r):
        c = curves[r["example_id"]]
        grid = np.asarray(c["p_yes"])
        j = args.u_grid.index(args.variance_u)
        logit = np.log(np.clip(grid[:, j], 1e-6, 1 - 1e-6) / np.clip(1 - grid[:, j], 1e-6, 1))
        return {"confidence": -c["factual"], "u50": crossing(args.u_grid, grid.mean(0)),
                "donor_variance": float(np.var(logit, ddof=1)) if len(logit) > 1 else 0.0}

    result = {"u_grid": args.u_grid, "group_sizes": {k: len(v) for k, v in groups.items()},
              "num_images_hallucinated": len({r["image_key"] for r in hallucinated}),
              "median_factual_p_yes": {k: float(np.median([r["factual_p_yes"] for r in v])) for k, v in groups.items()},
              "curves": {}, "auroc": {}}
    for name, members in groups.items():
        per_u = []
        for j in range(len(args.u_grid)):
            values = [np.mean(np.asarray(curves[r["example_id"]]["p_yes"])[:, j]) for r in members]
            mean, ci = cluster_bootstrap_mean(values, [r["image_key"] for r in members])
            per_u.append({"u": args.u_grid[j], "mean": mean, "ci95": ci})
        result["curves"][name] = per_u
    hal_sig = [signals(r) for r in hallucinated]
    for control in ("grounded_random", "grounded_matched"):
        con_sig = [signals(r) for r in groups[control]]
        result["auroc"][control] = {
            key: cluster_bootstrap_auroc([s[key] for s in hal_sig], [r["image_key"] for r in hallucinated],
                                         [s[key] for s in con_sig], [r["image_key"] for r in groups[control]])
            for key in ("confidence", "u50", "donor_variance")}
    result["examples"] = curves
    write_json(out / "confidence_control.json", result)
    figure(result, out)
    print(json.dumps(result["auroc"], indent=2))


def figure(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    style = {"hallucinated": ("o", "-", "#c05621"), "grounded_random": ("s", "--", "#2b6cb0"),
             "grounded_matched": ("^", ":", "#2f855a")}
    for name, points in result["curves"].items():
        marker, ls, color = style[name]
        u = [p["u"] for p in points]
        axes[0].plot(u, [p["mean"] for p in points], marker=marker, ls=ls, color=color,
                     label=f"{name.replace('_', ', ')} (n={result['group_sizes'][name]})")
        axes[0].fill_between(u, [p["ci95"][0] for p in points], [p["ci95"][1] for p in points], color=color, alpha=0.2)
    axes[0].axhline(0.5, color="0.5", lw=0.8, ls=":")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("u = weight on the image's own visual tokens")
    axes[0].set_ylabel("mean p(yes)")
    axes[0].legend(fontsize=7)
    labels = ["confidence", "u50", "donor_variance"]
    for offset, (control, marker) in enumerate((("grounded_random", "o"), ("grounded_matched", "^"))):
        values = result["auroc"][control]
        x = np.arange(len(labels)) + (offset - 0.5) * 0.15
        mid = [values[k]["auroc"] for k in labels]
        err = [[m - values[k]["ci95"][0] for k, m in zip(labels, mid)], [values[k]["ci95"][1] - m for k, m in zip(labels, mid)]]
        axes[1].errorbar(x, mid, yerr=err, fmt=marker, capsize=3, label=f"vs. {control.replace('_', ', ')}")
    axes[1].axhline(0.5, color="0.5", lw=0.8, ls=":")
    axes[1].set_xticks(range(len(labels)), ["confidence\n(low = flag)", "crossing\npoint $u_{50}$", "donor variance\nat u = 0.7"])
    axes[1].set_ylabel("AUROC for flagging hallucinations")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure1_confidence_control.pdf")
    fig.savefig(out / "figure1_confidence_control.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
