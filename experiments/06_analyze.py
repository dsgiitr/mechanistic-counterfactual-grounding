"""Step 6: every table, figure and statistic of the paper from the saved runs.

Reads results/<model>/<dataset>/{stability_K*.jsonl, benchmark*.jsonl,
budget_sensitivity.json} and writes to results/paper/:

  table1_retention.md             generated / retained counterfactuals          (Table 1)
  figure2_stability_<ds>.pdf      V_i and K_bar, grounded vs hallucinated        (Figure 2)
  stability_summary.{md,json}     group means, ratios, CIs, rank test, AUROC,
                                  K_bar = floor + slope * V consistency check,
                                  confidence-matched and label-free comparisons (Sec. 4.2, 6)
  figure_stability_distributions_<ds>.pdf   per-example distributions
  table2_correction.md            oracle-patch corrections with Wilson CIs      (Table 2)
  table3_benchmark.md             benchmark scores before / after patching,
                                  with corrected and newly broken counts        (Table 3)
  figure3_budget_sensitivity.pdf  circuit-size sensitivity                      (Figure 3)
  bound_constants.md              floor, slope, empirical-Bernstein and Hoeffding budgets
"""
import glob
import json
from pathlib import Path

import numpy as np

from common import config, parser

from mcg.bounds import bound_constants, example_statistics
from mcg.models import model_label
from mcg.stats import (auroc, bootstrap_ratio_ci, mann_whitney, match_on_confidence,
                       summarize, wilson_interval)
from mcg.utils import read_jsonl, write_json

DATASET_LABEL = {"pope": "POPE", "coco": "COCO", "hallusionbench": "HallusionBench"}


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def effects_of(record, use_all=False):
    circuit = record["circuit"]
    rows = [c for c in record["counterfactuals"] if "deltas" in c and (use_all or c["retained"])]
    return {f"{n['layer']}_{n['head']}": [c["deltas"][i] for c in rows] for i, n in enumerate(circuit)}


def example_rows(records, b, use_all=False):
    rows = []
    for r in records:
        if r.get("status") != "ok":
            continue
        s = example_statistics(effects_of(r, use_all), b["epsilon"], b["delta"], b["B"])
        if s is None:
            continue
        rows.append({"example_id": r["example_id"], "group": r["group"], "V": s["V"], "K_bar": s["K_bar"],
                     "K_bar_eb": s["K_bar_eb"], "factual_prob": r["factual_prob"],
                     "num_effects": s["num_effects"], "num_nodes": s["num_nodes"],
                     "range_exceeds_B": s["fraction_nodes_range_exceeds_B"]})
    return rows


def compare(rows, label=""):
    hall = [r for r in rows if r["group"] == "hallucinated"]
    grd = [r for r in rows if r["group"] == "grounded"]
    out = {"label": label, "n_hallucinated": len(hall), "n_grounded": len(grd)}
    for key in ("V", "K_bar"):
        h, g = [r[key] for r in hall], [r[key] for r in grd]
        out[key] = {"hallucinated": summarize(h), "grounded": summarize(g),
                    "ratio": float(np.mean(h) / np.mean(g)) if h and g and np.mean(g) > 0 else None,
                    "ratio_ci95": list(bootstrap_ratio_ci(h, g)) if h and g else None,
                    "mann_whitney": mann_whitney(h, g), "auroc": auroc(h, g)}
    return out


def stability(cfg, results, out):
    b = cfg["bounds"]
    constants = bound_constants(b["epsilon"], b["delta"], b["B"])
    summary, per_example = {}, {}
    for model in cfg["models"]["stability"] + [m for m in cfg["models"]["patching"]
                                                if m not in cfg["models"]["stability"]]:
        for dataset in cfg["datasets"]:
            for path in sorted(glob.glob(str(results / model / dataset / "stability_K*.jsonl"))):
                K = int(Path(path).stem.split("_K")[1])
                records = read_jsonl(path)
                rows = example_rows(records, b)
                if not rows:
                    continue
                key = f"{model}|{dataset}|{K}"
                entry = compare(rows, key)
                for group in ("hallucinated", "grounded"):
                    V = entry["V"][group].get("mean")
                    if V is not None:
                        entry["K_bar"][group]["predicted_from_V"] = constants["k_hat_floor"] + constants["k_hat_slope"] * V
                matched = match_on_confidence([r for r in rows if r["group"] == "hallucinated"],
                                              [r for r in rows if r["group"] == "grounded"])
                entry["confidence_matched"] = compare([p for pair in matched for p in pair], key + "|matched")
                entry["confidence_matched"]["pairs"] = len(matched)
                if any("deltas" in c and not c["retained"] for r in records for c in r.get("counterfactuals", [])):
                    entry["label_free"] = compare(example_rows(records, b, use_all=True), key + "|all")
                entry["status_counts"] = {s: sum(r.get("status") == s for r in records)
                                          for s in {r.get("status") for r in records}}
                entry["examples_without_two_retained"] = sum(
                    1 for r in records if r.get("status") == "ok" and r.get("num_retained", 0) < 2)
                summary[key] = entry
                per_example[key] = rows
    write_json(out / "stability_summary.json", {"bound_constants": constants, "comparisons": summary,
                                               "per_example": per_example})
    lines = ["# Counterfactual stability (Section 4.2)", "",
             f"Bounds: epsilon={b['epsilon']}, delta={b['delta']}, B={b['B']}; "
             f"K_hat floor={constants['k_hat_floor']:.2f}, slope={constants['k_hat_slope']:.1f}.", "",
             "| Model | Dataset | K | n (H/G) | V grounded | V halluc. | ratio V [95% CI] | K̄ grounded | "
             "K̄ halluc. | ratio K̄ | K̄ pred. from V (G/H) | AUROC K̄ | Mann–Whitney p | matched pairs | "
             "matched ratio K̄ | label-free ratio K̄ |",
             "|" + "---|" * 16]
    for key, e in summary.items():
        model, dataset, K = key.split("|")
        V, Kb = e["V"], e["K_bar"]
        ci = V["ratio_ci95"] or [None, None]
        lines.append(
            f"| {model_label(model)} | {DATASET_LABEL[dataset]} | {K} | {e['n_hallucinated']}/{e['n_grounded']} | "
            f"{fmt(V['grounded'].get('mean'), 4)} | {fmt(V['hallucinated'].get('mean'), 4)} | "
            f"{fmt(V['ratio'], 2)} [{fmt(ci[0], 2)}, {fmt(ci[1], 2)}] | {fmt(Kb['grounded'].get('mean'), 1)} | "
            f"{fmt(Kb['hallucinated'].get('mean'), 1)} | {fmt(Kb['ratio'], 2)} | "
            f"{fmt(Kb['grounded'].get('predicted_from_V'), 1)}/{fmt(Kb['hallucinated'].get('predicted_from_V'), 1)} | "
            f"{fmt(Kb['auroc'], 3)} | {fmt(Kb['mann_whitney']['p_value'], 4)} | "
            f"{e['confidence_matched']['pairs']} | {fmt(e['confidence_matched']['K_bar']['ratio'], 2)} | "
            f"{fmt(e.get('label_free', {}).get('K_bar', {}).get('ratio'), 2)} |")
    (out / "stability_summary.md").write_text("\n".join(lines) + "\n")
    return summary, per_example


def stability_figures(cfg, summary, per_example, out, K=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = cfg["models"]["stability"]
    for dataset in cfg["datasets"]:
        keys = [(m, f"{m}|{dataset}|{K}") for m in models if f"{m}|{dataset}|{K}" in summary]
        if not keys:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 0.6 * len(keys) + 1.6))
        for ax, metric, title in ((axes[0], "V", "(a) mean restoration-effect variance $V_i$"),
                                  (axes[1], "K_bar", r"(b) projected budget $\bar K$")):
            for row, (model, key) in enumerate(keys):
                g = summary[key][metric]["grounded"].get("mean")
                h = summary[key][metric]["hallucinated"].get("mean")
                if g is not None and h is not None:
                    ax.plot([g, h], [row, row], color="0.6", lw=2, zorder=1)
                if g is not None:
                    ax.scatter([g], [row], marker="o", s=60, color="#2b6cb0", zorder=2, label="grounded")
                if h is not None:
                    ax.scatter([h], [row], marker="D", s=60, color="#c53030", zorder=2, label="hallucinated")
            ax.set_yticks(range(len(keys)), [model_label(m) for m, _ in keys])
            ax.set_title(title, fontsize=10)
            ax.grid(axis="x", ls=":", alpha=0.6)
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[0].legend(unique.values(), unique.keys(), loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"figure2_stability_{dataset}.pdf")
        fig.savefig(out / f"figure2_stability_{dataset}.png", dpi=200)
        plt.close(fig)
        fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3), squeeze=False)
        for ax, (model, key) in zip(axes[0], keys):
            rows = per_example[key]
            data = [[r["K_bar"] for r in rows if r["group"] == g] for g in ("grounded", "hallucinated")]
            ax.boxplot(data, showfliers=False)
            for i, values in enumerate(data):
                jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(values))
                ax.scatter(np.full(len(values), i + 1) + jitter, values, s=8, alpha=0.6)
            ax.set_xticks([1, 2], ["G", "H"])
            ax.set_title(model_label(model), fontsize=9)
        axes[0][0].set_ylabel(r"$\bar K_i$")
        fig.tight_layout()
        fig.savefig(out / f"figure_stability_distributions_{dataset}.pdf")
        plt.close(fig)


def retention_table(cfg, results, out):
    lines = ["# Table 1: generated and retained counterfactuals", "",
             "Representative example = first hallucinated example (benchmark order) of the dataset.", "",
             "| Model | Dataset | Generated | Retained (repr.) | Rate % (repr.) | mean retained, halluc. | "
             "mean retained, grounded |", "|---|---|---|---|---|---|---|"]
    rows = []
    for model in cfg["models"]["stability"]:
        for dataset in cfg["datasets"]:
            examples_path = results / model / dataset / "examples.json"
            order = [e["example_id"] for e in json.loads(examples_path.read_text())] if examples_path.exists() else []
            for path in sorted(glob.glob(str(results / model / dataset / "stability_K*.jsonl")),
                               key=lambda p: int(Path(p).stem.split("_K")[1])):
                K = int(Path(path).stem.split("_K")[1])
                records = {r["example_id"]: r for r in read_jsonl(path) if r.get("status") == "ok"}
                hall = [records[i] for i in order if i in records and records[i]["group"] == "hallucinated"]
                grd = [r for r in records.values() if r["group"] == "grounded"]
                if not hall:
                    continue
                rep = hall[0]
                row = {"model": model, "dataset": dataset, "K": K, "retained": rep["num_retained"],
                       "rate": 100 * rep["num_retained"] / K, "example_id": rep["example_id"],
                       "mean_retained_hallucinated": float(np.mean([r["num_retained"] for r in hall])),
                       "mean_retained_grounded": float(np.mean([r["num_retained"] for r in grd])) if grd else None}
                rows.append(row)
                lines.append(f"| {model_label(model)} | {DATASET_LABEL[dataset]} | {K} | {row['retained']} | "
                             f"{row['rate']:.0f} | {row['mean_retained_hallucinated']:.1f} | "
                             f"{fmt(row['mean_retained_grounded'], 1)} |")
    (out / "table1_retention.md").write_text("\n".join(lines) + "\n")
    write_json(out / "table1_retention.json", rows)


def correction_table(cfg, results, out, K=50):
    lines = ["# Table 2: hallucinated examples corrected by oracle-assisted circuit patching", "",
             "| Model | Dataset | Corrected | % | 95% Wilson CI | no retained CF | grounded kept correct |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for model in cfg["models"]["stability"]:
        for dataset in cfg["datasets"]:
            records = read_jsonl(results / model / dataset / f"stability_K{K}.jsonl")
            hall = [r for r in records if r["group"] == "hallucinated"]
            if not hall:
                continue
            fixed = sum(bool(r.get("oracle_patch", {}).get("correct")) for r in hall)
            no_cf = sum(r.get("num_retained", 0) == 0 for r in hall)
            grd = [r for r in records if r["group"] == "grounded" and r.get("num_retained", 0) > 0]
            kept = sum(bool(r["oracle_patch"].get("correct")) for r in grd)
            lo, hi = wilson_interval(fixed, len(hall))
            rows.append({"model": model, "dataset": dataset, "corrected": fixed, "n": len(hall),
                         "ci95": [lo, hi], "no_retained": no_cf, "grounded_kept": kept, "grounded_n": len(grd)})
            lines.append(f"| {model_label(model)} | {DATASET_LABEL[dataset]} | {fixed}/{len(hall)} | "
                         f"{100 * fixed / len(hall):.0f} | [{100 * lo:.0f}, {100 * hi:.0f}] | {no_cf} | "
                         f"{kept}/{len(grd)} |")
    (out / "table2_correction.md").write_text("\n".join(lines) + "\n")
    write_json(out / "table2_correction.json", rows)


def benchmark_table(cfg, results, out):
    rows, lines = [], ["# Table 3: benchmark scores before and after oracle-assisted circuit patching", "",
                       "| Model | Dataset | n | Base | Patch | corrected (wrong→right) | broken (right→wrong) | "
                       "patched items |", "|---|---|---|---|---|---|---|---|"]
    for model in cfg["models"]["patching"]:
        for dataset in cfg["datasets"]:
            records = {}
            for path in glob.glob(str(results / model / dataset / "benchmark*.jsonl")):
                for r in read_jsonl(path):
                    records[r["example_id"]] = r
            if not records:
                continue
            recs = list(records.values())
            if dataset == "coco":
                base = [r["base"].get("recall") or 0.0 for r in recs]
                patch = [r["patched"].get("recall") or 0.0 for r in recs]
                score_b, score_p = 100 * np.mean(base), 100 * np.mean(patch)
            else:
                score_b = 100 * np.mean([bool(r["base"].get("correct")) for r in recs])
                score_p = 100 * np.mean([bool(r["patched"].get("correct")) for r in recs])
            fixed = sum(not r["base"].get("correct") and bool(r["patched"].get("correct")) for r in recs)
            broken = sum(bool(r["base"].get("correct")) and not r["patched"].get("correct") for r in recs)
            patched = sum(r["status"] == "patched" for r in recs)
            rows.append({"model": model, "dataset": dataset, "n": len(recs), "base": score_b, "patch": score_p,
                         "corrected": fixed, "broken": broken, "patched_items": patched,
                         "status_counts": {s: sum(r["status"] == s for r in recs) for s in {r["status"] for r in recs}}})
            lines.append(f"| {model_label(model)} | {DATASET_LABEL[dataset]} | {len(recs)} | {score_b:.1f} | "
                         f"{score_p:.1f} | {fixed} | {broken} | {patched} |")
    lines += ["", "POPE and HallusionBench: accuracy (%). COCO: context recall R_ctx (%); "
              "for COCO, corrected/broken refer to hallucination-free captions."]
    (out / "table3_benchmark.md").write_text("\n".join(lines) + "\n")
    write_json(out / "table3_benchmark.json", rows)


def budget_figure(cfg, results, out):
    bs = cfg["budget_sensitivity"]
    path = results / bs["model"] / bs["dataset"] / "budget_sensitivity.json"
    if not path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(path.read_text())
    styles = {"Global": ("o", "-"), "Layer Norm": ("^", "-"), "Iter Decay": ("s", "-"), "Random Noise": ("X", "--")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("faithfulness", "Faithfulness recovered (%)"),
              ("hallucination_remaining", "Hallucination signal remaining (%)"),
              ("accuracy_recovered", "Post-ablation accuracy recovered (%)")]
    for ax, (key, ylabel) in zip(axes, panels):
        for name, points in data["strategies"].items():
            marker, ls = styles.get(name, ("o", "-"))
            ax.plot([p["requested"] for p in points], [100 * p[key] for p in points], marker=marker, ls=ls, label=name)
        ax.set_xlabel("Nodes in circuit")
        ax.set_ylabel(ylabel)
        ax.grid(ls=":", alpha=0.6)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure3_budget_sensitivity.pdf")
    fig.savefig(out / "figure3_budget_sensitivity.png", dpi=200)
    plt.close(fig)


def constants_table(cfg, out):
    b = cfg["bounds"]
    c = bound_constants(b["epsilon"], b["delta"], b["B"])
    lines = ["# Bound constants (Section 3.4, Appendix B)", "", "| quantity | value |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in c.items()]
    (out / "bound_constants.md").write_text("\n".join(lines) + "\n")


def main():
    p = parser(__doc__, model=False, dataset=False)
    p.add_argument("--K", type=int, default=50, help="counterfactual pool used for Figure 2 and Table 2")
    args = p.parse_args()
    cfg = config(args)
    results = Path(args.results)
    out = results / "paper"
    out.mkdir(parents=True, exist_ok=True)
    constants_table(cfg, out)
    summary, per_example = stability(cfg, results, out)
    stability_figures(cfg, summary, per_example, out, K=args.K)
    retention_table(cfg, results, out)
    correction_table(cfg, results, out, K=args.K)
    benchmark_table(cfg, results, out)
    budget_figure(cfg, results, out)
    print(f"Wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
