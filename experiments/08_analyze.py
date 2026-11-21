"""Step 8: every table, figure and statistic of the paper.

Reads results/<model>/<dataset>/stability_K*.jsonl, results/timing.json,
results/llava-1.5-7b/pope/budget_sensitivity.json and writes to results/paper/:

  table1_retention.md              retained counterfactuals out of K            (Table 1)
  table2_variance.md               group-mean V, ratio, plug-in K_bar           (Table 2)
  table3_budget_sensitivity.md     K_bar ratio over (epsilon, B)                (Table 3)
  table5_rules.md                  budgets under plug-in, EB and Hoeffding      (Table 5)
  table6_audit_cost.md             counterfactuals, passes, GPU-seconds         (Table 6)
  table7_oracle_restoration.md     oracle-patch corrections, Wilson CIs         (Table 7)
  figure4_hoeffding_vs_bernstein   budget ratio over (epsilon, sigma^2/B^2)     (Figure 4)
  figure5_budget_rules             budget per node versus V                     (Figure 5)
  figure6_ratios                   hallucinated/grounded ratios per model       (Figure 6)
  figure7_circuit_construction     circuit-size sensitivity                     (Figure 7)
  stability_summary.{md,json}      per-example statistics, CIs, Mann-Whitney test, AUROC,
                                   confidence-matched and label-free comparisons (Section 6)
  bound_constants.md               floor, slope, EB and Hoeffding constants     (Section 3.4)

`--source reported` recomputes Tables 2, 3, 5, 6 and Figures 4-6 from the
quantities reported in the paper (configs/reported.yaml) instead of fresh runs.
"""
import glob
import json
from pathlib import Path

import numpy as np

from common import config, parser

from mcg.bounds import bound_constants, example_statistics
from mcg import report
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
    for model in cfg["models"]["stability"]:
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
        fig.savefig(out / f"figure_group_means_{dataset}.pdf")
        fig.savefig(out / f"figure_group_means_{dataset}.png", dpi=200)
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
    lines = ["# Table 7: oracle restoration diagnostic (hallucinated examples corrected)", "",
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
    (out / "table7_oracle_restoration.md").write_text("\n".join(lines) + "\n")
    write_json(out / "table7_oracle_restoration.json", rows)


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
    fig.savefig(out / "figure7_circuit_construction.pdf")
    fig.savefig(out / "figure7_circuit_construction.png", dpi=200)
    plt.close(fig)


def constants_table(cfg, out):
    b = cfg["bounds"]
    c = bound_constants(b["epsilon"], b["delta"], b["B"])
    lines = ["# Bound constants (Section 3.4, Appendix B)", "", "| quantity | value |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in c.items()]
    (out / "bound_constants.md").write_text("\n".join(lines) + "\n")


def budget_inputs(cfg, results, per_example, source, K):
    """Group-mean V, K=50 retention rate and timings per model (from runs or the paper)."""
    import yaml
    if source == "reported":
        rep = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "reported.yaml").read_text())
        retention = {m: v[50] / 50 for m, v in rep["retained_out_of_K"].items()}
        return rep["group_means_V"], retention, rep["timing_seconds"]
    means, retention = {}, {}
    for model in cfg["models"]["stability"]:
        rows = [r for key, v in per_example.items() if key.startswith(f"{model}|") and key.endswith(f"|{K}") for r in v]
        g = [r["V"] for r in rows if r["group"] == "grounded"]
        h = [r["V"] for r in rows if r["group"] == "hallucinated"]
        if g and h:
            means[model] = {"grounded": float(np.mean(g)), "hallucinated": float(np.mean(h))}
        recs = [r for d in cfg["datasets"] for r in read_jsonl(results / model / d / "stability_K50.jsonl")
                if r["group"] == "hallucinated" and r.get("status") == "ok"]
        if recs:
            retention[model] = float(np.mean([r["num_retained"] for r in recs])) / 50
    timing_path = results / "timing.json"
    timing = json.loads(timing_path.read_text())["models"] if timing_path.exists() else {}
    return means, retention, timing


def budget_tables(cfg, means, retention, timing, out, source):
    b = cfg["bounds"]
    eps, delta, B = b["epsilon"], b["delta"], b["B"]
    if not means:
        return
    label = {"reported": "inputs: values reported in the paper (configs/reported.yaml)", "runs": "inputs: fresh runs"}[source]
    t2 = report.table2(means, eps, delta, B)
    lines = ["# Table 2: mean restoration-effect variance and plug-in budget", "", label, "",
             "| Model | V grd. | V hal. | ratio | K̄ grd. → hal. | ratio |", "|---|---|---|---|---|---|"]
    lines += [f"| {model_label(r['model'])} | {r['V_grounded']:.3f} | {r['V_hallucinated']:.3f} | {r['V_ratio']:.1f}× | "
              f"{r['K_grounded']:.1f} → {r['K_hallucinated']:.1f} | {r['K_ratio']:.2f}× |" for r in t2]
    (out / "table2_variance.md").write_text("\n".join(lines) + "\n")
    t3 = report.table3(means, delta)
    models = list(means)
    lines = ["# Table 3: hallucinated-to-grounded plug-in budget ratio", "", label, "",
             "| ε | B | " + " | ".join(model_label(m) for m in models) + " |", "|---|---|" + "---|" * len(models)]
    lines += [f"| {r['epsilon']:.3f} | {r['B']:.1f} | " + " | ".join(f"{r[m]:.2f}" for m in models) + " |" for r in t3]
    (out / "table3_budget_sensitivity.md").write_text("\n".join(lines) + "\n")
    t5 = report.table5(means, eps, delta, B)
    lines = ["# Table 5: counterfactuals per node, grounded/hallucinated, under each rule", "", label, "",
             "| Rule | " + " | ".join(model_label(m) for m in models) + " |", "|---|" + "---|" * len(models)]
    for rule, values in t5.items():
        cells = [f"{values[m][0]:.1f}" if rule == "Hoeffding" else
                 "/".join(f"{x:.1f}" if isinstance(x, float) else str(x) for x in values[m]) for m in models]
        lines.append(f"| {rule} | " + " | ".join(cells) + " |")
    (out / "table5_rules.md").write_text("\n".join(lines) + "\n")
    t6 = []
    if all(m in retention and m in timing for m in models):
        t6 = report.table6(means, retention, timing, eps, delta, B)
        lines = ["# Table 6: cost of auditing one answer at its plug-in budget", "", label, "",
                 "| Model | K | gen. | seqs | TF passes (|S|=20/50) | GPU-s hal. | GPU-s grd. |", "|---|---|---|---|---|---|---|"]
        for r in t6:
            h, g = r["hallucinated"], r["grounded"]
            lines.append(f"| {model_label(r['model'])} | {h['K']} | {h['generated']} | {h['sequences']} | "
                         f"{h['tf_passes'][0]}/{h['tf_passes'][1]} | {h['gpu_seconds'][0]:.0f}/{h['gpu_seconds'][1]:.0f} | "
                         f"{g['gpu_seconds'][0]:.0f}/{g['gpu_seconds'][1]:.0f} |")
        (out / "table6_audit_cost.md").write_text("\n".join(lines) + "\n")
    write_json(out / "budget_tables.json", {"source": source, "means": means, "retention": retention,
                                            "table2": t2, "table3": t3, "table5": t5, "table6": t6})
    budget_figures(means, eps, delta, B, out)


def budget_figures(means, eps, delta, B, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mcg.bounds import eb_budget
    markers = ["o", "s", "^", "D"]
    # Figure 4
    E = np.logspace(-2, np.log10(0.3), 200)
    R = np.logspace(-3, np.log10(0.25), 200)
    Z = np.log10([[report.hoeffding_bernstein_ratio(e, r, delta) for e in E] for r in R])
    fig, ax = plt.subplots(figsize=(5, 4))
    lim = np.abs(Z).max()
    mesh = ax.pcolormesh(E, R, Z, cmap="RdBu", vmin=-lim, vmax=lim, shading="auto")
    cs = ax.contour(E, R, Z, levels=[0, np.log10(2), np.log10(5)], colors="k", linewidths=0.8)
    ax.clabel(cs, fmt={0: "1×", np.log10(2): "2×", np.log10(5): "5×"}, fontsize=7)
    for (model, m), mk in zip(means.items(), markers):
        ax.scatter([eps], [m["grounded"] / B ** 2], marker=mk, facecolors="none", edgecolors="k")
        ax.scatter([eps], [m["hallucinated"] / B ** 2], marker=mk, color="k")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tolerance ε"); ax.set_ylabel("σ²/B²")
    fig.colorbar(mesh, label="log10(K_Hoeff / K_Bern)")
    fig.tight_layout(); fig.savefig(out / "figure4_hoeffding_vs_bernstein.pdf"); fig.savefig(out / "figure4_hoeffding_vs_bernstein.png", dpi=200)
    plt.close(fig)
    # Figure 5
    V = np.linspace(0, 0.09, 200)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(V, [report.plugin(v, eps, delta, B) for v in V], "-", color="k", label="plug-in Bernstein (used)")
    ax.plot(V, [eb_budget(v, eps, delta, B) for v in V], "--", color="k", label="empirical Bernstein (one-sided)")
    ax.plot(V, [B ** 2 * np.log(2 / delta) / (2 * eps ** 2)] * len(V), ":", color="k", label="Hoeffding")
    for (model, m), mk in zip(means.items(), markers):
        ax.scatter([m["grounded"]], [report.plugin(m["grounded"], eps, delta, B)], marker=mk, facecolors="none",
                   edgecolors="#2b6cb0", label=model_label(model))
        ax.scatter([m["hallucinated"]], [report.plugin(m["hallucinated"], eps, delta, B)], marker=mk, color="#c53030")
    ax.set_xlabel("mean circuit variance V"); ax.set_ylabel("counterfactuals per node"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figure5_budget_rules.pdf"); fig.savefig(out / "figure5_budget_rules.png", dpi=200)
    plt.close(fig)
    # Figure 6
    models = list(means)
    var_ratio = [means[m]["hallucinated"] / means[m]["grounded"] for m in models]
    plug_ratio = [report.plugin(means[m]["hallucinated"], eps, delta, B) / report.plugin(means[m]["grounded"], eps, delta, B) for m in models]
    eb_ratio = [eb_budget(means[m]["hallucinated"], eps, delta, B) / eb_budget(means[m]["grounded"], eps, delta, B) for m in models]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = range(len(models))
    ax.plot(x, var_ratio, "o-", label="variance ratio")
    ax.plot(x, plug_ratio, "s-", label="plug-in K̄ ratio")
    ax.plot(x, eb_ratio, "^--", label="emp. Bernstein ratio (approx.)")
    for i, v in enumerate(var_ratio):
        ax.annotate(f"{v:.1f}×", (i, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    ax.set_xticks(list(x), [model_label(m) for m in models], fontsize=7)
    ax.set_ylabel("hallucinated / grounded"); ax.axhline(1, color="0.6", lw=0.8, ls=":"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figure6_ratios.pdf"); fig.savefig(out / "figure6_ratios.png", dpi=200)
    plt.close(fig)


def main():
    p = parser(__doc__, model=False, dataset=False)
    p.add_argument("--K", type=int, default=50, help="counterfactual pool behind V and Table 7")
    p.add_argument("--source", choices=["runs", "reported"], default="runs",
                   help="inputs for Tables 2, 3, 5, 6 and Figures 4-6")
    p.add_argument("--out", default=None, help="output directory (default results/paper, or "
                                               "results/paper/from_reported with --source reported)")
    args = p.parse_args()
    cfg = config(args)
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "paper" / ("from_reported" if args.source == "reported" else "")
    out.mkdir(parents=True, exist_ok=True)
    constants_table(cfg, out)
    summary, per_example = stability(cfg, results, out)
    stability_figures(cfg, summary, per_example, out, K=args.K)
    retention_table(cfg, results, out)
    correction_table(cfg, results, out, K=args.K)
    budget_figure(cfg, results, out)
    budget_tables(cfg, *budget_inputs(cfg, results, per_example, args.source, args.K), out, args.source)
    print(f"Wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
