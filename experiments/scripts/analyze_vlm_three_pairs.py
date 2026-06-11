import argparse
import math
import os
from datetime import datetime
from itertools import combinations

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError(
        "matplotlib is required for plots. Install with: pip install matplotlib"
    ) from exc


def calc_hoeffding_width(m, delta, b_bound):
    if m <= 0:
        return float("nan")
    return b_bound * math.sqrt((2.0 * math.log(2.0 / delta)) / m)


def calc_empirical_bernstein_width(m, delta, b_bound, sample_variance):
    if m <= 0:
        return float("nan")
    term1 = math.sqrt((2.0 * sample_variance * math.log(2.0 / delta)) / m)
    term2 = (2.0 * b_bound * math.log(2.0 / delta)) / (3.0 * m)
    return term1 + term2


def layer_family(layer_name):
    if layer_name.startswith("attn_layer_"):
        return "attn"
    if layer_name.startswith("mlp_layer_"):
        return "mlp"
    if "final" in layer_name:
        return "final"
    if "head_" in layer_name:
        return "circuit_head"
    return "other"


def rank_desc(values):
    order = np.argsort(-np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def pearson_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def spearman_corr(x, y):
    return pearson_corr(rank_desc(x), rank_desc(y))


def load_delta_data(delta_dir, num_files):
    files = sorted([f for f in os.listdir(delta_dir) if f.endswith(".pt")])[:num_files]
    if len(files) < 2:
        raise RuntimeError("Need at least 2 .pt files for consistency analysis.")

    pair_names = [os.path.splitext(f)[0] for f in files]
    loaded = [torch.load(os.path.join(delta_dir, f), map_location="cpu") for f in files]

    common_layers = sorted(set.intersection(*[set(obj.keys()) for obj in loaded]))
    if not common_layers:
        raise RuntimeError("No common layers across selected delta files.")

    return files, pair_names, loaded, common_layers


def shorten(name, max_len=30):
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def plot_heatmap(matrix, xlabels, ylabels, title, color_label, out_path):
    fig_w = max(8.0, 0.55 * len(xlabels))
    fig_h = max(4.0, 0.60 * len(ylabels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels([shorten(x) for x in xlabels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels([shorten(y) for y in ylabels])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Deep analysis and plots for small-sample VLM delta files."
    )
    parser.add_argument("--delta-dir", default=os.path.join("..", "results", "vlm_deltas"))
    parser.add_argument(
        "--out-dir",
        default=os.path.join("..", "results", "vlm_three_pair_analysis"),
    )
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-files", type=int, default=3)
    parser.add_argument("--sign-eps", type=float, default=1e-10)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    delta_dir = os.path.abspath(os.path.join(script_dir, args.delta_dir))
    out_dir = os.path.abspath(os.path.join(script_dir, args.out_dir))
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    files, pair_names, loaded, layers = load_delta_data(delta_dir, args.num_files)
    p_count = len(pair_names)
    l_count = len(layers)

    h_mat = np.zeros((p_count, l_count), dtype=float)
    eb_mat = np.zeros((p_count, l_count), dtype=float)
    var_mat = np.zeros((p_count, l_count), dtype=float)
    mu_mat = np.zeros((p_count, l_count), dtype=float)

    vectors_by_layer = {layer: [] for layer in layers}

    for i, pair_obj in enumerate(loaded):
        for j, layer in enumerate(layers):
            x = pair_obj[layer].detach().float().cpu().numpy().reshape(-1)
            vectors_by_layer[layer].append(x)

            m = int(x.size)
            mean = float(np.mean(x)) if m > 0 else 0.0
            sample_var = float(np.var(x, ddof=1)) if m > 1 else 0.0
            b_bound = float(np.max(np.abs(x))) if m > 0 else 0.0

            h_mat[i, j] = calc_hoeffding_width(m, args.delta, b_bound)
            eb_mat[i, j] = calc_empirical_bernstein_width(
                m, args.delta, b_bound, sample_var
            )
            var_mat[i, j] = sample_var
            mu_mat[i, j] = mean

    mean_h = h_mat.mean(axis=0)
    mean_eb = eb_mat.mean(axis=0)
    mean_var = var_mat.mean(axis=0)
    mean_mu = mu_mat.mean(axis=0)

    # Pairwise rank consistency via Spearman over layer ranks
    pair_spearman = np.eye(p_count, dtype=float)
    for i in range(p_count):
        for j in range(i + 1, p_count):
            rho = spearman_corr(eb_mat[i], eb_mat[j])
            pair_spearman[i, j] = rho
            pair_spearman[j, i] = rho

    # Leave-one-out ranking robustness
    top_k = min(args.top_k, l_count)
    full_rank_idx = np.argsort(-mean_eb)
    full_top_set = set(full_rank_idx[:top_k].tolist())
    loo_rows = []
    for left_out in range(p_count):
        keep = [idx for idx in range(p_count) if idx != left_out]
        loo_mean = eb_mat[keep, :].mean(axis=0)
        loo_rank_idx = np.argsort(-loo_mean)
        loo_top_set = set(loo_rank_idx[:top_k].tolist())
        overlap = len(full_top_set.intersection(loo_top_set)) / float(top_k)
        rho_full_loo = spearman_corr(mean_eb, loo_mean)
        loo_rows.append(
            {
                "left_out": pair_names[left_out],
                "overlap": overlap,
                "spearman": rho_full_loo,
                "loo_top_layers": [layers[idx] for idx in loo_rank_idx[:top_k]],
            }
        )

    # Family-level aggregates
    fam_indices = {}
    for idx, layer in enumerate(layers):
        fam = layer_family(layer)
        fam_indices.setdefault(fam, []).append(idx)

    fam_agg = {}
    for fam, idxs in fam_indices.items():
        fam_agg[fam] = {
            "avg_bernstein": float(mean_eb[idxs].mean()),
            "avg_variance": float(mean_var[idxs].mean()),
            "count": len(idxs),
        }

    # Sign consistency across dimensions across pairs
    sign_soft = {}
    sign_strict = {}
    for layer in layers:
        vecs = vectors_by_layer[layer]
        min_len = min(v.size for v in vecs)
        stacked = np.stack([v[:min_len] for v in vecs], axis=0)
        signs = np.sign(stacked)
        signs[np.abs(stacked) <= args.sign_eps] = 0.0

        soft = float(np.mean(np.abs(np.sum(signs, axis=0)) / float(p_count)))
        strict = float(np.mean(np.abs(np.sum(signs, axis=0)) == float(p_count)))
        sign_soft[layer] = soft
        sign_strict[layer] = strict

    # Plots
    plot_heatmap(
        eb_mat,
        layers,
        pair_names,
        "Bernstein Width by Pair and Layer",
        "Bernstein Width",
        os.path.join(plot_dir, "pair_layer_bernstein_heatmap.png"),
    )

    sens_idx = np.argsort(-mean_eb)
    sens_layers = [layers[i] for i in sens_idx]
    sens_vals = mean_eb[sens_idx]
    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * l_count), 4.8))
    ax.bar(np.arange(l_count), sens_vals)
    ax.set_xticks(np.arange(l_count))
    ax.set_xticklabels([shorten(x) for x in sens_layers], rotation=45, ha="right")
    ax.set_title("Average Bernstein Width by Layer (Descending)")
    ax.set_ylabel("Avg Bernstein Width")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "avg_bernstein_by_layer.png"), dpi=180)
    plt.close(fig)

    plot_heatmap(
        pair_spearman,
        pair_names,
        pair_names,
        "Pairwise Spearman Rank Correlation",
        "Spearman rho",
        os.path.join(plot_dir, "pairwise_spearman_heatmap.png"),
    )

    sc_sorted = sorted(layers, key=lambda x: sign_soft[x], reverse=True)
    sc_vals = [sign_soft[x] for x in sc_sorted]
    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * l_count), 4.8))
    ax.bar(np.arange(l_count), sc_vals)
    ax.set_xticks(np.arange(l_count))
    ax.set_xticklabels([shorten(x) for x in sc_sorted], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Sign Consistency Across Pairs by Layer")
    ax.set_ylabel("Consistency (0-1)")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "sign_consistency_by_layer.png"), dpi=180)
    plt.close(fig)

    fam_sorted = sorted(fam_agg.keys(), key=lambda f: fam_agg[f]["avg_bernstein"], reverse=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(np.arange(len(fam_sorted)), [fam_agg[f]["avg_bernstein"] for f in fam_sorted])
    ax.set_xticks(np.arange(len(fam_sorted)))
    ax.set_xticklabels(fam_sorted)
    ax.set_title("Average Bernstein Width by Layer Family")
    ax.set_ylabel("Avg Bernstein Width")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "family_avg_bernstein.png"), dpi=180)
    plt.close(fig)

    # Report
    report_path = os.path.join(out_dir, "three_pair_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Three-Pair VLM Delta Deep Analysis\n\n")
        f.write(f"- Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"- Input delta dir: {delta_dir}\n")
        f.write(f"- Files analyzed ({p_count}): {', '.join(files)}\n")
        f.write(f"- Common layers: {l_count}\n")
        f.write(f"- Confidence delta: {args.delta}\n\n")

        f.write("## Top Sensitive Layers (Avg Bernstein)\n")
        for idx in sens_idx[:top_k]:
            f.write(
                f"- {layers[idx]}: avg_bernstein={mean_eb[idx]:.6f}, avg_variance={mean_var[idx]:.6f}, avg_hoeffding={mean_h[idx]:.6f}, avg_mean={mean_mu[idx]:.6f}\n"
            )

        f.write("\n## Top Stable Layers (Avg Bernstein)\n")
        for idx in sens_idx[::-1][:top_k]:
            f.write(
                f"- {layers[idx]}: avg_bernstein={mean_eb[idx]:.6f}, avg_variance={mean_var[idx]:.6f}, avg_hoeffding={mean_h[idx]:.6f}, avg_mean={mean_mu[idx]:.6f}\n"
            )

        f.write("\n## Pairwise Rank Consistency (Spearman)\n")
        for i, j in combinations(range(p_count), 2):
            f.write(f"- {pair_names[i]} vs {pair_names[j]}: rho={pair_spearman[i, j]:.4f}\n")

        f.write(f"\n## Leave-One-Out Robustness (Top-{top_k})\n")
        for row in loo_rows:
            f.write(
                f"- left_out={row['left_out']}: overlap={row['overlap']:.3f}, spearman_full_vs_loo={row['spearman']:.4f}, loo_top={', '.join(row['loo_top_layers'])}\n"
            )

        f.write("\n## Family Aggregates\n")
        for fam in fam_sorted:
            rec = fam_agg[fam]
            f.write(
                f"- {fam}: avg_bernstein={rec['avg_bernstein']:.6f}, avg_variance={rec['avg_variance']:.6f}, layer_count={rec['count']}\n"
            )

        f.write("\n## Sign Consistency Across Pairs\n")
        for layer in sc_sorted:
            f.write(
                f"- {layer}: soft={sign_soft[layer]:.4f}, strict={sign_strict[layer]:.4f}\n"
            )

        f.write("\n## Generated Plots\n")
        f.write("- plots/pair_layer_bernstein_heatmap.png\n")
        f.write("- plots/avg_bernstein_by_layer.png\n")
        f.write("- plots/pairwise_spearman_heatmap.png\n")
        f.write("- plots/sign_consistency_by_layer.png\n")
        f.write("- plots/family_avg_bernstein.png\n")

    print("Done")
    print(f"Report: {report_path}")
    print(f"Plots: {plot_dir}")


if __name__ == "__main__":
    main()