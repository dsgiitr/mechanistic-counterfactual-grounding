import argparse
import json
import os
from datetime import datetime

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


def layer_effect_size(delta_tensor):
    """RMS-like magnitude for a layer delta vector."""
    x = delta_tensor.detach().float().cpu().numpy().reshape(-1)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


def shorten(name, max_len=30):
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def load_behavior_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Behavior metrics file not found: "
            + path
            + "\nGenerate it by rerunning run_vlm_extraction.py after pulling latest code."
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No behavior records found in {path}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Correlate layer delta magnitudes with yes/no margin shifts."
    )
    parser.add_argument("--delta-dir", default=os.path.join("..", "results", "vlm_deltas"))
    parser.add_argument(
        "--behavior-jsonl",
        default=os.path.join("..", "results", "vlm_behavior", "behavior_metrics.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("..", "results", "vlm_behavior_link"),
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    delta_dir = os.path.abspath(os.path.join(script_dir, args.delta_dir))
    behavior_jsonl = os.path.abspath(os.path.join(script_dir, args.behavior_jsonl))
    out_dir = os.path.abspath(os.path.join(script_dir, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    records = load_behavior_jsonl(behavior_jsonl)
    margins_by_delta_file = {
        r["delta_file"]: float(r["margin_delta_cf_minus_orig"]) for r in records
    }

    delta_files = sorted([f for f in os.listdir(delta_dir) if f.endswith(".pt")])
    if not delta_files:
        raise RuntimeError(f"No delta files found in {delta_dir}")

    # Use only files for which behavior margin exists.
    delta_files = [f for f in delta_files if f in margins_by_delta_file]
    if len(delta_files) < 2:
        raise RuntimeError(
            "Need at least 2 matched delta files with behavior metrics for correlation."
        )

    loaded = [torch.load(os.path.join(delta_dir, f), map_location="cpu") for f in delta_files]
    layers = sorted(set.intersection(*[set(obj.keys()) for obj in loaded]))
    if not layers:
        raise RuntimeError("No common layers across matched delta files.")

    y_margin_abs = np.array([abs(margins_by_delta_file[f]) for f in delta_files], dtype=float)

    layer_corr = {}
    layer_effect_matrix = np.zeros((len(delta_files), len(layers)), dtype=float)
    for j, layer in enumerate(layers):
        x_vals = []
        for i, obj in enumerate(loaded):
            effect = layer_effect_size(obj[layer])
            x_vals.append(effect)
            layer_effect_matrix[i, j] = effect
        layer_corr[layer] = pearson_corr(np.array(x_vals, dtype=float), y_margin_abs)

    # Ranking by absolute correlation strength.
    ranked_layers = sorted(
        layers,
        key=lambda ln: abs(layer_corr[ln]) if not np.isnan(layer_corr[ln]) else -1.0,
        reverse=True,
    )

    # Plot: correlation bar by layer.
    corr_vals = [layer_corr[ln] for ln in ranked_layers]
    fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(ranked_layers)), 4.8))
    ax.bar(np.arange(len(ranked_layers)), corr_vals)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(ranked_layers)))
    ax.set_xticklabels([shorten(x) for x in ranked_layers], rotation=45, ha="right")
    ax.set_ylabel("Pearson corr(effect_size, |margin_delta|)")
    ax.set_title("Layer-Behavior Correlation (small-sample exploratory)")
    fig.tight_layout()
    corr_plot = os.path.join(out_dir, "layer_behavior_correlation_bar.png")
    fig.savefig(corr_plot, dpi=180)
    plt.close(fig)

    # Plot: heatmap for per-pair layer effect sizes.
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(layers)), max(4.0, 0.60 * len(delta_files))))
    im = ax.imshow(layer_effect_matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([shorten(x) for x in layers], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(delta_files)))
    ax.set_yticklabels([shorten(x) for x in delta_files])
    ax.set_title("Per-pair Layer Delta Effect Size (RMS)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RMS Delta Magnitude")
    fig.tight_layout()
    heatmap_plot = os.path.join(out_dir, "pair_layer_effect_size_heatmap.png")
    fig.savefig(heatmap_plot, dpi=180)
    plt.close(fig)

    # Report
    report_path = os.path.join(out_dir, "behavior_layer_correlation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Behavior-Layer Correlation Report\n\n")
        f.write(f"- Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"- Matched pairs: {len(delta_files)}\n")
        f.write(f"- Layers analyzed: {len(layers)}\n")
        f.write("- Target behavior metric: |margin_delta_cf_minus_orig| for yes/no logits\n\n")

        f.write("## Pair-Level Behavior Deltas\n")
        for name in delta_files:
            f.write(
                f"- {name}: margin_delta_cf_minus_orig={margins_by_delta_file[name]:.6f}, abs={abs(margins_by_delta_file[name]):.6f}\n"
            )

        f.write("\n## Top Layers by |Correlation|\n")
        top_n = min(5, len(ranked_layers))
        for ln in ranked_layers[:top_n]:
            f.write(f"- {ln}: corr={layer_corr[ln]:.4f}\n")

        f.write("\n## Full Layer Correlations\n")
        for ln in ranked_layers:
            f.write(f"- {ln}: corr={layer_corr[ln]:.4f}\n")

        f.write("\n## Caveat\n")
        f.write(
            "- Correlations are exploratory because sample size is very small. Use larger pair counts before drawing strong causal conclusions.\n"
        )

        f.write("\n## Generated Plots\n")
        f.write("- layer_behavior_correlation_bar.png\n")
        f.write("- pair_layer_effect_size_heatmap.png\n")

    print("Done")
    print(f"Report: {report_path}")
    print(f"Plot: {corr_plot}")
    print(f"Plot: {heatmap_plot}")


if __name__ == "__main__":
    main()
