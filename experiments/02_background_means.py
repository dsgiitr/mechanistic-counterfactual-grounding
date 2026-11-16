"""Step 2: background mean activation of every attention head (CD-T / ablation reference).

Output  results/<model>/background_means.pt
"""
from pathlib import Path

from common import config, load_model, parser

from mcg.background import compute_background
from mcg.tasks import load_benchmark


def main():
    p = parser(__doc__, dataset=False)
    args = p.parse_args()
    cfg = config(args)
    bg = cfg["background"]
    vlm = load_model(args)
    bench = load_benchmark(bg["dataset"], cfg)
    stats = compute_background(vlm, bench, bg["num_items"], seed=cfg["seed"])
    path = Path(args.results) / args.model / "background_means.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.save(path, model=args.model, dataset=bg["dataset"], num_items=bg["num_items"])
    print(f"Saved background means to {path}")


if __name__ == "__main__":
    main()
