"""Command-line plumbing shared by the experiment scripts."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcg.utils import load_config  # noqa: E402


def parser(description, model=True, dataset=True):
    p = argparse.ArgumentParser(description=description)
    if model:
        kwargs = {"default": model} if isinstance(model, str) else {"required": True}
        p.add_argument("--model", help="registry key, e.g. llava-1.5-7b (see mcg/models.py)", **kwargs)
    if dataset:
        p.add_argument("--dataset", required=True, choices=["pope", "coco", "hallusionbench"])
    p.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"))
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config entries, e.g. --set counterfactuals.alpha=0.2 cdt.batch_size=8")
    return p


def config(args):
    import yaml
    cfg = load_config(args.config)
    for override in args.set:
        key, value = override.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def run_dir(args, model=None, dataset=None):
    path = Path(args.results) / (model or args.model) / (dataset or args.dataset)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model(args, dtype=None):
    import torch
    from mcg.models import VLM
    torch.set_grad_enabled(False)
    return VLM(args.model, device=args.device, dtype=dtype)
