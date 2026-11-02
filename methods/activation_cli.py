"""Export descriptive native decoder activations for supplied image pairs."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from models.factory import load_wrapper
from models.native_utils import prepare_inputs


def main(default_model_id):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=default_model_id)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_dir", type=Path, default=root / "images")
    parser.add_argument("--questions_path", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    questions = json.loads((args.questions_path or args.image_dir / "counterfactual_questions.json").read_text())
    wrapper = load_wrapper(args.model_id, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def features(path, question, prefix):
        with Image.open(path) as image:
            inputs = prepare_inputs(wrapper, image.convert("RGB"), question, prefix)
        wrapper.clear_activations()
        with torch.no_grad():
            wrapper.model(**inputs, use_cache=False)
        return {f"layer_{layer}_head_{head}": tensor[:, -1, head].detach().cpu().clone()
                for layer, tensor in wrapper.get_activations().items() for head in range(wrapper.num_heads)}
    try:
        for example, item in questions.items():
            directory = args.image_dir / example
            originals = sorted(directory.glob("original*"))
            cfs = sorted(directory.glob("counterfactual*"))
            if len(originals) != 1 or not cfs:
                raise ValueError(f"{directory} needs one original and at least one counterfactual")
            question = item["question"] + " " + item.get("prompt", "")
            original = features(originals[0], question, item.get("answer_prefix", ""))
            for index, path in enumerate(cfs):
                cf = features(path, question, item.get("answer_prefix", ""))
                torch.save({key: original[key]-cf[key] for key in original}, args.output_dir / f"{example}_cf_{index}_deltas.pt")
    finally:
        wrapper.remove_hooks()
    print("Exported activation differences; these are descriptive, not restoration effects")
