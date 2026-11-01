"""Train a bounded visual-latent generator against grounded answer supervision."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from models.factory import load_wrapper
from models.native_utils import prepare_inputs
from methods.restoration_effects import append_target_prefix
from methods.latent_counterfactual import (
    GeneratorConfig, LatentCounterfactualGenerator, generator_training_step,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="JSON list of image, question, correct_answer (or correct_token_ids)")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--family", choices=["auto", "llava", "qwen3_vl"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max_relative_shift", type=float, default=.1)
    parser.add_argument("--distance_weight", type=float, default=1.)
    parser.add_argument("--diversity_weight", type=float, default=.1)
    parser.add_argument("--min_diversity", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("epochs and learning_rate must be positive")
    raw = args.dataset_path.read_bytes()
    examples = json.loads(raw)
    if not isinstance(examples, list) or not examples:
        parser.error("Training manifest must be a nonempty JSON list")
    for example in examples:
        if not all(k in example for k in ("image", "question")) or not (
                "correct_answer" in example or "correct_token_ids" in example):
            parser.error("Every training example needs image, question and a grounded correct answer")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    wrapper = load_wrapper(args.model_id, args.device, args.family)
    wrapper.remove_hooks()
    config = GeneratorConfig(wrapper.config.hidden_size, args.noise_dim, args.rank, args.max_relative_shift)
    generator = LatentCounterfactualGenerator(config).to(args.device).train()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.learning_rate)
    random_state = torch.Generator(device="cpu").manual_seed(args.seed)
    history = []
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        for index in order:
            item = examples[index]
            path = Path(item["image"])
            if not path.is_absolute():
                path = args.dataset_path.parent / path
            with Image.open(path) as image:
                inputs = prepare_inputs(wrapper, image.convert("RGB"), item["question"], item.get("answer_prefix", ""),
                                        item.get("answer_prefix_token_ids"))
            targets = item.get("correct_token_ids")
            if targets is None:
                # Token IDs can be supplied to fix answer-boundary tokenization.
                text = (" " if args.family == "llava" or "llava" in args.model_id.lower() else "") + item["correct_answer"]
                targets = wrapper.tokenizer.encode(text, add_special_tokens=False)
            inputs = append_target_prefix(inputs, targets)
            noises = [torch.randn(1, args.noise_dim, generator=random_state) for _ in range(2)]
            metrics = generator_training_step(
                wrapper.model, inputs, targets, generator, optimizer, *noises,
                distance_weight=args.distance_weight, diversity_weight=args.diversity_weight,
                min_diversity=args.min_diversity,
            )
            record = {"epoch": epoch + 1, "example_index": index, **metrics}
            history.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
        generator.save(args.output, args.model_id, seed=args.seed, epochs_completed=epoch+1,
                       training_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                       objective={"name": "grounded_answer_nll_plus_locality_and_diversity",
                                  "distance_weight": args.distance_weight,
                                  "diversity_weight": args.diversity_weight,
                                  "min_diversity": args.min_diversity})
    args.output.with_suffix(".training.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False)+"\n" for row in history))
    print(f"Saved trained generator to {args.output}")


if __name__ == "__main__":
    main()
