"""Native-model restoration experiments with image or learned latent counterfactuals."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from models.factory import load_wrapper
from models.native_utils import prepare_inputs
from methods.latent_counterfactual import LatentCounterfactualGenerator, visual_intervention
from methods.restoration_effects import FactualRestoration, append_target_prefix


def extract_restoration_records(wrapper, questions, image_dir, nodes, *, mode="images",
                                generator=None, num_counterfactuals=50, seed=42):
    if mode not in ("images", "latent"):
        raise ValueError("counterfactual mode must be images or latent")
    if mode == "latent" and (generator is None or num_counterfactuals < 1):
        raise ValueError("Latent mode requires a trained generator and a positive sample budget")
    wrapper.remove_hooks()
    extractor = FactualRestoration(wrapper.model, nodes)
    random_state = torch.Generator(device="cpu").manual_seed(seed)
    records = []
    for example_id, item in questions.items():
        directory = Path(image_dir) / example_id
        originals = sorted(directory.glob("original*"))
        if len(originals) != 1:
            raise ValueError(f"{directory} requires exactly one original image")
        question = item["question"] + " " + item.get("prompt", "")
        def prepare(path):
            with Image.open(path) as image:
                return prepare_inputs(wrapper, image.convert("RGB"), question, item.get("answer_prefix", ""),
                                      item.get("answer_prefix_token_ids"))
        factual = prepare(originals[0])
        target = item.get("target_token_ids")
        if target is None and "target_answer" in item:
            prefix = " " if wrapper.model.config.model_type == "llava" else ""
            target = wrapper.tokenizer.encode(prefix + item["target_answer"], add_special_tokens=False)
        if target is None:
            with torch.no_grad():
                target = [wrapper.model(**factual, use_cache=False).logits[0, -1].argmax().item()]
        factual_forced = append_target_prefix(factual, target)
        factual_score = extractor.capture(factual_forced, target)
        factual_image_hash = hashlib.sha256(originals[0].read_bytes()).hexdigest()
        if mode == "images":
            samples = sorted(directory.glob("counterfactual*"))
            if not samples:
                raise ValueError(f"No counterfactual images in {directory}")
        else:
            samples = range(num_counterfactuals)
        for index, sample in enumerate(samples):
            metadata = {"counterfactual_mode": mode}
            if mode == "images":
                cf = prepare(sample)
                if not torch.equal(factual["input_ids"], cf["input_ids"]):
                    raise ValueError("Counterfactual must preserve prompt and visual-token layout")
                cf = append_target_prefix(cf, target)
                context = nullcontext
                metadata["counterfactual_image"] = str(sample)
            else:
                noise = torch.randn(1, generator.config.noise_dim, generator=random_state)
                cf = factual_forced
                def context():
                    return visual_intervention(wrapper.model, cf, generator, noise)
                metadata.update({"latent_noise": noise[0].tolist(), "seed": seed})
            baseline, effects = extractor.effects(cf, target, forward_context=context)
            records.append({
                "example_id": example_id, "counterfactual_index": index,
                "model_id": wrapper.model_id, "target_token_ids": target,
                "model_revision": getattr(wrapper.model.config, "_commit_hash", None),
                "question": question, "factual_image": str(originals[0]),
                "factual_image_sha256": factual_image_hash,
                "target_scope": "provided_answer" if "target_answer" in item or "target_token_ids" in item else "first_generated_token",
                "answer_prefix": item.get("answer_prefix", ""),
                "answer_prefix_token_ids": item.get("answer_prefix_token_ids"),
                "factual_log_probability": factual_score, "cf_log_probability": baseline,
                "node_patch_deltas": effects, **metadata,
            })
    if not records:
        raise ValueError("No counterfactual observations extracted")
    return records


def main(default_model_id="llava-hf/llava-1.5-7b-hf"):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit_path", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, default=root / "images")
    parser.add_argument("--questions_path", type=Path)
    parser.add_argument("--output", type=Path, default=root / "output/circuit_behavior_metrics.jsonl")
    parser.add_argument("--model_id", default=default_model_id)
    parser.add_argument("--family", choices=["auto", "llava", "qwen3_vl"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--counterfactual_mode", choices=["images", "latent"], default="images")
    parser.add_argument("--generator_checkpoint", type=Path)
    parser.add_argument("--num_counterfactuals", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.counterfactual_mode == "latent" and args.generator_checkpoint is None:
        parser.error("latent mode requires --generator_checkpoint trained with training/train_generator.py")
    nodes = json.loads(args.circuit_path.read_text())["circuit"]
    questions = json.loads((args.questions_path or args.image_dir / "counterfactual_questions.json").read_text())
    wrapper = load_wrapper(args.model_id, args.device, args.family)
    generator = None
    if args.counterfactual_mode == "latent":
        generator = LatentCounterfactualGenerator.load(args.generator_checkpoint, args.model_id,
                                                       wrapper.config.hidden_size, args.device)
    records = extract_restoration_records(
        wrapper, questions, args.image_dir, nodes, mode=args.counterfactual_mode,
        generator=generator, num_counterfactuals=args.num_counterfactuals, seed=args.seed)
    circuit_hash = hashlib.sha256(args.circuit_path.read_bytes()).hexdigest()
    for record in records:
        record["circuit_sha256"] = circuit_hash
    if generator is not None:
        checkpoint_hash = hashlib.sha256(args.generator_checkpoint.read_bytes()).hexdigest()
        for record in records:
            record["generator_checkpoint_sha256"] = checkpoint_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in records))
    print(f"Saved {len(records)} real restoration observations to {args.output}")
