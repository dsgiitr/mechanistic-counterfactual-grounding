"""Collect actual binary hallucinations and grounded examples from native VLM runs."""
import argparse
import json
from pathlib import Path
from PIL import Image
from models.factory import load_wrapper
from methods.generation import generate_binary_answer


def main(default_model_id):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=default_model_id)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset_path", type=Path, help="Local JSON list with image, question, answer")
    parser.add_argument("--dataset_name", default="lmms-lab/POPE")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=Path, default=Path("data/hallucinations"))
    parser.add_argument("--correct_dir", type=Path, default=Path("data/correct"))
    args = parser.parse_args()
    if args.max_samples <= 0 or args.max_new_tokens <= 0:
        parser.error("Sample and token budgets must be positive")
    if args.dataset_path:
        examples = json.loads(args.dataset_path.read_text())
    else:
        from datasets import load_dataset
        examples = load_dataset(args.dataset_name, name=args.dataset_config, split=args.dataset_split)
    wrapper = load_wrapper(args.model_id, args.device)
    wrapper.remove_hooks()
    hallucinated, grounded = [], []
    for index, item in enumerate(examples):
        label = str(item.get("answer", item.get("label", ""))).lower().strip()
        if label not in ("yes", "no"):
            raise ValueError("Binary extraction requires explicit yes/no reference answers")
        image = item["image"]
        if not isinstance(image, Image.Image):
            path = Path(image)
            if args.dataset_path and not path.is_absolute():
                path = args.dataset_path.parent / path
            with Image.open(path) as loaded:
                image = loaded.convert("RGB")
        question = item["question"] + " Please answer with yes or no."
        result = generate_binary_answer(wrapper, image, question, args.max_new_tokens)
        if result is None:
            continue
        is_hallucination = result["answer"] == "yes" and label == "no"
        is_grounded = result["answer"] == label
        if not is_hallucination and not is_grounded:
            continue
        collection = hallucinated if is_hallucination else grounded
        if len(collection) >= args.max_samples:
            continue
        directory = args.output_dir if is_hallucination else args.correct_dir
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"example_{index}.png"
        image.convert("RGB").save(image_path)
        collection.append({"id": index, "image": str(image_path.resolve()), "question": question,
                           "hallucinated_answer": result["answer"], "correct_answer": label,
                           "hallucinated_token_id": result["answer_token_id"],
                           "correct_token_id": wrapper.get_correct_token_id(label),
                           "answer_prefix": result["answer_prefix"],
                           "answer_prefix_token_ids": result["answer_prefix_token_ids"],
                           "generated_text": result["generated_text"]})
        print(f"Scanned {index+1}: hallucinated={len(hallucinated)}, grounded={len(grounded)}", flush=True)
        if min(len(hallucinated), len(grounded)) >= args.max_samples:
            break
    for directory, name, records in ((args.output_dir, "hallucinations.json", hallucinated),
                                      (args.correct_dir, "correct_examples.json", grounded)):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(json.dumps(records, indent=2)+"\n")
    print(f"Collected {len(hallucinated)} hallucinations and {len(grounded)} grounded examples")


if __name__ == "__main__":
    main("Qwen/Qwen3-VL-8B-Instruct")
