import argparse
import torch
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.llava_cvl import LLaVACVL
from training.vqa_dataset import VQADataset, AnswerVocabulary
from training.collator import VQACollator, build_dataloader
from training.evaluator import VQAEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CVL-LLaVA on VQA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load answer vocab
    vocab = AnswerVocabulary(min_freq=config["data"]["min_answer_freq"])
    vocab.load(config["data"]["answer_vocab_path"])

    # Load processor
    from transformers import LlavaProcessor
    processor = LlavaProcessor.from_pretrained(config["model"]["llava_model_name"])

    # Load dataset
    dataset = VQADataset(
        data_dir=config["data"]["data_dir"],
        images_dir=config["data"]["images_dir"],
        split=args.split,
        dataset_name=config["data"]["dataset_name"],
        answer_vocab=vocab,
        processor=processor,
        max_length=config["model"].get("max_length", 512),
        image_size=config["data"].get("image_size", 336),
        max_answers=config["data"].get("max_answers_per_question", 10),
        cache_dir=config.get("cache", {}).get("dataset_cache_dir"),
        use_cache=config.get("cache", {}).get("use_dataset_cache", True),
    )

    # Dataloader
    eval_batch_size = args.batch_size or config.get("evaluation", {}).get("eval_batch_size", 32)
    collator = VQACollator(processor, max_length=config["model"].get("max_length", 512))
    dataloader = build_dataloader(
        dataset,
        collator=collator,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config["training"].get("dataloader_num_workers", 4),
    )

    # Load model
    model = LLaVACVL(
        llava_model_name=config["model"]["llava_model_name"],
        num_answers=len(vocab),
        use_lora=config["model"].get("use_lora", True),
        lora_r=config["model"].get("lora_r", 16),
        lora_alpha=config["model"].get("lora_alpha", 32),
        lora_dropout=config["model"].get("lora_dropout", 0.05),
        lora_target_modules=config["model"].get("lora_target_modules"),
        freeze_vision_encoder=config["model"].get("freeze_vision_encoder", True),
        load_in_4bit=config["model"].get("load_in_4bit", False),
        bnb_4bit_quant_type=config["model"].get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=config["model"].get("bnb_4bit_compute_dtype", "bfloat16"),
        bnb_4bit_use_double_quant=config["model"].get("bnb_4bit_use_double_quant", True),
        device_map=config["model"].get("device_map"),
        cache_dir=config.get("cache", {}).get("hf_cache_dir"),
        local_files_only=config.get("cache", {}).get("local_files_only", False),
        epsilon=config["counterfactual"].get("epsilon", 0.1),
        cf_grad_steps=config["counterfactual"].get("cf_grad_steps", 5),
        cf_lr=config["counterfactual"].get("cf_lr", 0.01),
        lambda_cf=config["counterfactual"].get("lambda_cf", 0.5),
        clip_M=config["counterfactual"].get("clip_M", 10.0),
        beta_alpha0=config["counterfactual"].get("beta_alpha0", 0.1),
        beta_beta0=config["counterfactual"].get("beta_beta0", 0.1),
        intervene_vision=config["counterfactual"].get("intervene_vision", True),
        intervene_language=config["counterfactual"].get("intervene_language", True),
    )
    model.to(device)
    model.eval()

    # Load checkpoint
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)

    # Evaluation loop
    evaluator = VQAEvaluator()
    all_predictions = []
    all_targets = []
    all_qtypes = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            answer_scores = batch["answer_scores"]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1).cpu()
            all_predictions.extend(preds.tolist())
            all_targets.extend(answer_scores.tolist())
            # Optionally, collect question types if available
            if "question_types" in batch:
                all_qtypes.extend(batch["question_types"])

    # Compute metrics
    results = evaluator.compute_accuracy(all_predictions, all_targets)
    print("VQA Accuracy: {:.4f}".format(results["vqa_accuracy"]))
    if all_qtypes:
        per_type = evaluator.compute_per_type_accuracy(all_predictions, all_targets, all_qtypes)
        for k, v in per_type.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
