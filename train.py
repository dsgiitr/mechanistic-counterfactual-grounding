"""
Main training script for CVL-LLaVA.

Usage:
    python scripts/train.py --config configs/cvl_llava_vqa.yaml \
        --dataset vqa_cp \
        --data_dir /path/to/vqa_cp \
        --images_dir /path/to/coco/images \
        --output_dir ./outputs/cvl_llava_vqacp

    # Resume from checkpoint:
    python scripts/train.py --config configs/cvl_llava_vqa.yaml \
        --resume ./outputs/cvl_llava_vqacp/checkpoint-500/checkpoint.pt

    # Debug run (small subset):
    python scripts/train.py --config configs/cvl_llava_vqa.yaml \
        --debug --debug_samples 100
"""

import sys
import os
import argparse
import random
import logging

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import LlavaProcessor

from llava_cvl import LLaVACVL
from data.vqa_dataset import VQADataset, AnswerVocabulary
from collator import VQACollator, build_dataloader
from trainer import CVLTrainer
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CVL-LLaVA for VQA")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, choices=["vqa_v2", "vqa_cp"],
                        help="Override config dataset name")
    parser.add_argument("--data_dir", type=str,
                        help="Override config data directory")
    parser.add_argument("--images_dir", type=str,
                        help="Path to COCO images directory")
    parser.add_argument("--output_dir", type=str,
                        help="Override config output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: use small subset of data")
    parser.add_argument("--debug_samples", type=int, default=200,
                        help="Number of samples in debug mode")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_answer_vocab(
    config: dict,
    data_dir: str,
    dataset_name: str,
) -> AnswerVocabulary:
    """Build or load answer vocabulary."""
    import json
    from pathlib import Path

    vocab_path = config["data"]["answer_vocab_path"]
    vocab = AnswerVocabulary(min_freq=config["data"]["min_answer_freq"])

    if Path(vocab_path).exists():
        logger.info(f"Loading answer vocab from {vocab_path}")
        vocab.load(vocab_path)
    else:
        logger.info("Building answer vocabulary from training annotations...")
        # Load training annotations
        splits = VQADataset.SPLITS[dataset_name]
        _, a_file, _ = splits["train"]
        ann_path = Path(data_dir) / a_file

        with open(ann_path) as f:
            ann_data = json.load(f)
        annotations = ann_data.get("annotations", ann_data)

        vocab.build_from_annotations(annotations)
        vocab.save(vocab_path)
        logger.info(f"Built vocab with {len(vocab)} answers, saved to {vocab_path}")

    return vocab


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.dataset:
        config["data"]["dataset_name"] = args.dataset
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.images_dir:
        config["data"]["images_dir"] = args.images_dir
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    output_dir = config["training"]["output_dir"]
    setup_logging(output_dir)

    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Processor ─────────────────────────────────────────────────────────
    model_name = config["model"]["llava_model_name"]
    logger.info(f"Loading LLaVA processor: {model_name}")
    processor = LlavaProcessor.from_pretrained(model_name)

    # ── Answer Vocabulary ──────────────────────────────────────────────────
    dataset_name = config["data"]["dataset_name"]
    data_dir = config["data"]["data_dir"]
    vocab = build_answer_vocab(config, data_dir, dataset_name)
    num_answers = len(vocab)
    logger.info(f"Answer vocabulary size: {num_answers}")

    # ── Datasets ───────────────────────────────────────────────────────────
    logger.info(f"Loading {dataset_name} datasets...")
    train_dataset = VQADataset(
        data_dir=data_dir,
        images_dir=config["data"]["images_dir"],
        split="train",
        dataset_name=dataset_name,
        answer_vocab=vocab,
        processor=processor,
        max_length=config["model"]["max_length"],
        image_size=config["data"]["image_size"],
    )
    val_dataset = VQADataset(
        data_dir=data_dir,
        images_dir=config["data"]["images_dir"],
        split="val",
        dataset_name=dataset_name,
        answer_vocab=vocab,
        processor=processor,
        max_length=config["model"]["max_length"],
        image_size=config["data"]["image_size"],
    )

    if args.debug:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(args.debug_samples, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(args.debug_samples // 4, len(val_dataset))))
        logger.info(f"Debug mode: {len(train_dataset)} train, {len(val_dataset)} val samples")

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── Dataloaders ────────────────────────────────────────────────────────
    collator = VQACollator(processor, config["model"]["max_length"])

    train_loader = build_dataloader(
        train_dataset, collator,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["dataloader_num_workers"],
    )
    val_loader = build_dataloader(
        val_dataset, collator,
        batch_size=config["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["training"]["dataloader_num_workers"],
    )

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Initializing LLaVA-CVL model...")
    mcfg = config["model"]
    cfcfg = config["counterfactual"]

    model = LLaVACVL(
        llava_model_name=mcfg["llava_model_name"],
        num_answers=num_answers,
        use_lora=mcfg["use_lora"],
        lora_r=mcfg["lora_r"],
        lora_alpha=mcfg["lora_alpha"],
        lora_dropout=mcfg["lora_dropout"],
        lora_target_modules=mcfg["lora_target_modules"],
        freeze_vision_encoder=mcfg["freeze_vision_encoder"],
        epsilon=cfcfg["epsilon"],
        cf_grad_steps=cfcfg["cf_grad_steps"],
        cf_lr=cfcfg["cf_lr"],
        lambda_cf=cfcfg["lambda_cf"],
        clip_M=cfcfg["clip_M"],
        beta_alpha0=cfcfg["beta_alpha0"],
        beta_beta0=cfcfg["beta_beta0"],
        intervene_vision=cfcfg["intervene_vision"],
        intervene_language=cfcfg["intervene_language"],
    )

    # Print parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total "
                f"({100*trainable/total:.1f}%)")

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = CVLTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_dataset=train_dataset,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # ── Train ──────────────────────────────────────────────────────────────
    trainer.train()


if __name__ == "__main__":
    main()