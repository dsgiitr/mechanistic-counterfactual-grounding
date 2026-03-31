"""
Collation utilities for batching VQA samples with LLaVA processor.
Handles image preprocessing and tokenization jointly.
"""

from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader
from transformers import LlavaProcessor


class VQACollator:
    """
    Collates a list of VQA samples into a batch suitable for LLaVA.

    Handles:
        - Image preprocessing via processor
        - Question tokenization with padding
        - Soft label stacking
        - Metadata aggregation
    """

    def __init__(self, processor: LlavaProcessor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [s["image"] for s in samples]
        questions = [s["question"] for s in samples]

        # Process images + text together through LLaVA processor
        encoding = self.processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        batch = {
            # LLaVA model inputs
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "pixel_values": encoding["pixel_values"],
            # Labels
            "answer_scores": torch.stack([s["answer_scores"] for s in samples]),
            "answer_idx": torch.tensor([s["answer_idx"] for s in samples], dtype=torch.long),
            # Metadata (kept as lists, not tensored)
            "question_ids": [s["question_id"] for s in samples],
            "image_ids": [s["image_id"] for s in samples],
            "raw_answers": [s["raw_answers"] for s in samples],
        }

        return batch


def build_dataloader(
    dataset,
    collator: VQACollator,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # Drop last incomplete batch during training
    )