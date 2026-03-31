"""
VQA Dataset Loaders for CVL-LLaVA.
Supports VQA v2 and VQA-CP v2 (the bias-diagnostic split).

VQA-CP v2 paper: Agrawal et al., "Don't Just Assume; Look and Answer"
VQA v2 paper: Goyal et al., "Making the V in VQA Matter"
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import LlavaProcessor


class AnswerVocabulary:
    """
    Builds and manages the answer vocabulary for closed-set VQA.
    Standard VQA uses soft labels (multiple annotators may give different answers).
    """

    def __init__(self, min_freq: int = 9):
        self.min_freq = min_freq
        self.answer2idx: Dict[str, int] = {}
        self.idx2answer: Dict[int, str] = {}
        self._freq: Dict[str, int] = {}

    def build_from_annotations(self, annotations: List[Dict]) -> None:
        """Count answer frequencies and keep those above min_freq."""
        for ann in annotations:
            for ans_entry in ann.get("answers", []):
                ans = self._normalize_answer(ans_entry["answer"])
                self._freq[ans] = self._freq.get(ans, 0) + 1

        # Filter by frequency and build mappings
        self.answer2idx = {"<unk>": 0}
        self.idx2answer = {0: "<unk>"}
        for ans, freq in sorted(self._freq.items(), key=lambda x: -x[1]):
            if freq >= self.min_freq:
                idx = len(self.answer2idx)
                self.answer2idx[ans] = idx
                self.idx2answer[idx] = ans

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        answer = answer.lower().strip()
        # Remove trailing period
        if answer.endswith("."):
            answer = answer[:-1]
        return answer

    def encode(self, answer: str) -> int:
        return self.answer2idx.get(self._normalize_answer(answer), 0)

    def decode(self, idx: int) -> str:
        return self.idx2answer.get(idx, "<unk>")

    def __len__(self) -> int:
        return len(self.answer2idx)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"answer2idx": self.answer2idx, "idx2answer": {
                str(k): v for k, v in self.idx2answer.items()
            }}, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.answer2idx = data["answer2idx"]
        self.idx2answer = {int(k): v for k, v in data["idx2answer"].items()}


class VQADataset(Dataset):
    """
    Dataset for VQA v2 and VQA-CP v2.

    Each item returns:
        - image: PIL Image
        - question: str
        - question_id: int
        - answer_scores: FloatTensor of shape (num_answers,) — soft labels
        - answer_idx: int — index of most common answer
        - raw_answers: List[str] — all provided answers (for evaluation)
        - image_id: int
    """

    SPLITS = {
        "vqa_v2": {
            "train": ("v2_OpenEnded_mscoco_train2014_questions.json",
                      "v2_mscoco_train2014_annotations.json",
                      "train2014"),
            "val": ("v2_OpenEnded_mscoco_val2014_questions.json",
                    "v2_mscoco_val2014_annotations.json",
                    "val2014"),
        },
        "vqa_cp": {
            "train": ("vqacp_v2_train_questions.json",
                      "vqacp_v2_train_annotations.json",
                      "train2014"),
            "val": ("vqacp_v2_test_questions.json",
                    "vqacp_v2_test_annotations.json",
                    "val2014"),
        },
    }

    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        split: str,
        dataset_name: str,
        answer_vocab: AnswerVocabulary,
        processor: LlavaProcessor,
        max_length: int = 512,
        image_size: int = 336,
        max_answers: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.split = split
        self.dataset_name = dataset_name
        self.vocab = answer_vocab
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.max_answers = max_answers

        # Load questions and annotations
        q_file, a_file, self.img_dir_name = self.SPLITS[dataset_name][split]
        self.samples = self._load_samples(
            self.data_dir / q_file,
            self.data_dir / a_file,
        )

    def _load_samples(self, q_path: Path, a_path: Path) -> List[Dict]:
        """Merge questions and annotations into unified sample list."""
        with open(q_path) as f:
            q_data = json.load(f)
        with open(a_path) as f:
            a_data = json.load(f)

        # Build question_id -> annotation mapping
        ann_map = {}
        for ann in a_data.get("annotations", a_data):
            ann_map[ann["question_id"]] = ann

        samples = []
        for q in q_data.get("questions", q_data):
            qid = q["question_id"]
            ann = ann_map.get(qid, {})

            raw_answers = [
                AnswerVocabulary._normalize_answer(a["answer"])
                for a in ann.get("answers", [])
            ]

            # Soft score: fraction of annotators who gave that answer
            score_map: Dict[str, float] = {}
            for a in raw_answers:
                score_map[a] = score_map.get(a, 0) + 1.0 / len(raw_answers)

            # Build soft label vector
            answer_scores = torch.zeros(len(self.vocab))
            for ans, score in score_map.items():
                idx = self.vocab.encode(ans)
                answer_scores[idx] = min(score * 3, 1.0)  # VQA standard scaling

            # Most common answer index
            most_common_ans = max(score_map, key=score_map.get) if score_map else "<unk>"
            answer_idx = self.vocab.encode(most_common_ans)

            samples.append({
                "question_id": qid,
                "image_id": q["image_id"],
                "question": q["question"],
                "answer_scores": answer_scores,
                "answer_idx": answer_idx,
                "raw_answers": raw_answers,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_image_path(self, image_id: int) -> Path:
        """Resolve COCO image path from image_id."""
        # COCO naming convention: COCO_train2014_000000XXXXXX.jpg
        filename = f"COCO_{self.img_dir_name}_{image_id:012d}.jpg"
        return self.images_dir / self.img_dir_name / filename

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image
        img_path = self._get_image_path(sample["image_id"])
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback: create blank image (useful for debugging)
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))

        # Format question as LLaVA prompt
        question_text = (
            f"USER: <image>\n{sample['question']}\nASSISTANT:"
        )

        return {
            "image": image,
            "question": question_text,
            "question_id": sample["question_id"],
            "image_id": sample["image_id"],
            "answer_scores": sample["answer_scores"],
            "answer_idx": sample["answer_idx"],
            "raw_answers": sample["raw_answers"],
        }

    def get_random_sample_embeddings(self, n: int) -> List[int]:
        """Return n random indices — used for counterfactual interpolation."""
        return random.sample(range(len(self)), n)