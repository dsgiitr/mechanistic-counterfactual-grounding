"""
VQA Evaluation Metrics.

Standard VQA accuracy (soft):
    Acc(answer) = min(#humans that said answer / 3, 1.0)

For a prediction to be "correct", the model's answer must match
at least 3 out of 10 annotators' answers.
"""

from typing import List, Dict
import torch


class VQAEvaluator:
    """
    Computes standard VQA soft accuracy.

    Given a predicted answer index and the ground-truth soft scores,
    the accuracy for sample i is simply the soft score of the predicted answer.
    """

    def compute_accuracy(
        self,
        predictions: List[int],
        answer_scores: List[List[float]],
    ) -> Dict[str, float]:
        """
        Compute VQA accuracy.

        Args:
            predictions:   list of predicted answer indices (length N)
            answer_scores: list of soft label vectors (N x num_answers)

        Returns:
            dict with "vqa_accuracy" key
        """
        assert len(predictions) == len(answer_scores)

        total_acc = 0.0
        for pred_idx, scores in zip(predictions, answer_scores):
            # Soft score for the predicted answer
            if pred_idx < len(scores):
                acc = scores[pred_idx]
            else:
                acc = 0.0
            total_acc += acc

        vqa_acc = total_acc / len(predictions) if predictions else 0.0
        return {"vqa_accuracy": vqa_acc}

    def compute_per_type_accuracy(
        self,
        predictions: List[int],
        answer_scores: List[List[float]],
        question_types: List[str],
    ) -> Dict[str, float]:
        """
        Compute per-question-type accuracy (yes/no, number, other).
        Useful for VQA-CP analysis.
        """
        type_totals: Dict[str, float] = {}
        type_counts: Dict[str, int] = {}

        for pred_idx, scores, qtype in zip(predictions, answer_scores, question_types):
            acc = scores[pred_idx] if pred_idx < len(scores) else 0.0
            type_totals[qtype] = type_totals.get(qtype, 0.0) + acc
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

        results = {
            f"vqa_accuracy_{qtype}": type_totals[qtype] / type_counts[qtype]
            for qtype in type_counts
        }
        results["vqa_accuracy"] = sum(type_totals.values()) / sum(type_counts.values())
        return results