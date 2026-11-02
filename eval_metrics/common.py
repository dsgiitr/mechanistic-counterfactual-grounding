"""Shared input validation and binary answer parsing for benchmark evaluators."""
import json
import re


def aligned_records(predictions_file, answers_file):
    with open(predictions_file) as stream:
        predictions = [json.loads(line) for line in stream if line.strip()]
    with open(answers_file) as stream:
        answers = [json.loads(line) for line in stream if line.strip()]
    if not answers or len(predictions) != len(answers):
        raise ValueError("Predictions and references must be nonempty and have equal lengths")
    for prediction, answer in zip(predictions, answers):
        for field in ("question_id", "id"):
            if field in prediction and field in answer and prediction[field] != answer[field]:
                raise ValueError(f"Mismatched {field}; align predictions with references before evaluation")
    return predictions, answers


def binary_answer(text):
    match = re.match(r'''^\s*["']?(yes|no)\b''', str(text), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def binary_accuracy(predictions_file, answers_file):
    predictions, answers = aligned_records(predictions_file, answers_file)
    correct = 0
    for prediction, answer in zip(predictions, answers):
        label = str(answer.get("label", "")).strip().lower()
        if label not in ("yes", "no"):
            raise ValueError(f"Expected yes/no reference label, got {label!r}")
        correct += binary_answer(prediction.get("text", "")) == label
    return correct / len(answers)
