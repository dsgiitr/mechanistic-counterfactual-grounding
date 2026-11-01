import json
import argparse
import sys

try:
    from .common import binary_accuracy
except ImportError:
    from common import binary_accuracy


def evaluate_pope(predictions_file, answers_file):
    accuracy = binary_accuracy(predictions_file, answers_file)
    print(f"POPE Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate POPE Accuracy")
    parser.add_argument("--predictions", type=str, required=True, help="JSONL file with model predictions")
    parser.add_argument("--answers", type=str, required=True, help="JSONL file with ground truth answers")
    args = parser.parse_args()
    
    evaluate_pope(args.predictions, args.answers)
