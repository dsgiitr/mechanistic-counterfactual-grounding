import json
import argparse
import sys

def evaluate_pope(predictions_file, answers_file):
    """
    Evaluates POPE (Predicting Object Presence in Everything) Accuracy.
    POPE consists of binary Yes/No questions regarding the presence of objects.
    """
    with open(predictions_file, 'r') as f:
        preds = [json.loads(line) for line in f]
        
    with open(answers_file, 'r') as f:
        answers = [json.loads(line) for line in f]
        
    if len(preds) != len(answers):
        print("Warning: Number of predictions does not match number of answers.")
        
    correct = 0
    total = len(preds)
    
    for p, a in zip(preds, answers):
        # Clean predictions
        pred_text = p.get("text", "").strip().lower()
        ans_text = a.get("label", "").strip().lower()
        
        # Simple extraction for yes/no
        if pred_text.startswith("yes") and ans_text == "yes":
            correct += 1
        elif pred_text.startswith("no") and ans_text == "no":
            correct += 1
            
    accuracy = correct / total if total > 0 else 0
    print(f"POPE Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate POPE Accuracy")
    parser.add_argument("--predictions", type=str, required=True, help="JSONL file with model predictions")
    parser.add_argument("--answers", type=str, required=True, help="JSONL file with ground truth answers")
    args = parser.parse_args()
    
    evaluate_pope(args.predictions, args.answers)
