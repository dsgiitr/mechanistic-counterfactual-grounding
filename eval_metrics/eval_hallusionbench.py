import json
import argparse

def evaluate_hallusionbench(predictions_file, answers_file):
    """
    Evaluates HallusionBench Accuracy.
    HallusionBench tests logic and visual illusions.
    """
    with open(predictions_file, 'r') as f:
        preds = [json.loads(line) for line in f]
        
    with open(answers_file, 'r') as f:
        answers = [json.loads(line) for line in f]
        
    correct = 0
    total = len(preds)
    
    for p, a in zip(preds, answers):
        pred_text = p.get("text", "").strip().lower()
        ans_text = a.get("label", "").strip().lower()
        
        # Exact match or specific heuristic for HallusionBench
        if pred_text == ans_text or ans_text in pred_text:
            correct += 1
            
    accuracy = correct / total if total > 0 else 0
    print(f"HallusionBench Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HallusionBench")
    parser.add_argument("--predictions", type=str, required=True, help="JSONL file with model predictions")
    parser.add_argument("--answers", type=str, required=True, help="JSONL file with ground truth answers")
    args = parser.parse_args()
    
    evaluate_hallusionbench(args.predictions, args.answers)
