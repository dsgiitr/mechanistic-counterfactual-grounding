import json
import argparse

def evaluate_coco_context_recall(predictions_file, ground_truth_file):
    """
    Evaluates COCO Context Recall (Recall_ctx).
    Recall_ctx = |C_pred \cap C_gold| / |C_gold|
    where C_pred is the set of grounded concepts recovered by the model,
    and C_gold is the reference context set.
    """
    with open(predictions_file, 'r') as f:
        preds = [json.loads(line) for line in f]
        
    with open(ground_truth_file, 'r') as f:
        truths = [json.loads(line) for line in f]
        
    total_recall = 0.0
    count = len(preds)
    
    for p, t in zip(preds, truths):
        # Extract sets of concepts. Assume they are provided as lists in JSON.
        c_pred = set(p.get("concepts", []))
        c_gold = set(t.get("concepts", []))
        
        if not c_gold:
            continue
            
        intersection = c_pred.intersection(c_gold)
        recall = len(intersection) / len(c_gold)
        total_recall += recall
        
    avg_recall = total_recall / count if count > 0 else 0
    print(f"COCO Context Recall (Recall_ctx): {avg_recall:.4f}")
    return avg_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate COCO Context Recall")
    parser.add_argument("--predictions", type=str, required=True, help="JSONL file with predicted concepts")
    parser.add_argument("--ground_truth", type=str, required=True, help="JSONL file with gold concepts")
    args = parser.parse_args()
    
    evaluate_coco_context_recall(args.predictions, args.ground_truth)
