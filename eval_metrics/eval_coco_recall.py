import json
import argparse
try:
    from .common import aligned_records
except ImportError:
    from common import aligned_records

def evaluate_coco_context_recall(predictions_file, ground_truth_file):
    """
    Evaluates COCO Context Recall (Recall_ctx).
    Recall_ctx = |C_pred intersect C_gold| / |C_gold|
    where C_pred is the set of grounded concepts recovered by the model,
    and C_gold is the reference context set.
    """
    preds, truths = aligned_records(predictions_file, ground_truth_file)
        
    total_recall = 0.0
    count = 0
    
    for p, t in zip(preds, truths):
        # Extract sets of concepts. Assume they are provided as lists in JSON.
        c_pred = set(p.get("concepts", []))
        c_gold = set(t.get("concepts", []))
        
        if not c_gold:
            continue
            
        count += 1
        intersection = c_pred.intersection(c_gold)
        recall = len(intersection) / len(c_gold)
        total_recall += recall
        
    if count == 0:
        raise ValueError("No nonempty reference concept sets to evaluate")
    avg_recall = total_recall / count
    print(f"COCO Context Recall (Recall_ctx): {avg_recall:.4f}")
    return avg_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate COCO Context Recall")
    parser.add_argument("--predictions", type=str, required=True, help="JSONL file with predicted concepts")
    parser.add_argument("--ground_truth", type=str, required=True, help="JSONL file with gold concepts")
    args = parser.parse_args()
    
    evaluate_coco_context_recall(args.predictions, args.ground_truth)
