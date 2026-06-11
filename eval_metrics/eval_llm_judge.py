import json
import argparse

def evaluate_llm_judge(predictions_file):
    """
    Evaluates LLM-as-a-judge scores.
    Expects a file where an LLM (e.g. GPT-4) has scored the output on a scale (e.g., 1-10 or 0-100).
    Outputs the Averaged Out score.
    """
    with open(predictions_file, 'r') as f:
        scores_data = [json.loads(line) for line in f]
        
    total_score = 0.0
    count = 0
    
    for item in scores_data:
        score = item.get("llm_score")
        if score is not None:
            total_score += float(score)
            count += 1
            
    avg_score = total_score / count if count > 0 else 0
    print(f"LLM Judge (Averaged Out) Score: {avg_score:.4f}")
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM Judge Score")
    parser.add_argument("--scores_file", type=str, required=True, help="JSONL file with LLM judge scores")
    args = parser.parse_args()
    
    evaluate_llm_judge(args.scores_file)
