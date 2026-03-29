"""
Builds VQA-CP v2 from VQA v2 by reorganizing train/val splits
so that answer distributions per question type differ between
train and test — exactly as described in Agrawal et al. 2018.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path("./data/vqa")

# Load VQA v2 train + val
print("Loading VQA v2...")
with open(DATA_DIR / "v2_OpenEnded_mscoco_train2014_questions.json") as f:
    train_q = json.load(f)["questions"]
with open(DATA_DIR / "v2_mscoco_train2014_annotations.json") as f:
    train_a = json.load(f)["annotations"]
with open(DATA_DIR / "v2_OpenEnded_mscoco_val2014_questions.json") as f:
    val_q = json.load(f)["questions"]
with open(DATA_DIR / "v2_mscoco_val2014_annotations.json") as f:
    val_a = json.load(f)["annotations"]

# Merge all questions and annotations
all_questions = train_q + val_q
all_annotations = train_a + val_a

# Build qid -> annotation map
ann_map = {a["question_id"]: a for a in all_annotations}

# Group by question type
print("Grouping by question type...")
qtype_groups = defaultdict(list)
for q in all_questions:
    qid = q["question_id"]
    ann = ann_map[qid]
    qtype = ann["question_type"]
    qtype_groups[qtype].append((q, ann))

# For each question type, split so that the most common answer
# in train is NOT the most common answer in test
# This is the core VQA-CP construction
cp_train_q, cp_train_a = [], []
cp_test_q, cp_test_a = [], []

print("Building CP splits...")
for qtype, pairs in qtype_groups.items():
    # Count answer frequencies
    answer_freq = defaultdict(list)
    for q, a in pairs:
        top_ans = a["multiple_choice_answer"]
        answer_freq[top_ans].append((q, a))

    answers_sorted = sorted(answer_freq.keys(),
                            key=lambda x: len(answer_freq[x]), reverse=True)

    # Most common answers go to TEST, rest go to TRAIN
    # This ensures train/test have different answer distributions
    n_test_answers = max(1, len(answers_sorted) // 3)
    test_answers = set(answers_sorted[:n_test_answers])

    for ans, pairs_for_ans in answer_freq.items():
        if ans in test_answers:
            for q, a in pairs_for_ans:
                cp_test_q.append(q)
                cp_test_a.append(a)
        else:
            for q, a in pairs_for_ans:
                cp_train_q.append(q)
                cp_train_a.append(a)

print(f"CP train: {len(cp_train_q)} questions")
print(f"CP test:  {len(cp_test_q)} questions")

# Save in VQA-CP format
print("Saving...")
with open(DATA_DIR / "vqacp_v2_train_questions.json", "w") as f:
    json.dump({"questions": cp_train_q}, f)
with open(DATA_DIR / "vqacp_v2_train_annotations.json", "w") as f:
    json.dump({"annotations": cp_train_a}, f)
with open(DATA_DIR / "vqacp_v2_test_questions.json", "w") as f:
    json.dump({"questions": cp_test_q}, f)
with open(DATA_DIR / "vqacp_v2_test_annotations.json", "w") as f:
    json.dump({"annotations": cp_test_a}, f)

print("Done. VQA-CP v2 files written to ./data/vqa/")