"""Benchmarks, answer extraction, hallucination labels and the retention rule.

Datasets (Section 4.1)
  pope            POPE (lmms-lab/POPE, config "Full", split "random": 3,000 yes/no questions)
  hallusionbench  HallusionBench (lmms-lab/HallusionBench, splits "image" + "non_image")
  coco            COCO val2014 captioning, objects extracted CHAIR-style

Answer y (what log p_theta(y) scores, Eq. 3)
  binary  : the token(s) of the model's yes/no word, after any reasoning trace
  caption : the first mention of a hallucinated object (hallucinated example) or of
            an annotated object (grounded example), teacher-forced after the caption
            prefix that precedes it
For Qwen3-VL-Thinking the reasoning trace is generated greedily on the factual
input and is part of the fixed (teacher-forced) prefix in every counterfactual
and patched run.

Retention (Section 4.1): a counterfactual is retained if the model gives the
ground-truth answer (binary) / mentions no absent object (caption) in at least
`threshold` of `decodings` sampled decodings. The same rule is applied to
hallucinated and grounded examples.
"""
import json
import random
import re
from pathlib import Path

from PIL import Image

PROMPTS = {
    "binary": "{question} Please answer yes or no.",
    "caption": "Please describe this image in detail.",
}

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

_IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child",
              "mice": "mouse", "knives": "knife", "geese": "goose", "teeth": "tooth",
              "feet": "foot", "skis": "ski", "glasses": "glass", "buses": "bus",
              "boxes": "box", "benches": "bench", "sandwiches": "sandwich",
              "couches": "couch", "dishes": "dish", "watches": "watch", "shelves": "shelf",
              "leaves": "leaf", "scissors": "scissors", "giraffes": "giraffe"}


def singularize(word):
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


class CocoObjects:
    """CHAIR-style object mention extraction.

    With the official CHAIR synonym list (`synonyms_path`; one line per category:
    "category, synonym, synonym, ...") this follows Rohrbach et al. (2018).
    Without it, only the 80 category names (singular/plural) are matched.
    """

    def __init__(self, synonyms_path=None):
        self.lookup = {}
        if synonyms_path:
            for line in Path(synonyms_path).read_text().splitlines():
                words = [w.strip().lower() for w in line.split(",") if w.strip()]
                if words:
                    for synonym in words:
                        self.lookup[self._norm(synonym)] = words[0]
        else:
            for name in COCO_CATEGORIES:
                self.lookup[self._norm(name)] = name
        self.categories = sorted(set(self.lookup.values()))

    @staticmethod
    def _norm(phrase):
        return " ".join(singularize(w) for w in phrase.lower().split())

    def mentions(self, caption):
        """Ordered list of (category, char_start, char_end) mentions."""
        words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[a-z]+", caption.lower())]
        found, i = [], 0
        while i < len(words):
            if i + 1 < len(words):
                pair = singularize(words[i][0]) + " " + singularize(words[i + 1][0])
                if pair in self.lookup:
                    found.append((self.lookup[pair], words[i][1], words[i + 1][2]))
                    i += 2
                    continue
            single = singularize(words[i][0])
            if single in self.lookup:
                found.append((self.lookup[single], words[i][1], words[i][2]))
            i += 1
        return found


def context_recall(mentioned, gold):
    """Recall_ctx = |C_pred ∩ C_gold| / |C_gold|."""
    gold = set(gold)
    return len(set(mentioned) & gold) / len(gold) if gold else None


# ---------------------------------------------------------------------------- benchmarks
class Benchmark:
    task = "binary"

    def __len__(self):
        return len(self.rows)

    def item(self, index):
        raise NotImplementedError

    def prompt(self, item):
        return PROMPTS[self.task].format(question=item.get("question", ""))


class POPE(Benchmark):
    def __init__(self, dataset_name="lmms-lab/POPE", config="Full", split="random"):
        from datasets import load_dataset
        self.rows = load_dataset(dataset_name, config, split=split)
        self.name = "pope"

    def item(self, index):
        row = self.rows[index]
        return {"index": index, "example_id": f"pope:{index}", "image": row["image"].convert("RGB"),
                "image_key": row["image_source"], "question": row["question"].strip(),
                "gt": row["answer"].strip().lower()}


class HallusionBench(Benchmark):
    def __init__(self, dataset_name="lmms-lab/HallusionBench", splits=("image", "non_image")):
        from datasets import load_dataset, concatenate_datasets
        parts = [load_dataset(dataset_name, split=split) for split in splits]
        self.rows = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        self.name = "hallusionbench"

    def item(self, index):
        row = self.rows[index]
        has_image = str(row.get("visual_input", "1")) != "0" and row.get("image") is not None
        return {"index": index, "example_id": f"hallusionbench:{index}",
                "image": row["image"].convert("RGB") if has_image else None,
                "image_key": row.get("filename") if has_image else None,
                "question": row["question"].strip(), "gt": "yes" if str(row["gt_answer"]) == "1" else "no",
                "category": row.get("category"), "subcategory": row.get("subcategory")}


class COCOCaptions(Benchmark):
    task = "caption"

    def __init__(self, image_dir, instances_path, num_images=500, seed=0, synonyms_path=None):
        data = json.loads(Path(instances_path).read_text())
        names = {c["id"]: c["name"] for c in data["categories"]}
        self.objects = CocoObjects(synonyms_path)
        gold = {}
        for annotation in data["annotations"]:
            name = self.objects.lookup.get(CocoObjects._norm(names[annotation["category_id"]]),
                                           names[annotation["category_id"]])
            gold.setdefault(annotation["image_id"], set()).add(name)
        images = sorted((im for im in data["images"] if im["id"] in gold), key=lambda im: im["id"])
        random.Random(seed).shuffle(images)
        self.rows = images[:num_images] if num_images else images
        self.gold = gold
        self.image_dir = Path(image_dir)
        self.name = "coco"

    def item(self, index):
        row = self.rows[index]
        return {"index": index, "example_id": f"coco:{row['id']}",
                "image": Image.open(self.image_dir / row["file_name"]).convert("RGB"),
                "image_key": row["file_name"], "question": "", "gt": sorted(self.gold[row["id"]])}


def load_benchmark(name, cfg):
    data = cfg["data"]
    if name == "pope":
        return POPE(data["pope"]["dataset"], data["pope"]["config"], data["pope"]["split"])
    if name == "hallusionbench":
        return HallusionBench(data["hallusionbench"]["dataset"], tuple(data["hallusionbench"]["splits"]))
    if name == "coco":
        coco = data["coco"]
        return COCOCaptions(coco["image_dir"], coco["instances"], coco["num_images"], coco["seed"],
                            coco.get("chair_synonyms"))
    raise ValueError(name)


# ---------------------------------------------------------------------------- answers
_BINARY = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_binary(text):
    match = _BINARY.search(text)
    return match.group(1).lower() if match else None


def analyse_generation(vlm, bench, item, generated):
    """Label one factual generation and define y and its prefixes.

    Returns None if the output cannot be scored (unfinished reasoning, no yes/no,
    or a caption that mentions no COCO object).
    """
    split = vlm.split_reasoning(generated)
    if split is None:
        return None
    trace, rest = split
    text = vlm.decode(rest)
    record = {"generated_ids": list(generated), "gen_prefix_ids": trace, "answer_text": text}
    if bench.task == "binary":
        match = _BINARY.search(text)
        if match is None:
            return None
        answer = match.group(1).lower()
        first, last = vlm.span_tokens(rest, match.start(1), match.end(1))
        record.update(answer=answer, correct=answer == item["gt"], hallucinated=answer != item["gt"],
                      score_prefix_ids=trace + rest[:first], target_ids=rest[first:last])
        return record
    mentions = bench.objects.mentions(text)
    gold = set(item["gt"])
    mentioned = [m[0] for m in mentions]
    hallucinated_objects = sorted({c for c in mentioned if c not in gold})
    record.update(answer=text, mentioned_objects=sorted(set(mentioned)),
                  hallucinated_objects=hallucinated_objects, hallucinated=bool(hallucinated_objects),
                  correct=not hallucinated_objects, recall=context_recall(mentioned, gold))
    target = next((m for m in mentions if m[0] not in gold), None) if hallucinated_objects \
        else next((m for m in mentions if m[0] in gold), None)
    if target is None:
        record.update(score_prefix_ids=None, target_ids=None, target_object=None)
        return record
    first, last = vlm.span_tokens(rest, target[1], target[2])
    record.update(score_prefix_ids=trace + rest[:first], target_ids=rest[first:last], target_object=target[0])
    return record


def sample_is_correct(vlm, bench, item, sample_ids):
    """Retention predicate for one sampled decoding (continuation after the fixed prefix)."""
    text = vlm.decode(sample_ids)
    if bench.task == "binary":
        return parse_binary(text) == item["gt"]
    gold = set(item["gt"])
    return all(category in gold for category, _, _ in bench.objects.mentions(text))


def load_donor_pool(cfg):
    """Donor pool of the counterfactual generator (POPE images + COCO annotations)."""
    from mcg.counterfactuals import DonorPool
    data = cfg["data"]
    pope = POPE(data["pope"]["dataset"], data["pope"]["config"], data["pope"]["split"])
    return DonorPool.from_pope(pope, data["coco"]["instances"], CocoObjects(data["coco"].get("chair_synonyms")),
                               cfg["counterfactuals"]["donor_exclude_top"])
