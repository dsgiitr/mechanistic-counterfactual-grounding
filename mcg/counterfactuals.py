"""Counterfactual generator: CVL donor mixing of projected visual tokens (Section 3.1, Eq. 1).

    h~ = u h(I) + (1 - u) h(I')   if u < 1 - eps_g
    h~ = h(I)                     otherwise

h(I) is the output of the VLM's multimodal projector (for Qwen3-VL also the
DeepStack merger outputs), I' is a donor image and u in [0, 1) an exogenous
variable. For probing, the VLM is frozen and u_1..u_K are drawn i.i.d. from a
Beta distribution: the trained CVL posterior (`training/`, read from a CVL
checkpoint) or the configured parameters. The question and all text tokens are
unchanged. The donor is resized to the factual image size before
preprocessing, so both produce the same number of visual tokens.

Draw k of example e uses (u_k, donor_k) seeded by (seed, e, k): the
counterfactual baseline, every restoration run and every sampled decoding of
draw k see the same visual tokens.
"""
import json
import re
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from mcg.models import HookSet
from mcg.utils import derive_seed


# --------------------------------------------------------------------------- projector outputs
@torch.no_grad()
def projector_outputs(vlm, inputs):
    """h(I): outputs of every projector module in one forward pass {stream: tensor}."""
    outputs = {}

    def recorder(stream):
        def hook(module, args, output):
            outputs[stream] = output.detach()
        return hook

    with HookSet() as hooks:
        for stream, module in enumerate(vlm.projector_modules()):
            hooks.post(module, recorder(stream))
        vlm.forward_logits(inputs, keep=1)
    if not outputs:
        raise ValueError("The input contains no image: a visual counterfactual is undefined")
    return outputs


@contextmanager
def mixing(vlm, donor, u):
    """Replace h(I) by u h(I) + (1 - u) h(I') inside the context (donor = projector_outputs of I')."""
    def mixer(stream):
        def hook(module, args, output):
            d = donor[stream].to(output.device, output.dtype)
            if d.shape != output.shape:
                repeats = output.shape[0] // d.shape[0]
                if repeats * d.shape[0] != output.shape[0] or d.shape[1:] != output.shape[1:]:
                    raise ValueError(f"Donor tokens {tuple(d.shape)} do not match factual {tuple(output.shape)}")
                d = d.repeat(repeats, *([1] * (d.dim() - 1)))
            return u * output + (1 - u) * d
        return hook

    with HookSet() as hooks:
        for stream, module in enumerate(vlm.projector_modules()):
            if stream in donor:
                hooks.post(module, mixer(stream))
        yield


def donor_inputs(vlm, donor_image, factual_image, question):
    return vlm.prepare(donor_image.convert("RGB").resize(factual_image.size), question)


# --------------------------------------------------------------------------- donors
class DonorPool:
    """Donor images: POPE images whose COCO annotations contain neither the queried
    object nor its `exclude_top` most frequently co-occurring categories (Fig. 1 rule)."""

    def __init__(self, images, categories, cooccurrence, objects, exclude_top=5):
        self.images = images            # list of (key, loader)
        self.categories = categories    # key -> set of category names
        self.cooccurrence = cooccurrence
        self.objects = objects          # CocoObjects, maps words to categories
        self.exclude_top = exclude_top

    @classmethod
    def from_pope(cls, pope, instances_path, objects, exclude_top=5):
        data = json.loads(Path(instances_path).read_text())
        names = {c["id"]: c["name"] for c in data["categories"]}
        per_image = {}
        for a in data["annotations"]:
            name = objects.lookup.get(objects._norm(names[a["category_id"]]), names[a["category_id"]])
            per_image.setdefault(a["image_id"], set()).add(name)
        pairs = Counter()
        for cats in per_image.values():
            for a in cats:
                for b in cats:
                    if a != b:
                        pairs[(a, b)] += 1
        cooc = {}
        for (a, b), n in pairs.items():
            cooc.setdefault(a, Counter())[b] = n
        images, categories, seen = [], {}, set()
        for index in range(len(pope.rows)):
            key = pope.rows[index]["image_source"]
            if key in seen:
                continue
            seen.add(key)
            coco_id = int(key.rsplit("_", 1)[-1])
            categories[key] = per_image.get(coco_id, set())
            images.append((key, lambda i=index: pope.rows[i]["image"]))
        return cls(images, categories, cooc, objects, exclude_top)

    def excluded(self, queried):
        if queried is None:
            return set()
        top = [c for c, _ in self.cooccurrence.get(queried, Counter()).most_common(self.exclude_top)]
        return {queried, *top}

    def candidates(self, queried, factual_key=None):
        banned = self.excluded(queried)
        return [i for i, (key, _) in enumerate(self.images)
                if key != factual_key and not (self.categories.get(key, set()) & banned)]

    def queried_object(self, item, record=None):
        """Category queried by a POPE question, or the scored object of a caption."""
        if record is not None and record.get("target_object"):
            return record["target_object"]
        match = re.search(r"is there an? (.+?)(?:\s+in (?:the|this) (?:image|picture|photo))?\s*\?",
                          item.get("question", ""), re.IGNORECASE)
        if not match:
            return None
        mentions = self.objects.mentions(match.group(1))
        return mentions[0][0] if mentions else None


def load_beta(cfg):
    """(alpha, beta) of the u distribution: trained CVL posterior if a checkpoint is given."""
    c = cfg["counterfactuals"]
    path = c.get("beta_checkpoint")
    if path:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "exogenous_alpha_v" in state:
            return float(state["exogenous_alpha_v"]), float(state["exogenous_beta_v"])
        sd = state.get("model_state_dict", state)
        alpha = next(v for k, v in sd.items() if k.endswith("alpha_v"))
        beta = next(v for k, v in sd.items() if k.endswith("beta_v"))
        return float(alpha), float(beta)
    return float(c["beta_alpha"]), float(c["beta_beta"])


# --------------------------------------------------------------------------- generator
class DonorMixingCounterfactuals:
    def __init__(self, vlm, pool, beta, gate, seed, example_key, image, question, queried=None,
                 factual_key=None):
        self.vlm = vlm
        self.pool = pool
        self.alpha, self.beta = beta
        self.gate = gate
        self.seed = seed
        self.example_key = str(example_key)
        self.image = image
        self.question = question
        self.candidates = pool.candidates(queried, factual_key)
        if not self.candidates:
            raise ValueError("No admissible donor image")
        self.queried = queried
        self._donor_cache = {}

    def observe(self, inputs):
        """Kept for interface symmetry: checks that the input has visual tokens."""
        projector_outputs(self.vlm, inputs)

    def sample(self, k):
        rng = np.random.default_rng(derive_seed(self.seed, self.example_key, k, "u-donor"))
        u = float(rng.beta(self.alpha, self.beta))
        donor = int(self.candidates[rng.integers(len(self.candidates))])
        return {"u": u, "donor": self.pool.images[donor][0], "donor_index": donor,
                "factual_gate": u >= 1 - self.gate}

    def donor_features(self, donor_index):
        if donor_index not in self._donor_cache:
            image = self.pool.images[donor_index][1]()
            self._donor_cache[donor_index] = projector_outputs(
                self.vlm, donor_inputs(self.vlm, image, self.image, self.question))
        return self._donor_cache[donor_index]

    def clear(self):
        if len(self._donor_cache) > 8:
            self._donor_cache.clear()

    def draw(self, k):
        s = self.sample(k)
        if s["factual_gate"]:
            return nullcontext()
        return mixing(self.vlm, self.donor_features(s["donor_index"]), s["u"])
