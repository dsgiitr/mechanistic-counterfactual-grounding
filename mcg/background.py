"""Background mean activations of every attention head.

These means are the reference point for (i) the CD-T source decomposition
(beta = activation - mean, gamma = mean at the source head) and (ii) mean
ablation in the faithfulness analysis (Appendix D).

Means are kept per region so they can be aligned to any example:
  visual tokens  -> mean at the same visual-token offset (falls back to the
                    visual-region mean for offsets unseen in the background set)
  final position -> mean at the last prompt position
  other text     -> text-region mean
"""
import torch
from tqdm import tqdm

from mcg.models import HookSet


class BackgroundMeans:
    def __init__(self, num_layers, num_heads, head_dim, max_visual=4096):
        shape = (num_layers, num_heads, head_dim)
        self.visual_sum = torch.zeros(num_layers, 0, num_heads, head_dim, dtype=torch.float32)
        self.visual_count = torch.zeros(0, dtype=torch.long)
        self.region_sum = {r: torch.zeros(shape, dtype=torch.float32) for r in ("visual", "text", "final")}
        self.region_count = {r: 0 for r in ("visual", "text", "final")}
        self.max_visual = max_visual

    @torch.no_grad()
    def accumulate(self, vlm, inputs):
        acts = {}

        def capture(layer):
            def hook(module, args):
                x = args[0][0].float().cpu()
                acts[layer] = x.view(x.shape[0], vlm.num_heads, vlm.head_dim)
            return hook

        with HookSet() as hooks:
            for layer in range(vlm.num_layers):
                hooks.pre(vlm.o_proj(layer), capture(layer))
            vlm.forward_logits(inputs, keep=1)
        stacked = torch.stack([acts[l] for l in range(vlm.num_layers)])  # [L, S, H, D]
        visual = (inputs["input_ids"][0] == vlm.image_token_id).cpu()
        text = ~visual
        text[-1] = False
        nvis = min(int(visual.sum()), self.max_visual)
        if nvis > self.visual_sum.shape[1]:
            pad = nvis - self.visual_sum.shape[1]
            self.visual_sum = torch.cat((self.visual_sum, torch.zeros(
                stacked.shape[0], pad, *stacked.shape[2:], dtype=torch.float32)), 1)
            self.visual_count = torch.cat((self.visual_count, torch.zeros(pad, dtype=torch.long)))
        if nvis:
            self.visual_sum[:, :nvis] += stacked[:, visual][:, :nvis]
            self.visual_count[:nvis] += 1
            self.region_sum["visual"] += stacked[:, visual].sum(1)
            self.region_count["visual"] += int(visual.sum())
        self.region_sum["text"] += stacked[:, text].sum(1)
        self.region_count["text"] += int(text.sum())
        self.region_sum["final"] += stacked[:, -1]
        self.region_count["final"] += 1

    def state_dict(self):
        means = {r: (self.region_sum[r] / max(1, self.region_count[r])).float()
                 for r in self.region_sum}
        count = self.visual_count.clamp_min(1).float()[None, :, None, None]
        return {"visual_offset": (self.visual_sum / count).half(), "visual_offset_count": self.visual_count,
                "region": {r: m.half() for r, m in means.items()},
                "region_count": dict(self.region_count)}

    def save(self, path, **meta):
        torch.save({**self.state_dict(), "meta": meta}, path)


class MeanCache:
    """Loaded background means; builds per-example [S, H, D] references per layer."""

    def __init__(self, path, device):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.visual_offset = state["visual_offset"]
        self.visual_count = state["visual_offset_count"]
        self.region = state["region"]
        self.device = device
        self.meta = state.get("meta", {})

    def for_inputs(self, vlm, input_ids, final_position=None):
        """dict layer -> tensor [S, H, D] aligned with this example's tokens."""
        ids = input_ids[0].cpu()
        S = ids.shape[0]
        final_position = S - 1 if final_position is None else final_position
        out = self.region["text"].float()[:, None].expand(-1, S, -1, -1).clone()  # [L, S, H, D]
        visual = (ids == vlm.image_token_id).nonzero(as_tuple=True)[0]
        if len(visual):
            out[:, visual] = self.region["visual"].float()[:, None]
            known = min(len(visual), int((self.visual_count > 0).sum()))
            out[:, visual[:known]] = self.visual_offset[:, :known].float()
        out[:, final_position] = self.region["final"].float()
        out = out.to(self.device, vlm.dtype)
        return {layer: out[layer] for layer in range(out.shape[0])}


def compute_background(vlm, bench, num_items, seed=0):
    """Accumulate means over `num_items` benchmark prompts (factual inputs only)."""
    import random
    stats = BackgroundMeans(vlm.num_layers, vlm.num_heads, vlm.head_dim)
    indices = list(range(len(bench)))
    random.Random(seed).shuffle(indices)
    used = 0
    for index in tqdm(indices, desc="background means"):
        item = bench.item(index)
        if item["image"] is None:
            continue
        stats.accumulate(vlm, vlm.prepare(item["image"], bench.prompt(item)))
        used += 1
        if used >= num_items:
            break
    return stats
