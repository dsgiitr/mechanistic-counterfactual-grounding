"""Mechanical tests on tiny randomly initialised LLaVA, Idefics3 (SmolVLM) and Qwen3-VL models.

No weights or network access are needed. The tests check that the CD-T
decomposition reproduces the native forward pass, that restoration effects equal
an explicit single-node patch, that latent counterfactuals are deterministic and
consistent under batching, that patching only touches the prefill, and the
numerical constants of the bounds used in the paper.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcg.bounds import bernstein_budget, bound_constants, eb_radius, example_statistics  # noqa: E402
from mcg.cdt import capture_context, propagate, sweep  # noqa: E402
from mcg.circuits import iterative_search, layer_normalized_topk, global_topk, iterative_decay  # noqa: E402
from collections import Counter  # noqa: E402
from mcg.counterfactuals import DonorMixingCounterfactuals, DonorPool, mixing, projector_outputs  # noqa: E402
from mcg.models import VLM, HookSet, append_tokens  # noqa: E402
from mcg.patching import mean_ablation, patch_heads  # noqa: E402
from mcg.restoration import Restorer  # noqa: E402
from mcg.stats import auroc, wilson_interval  # noqa: E402
from mcg.tasks import CocoObjects, context_recall  # noqa: E402

@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


STUB =SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=None, pad_token_id=0, encode=lambda *a, **k: [1]))


def tiny_llava():
    from transformers import LlavaConfig, LlavaForConditionalGeneration
    torch.manual_seed(0)
    config = LlavaConfig(
        text_config={"model_type": "llama", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64, "pad_token_id": 0},
        vision_config={"model_type": "clip_vision_model", "hidden_size": 16, "intermediate_size": 32,
                       "num_hidden_layers": 2, "num_attention_heads": 4, "image_size": 8, "patch_size": 4},
        image_token_index=63, vision_feature_layer=-2, pad_token_id=0)
    model = LlavaForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 63, 63, 63, 63, 5, 6, 7]])
    return model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": torch.randn(1, 3, 8, 8)}


def tiny_idefics3():
    from transformers import Idefics3Config, Idefics3ForConditionalGeneration
    torch.manual_seed(1)
    config = Idefics3Config(
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                       "num_attention_heads": 4, "image_size": 16, "patch_size": 4},
        text_config={"model_type": "llama", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64, "pad_token_id": 0},
        scale_factor=2, image_token_id=63)
    model = Idefics3ForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 63, 63, 63, 63, 5, 6, 7]])
    return model, {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                   "pixel_values": torch.randn(1, 1, 3, 16, 16)}


def tiny_qwen3():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    torch.manual_seed(2)
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=4, patch_size=2,
                           temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32,
                           num_position_embeddings=16, deepstack_visual_indexes=[0, 1]),
        image_token_id=62, video_token_id=63, vision_start_token_id=60, vision_end_token_id=61)
    model = Qwen3VLForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 60, 62, 62, 62, 62, 61, 5, 6]])
    return model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": torch.randn(16, 24),
                   "image_grid_thw": torch.tensor([[1, 4, 4]])}


FAMILIES = {"llava": tiny_llava, "idefics3": tiny_idefics3, "qwen3_vl": tiny_qwen3}


def build(family):
    model, inputs = FAMILIES[family]()
    return VLM("tiny", device="cpu", model=model, processor=STUB, family=family), inputs


def zero_means(vlm, length):
    return {l: torch.zeros(length, vlm.num_heads, vlm.head_dim) for l in range(vlm.num_layers)}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_cdt_decomposition_reproduces_native_forward(family):
    vlm, inputs = build(family)
    ctx = capture_context(vlm, inputs)
    S = inputs["input_ids"].shape[1]
    means = zero_means(vlm, S)
    for start, sources in ((0, [[]]), (1, [[0], [3]])):
        rel, irrel = propagate(vlm, ctx, start, sources, means)
        logits = (rel + irrel)[:, -1] @ vlm.lm_head.weight.T
        torch.testing.assert_close(logits, ctx["logits"][-1].expand_as(logits), atol=1e-4, rtol=1e-4)
    rel, _ = propagate(vlm, ctx, 1, [[0]], means)
    assert rel.abs().sum() > 0
    scores = sweep(vlm, ctx, means, S - 1, vlm.lm_head.weight[3] - vlm.lm_head.weight[4], batch_size=3)
    N = vlm.num_layers * vlm.num_heads
    assert scores["s_head"].shape == (N, N)
    H = vlm.num_heads
    for j in range(N):  # no edge from a head to the same or an earlier layer
        assert scores["s_head"][j, : (j // H + 1) * H].abs().sum() == 0


@pytest.mark.parametrize("family", list(FAMILIES))
def test_restoration_equals_explicit_patch_and_is_batch_invariant(family):
    vlm, inputs = build(family)
    target = [7, 9]
    inputs = append_tokens(inputs, target[:-1])
    ctx = capture_context(vlm, inputs)
    nodes = [{"layer": 0, "head": 1}, {"layer": 2, "head": 3}, {"layer": 1, "head": 0}]
    restorer = Restorer(vlm, nodes, ctx["acts"])
    base, deltas, _ = restorer.run(inputs, target)
    assert max(abs(v) for v in deltas.values()) < 1e-5  # restoring on the factual input changes nothing
    donor = projector_outputs(vlm, donor_of(inputs))
    with mixing(vlm, donor, 0.4):
        b1, d1, acts = restorer.run(inputs, target, batch_size=8)
        b2, d2, _ = restorer.run(inputs, target, batch_size=1)
        D = vlm.head_dim
        l, h = 2, 3

        def manual(module, args):
            x = args[0].clone()
            x[:, :, h * D:(h + 1) * D] = restorer.factual[(l, h)]
            return (x,)

        with HookSet() as hooks:
            hooks.pre(vlm.o_proj(l), manual)
            patched = float(vlm.target_logprob(inputs, target)[0])
    assert abs(b1 - b2) < 1e-5 and abs(b1 - base) > 1e-6
    assert all(abs(d1[k] - d2[k]) < 1e-5 for k in d1)
    assert abs((patched - b1) - d1[(2, 3)]) < 1e-5
    assert set(acts) == {(0, 1), (2, 3), (1, 0)}


def donor_of(inputs, shift=1.0):
    """Same token layout, different image."""
    return dict(inputs, pixel_values=inputs["pixel_values"] + shift)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_donor_mixing_interpolates_projected_tokens_and_tiles(family):
    vlm, inputs = build(family)
    factual = vlm.forward_logits(inputs)[0, -1]
    donor_inputs = donor_of(inputs)
    donor = projector_outputs(vlm, donor_inputs)
    with mixing(vlm, donor, 1.0):
        torch.testing.assert_close(vlm.forward_logits(inputs)[0, -1], factual)
    with mixing(vlm, donor, 0.0):  # all visual tokens (and DeepStack features) come from the donor
        torch.testing.assert_close(vlm.forward_logits(inputs)[0, -1], vlm.forward_logits(donor_inputs)[0, -1],
                                   atol=1e-5, rtol=1e-5)
    with mixing(vlm, donor, 0.5):
        mixed = vlm.forward_logits(inputs)[0, -1]
        batched = vlm.forward_logits(vlm.expand(inputs, 3))[:, -1]
    assert (mixed - factual).abs().max() > 1e-6
    torch.testing.assert_close(batched, mixed.expand_as(batched), atol=1e-5, rtol=1e-5)


def test_donor_pool_rule_and_deterministic_u_draws():
    from mcg.tasks import CocoObjects
    cooc = {"dog": Counter({"person": 9, "frisbee": 5, "car": 1})}
    pool = DonorPool([(k, None) for k in ("a", "b", "c", "d")],
                     {"a": {"dog"}, "b": {"person"}, "c": {"car"}, "d": {"cat"}}, cooc, CocoObjects(), exclude_top=2)
    assert pool.excluded("dog") == {"dog", "person", "frisbee"}
    assert [pool.images[i][0] for i in pool.candidates("dog", factual_key="d")] == ["c"]
    assert pool.queried_object({"question": "Is there a dog in the image?"}) == "dog"
    vlm = SimpleNamespace()
    gen = DonorMixingCounterfactuals(vlm, pool, (0.1, 0.1), 0.1, seed=3, example_key="e", image=None,
                                     question="", queried="dog", factual_key="d")
    draws = [gen.sample(k) for k in range(200)]
    assert draws[:5] == [gen.sample(k) for k in range(5)]
    assert all(d["donor"] == "c" for d in draws)
    assert all(d["factual_gate"] == (d["u"] >= 0.9) for d in draws)
    assert 0 < sum(d["factual_gate"] for d in draws) < 200   # Beta(0.1, 0.1) is bimodal


def test_patch_heads_only_touches_prefill_and_mean_ablation_identities():
    vlm, inputs = build("llava")
    S = inputs["input_ids"].shape[1]
    ctx = capture_context(vlm, inputs)
    factual = {(1, 2): ctx["acts"][1][:, 2]}
    with patch_heads(vlm, factual):
        torch.testing.assert_close(vlm.forward_logits(inputs)[0, -1], ctx["logits"][-1])
    seen = []  # (sequence length, |head 0 input|) after the patch hook, for every call
    with patch_heads(vlm, {(0, 0): torch.zeros(S, vlm.head_dim)}), HookSet() as hooks:
        hooks.pre(vlm.o_proj(0), lambda m, a: seen.append((a[0].shape[1], a[0][0, :, :vlm.head_dim].abs().sum().item())))
        vlm.generate(inputs, max_new_tokens=3)
    assert seen[0] == (S, 0.0)                           # prefill patched
    assert all(n == 1 and v > 0 for n, v in seen[1:])   # cached decoding steps untouched
    means = {l: ctx["acts"][l] for l in range(vlm.num_layers)}
    with mean_ablation(vlm, means, keep=set()):
        torch.testing.assert_close(vlm.forward_logits(inputs)[0, -1], ctx["logits"][-1], atol=1e-5, rtol=1e-5)
    with mean_ablation(vlm, zero_means(vlm, S), ablate={(0, 0)}):
        changed = vlm.forward_logits(inputs)[0, -1]
    assert (changed - ctx["logits"][-1]).abs().max() > 1e-6


def test_iterative_search_follows_appendix_c():
    L, H = 3, 2
    N = L * H
    s_logit = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9, 0.8])
    beta_logit = torch.tensor([1., 1., 1., 1., 1., -1.])
    s_head = torch.zeros(N, N)
    beta_head = torch.zeros(N, N)
    s_head[2, 4], beta_head[2, 4] = 0.5, 1.0   # layer1 head0 -> layer2 head0
    s_head[3, 4], beta_head[3, 4] = 0.9, -1.0  # negative contribution: filtered
    s_head[0, 2], beta_head[0, 2] = 0.7, 1.0   # layer0 head0 -> layer1 head0
    scores = {"s_logit": s_logit, "beta_logit": beta_logit, "s_head": s_head, "beta_head": beta_head,
              "num_layers": L, "num_heads": H}
    circuit = iterative_search(scores, k0=1, k_schedule=lambda i: 1)
    assert [(n["layer"], n["head"]) for n in circuit] == [(2, 0), (1, 0), (0, 0)]
    assert [(n["layer"], n["head"]) for n in global_topk(scores, 2)] == [(2, 0), (1, 1)]
    assert len(layer_normalized_topk(scores, 3)) == 3
    assert len(iterative_decay(scores, 2)) <= 2


def test_bound_constants_match_the_paper():
    c = bound_constants(0.185, 0.05, 0.6)
    assert c["k_hat_floor"] == pytest.approx(7.98, abs=0.01)
    assert c["k_hat_min"] == 8
    assert c["k_hat_slope"] == pytest.approx(216, abs=1)
    assert round(c["eb_min_budget"]) == 29
    assert c["max_variance_for_range"] == pytest.approx(0.09)
    L = math.log(40)
    assert bernstein_budget(0.0, 0.185, 0.05, 0.6) == math.ceil(2 / 3 * 0.6 * 0.185 / 0.185 ** 2 * L)
    assert bernstein_budget(0.05, 0.185, 0.05, 0.6) > bernstein_budget(0.01, 0.185, 0.05, 0.6)
    assert eb_radius(10, 0.01, 0.6, 0.05) > eb_radius(100, 0.01, 0.6, 0.05)
    stats = example_statistics({"0_0": [0.1, 0.2, 0.3], "1_1": [0.0, 0.0, 0.3]}, 0.185, 0.05, 0.6)
    assert stats["V"] == pytest.approx((0.01 + 0.03) / 2)
    assert example_statistics({"0_0": [0.1]}, 0.185, 0.05, 0.6) is None


def test_coco_objects_and_statistics():
    objects = CocoObjects()
    mentions = objects.mentions("Two dogs sit near a hot dog and a dining table with cups.")
    assert [m[0] for m in mentions] == ["dog", "hot dog", "dining table", "cup"]
    assert context_recall(["dog", "cup"], ["dog", "cat"]) == 0.5
    lo, hi = wilson_interval(34, 50)
    assert (round(100 * lo), round(100 * hi)) == (54, 79)
    assert auroc([3, 4], [1, 2]) == 1.0 and auroc([1], [1]) == 0.5
