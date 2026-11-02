import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaModel, LlavaConfig, LlavaForConditionalGeneration

from methods.bernstein import empirical_radius, estimate_budget, evaluate_records
from methods.counterfactual_gen import CounterFactualGenerator
from methods.hallucination_circuit import prop_LLaVA, causal_attention_mask
from methods.restoration_effects import FactualRestoration, append_target_prefix, target_log_probability
from methods.faithfulness import mean_ablation
from pyfunctions.nodes import Node
from training.exogenous_module import ExogenousVariableModule


def tiny_llava():
    torch.manual_seed(12)
    config = LlavaConfig(
        text_config={"model_type": "llama", "hidden_size": 16,
                     "intermediate_size": 32, "num_hidden_layers": 2,
                     "num_attention_heads": 4, "num_key_value_heads": 2,
                     "vocab_size": 64, "pad_token_id": 0},
        vision_config={"model_type": "clip_vision_model", "hidden_size": 16,
                       "intermediate_size": 32, "num_hidden_layers": 2,
                       "num_attention_heads": 4, "image_size": 8, "patch_size": 4},
        image_token_index=63, vision_feature_layer=-2, pad_token_id=0,
    )
    model = LlavaForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 63, 63, 63, 63, 5, 6]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
              "pixel_values": torch.randn(1, 3, 8, 8)}
    return model, inputs


@pytest.mark.parametrize("with_source", [False, True])
def test_decomposition_preserves_native_decoder_and_causality(with_source):
    torch.manual_seed(7)
    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=64)
    model = LlamaModel(config).eval()
    wrapper = SimpleNamespace(language_model=model, device="cpu")
    dataset = SimpleNamespace(wrapper=wrapper)
    embeddings = torch.randn(1, 5, 16)
    means = {i: torch.zeros(1, 5, 4, 4) for i in range(2)}
    sources = [[Node(0, -1, 1)]] if with_source else [[]]
    with torch.no_grad():
        expected = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        rel, irrel = prop_LLaVA(dataset, sources, embeddings, means)
        torch.testing.assert_close(rel + irrel, expected, atol=2e-6, rtol=2e-5)
        altered = embeddings.clone()
        altered[:, 2:] += 10
        other_rel, other_irrel = prop_LLaVA(dataset, sources, altered, means)
        torch.testing.assert_close((rel+irrel)[:, :2], (other_rel+other_irrel)[:, :2])
    mask = causal_attention_mask(3, "cpu", torch.float32)
    assert mask[0, 0, 0, 2] < -1000
    assert mask[0, 0, 2, 0] == 0


def test_exact_restoration_uses_fixed_target_and_removes_hooks():
    model, inputs = tiny_llava()
    target = [7, 8]
    factual = append_target_prefix(inputs, target)
    extractor = FactualRestoration(model, [{"layer": 0, "head": 1}])
    score = extractor.capture(factual, target)
    baseline, effects = extractor.effects(factual, target)
    assert baseline == score
    assert effects["0_1"] == pytest.approx(0, abs=1e-7)
    cf = dict(factual, pixel_values=inputs["pixel_values"] + 1)
    baseline, effects = extractor.effects(cf, target)
    projection = extractor.decoder.layers[0].self_attn.o_proj
    reference = extractor.factual["0_1"][1]
    def manual_patch(module, args):
        hidden = args[0].clone()
        hidden[:, :, 4:8] = reference
        return (hidden,)
    handle = projection.register_forward_pre_hook(manual_patch)
    try:
        manual = target_log_probability(model, cf, target)
    finally:
        handle.remove()
    assert effects["0_1"] == pytest.approx(manual-baseline)
    assert not projection._forward_pre_hooks
    with pytest.raises(ValueError, match="layouts"):
        extractor.effects(inputs, target)
    assert not projection._forward_pre_hooks
    with pytest.raises(ValueError, match="same target"):
        extractor.effects(factual, [9])
    with torch.no_grad():
        logits = model(**factual).logits[0, -2:].float().log_softmax(-1)
    assert score == pytest.approx((logits[0, 7] + logits[1, 8]).item())


def test_native_mean_ablation_keeps_all_heads_and_changes_empty_circuit():
    model, inputs = tiny_llava()
    means = {i: torch.zeros(1, 7, 4, 4) for i in range(2)}
    with torch.no_grad():
        original = model(**inputs).logits
        with mean_ablation(model, means, [Node(l, -1, h) for l in range(2) for h in range(4)]):
            recovered = model(**inputs).logits
        with mean_ablation(model, means, []):
            baseline = model(**inputs).logits
    torch.testing.assert_close(recovered, original)
    assert not torch.allclose(baseline, original)
    assert not model.model.language_model.layers[0].self_attn.o_proj._forward_pre_hooks


def test_generator_broadcasts_and_inner_optimization_leaves_model_gradients_clean():
    generator = CounterFactualGenerator(cf_grad_steps=3, intervene_language=False)
    original = torch.ones(2, 3, 4)
    donor = torch.zeros_like(original)
    vision, language = generator.generate(original, original, donor, donor, torch.tensor([0., 1.]), torch.zeros(2))
    torch.testing.assert_close(vision[0], donor[0])
    torch.testing.assert_close(vision[1], original[1])
    torch.testing.assert_close(language, original)
    classifier = torch.nn.Linear(4, 2)
    before = [p.clone() for p in classifier.parameters()]
    def fusion(v, q):
        return classifier(v.mean(1))
    u, _ = generator.optimize_u_for_intervention(
        original, original, donor, donor, torch.full((2,), .5), torch.full((2,), .5),
        torch.ones(2, dtype=torch.long), fusion)
    assert ((0 <= u) & (u < 1)).all()
    assert not torch.equal(u, torch.full((2,), .5))
    for parameter, saved in zip(classifier.parameters(), before):
        assert parameter.grad is None
        torch.testing.assert_close(parameter, saved)


def test_posterior_sampling_and_checkpoint_roundtrip():
    module = ExogenousVariableModule()
    module.update_posterior(torch.tensor([[2., 1.], [1., 2.]]), torch.tensor([0, 0]))
    assert module.n_updates.item() == 1
    vision, language = module(10, torch.device("cpu"))
    assert vision.shape == language.shape == (10,)
    assert (vision < 1).all() and (vision >= 0).all()
    loaded = ExogenousVariableModule()
    loaded.load_state_dict(module.state_dict())
    assert loaded.get_posterior_mean() == module.get_posterior_mean()


def records(example, count, value):
    return [{"example_id": example, "counterfactual_index": i,
             "target_token_ids": [2], "node_patch_deltas": {"0_0": value}}
            for i in range(count)]


def test_bernstein_actual_sample_budget_and_family_confidence():
    nodes = [{"layer": 0, "head": 0}]
    rows = records("one", 200, .01) + records("two", 200, .02)
    result = evaluate_records(rows, nodes, .05, .05, -.1, .1)
    assert len(result["examples"]) == 2
    for example in result["examples"]:
        node = example["nodes"]["0_0"]
        count = node["required_budget"]
        assert 2 <= count <= 200
        delta = .05 / (2 * 199)
        assert node["delta_per_prefix"] == pytest.approx(delta)
        assert empirical_radius(count, 0, .2, delta) <= .05
        assert empirical_radius(count-1, 0, .2, delta) > .05
    result = evaluate_records(records("short", 2, 0), nodes, .001, .05, -1, 1)
    assert result["examples"][0]["m_bar"] is None
    assert result["examples"][0]["status"] == "insufficient_counterfactuals"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "outside", "nonfinite", "one_sample"])
def test_bernstein_rejects_invalid_or_mock_observations(problem):
    rows = records("one", 3, .1)
    if problem == "missing":
        rows[0].pop("node_patch_deltas")
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "outside":
        rows[0]["node_patch_deltas"]["0_0"] = 3
    elif problem == "nonfinite":
        rows[0]["node_patch_deltas"]["0_0"] = float("nan")
    else:
        rows = rows[:1]
    with pytest.raises(ValueError):
        evaluate_records(rows, [{"layer": 0, "head": 0}], .1, .05, -1, 1)


def test_benchmark_answers_and_alignment(tmp_path):
    from eval_metrics.eval_hallusionbench import evaluate_hallusionbench
    from eval_metrics.eval_pope import evaluate_pope
    from eval_metrics.eval_coco_recall import evaluate_coco_context_recall
    predictions, answers = tmp_path / "pred.jsonl", tmp_path / "answers.jsonl"
    def write(preds, gold):
        predictions.write_text("".join(json.dumps(r)+"\n" for r in preds))
        answers.write_text("".join(json.dumps(r)+"\n" for r in gold))
    write([{"text": "yesterday"}], [{"label": "yes"}])
    assert evaluate_hallusionbench(predictions, answers) == 0
    write([{"text": "No, there is no cat."}], [{"label": "no"}])
    assert evaluate_pope(predictions, answers) == 1
    write([{"id": 1, "text": "yes"}], [{"id": 2, "label": "yes"}])
    with pytest.raises(ValueError, match="Mismatched"):
        evaluate_pope(predictions, answers)
    write([{"text": "yes"}], [{"label": "yes"}, {"label": "no"}])
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_pope(predictions, answers)
    write([{"concepts": ["cat"]}, {"concepts": []}], [{"concepts": ["cat"]}, {"concepts": []}])
    assert evaluate_coco_context_recall(predictions, answers) == 1


def test_qwen_wrappers_do_not_substitute_other_models():
    from models.qwen3_instruct_wrapper import Qwen3InstructWrapper
    from models.qwen3_thinking_wrapper import Qwen3ThinkingWrapper
    model, _ = tiny_llava()
    for wrapper in [Qwen3InstructWrapper, Qwen3ThinkingWrapper]:
        with pytest.raises(ValueError, match="native Qwen3"):
            wrapper(device="cpu", model=model, processor=SimpleNamespace(tokenizer=None))


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("language_intervention", [False, True])
def test_training_uses_learned_interpolation_and_preserves_gradients(checkpointing, language_intervention):
    from training.llava_cvl import LLaVACVL, VQAHead
    model = LLaVACVL.__new__(LLaVACVL)
    torch.nn.Module.__init__(model)
    model.llava, inputs = tiny_llava()
    model.llava.config.use_cache = False
    model.vqa_head = VQAHead(16, 3, dropout=0)
    model.exogenous_module = ExogenousVariableModule()
    model.cf_generator = CounterFactualGenerator(cf_grad_steps=2, intervene_language=language_intervention)
    model.clip_M = 10
    model.train()
    if checkpointing:
        model.llava.gradient_checkpointing_enable()
    original_embedding_hooks = set(model.llava.get_input_embeddings()._forward_hooks)
    donor = dict(inputs)
    donor["input_ids"] = inputs["input_ids"].clone()
    donor["input_ids"][:, -2:] = torch.tensor([10, 11])
    donor["pixel_values"] = torch.randn_like(inputs["pixel_values"])
    observed_ids = []
    def capture_inputs(module, args, kwargs):
        observed_ids.append(kwargs["input_ids"].clone())
    handle = model.llava.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    try:
        output = model(**inputs, answer_idx=torch.tensor([1]), run_counterfactual=True,
                       rand_input_ids=donor["input_ids"], rand_attention_mask=donor["attention_mask"],
                       rand_pixel_values=donor["pixel_values"])
    finally:
        handle.remove()
    assert model.llava.is_gradient_checkpointing == checkpointing
    assert output["logits"].shape == output["logits_cf"].shape == (1, 3)
    assert torch.isfinite(output["omega"]).all()
    assert len(observed_ids) == 5  # factual, donor, two inner steps, final CF
    assert torch.equal(observed_ids[1], donor["input_ids"])
    assert all(torch.equal(ids, inputs["input_ids"]) for i, ids in enumerate(observed_ids) if i != 1)
    assert all(parameter.grad is None for parameter in model.parameters())
    (output["logits"].sum() + output["logits_cf"].sum()).backward()
    assert model.vqa_head.classifier[0].weight.grad is not None
    assert not model.llava.model.multi_modal_projector._forward_hooks
    assert set(model.llava.get_input_embeddings()._forward_hooks) == original_embedding_hooks


def test_offline_circuit_search_excludes_nonpromoters():
    import importlib.util
    filename = Path(__file__).resolve().parents[1] / "vlm_experiments/LLaVA3.5-7B/find_circuit_from_matrix.py"
    spec = importlib.util.spec_from_file_location("circuit_search", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adjacency = {"head_combined": torch.zeros(2, 2), "head_mean_beta": torch.zeros(2, 2),
                 "logit_combined": torch.tensor([1., 2.]), "logit_mean_beta": torch.tensor([1., -1.])}
    circuit, _ = module.find_circuit(adjacency, num_layers=1, num_heads=2, num_outliers=2)
    assert [(n["layer"], n["head"]) for n in circuit] == [(0, 0)]
