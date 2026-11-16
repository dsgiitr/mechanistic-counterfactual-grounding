"""Tests of the separate CVL (counterfactual vision-and-language) training utility in training/."""
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlavaConfig, LlavaForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods.counterfactual_gen import CounterFactualGenerator  # noqa: E402
from training.exogenous_module import ExogenousVariableModule  # noqa: E402


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
