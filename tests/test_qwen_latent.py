from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from methods.hallucination_circuit import prop_decoder, build_adjacency_matrix
from methods.latent_counterfactual import (
    GeneratorConfig, LatentCounterfactualGenerator, generator_training_step, visual_intervention,
)
from models.native_utils import capture_decoder_context
from methods.experiment_cli import extract_restoration_records
from methods.restoration_effects import FactualRestoration, append_target_prefix, target_log_probability
from models.qwen3_instruct_wrapper import Qwen3InstructWrapper
from models.qwen3_thinking_wrapper import Qwen3ThinkingWrapper
from pyfunctions.nodes import Node
from test_grounding_pipeline import tiny_llava


def tiny_qwen():
    torch.manual_seed(23)
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, rope_parameters={"rope_type": "default", "rope_theta": 10000., "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=4,
                           patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                           out_hidden_size=32, num_position_embeddings=16,
                           deepstack_visual_indexes=[0, 1]),
        image_token_id=62, video_token_id=63, vision_start_token_id=60, vision_end_token_id=61,
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 60, 62, 62, 62, 62, 61, 5, 6]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  pixel_values=torch.randn(16, 24), image_grid_thw=torch.tensor([[1, 4, 4]]),
                  mm_token_type_ids=(ids == 62).long())
    return model, inputs


class LocalProcessor:
    """Processor fixture; all vision/decoder computation still uses the actual model."""
    def __init__(self, inputs):
        self.inputs = inputs
        self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [7, 8] if " " in text.strip() else [7])
        self.last_messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        return "<user>image:" + messages[0]["content"][1]["text"] + "<assistant>"

    def __call__(self, text, images, **kwargs):
        image = images[0] if isinstance(images, list) else images
        offset = sum(image.getpixel((0, 0))) / (3 * 255)
        inputs = {key: value.clone() for key, value in self.inputs.items()}
        inputs["pixel_values"] += offset
        return inputs


@pytest.mark.parametrize("wrapper_type", [Qwen3InstructWrapper, Qwen3ThinkingWrapper])
def test_native_wrappers_capture_real_visual_heads(wrapper_type):
    model, inputs = tiny_qwen()
    processor = LocalProcessor(inputs)
    wrapper = wrapper_type(device="cpu", model=model, processor=processor)
    token, logits, _ = wrapper.run_greedy(Image.new("RGB", (8, 8)), "Is there a cat?")
    assert wrapper.model is model
    assert token == logits.argmax().item()
    assert wrapper.visual_positions == [2, 3, 4, 5]
    assert wrapper.N_vis == 4
    assert wrapper.answer_position == 8
    assert wrapper.activations[0].shape == (1, 9, 4, 8)
    assert processor.last_messages[0]["content"][0]["type"] == "image"
    assert len(wrapper.propagation_context["deepstack_visual_embeds"]) == 2
    wrapper.remove_hooks()
    assert not model.model.language_model.layers[0].self_attn.o_proj._forward_pre_hooks


@pytest.mark.parametrize("with_source", [False, True])
def test_qwen_decomposition_preserves_native_mrope_qknorm_deepstack(with_source):
    model, inputs = tiny_qwen()
    output, context = capture_decoder_context(model, inputs)
    dataset = SimpleNamespace(wrapper=SimpleNamespace(language_model=model.model.language_model,
                                                       propagation_context=context))
    means = {layer: torch.zeros(1, 9, 4, 8) for layer in range(3)}
    sources = [[Node(0, -1, 1)], [Node(1, 8, 2)]] if with_source else [[]]
    with torch.no_grad():
        rel, irrel, intermediates = prop_decoder(dataset, sources, context["base_embeds"], means, True)
    torch.testing.assert_close(rel+irrel, output.hidden_states[-1].expand_as(rel), atol=2e-6, rtol=2e-5)
    assert len(intermediates) == 3
    if with_source:
        assert rel.abs().sum() > 0
    assert not model.model.language_model._forward_pre_hooks


@pytest.mark.parametrize("architecture", ["qwen", "llava"])
def test_learned_generator_updates_without_changing_vlm_and_roundtrips(tmp_path, architecture):
    model, inputs = tiny_qwen() if architecture == "qwen" else tiny_llava()
    hidden_size = model.config.text_config.hidden_size
    torch.manual_seed(31)
    generator = LatentCounterfactualGenerator(GeneratorConfig(hidden_size, noise_dim=4, rank=8))
    optimizer = torch.optim.Adam(generator.parameters(), lr=.01)
    saved_model = {name: value.clone() for name, value in model.state_dict().items()}
    saved_generator = {name: value.clone() for name, value in generator.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    noises = [torch.randn(1, 4), torch.randn(1, 4)]
    target = [7, 8]
    forced = append_target_prefix(inputs, target)
    metrics = generator_training_step(model, forced, target, generator, optimizer, *noises)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.equal(value, generator.state_dict()[name]) for name, value in saved_generator.items())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in saved_model.items())
    assert all(p.grad is None for p in model.parameters())
    assert [p.requires_grad for p in model.parameters()] == flags
    assert not model.training
    generator.save(tmp_path / "generator.pt", "tiny")
    loaded = LatentCounterfactualGenerator.load(tmp_path / "generator.pt", "tiny", hidden_size)
    with pytest.raises(ValueError, match="different model"):
        LatentCounterfactualGenerator.load(tmp_path / "generator.pt", "other", hidden_size)
    image_id = getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", None))
    mask = forced["input_ids"] == image_id
    captured = []
    decoder = model.model.language_model
    def capture(module, args, kwargs):
        captured.append(kwargs["inputs_embeds"].detach().clone())
    handle = decoder.register_forward_pre_hook(capture, with_kwargs=True)
    target_log_probability(model, forced, target)
    handle.remove()
    with visual_intervention(model, forced, loaded, noises[0]) as state:
        first = target_log_probability(model, forced, target)
    with visual_intervention(model, forced, loaded, noises[0]):
        second = target_log_probability(model, forced, target)
    assert first == second
    assert len(state["generated"]) == (3 if architecture == "qwen" else 1)
    for original, generated in zip(state["original"], state["generated"]):
        rms = original.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        assert ((generated-original).abs() <= .1 * rms + 1e-6).all()
    before = captured[0]
    # Observe the decoder input after the generator hook, not before it.
    with visual_intervention(model, forced, loaded, noises[0]):
        handle = decoder.register_forward_pre_hook(capture, with_kwargs=True)
        target_log_probability(model, forced, target)
        handle.remove()
    torch.testing.assert_close(captured[-1][~mask], before[~mask])
    assert not torch.equal(captured[-1][mask], before[mask])
    assert not decoder._forward_pre_hooks


def test_latent_restoration_is_reproducible_and_uses_same_cf_for_every_patch(tmp_path):
    model, inputs = tiny_qwen()
    wrapper = Qwen3InstructWrapper(device="cpu", model=model, processor=LocalProcessor(inputs))
    generator = LatentCounterfactualGenerator(GeneratorConfig(32, noise_dim=4, rank=8)).eval()
    directory = tmp_path / "one"
    directory.mkdir()
    Image.new("RGB", (8, 8), (100, 90, 80)).save(directory / "original.png")
    questions = {"one": {"question": "Is it present?", "target_token_ids": [7, 8]}}
    nodes = [{"layer": 0, "head": 1}, {"layer": 1, "head": 2}]
    first = extract_restoration_records(wrapper, questions, tmp_path, nodes,
                                       mode="latent", generator=generator, num_counterfactuals=3, seed=5)
    second = extract_restoration_records(wrapper, questions, tmp_path, nodes,
                                        mode="latent", generator=generator, num_counterfactuals=3, seed=5)
    assert first == second
    assert len(first) == 3
    assert first[0]["latent_noise"] != first[1]["latent_noise"]
    assert all(set(record["node_patch_deltas"]) == {"0_1", "1_2"} for record in first)
    factual = wrapper.prepare_inputs(Image.new("RGB", (8, 8), (100, 90, 80)), "Is it present? ")
    forced = append_target_prefix(factual, [7, 8])
    extractor = FactualRestoration(model, nodes)
    extractor.capture(forced, [7, 8])
    noise = torch.tensor([first[0]["latent_noise"]])
    baseline, deltas = extractor.effects(forced, [7, 8], lambda: visual_intervention(model, forced, generator, noise))
    assert first[0]["cf_log_probability"] == baseline
    assert first[0]["node_patch_deltas"] == deltas
    assert not model.model.language_model._forward_pre_hooks


def test_native_qwen_adjacency_uses_current_context_and_answer_prefix():
    from tasks.hallucination_vlm import HallucinationDataset
    from methods.faithfulness import evaluate_faithfulness
    model, inputs = tiny_qwen()
    wrapper = Qwen3InstructWrapper(device="cpu", model=model, processor=LocalProcessor(inputs))
    example = {"image": Image.new("RGB", (8, 8)), "question": "Is it present?",
               "hallucinated_token_id": 7, "correct_token_id": 8,
               "answer_prefix_token_ids": [9, 10]}
    dataset = HallucinationDataset([example], wrapper)
    sources = [Node(0, -1, 0), Node(1, -1, 1)]
    with torch.no_grad():
        adjacency = build_adjacency_matrix(dataset, sources, batch_size=2)
    assert adjacency["head_combined"].shape == (12, 12)
    assert torch.isfinite(adjacency["head_combined"]).all()
    assert torch.isfinite(adjacency["logit_combined"]).all()
    assert wrapper.propagation_context["base_embeds"].shape[1] == 11
    all_heads = [Node(layer, -1, head) for layer in range(3) for head in range(4)]
    faithfulness = evaluate_faithfulness(dataset, all_heads)
    assert faithfulness["overall_faithfulness"] == pytest.approx(1.)
    wrapper.remove_hooks()


def test_generation_preserves_exact_thinking_prefix_and_rejects_unfinished_reasoning():
    from methods.generation import generate_binary_answer
    class CharacterTokenizer:
        def decode(self, tokens, **kwargs):
            return "".join(chr(token) for token in tokens)
    class GeneratingModel:
        text = "<think>inspect image</think>\nyes"
        def generate(self, **kwargs):
            return torch.cat((kwargs["input_ids"], torch.tensor([[ord(c) for c in self.text]])), dim=1)
    class Wrapper:
        tokenizer = CharacterTokenizer()
        model = GeneratingModel()
        def prepare_inputs(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}
    wrapper = Wrapper()
    result = generate_binary_answer(wrapper, None, "question")
    assert result["answer"] == "yes"
    assert result["answer_prefix"] == "<think>inspect image</think>\n"
    assert result["answer_prefix_token_ids"] == [ord(c) for c in result["answer_prefix"]]
    assert result["answer_token_id"] == ord("y")
    wrapper.model.text = "<think>yes is only an internal hypothesis"
    assert generate_binary_answer(wrapper, None, "question") is None
    wrapper.model.text = "yes, but I am still reasoning"
    wrapper.prepare_inputs = lambda *args, **kwargs: {
        "input_ids": torch.tensor([[ord(c) for c in "prompt<think>"]])}
    assert generate_binary_answer(wrapper, None, "question") is None


def test_generator_hook_cleanup_on_forward_failure():
    model, inputs = tiny_qwen()
    generator = LatentCounterfactualGenerator(GeneratorConfig(32, noise_dim=4, rank=8))
    with pytest.raises(RuntimeError, match="failed"):
        with visual_intervention(model, inputs, generator, torch.zeros(1, 4)):
            raise RuntimeError("failed forward")
    assert not model.model.language_model._forward_pre_hooks


def test_training_and_latent_extraction_cli_roundtrip(tmp_path, monkeypatch):
    import json
    import sys
    import training.train_generator as training_cli
    import methods.experiment_cli as extraction_cli
    model, inputs = tiny_qwen()
    wrapper = Qwen3InstructWrapper(device="cpu", model=model, processor=LocalProcessor(inputs))
    directory = tmp_path / "one"
    directory.mkdir()
    Image.new("RGB", (8, 8), (90, 80, 70)).save(directory / "original.png")
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps([{"image": "one/original.png", "question": "Is it present?", "correct_token_ids": [7, 8]}]))
    checkpoint = tmp_path / "generator.pt"
    monkeypatch.setattr(training_cli, "load_wrapper", lambda *args: wrapper)
    monkeypatch.setattr(sys, "argv", ["train_generator", "--dataset_path", str(manifest),
                                     "--model_id", wrapper.model_id, "--device", "cpu", "--output", str(checkpoint),
                                     "--epochs", "1", "--noise_dim", "4", "--rank", "8"])
    training_cli.main()
    assert checkpoint.exists()
    training_record = json.loads(checkpoint.with_suffix(".training.jsonl").read_text().strip())
    assert training_record["epoch"] == 1
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"one": {"question": "Is it present?", "target_token_ids": [7, 8]}}))
    circuit = tmp_path / "circuit.json"
    circuit.write_text(json.dumps({"circuit": [{"layer": 0, "head": 1}]}))
    output = tmp_path / "effects.jsonl"
    monkeypatch.setattr(extraction_cli, "load_wrapper", lambda *args: wrapper)
    monkeypatch.setattr(sys, "argv", ["extract", "--model_id", wrapper.model_id, "--device", "cpu",
                                     "--circuit_path", str(circuit), "--image_dir", str(tmp_path),
                                     "--questions_path", str(questions), "--output", str(output),
                                     "--counterfactual_mode", "latent", "--generator_checkpoint", str(checkpoint),
                                     "--num_counterfactuals", "2"])
    extraction_cli.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 2
    assert all(len(record["generator_checkpoint_sha256"]) == 64 for record in records)
    assert all(len(record["circuit_sha256"]) == 64 for record in records)
    assert all(record["counterfactual_mode"] == "latent" for record in records)
    from methods.bernstein import evaluate_records
    statistics = evaluate_records(records, [{"layer": 0, "head": 1}], .1, .05, -100, 100)
    assert statistics["examples"][0]["status"] == "insufficient_counterfactuals"
    records[1]["generator_checkpoint_sha256"] = "different_generator"
    with pytest.raises(ValueError, match="Cannot pool"):
        evaluate_records(records, [{"layer": 0, "head": 1}], .1, .05, -100, 100)
