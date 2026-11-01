from models.qwen3_wrapper import Qwen3VLWrapper


class Qwen3ThinkingWrapper(Qwen3VLWrapper):
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Thinking", device="cuda", **kwargs):
        super().__init__(model_id, device, **kwargs)
