from models.qwen3_wrapper import Qwen3VLWrapper


class Qwen3InstructWrapper(Qwen3VLWrapper):
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda", **kwargs):
        super().__init__(model_id, device, **kwargs)
