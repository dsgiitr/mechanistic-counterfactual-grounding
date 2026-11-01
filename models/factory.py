"""Select a native model wrapper without substituting architectures."""


def load_wrapper(model_id, device="cuda", family="auto"):
    if family == "auto":
        name = model_id.lower()
        if "qwen3" in name:
            family = "qwen3_vl"
        elif "llava" in name:
            family = "llava"
        else:
            raise ValueError("Specify --family llava or qwen3_vl for a custom checkpoint path")
    if family == "llava":
        from models.llava_wrapper import LlavaWrapper
        return LlavaWrapper(model_id, device)
    if family == "qwen3_vl":
        from models.qwen3_wrapper import Qwen3VLWrapper
        return Qwen3VLWrapper(model_id, device)
    raise ValueError(f"Unsupported model family: {family}")
