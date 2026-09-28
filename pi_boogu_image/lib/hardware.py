"""Capability-based device selection. No drivers or packages are installed here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    device: str
    backend: str
    dtype: object
    total_bytes: int
    portable: bool


def detect(torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("Boogu needs a GPU exposed by this Forge PyTorch installation through CUDA or HIP/ROCm. "
            "CPU-only and DirectML builds are not supported. No driver or package will be installed automatically.")
    index = torch.cuda.current_device()
    backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    try:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        bf16 = torch.cuda.is_bf16_supported()
    # FP16 can overflow BF16-trained Qwen/Boogu weights. FP32 is slower and
    # consumes more RAM, but does not silently lower the numerical range.
    dtype = torch.bfloat16 if bf16 else torch.float32
    return Hardware(f"cuda:{index}", backend, dtype,
        torch.cuda.get_device_properties(index).total_memory, backend == "rocm" or not bf16)


def validate_weights(path, hardware):
    if not hardware.portable:
        return
    if path.name == "model_index.json":
        import json
        for name in ("transformer", "mllm", "vae"):
            component = path.parent / name
            config = component / "config.json"
            if config.is_file() and json.loads(config.read_text()).get("quantization_config"):
                raise RuntimeError(f"Compatibility mode requires unquantized {name} weights.")
            for shard in component.glob("*.safetensors"):
                validate_weights(shard, hardware)
        return
    from .assets import header
    data = header(path)
    metadata = data.get("__metadata__", {})
    packed = bool(metadata.get("_quantization_metadata")) or any(
        name.endswith((".weight_scale", ".comfy_quant", ".weight_scale_2")) or
        (isinstance(value, dict) and value.get("dtype", "").startswith("F8"))
        for name, value in data.items())
    if packed:
        raise RuntimeError(f"{hardware.backend.upper()} compatibility mode requires unquantized BF16/FP32 weights; "
            f"{path.name} uses a packed format whose kernels are not verified on this device. "
            "Select existing BF16 Boogu and Qwen3-VL weights. Downloads still require your confirmation.")
