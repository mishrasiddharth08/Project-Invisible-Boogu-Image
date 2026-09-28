"""Strict streaming loaders for official tensors and Comfy-Org quantized tensors.

Packed Linear modules retain disk precision. Dequantization uses Comfy-Kitchen's
native layouts (including inverse ConvRot), one layer at a time. This is not an
SD/Flux model loader and does not replace global Torch/Forge operations.
"""
from __future__ import annotations
import json
import torch
from torch import nn
from safetensors import safe_open
from accelerate.utils import set_module_tensor_to_device

class PackedLinear(nn.Module):
    def __init__(self, source, data, conf, scale, scale2=None, bias=None):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.conf = conf
        # Parameters (never trainable), not buffers: Accelerate's official
        # sequential offloader must see and evict packed weights after forward.
        self.register_parameter("packed", nn.Parameter(data, requires_grad=False))
        self.register_parameter("scale", nn.Parameter(scale, requires_grad=False))
        self.register_parameter("scale2", nn.Parameter(scale2, requires_grad=False) if scale2 is not None else None)
        self.register_parameter("bias", nn.Parameter(bias, requires_grad=False) if bias is not None else None)

    def unpack(self, device=None, dtype=torch.bfloat16):
        device = device or self.packed.device
        data, scale = self.packed.to(device), self.scale.to(device)
        fmt = self.conf["format"]
        if fmt in ("float8_e4m3fn", "float8_e5m2"):
            fp8 = torch.float8_e4m3fn if fmt == "float8_e4m3fn" else torch.float8_e5m2
            value = data.view(fp8).float() * scale.float()
        else:
            from comfy_kitchen.tensor import TensorWiseINT8Layout, TensorCoreNVFP4Layout
            shape = (self.out_features, self.in_features)
            if fmt == "int8_tensorwise":
                nested = self.conf.get("params", {})
                params = TensorWiseINT8Layout.Params(scale=scale, orig_dtype=dtype, orig_shape=shape,
                    convrot=self.conf.get("convrot", nested.get("convrot", False)),
                    convrot_groupsize=self.conf.get("convrot_groupsize", nested.get("convrot_groupsize", 256)))
                value = TensorWiseINT8Layout.dequantize(data, params)
            elif fmt == "nvfp4":
                params = TensorCoreNVFP4Layout.Params(scale=self.scale2.to(device),
                    block_scale=scale.view(torch.float8_e4m3fn), orig_dtype=dtype, orig_shape=shape)
                value = TensorCoreNVFP4Layout.dequantize(data, params)
            else:
                raise ValueError("Unverified quantization format: " + str(fmt))
        return value[:self.out_features, :self.in_features].to(dtype)

    @property
    def weight(self):
        # Upstream modules occasionally inspect weight.dtype/device.
        return self.unpack()

    def forward(self, value):
        weight = self.unpack(value.device, value.dtype)
        bias = self.bias.to(value.device, value.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(value, weight, bias)

def te_key(key):
    # Comfy's Qwen3-VL export flattens language_model but retains the visual trunk.
    if key.startswith("model.language_model.") or key.startswith("model.visual."):
        return key
    if key.startswith("model."):
        return "model.language_model." + key[len("model."):]
    if key.startswith("visual."):
        return "model." + key
    return key

def load_single(model, path, mapper=lambda k: k, dtype=torch.bfloat16):
    expected = dict(model.named_parameters())
    expected.update(dict(model.named_buffers()))
    loaded, consumed = set(), set()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        layers = json.loads((handle.metadata() or {}).get("_quantization_metadata", "{}" )).get("layers", {})
        for original in sorted(keys):
            mapped = mapper(original)
            if mapped not in expected:
                continue
            tensor = handle.get_tensor(original)
            prefix = original.rsplit(".", 1)[0]
            conf = layers.get(prefix)
            qkey = prefix + ".comfy_quant"
            if qkey in keys:
                conf = json.loads(bytes(handle.get_tensor(qkey).tolist()))
                consumed.add(qkey)
            scale_key = prefix + ".weight_scale"
            if original.endswith(".weight") and scale_key in keys:
                if conf is None and tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    conf = {"format": "float8_e4m3fn" if tensor.dtype == torch.float8_e4m3fn else "float8_e5m2"}
                if conf is None:
                    raise ValueError("Quantized weight has no format metadata: " + original)
                module_name = mapped.rsplit(".", 1)[0]
                module = model.get_submodule(module_name)
                if not isinstance(module, nn.Linear):
                    raise ValueError("Quantized non-Linear layer is not supported: " + module_name)
                scale2_key = prefix + ".weight_scale_2"
                bias_key = prefix + ".bias"
                replacement = PackedLinear(module, tensor, conf, handle.get_tensor(scale_key),
                    handle.get_tensor(scale2_key) if scale2_key in keys else None,
                    handle.get_tensor(bias_key).to(dtype) if bias_key in keys else None)
                replacement.compute_dtype = dtype
                parent, _, leaf = module_name.rpartition(".")
                setattr(model.get_submodule(parent) if parent else model, leaf, replacement)
                consumed.update(k for k in (scale_key, scale2_key, bias_key) if k in keys)
                loaded.add(mapped)
                if bias_key in keys:
                    loaded.add(mapper(bias_key))
            else:
                if tuple(tensor.shape) != tuple(expected[mapped].shape):
                    raise ValueError(f"Tensor shape mismatch: {original}: {tuple(tensor.shape)} != {tuple(expected[mapped].shape)}")
                set_module_tensor_to_device(model, mapped, "cpu", value=tensor,
                dtype=dtype if tensor.is_floating_point() else tensor.dtype)
                loaded.add(mapped)
            consumed.add(original)
        # Rotary buffers are non-persistent and constructed on CPU outside meta init.
        missing = {k for k in expected if k not in loaded and k in dict(model.named_parameters())}
        extra = keys - consumed
        if missing or extra:
            raise ValueError(f"Weight contract mismatch: missing={sorted(missing)[:8]}, unexpected={sorted(extra)[:8]}")
    model.requires_grad_(False).eval()
    return model

def materialize_lora_targets(model, targets=None):
    """PEFT needs ordinary Linear parameters. Called only for explicit LoRAs."""
    for name, module in list(model.named_modules()):
        if isinstance(module, PackedLinear) and (targets is None or name in targets):
            dtype = getattr(module, "compute_dtype", torch.bfloat16)
            ordinary = nn.Linear(module.in_features, module.out_features, module.bias is not None,
                                 device="cpu", dtype=dtype)
            ordinary.weight = nn.Parameter(module.unpack("cpu", dtype), requires_grad=False)
            if module.bias is not None:
                ordinary.bias = nn.Parameter(module.bias.cpu(), requires_grad=False)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, ordinary)
