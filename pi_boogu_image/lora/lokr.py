"""Streaming-safe inference support for LyCORIS LoKr linear adapters."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from safetensors import safe_open


_SUFFIXES = ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "alpha", "scale")


class LokrLinear(nn.Module):
    """Add a LoKr branch while leaving the original (possibly packed) layer intact."""

    def __init__(self, base: nn.Module, tensors: Dict[str, torch.Tensor], strength: float):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.strength = float(strength)
        for name, value in tensors.items():
            self.register_parameter(name, nn.Parameter(value, requires_grad=False))

        w1, rank1 = self._matrix("lokr_w1")
        w2, rank2 = self._matrix("lokr_w2")
        expected = (base.out_features, base.in_features)
        actual = (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1])
        if actual != expected:
            raise ValueError(f"LoKr shape mismatch: reconstructed {actual}, target {expected}")
        rank = rank2 or rank1
        alpha = getattr(self, "alpha", None)
        scale = getattr(self, "scale", None)
        if alpha is not None and alpha.numel() != 1:
            raise ValueError("LoKr alpha must be a scalar")
        if scale is not None and scale.numel() != 1:
            raise ValueError("LoKr scale must be a scalar")
        alpha_scale = float(alpha.item()) / rank if alpha is not None and rank else 1.0
        self.network_scale = alpha_scale * (float(scale.item()) if scale is not None else 1.0)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return getattr(self.base, "bias", None)

    def _matrix(self, stem, device=None, dtype=None):
        full = getattr(self, stem, None)
        left = getattr(self, stem + "_a", None)
        right = getattr(self, stem + "_b", None)
        if full is not None:
            if left is not None or right is not None:
                raise ValueError(f"LoKr {stem} cannot be both full and factorized")
            if full.ndim != 2:
                raise ValueError(f"LoKr {stem} must be a matrix")
            return full.to(device=device, dtype=dtype) if device is not None else full, None
        if left is None or right is None:
            raise ValueError(f"LoKr {stem} requires either a full matrix or both A/B factors")
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError(f"LoKr {stem} factors have incompatible shapes")
        if device is not None:
            left, right = left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)
        elif left.dtype != right.dtype:
            common = torch.promote_types(left.dtype, right.dtype)
            left, right = left.to(common), right.to(common)
        return left @ right, right.shape[0]

    def forward(self, value):
        result = self.base(value)
        if self.strength == 0:
            return result
        w1, _ = self._matrix("lokr_w1", value.device, value.dtype)
        w2, _ = self._matrix("lokr_w2", value.device, value.dtype)
        # Apply the Kronecker factors without allocating the full weight matrix.
        branch = value.reshape(*value.shape[:-1], w1.shape[1], w2.shape[1])
        branch = torch.nn.functional.linear(branch, w2).transpose(-1, -2)
        branch = torch.nn.functional.linear(branch, w1).transpose(-1, -2)
        return result + branch.reshape(*value.shape[:-1], self.out_features) * (self.strength * self.network_scale)


def _split_key(key):
    for suffix in _SUFFIXES:
        marker = "." + suffix
        if key.endswith(marker):
            return key[:-len(marker)], suffix
    raise ValueError("Unsupported LoKr tensor key: " + key)


def _targets(pipe):
    aliases = {}
    roots = [(name, getattr(pipe, name)) for name in ("transformer", "prompt_embedding")
             if getattr(pipe, name, None) is not None]
    for root_name, root in roots:
        for local, module in root.named_modules():
            if not local or not all(hasattr(module, field) for field in ("in_features", "out_features")):
                continue
            dotted = root_name + "." + local
            normalized = local.replace(".", "_")
            names = {
                dotted, local, dotted.replace(".", "_"),
                "base_model.model." + dotted,
                "lora_" + root_name + "_" + normalized,
            }
            if root_name == "transformer":
                names.add("lora_unet_" + normalized)
                names.add("diffusion_model." + local)
            else:
                names.add("lora_te_" + normalized)
            for name in names:
                aliases.setdefault(name, []).append((root, local, module))
    return aliases


def attach(pipe, path, strength=1.0):
    """Attach every LoKr group in a local safetensors file to ``pipe``.

    Call this before Accelerate installs CPU offload hooks. The returned wrappers
    are also owned by the pipeline module tree, so releasing the pipeline releases
    the adapter parameters.
    """
    if not math.isfinite(float(strength)):
        raise ValueError("LoKr strength must be finite")
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("LoKr requires a local .safetensors file")
    groups = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            group, suffix = _split_key(key)
            if suffix in groups.setdefault(group, {}):
                raise ValueError("Duplicate LoKr tensor: " + key)
            tensor = handle.get_tensor(key)
            if not tensor.is_floating_point():
                raise ValueError("LoKr tensors must be floating point: " + key)
            groups[group][suffix] = tensor
    if not groups:
        raise ValueError("LoKr file contains no tensors")

    aliases = _targets(pipe)
    planned = []
    for group, tensors in groups.items():
        matches = aliases.get(group, [])
        if len(matches) != 1:
            reason = "not found" if not matches else "ambiguous"
            raise ValueError(f"LoKr target {reason}: {group}")
        parent_root, local, module = matches[0]
        wrapper = LokrLinear(module, tensors, strength)
        parent_name, _, leaf = local.rpartition(".")
        parent = parent_root.get_submodule(parent_name) if parent_name else parent_root
        planned.append((parent, leaf, wrapper))
    attached = []
    for parent, leaf, wrapper in planned:
        setattr(parent, leaf, wrapper)
        attached.append(wrapper)
    return attached
