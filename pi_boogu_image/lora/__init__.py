"""Native Extra Networks selection and local Boogu adapter application."""
from pathlib import Path
import math
import re
from ..lib.assets import header, trunk, config

_LORA = re.compile(r"^(.*)\.lora_(A|B)\.weight$", re.I)
_LOKR = re.compile(r"^(.*)\.lokr_(w1|w2|w1_a|w1_b|w2_a|w2_b)(?:\.weight)?$", re.I)
_BAD = ("dora_scale", "hada_", "loha_", "ia3_", "oft_", "boft_", "lokr_t2")

def _kind(data):
    keys = [k for k in data if k != "__metadata__"]
    low = [k.lower() for k in keys]
    if any(marker in key for key in low for marker in _BAD):
        raise ValueError("Unsupported adapter type; Boogu supports Linear LoRA and LoKr only")
    has_lora = any(_LORA.match(k) for k in keys)
    has_lokr = any(_LOKR.match(k) for k in keys)
    if has_lora and has_lokr:
        raise ValueError("Hybrid LoRA/LoKr files are unsupported; use separate files")
    if not has_lora and not has_lokr:
        raise ValueError("No supported Linear LoRA or LoKr tensors found")
    return "lokr" if has_lokr else "lora"

def _lora_targets(state_dict):
    pairs = {}
    for key in state_dict:
        match = _LORA.match(key)
        if match:
            target, side = match.groups()
            if target.startswith("transformer."):
                target = target[len("transformer."):]
            elif target.startswith("prompt_embedding."):
                raise ValueError("Prompt-embedding LoRA is not supported")
            pairs.setdefault(target, set()).add(side.upper())
        else:
            raise ValueError("Unsupported tensor in LoRA checkpoint: " + key)
    incomplete = sorted(k for k, sides in pairs.items() if sides != {"A", "B"})
    if not pairs or incomplete:
        raise ValueError("Incomplete LoRA tensor pairs: " + ", ".join(incomplete[:4]))
    return set(pairs)

def _normalize_lora_state(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        match = _LORA.match(key)
        if not match:
            raise ValueError("Unsupported tensor in LoRA checkpoint: " + key)
        target, side = match.groups()
        if target.startswith("prompt_embedding."):
            raise ValueError("Prompt-embedding LoRA is not supported")
        if target.startswith("transformer."):
            target = target[len("transformer."):]
        canonical = "transformer." + target + ".lora_" + side.upper() + ".weight"
        if canonical in normalized:
            raise ValueError("Duplicate LoRA tensor: " + canonical)
        normalized[canonical] = value
    return normalized


def _validate_lora_targets(model, state_dict):
    targets = _lora_targets(state_dict)
    modules = dict(model.named_modules())
    for target in targets:
        module = modules.get(target)
        if module is None or not all(hasattr(module, x) for x in ("in_features", "out_features")):
            raise ValueError("LoRA target is not a model Linear: " + target)
        a = state_dict.get("transformer." + target + ".lora_A.weight",
                           state_dict.get(target + ".lora_A.weight"))
        b = state_dict.get("transformer." + target + ".lora_B.weight",
                           state_dict.get(target + ".lora_B.weight"))
        if getattr(a, "ndim", None) != 2 or getattr(b, "ndim", None) != 2:
            raise ValueError("LoRA tensors must be matrices: " + target)
        if a.shape[1] != module.in_features or b.shape[0] != module.out_features or a.shape[0] != b.shape[1]:
            raise ValueError("LoRA shape mismatch for target: " + target)
    return targets

def _load_official(pipe, state, name, materialized):
    """PEFT 0.18 probes broken optional TorchAO even for ordinary Linear layers."""
    import peft
    if peft.__version__ != "0.18.0":
        return pipe.load_lora_weights(state, adapter_name=name, local_files_only=True)
    import torch
    from peft.tuners.lora import model as peft_lora_model
    original = peft_lora_model.dispatch_torchao

    def scoped(target, *args, **kwargs):
        base = target.get_base_layer() if hasattr(target, "get_base_layer") else target
        weight = getattr(base, "weight", None)
        if base in materialized and type(weight) in (torch.Tensor, torch.nn.Parameter):
            return None
        return original(target, *args, **kwargs)

    peft_lora_model.dispatch_torchao = scoped
    try:
        return pipe.load_lora_weights(state, adapter_name=name, local_files_only=True)
    finally:
        if peft_lora_model.dispatch_torchao is scoped:
            peft_lora_model.dispatch_torchao = original

def classification(path):
    path = Path(path)
    data = header(path)
    meta, name = data.get("__metadata__", {}), path.stem.lower()
    if meta.get("ss_base_model") and "boogu" not in str(meta["ss_base_model"]).lower():
        return None, False
    family = str(meta.get("ss_base_model", "")) + " " + str(meta.get("ss_base_model_version", ""))
    if "boogu" not in family.lower() and not (
            "boogu" in name and ("lora" in name or "lokr" in name)):
        return None, False
    declared = meta.get("boogu_trunk", meta.get("ss_boogu_trunk"))
    if not declared:
        declared = config().get("adapter_trunks", {}).get(path.name)
    target = str(declared).lower().replace("_", "-") if declared else trunk(name)
    if not declared and not re.search(r"(?:^|[^a-z0-9])(?:base|turbo|edit|hotfix)(?:$|[^a-z0-9])", name):
        _kind(data)
        return "unknown", False
    if target not in ("base", "turbo", "edit", "edit-turbo", "all"):
        return None, False
    kind = _kind(data)
    distill = kind == "lora" and name.startswith("boogu_image_turbo") and "rank_128" in name
    if distill:
        ranks = [v["shape"][0] for k, v in data.items() if k != "__metadata__"
                 and _LORA.match(k) and k.lower().endswith("lora_a.weight")
                 and len(v.get("shape", [])) == 2]
        if not ranks or any(rank != 128 for rank in ranks):
            raise ValueError("Turbo distillation adapter must be rank 128")
    return target, distill

def resolve(prompt, selected_trunk):
    from modules import extra_networks
    clean, parsed = extra_networks.parse_prompt(prompt)
    if set(parsed) - {"lora"}:
        raise ValueError("Only native LoRA Extra Networks are supported by Boogu")
    if not parsed.get("lora"):
        return clean, []
    import networks
    result = []
    for params in parsed.get("lora", []):
        args, name = params.positional, params.positional[0]
        item = networks.available_networks.get(name) or networks.available_network_aliases.get(name)
        if item is None:
            raise ValueError("LoRA not found: " + name)
        target, distill = classification(item.filename)
        if target == "unknown":
            raise ValueError(f"{name}: Boogu adapter recognised, but its Base/Turbo/Edit model type is missing. Confirm its trained model before use.")
        if target not in (selected_trunk, "all"):
            raise ValueError(f"{name}: not a Boogu LoRA for the {selected_trunk} trunk")
        if distill:
            raise ValueError("Turbo checkpoint already contains distillation; remove the Turbo distillation LoRA")
        if len(args) > 2 or getattr(params, "named", {}):
            raise ValueError("Boogu supports one native LoRA strength, without block modifiers")
        strength = float(args[1]) if len(args) > 1 else 1.0
        if not math.isfinite(strength) or not -4 <= strength <= 4:
            raise ValueError("LoRA strength must be finite and between -4 and 4")
        result.append((str(Path(item.filename).resolve()), strength))
    return clean, result

def apply(pipe, adapters):
    if not adapters:
        return
    described = [(p, s, _kind(header(p))) for p, s in adapters]
    loras = [(p, s) for p, s, kind in described if kind == "lora"]
    lokrs = [(p, s) for p, s, kind in described if kind == "lokr"]
    if loras:
        try:
            import peft  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                'Boogu LoRA needs PEFT in Forge Python. Install "peft>=0.17,<0.19", then restart Forge.'
            ) from error
        from ..lib.weights import materialize_lora_targets
        names, strengths = [], []
        for index, (path, strength) in enumerate(loras):
            name = f"boogu_{index}"
            try:
                state = _normalize_lora_state(pipe.lora_state_dict(path, local_files_only=True))
                targets = _validate_lora_targets(pipe.transformer, state)
                materialize_lora_targets(pipe.transformer, targets)
                modules = dict(pipe.transformer.named_modules())
                _load_official(pipe, state, name, {modules[target] for target in targets})
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    'Boogu LoRA dependencies are incomplete. Install "peft>=0.17,<0.19" in Forge Python, then restart.'
                ) from error
            names.append(name); strengths.append(strength)
        pipe.set_adapters(names, adapter_weights=strengths)
    if lokrs:
        from .lokr import attach
        for path, strength in lokrs:
            attach(pipe, path, strength)
