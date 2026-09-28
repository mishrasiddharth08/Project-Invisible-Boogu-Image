"""Read-only discovery. No downloads and no Torch imports."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import struct

TRUNKS = ("base", "turbo", "edit", "edit-turbo")
EXT = Path(__file__).resolve().parents[2]

def config():
    return json.loads((EXT / "config.json").read_text(encoding="utf-8"))

def forge_root():
    # An archive copy must never infer its parent as the running Forge.
    try:
        from modules.paths_internal import script_path
        return Path(script_path)
    except ImportError:
        configured = config().get("live_forge")
        if configured:
            return Path(configured)
        candidate = EXT.parent.parent
        if EXT.parent.name == "extensions" and (candidate / "modules").is_dir():
            return candidate
        raise RuntimeError("Install this folder inside Forge/extensions or set live_forge in config.json.")

def roots():
    models = forge_root() / "models"
    names = ("Boogu", "Stable-diffusion", "diffusion_models", "text_encoder",
             "text_encoders", "VAE", "Lora")
    return [models / n for n in names] + [Path(p) for p in config().get("extra_search_paths", [])]

def trunk(path):
    text = re.sub(r"[^a-z0-9]+", "_", str(path).lower())
    edit = "edit" in text
    turbo = "turbo" in text or "hotfix" in text
    return "edit-turbo" if edit and turbo else "edit" if edit else "turbo" if turbo else "base"

def quant(path):
    text = str(path).lower()
    return next((v for v in ("nvfp4", "int8_convrot", "fp8_scaled", "bf16") if v in text), "bf16")

@lru_cache(maxsize=512)
def _header(filename, size, modified):
    with open(filename, "rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Incomplete safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if not 2 <= n <= 32 * 1024 * 1024 or n + 8 > size:
            raise ValueError("Invalid safetensors header")
        data = json.loads(stream.read(n))
    offsets = []
    for key, value in data.items():
        if key == "__metadata__":
            continue
        start, end = value["data_offsets"]
        if not 0 <= start <= end <= size - n - 8:
            raise ValueError("Incomplete tensor payload: " + key)
        offsets.append((start, end))
    offsets.sort()
    if any(b[0] < a[1] for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Overlapping tensor payloads")
    return data

def header(path):
    path = Path(path)
    stat = path.stat()
    return _header(str(path.resolve()), stat.st_size, stat.st_mtime_ns)

def is_boogu(path):
    path = Path(path)
    if path.name == "model_index.json":
        try:
            return "BooguImage" in json.loads(path.read_text())["_class_name"]
        except (OSError, ValueError, KeyError):
            return False
    if path.suffix.lower() != ".safetensors" or "boogu" not in str(path).lower() or "lora" in path.name.lower():
        return False
    try:
        data = header(path)
        # Distinguish a real Boogu transformer from TE/LoRA/VAE named Boogu.
        return ("context_refiner.0.attn.to_q.weight" in data and
                "double_stream_layers.0.img_self_attn.to_q.weight" in data)
    except (OSError, ValueError, KeyError, TypeError):
        return False

@dataclass(frozen=True)
class Asset:
    path: Path
    trunk: str
    quant: str

def scan():
    found, seen = [], set()
    for root in roots():
        if not root.is_dir():
            continue
        for pattern in ("*.safetensors", "model_index.json"):
            for path in root.rglob(pattern):
                key = str(path.resolve()).casefold()
                if key in seen:
                    continue
                seen.add(key)
                if is_boogu(path):
                    found.append(Asset(path, trunk(path), quant(path)))
    # Hotfixes always precede older alternatives; preserve each precision choice.
    return sorted(found, key=lambda a: (a.trunk, "hotfix" not in a.path.name.lower(),
                                        "20260708" not in str(a.path), str(a.path)))

def component(kind, unquantized_only=False):
    candidates = []
    for root in roots():
        if not root.exists():
            continue
        for path in root.rglob("*.safetensors"):
            name = path.name.lower().replace("-", "_").replace(".", "_")
            if kind == "te" and not ("qwen3" in name and "vl" in name and "8b" in name):
                continue
            if kind == "vae" and not (name == "ae_safetensors" or "flux1" in name or "flux_1" in name):
                continue
            try:
                data = header(path)
                if unquantized_only and (bool(data.get("__metadata__", {}).get("_quantization_metadata")) or any(
                    key.endswith((".weight_scale", ".comfy_quant", ".weight_scale_2")) or
                    (isinstance(value, dict) and value.get("dtype", "").startswith("F8"))
                    for key, value in data.items())):
                    continue
                if kind == "te":
                    embedding = data.get("model.embed_tokens.weight", data.get("model.language_model.embed_tokens.weight", {}))
                    if embedding.get("shape", [0, 0])[-1] != 4096 or not any("visual." in k for k in data):
                        continue
                else:
                    if data.get("decoder.conv_in.weight", {}).get("shape", [0, 0])[1] != 16:
                        continue
                candidates.append(path)
            except (OSError, ValueError, KeyError, IndexError):
                continue
    return sorted(candidates, key=lambda p: ("uncensored" in p.name.lower(),
                    "fp8" not in p.name.lower(), "bf16" not in p.name.lower(), str(p)))

def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns
