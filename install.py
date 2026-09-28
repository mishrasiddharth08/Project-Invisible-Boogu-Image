"""Read-only dependency check. Never upgrade Forge or download models at startup."""
from importlib import metadata

def check():
    missing = []
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "einops", "huggingface-hub", "peft"):
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        print("[Invisible-Boogu] Missing packages: " + ", ".join(missing) + ". See README; no packages or models were downloaded.")
    else:
        print("[Invisible-Boogu] Dependencies found; models load only on Generate.")
    return missing

if __name__ == "__main__":
    check()
