"""Scan first, then optionally download explicitly selected Boogu files."""
from __future__ import annotations
import argparse
import json
from ..lib import assets

REPO = "Comfy-Org/Boogu-Image"
FILES = {
    "base": {"bf16": "boogu_image_base_bf16.safetensors", "fp8_scaled": "boogu_image_base_fp8_scaled.safetensors", "nvfp4": "boogu_image_base_nvfp4.safetensors"},
    "turbo": {"bf16": "boogu_image_turbo_hotfix_bf16.safetensors", "fp8_scaled": "boogu_image_turbo_fp8_scaled.safetensors", "int8_convrot": "boogu_image_turbo_hotfix_int8_convrot.safetensors", "nvfp4": "boogu_image_turbo_hotfix_nvfp4.safetensors"},
    "edit": {"bf16": "boogu_image_edit_bf16.safetensors", "fp8_scaled": "boogu_image_edit_fp8_scaled.safetensors", "int8_convrot": "boogu_image_edit_int8_convrot.safetensors", "nvfp4": "boogu_image_edit_nvfp4.safetensors"},
    "edit-turbo": {"bf16": "boogu_image_edit_turbo_hotfix_1k_20260708_bf16.safetensors", "int8_convrot": "boogu_image_edit_turbo_hotfix_1k_20260708_int8_convrot.safetensors"},
}
OPTIONS = tuple((trunk, precision) for trunk, choices in FILES.items() for precision in choices)

def plan(trunk, precision):
    root = assets.forge_root() / "models/Boogu"
    jobs, reused = [], []
    available = [a for a in assets.scan() if a.trunk == trunk and a.quant == precision]
    try:
        name = FILES[trunk][precision]
    except KeyError as error:
        raise ValueError(f"No verified Comfy-Org {precision} file for {trunk}; choose from {list(FILES[trunk])}") from error
    preferred = [a for a in available if "hotfix" not in name or "hotfix" in str(a.path).lower()]
    if trunk == "edit-turbo":
        preferred = [a for a in preferred if "20260708" in str(a.path)]
    if preferred:
        reused.append(str(preferred[0].path))
    else:
        jobs.append((REPO, "diffusion_models/" + name, root))
    for kind, filename in (("te", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"), ("vae", "vae/flux1_vae_bf16.safetensors")):
        existing = assets.component(kind)
        if existing:
            reused.append(str(existing[0]))
        else:
            jobs.append((REPO, filename, root))
    manifest = json.loads((assets.EXT / "sources.json").read_text(encoding="utf-8"))
    from ..lib.pipeline import CONFIGS
    processor_files = manifest["processor_files"]
    if (CONFIGS / "processor/tokenizer.json").is_file() and (CONFIGS / "processor/preprocessor_config.json").is_file():
        reused.append(str(CONFIGS / "processor"))
        processor_files = []
    for filename in processor_files:
        target = root / "support" / filename
        if not target.is_file() or target.stat().st_size == 0:
            jobs.append(("Boogu/Boogu-Image-0.1-Base", filename, root / "support"))
        else:
            reused.append(str(target))
    return jobs, reused

def plan_many(selections):
    jobs, reused, seen_jobs, seen_reused = [], [], set(), set()
    for trunk, precision in selections:
        selected_jobs, selected_reused = plan(trunk, precision)
        for job in selected_jobs:
            key = (job[0], job[1], str(job[2]))
            if key not in seen_jobs:
                seen_jobs.add(key); jobs.append(job)
        for path in selected_reused:
            if path not in seen_reused:
                seen_reused.add(path); reused.append(path)
    return jobs, reused

def choose_models():
    print("\nChoose one or more verified Boogu models.")
    print("Enter numbers separated by commas, or press Enter to cancel.\n")
    for number, (trunk, precision) in enumerate(OPTIONS, 1):
        print(f" {number:2}. {trunk:10}  {precision}")
    raw = input("\nModel numbers: ").strip()
    if not raw:
        return []
    try:
        numbers = [int(value.strip()) for value in raw.split(",")]
    except ValueError:
        print("Invalid selection; no download started."); return []
    if any(number < 1 or number > len(OPTIONS) for number in numbers):
        print("Invalid selection; no download started."); return []
    return [OPTIONS[number - 1] for number in dict.fromkeys(numbers)]

def choose_method():
    print("\nBoogu model setup")
    print(" 1. Download manually (recommended)")
    print(" 2. Choose models for automatic download")
    raw = input("\nChoice (press Enter to cancel): ").strip()
    if raw == "1":
        print("\nNo network action was started.")
        print("Open this guide:", assets.EXT / "MANUAL-DOWNLOADS.md")
        return "manual"
    if raw == "2":
        return "automatic"
    print("Cancelled; no download started.")
    return "cancel"

def main(argv=None):
    parser = argparse.ArgumentParser(description="Boogu: scan first; explicit confirmation before any model download")
    parser.add_argument("--trunk", choices=FILES)
    parser.add_argument("--precision", choices=("bf16", "fp8_scaled", "int8_convrot", "nvfp4"))
    parser.add_argument("--plan", action="store_true", help="Print plan only; never download")
    args = parser.parse_args(argv)
    # Every command-line flag is noninteractive except the final download approval.
    # In particular, --plan alone prints the former default Turbo/BF16 plan.
    if args.trunk or args.precision or args.plan:
        selections = [(args.trunk or "turbo", args.precision or "bf16")]
    else:
        method = choose_method()
        if method != "automatic":
            return
        selections = choose_models()
    if not selections:
        print("Cancelled; no download started."); return
    try:
        jobs, reused = plan_many(selections)
    except ValueError as error:
        parser.error(str(error))
    print("\nSelected:", ", ".join(f"{trunk} / {precision}" for trunk, precision in selections))
    for path in reused: print("REUSE", path)
    for repo, filename, destination in jobs: print("DOWNLOAD", repo + "/" + filename, "->", destination / filename)
    if not jobs or args.plan:
        print("No downloads performed."); return
    print("\nModel weights are large. Downloads use Forge's existing Python and Hugging Face cache.")
    if input("Type DOWNLOAD to approve precisely the files above (anything else cancels): ").strip() != "DOWNLOAD":
        print("Cancelled; no download started."); return
    from huggingface_hub import hf_hub_download
    manifest = json.loads((assets.EXT / "sources.json").read_text(encoding="utf-8"))
    for repo, filename, destination in jobs:
        path = hf_hub_download(repo, filename, revision=manifest["models"][repo], local_dir=str(destination))
        if filename.endswith(".safetensors"): assets.header(path)
        print("READY", path)
    print("Setup finished. Restart Forge or refresh the checkpoint dropdown.")

if __name__ == "__main__": main()
