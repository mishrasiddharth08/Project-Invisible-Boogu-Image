"""Only the vendored, revision-pinned official Boogu pipeline generates images."""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import sys
import threading
from contextlib import contextmanager, nullcontext
from . import assets

LOCK = threading.RLock()
_pipe = None
_identity = None
_component_cache = {}
CONFIGS = Path(__file__).parent / "configs"

@contextmanager
def safe_console():
    # Upstream logs Unicode checkmarks; legacy Windows consoles can be CP1252.
    streams = []
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                streams.append((stream, stream.errors))
                stream.reconfigure(errors="backslashreplace")
            except (OSError, ValueError):
                pass
    try:
        yield
    finally:
        for stream, errors in streams:
            try:
                stream.reconfigure(errors=errors)
            except (OSError, ValueError):
                pass

def invoke(pipe, **kwargs):
    import torch
    generator = kwargs.get("generator")
    if generator is not None:
        kwargs.setdefault("device", str(generator.device))
    # The official edit path samples the reference VAE posterior without passing
    # its generator. Scope that RNG to Forge's seed and restore it afterward.
    seeded_reference = generator is not None and kwargs.get("input_images") is not None
    device = generator.device if seeded_reference else None
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device is not None and device.type == "cuda" else []
    rng_scope = torch.random.fork_rng(devices=cuda_devices) if seeded_reference else nullcontext()
    with safe_console(), torch.inference_mode(), rng_scope:
        if seeded_reference:
            torch.default_generator.manual_seed(generator.initial_seed())
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(generator.initial_seed())
        return pipe(**kwargs)


def invoke_with_recovery(asset, adapters, width, height, seed, kwargs, state, opts):
    import torch
    from .runtime import SkipImage, progress_hooks
    kwargs = dict(kwargs)
    pipe = get(asset, adapters, width, height)
    for attempt in range(2):
        try:
            with progress_hooks(pipe, state, opts):
                return invoke(pipe, **kwargs), pipe._boogu_offload
        except SkipImage:
            pipe.maybe_free_model_hooks()
            raise
        except torch.cuda.OutOfMemoryError:
            if attempt or getattr(pipe, "_boogu_offload", "sequential") != "model":
                raise
            pipe = None
        # Leave the exception scope before freeing memory: the traceback may
        # retain GPU tensors. Retry once with official sequential offload.
        release()
        state.textinfo = "Boogu: retrying with lower VRAM use"
        state.sampling_step = 0
        pipe = get(asset, adapters, width, height, offload="sequential")
        kwargs["generator"] = torch.Generator(kwargs["generator"].device).manual_seed(seed)

def release_if_changed(asset):
    with LOCK:
        if _pipe is None:
            return
        if asset is not None and _identity and _identity[0] == assets.fingerprint(asset.path):
            return
        release()


def release():
    global _pipe, _identity
    with LOCK:
        if _pipe is not None:
            try:
                cache = getattr(_pipe, "_boogu_prompt_cache", None)
                if cache is not None:
                    cache.clear()
                if getattr(_pipe, "_boogu_has_adapters", False):
                    try:
                        _pipe.unload_lora_weights()
                    except Exception as error:
                        print(f"[Invisible-Boogu] Adapter cleanup: {error}")
                try:
                    if hasattr(_pipe, "remove_all_hooks"):
                        _pipe.remove_all_hooks()
                except Exception as error:
                    print(f"[Invisible-Boogu] Offload cleanup: {error}")
            finally:
                _pipe = None
                _identity = None
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

def official_classes(portable=False):
    vendor = str(Path(__file__).parent / "vendor")
    # Never overwrite another installed Boogu package in sys.modules.
    existing = sys.modules.get("boogu")
    if existing is not None and not str(getattr(existing, "__file__", "")).startswith(vendor):
        raise RuntimeError("Another Boogu Python package is already loaded; restart with one Boogu engine enabled")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    if portable:
        # Disable optional fused backends only inside our private Boogu package.
        from boogu.utils import import_utils
        import_utils._flash_attn_available = False
        import_utils._triton_available = False
    from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
    from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
    if portable:
        import torch
        from boogu.models.transformers import block_lumina2, transformer_boogu
        block_lumina2.RMSNorm = torch.nn.RMSNorm
        block_lumina2.swiglu = block_lumina2.torch_swiglu
        transformer_boogu.RMSNorm = torch.nn.RMSNorm
    return BooguImagePipeline, BooguImageTurboPipeline

def _json(component):
    return json.loads((CONFIGS / component / ("scheduler_config.json" if component == "scheduler" else "config.json")).read_text())

def processor_path():
    bundled = CONFIGS / "processor"
    if assets.processor_complete(bundled):
        return bundled
    for root in assets.roots():
        for candidate in (root / "support/processor", root / "processor"):
            if assets.processor_complete(candidate):
                return candidate
    # An incomplete older bundle should not hide an already downloaded official processor.
    # This uses the pinned local cache only; setup is still explicit when the cache is missing.
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        snapshot_download = None
    if snapshot_download is not None:
        repo = "Boogu/Boogu-Image-0.1-Base"
        revision = json.loads((assets.EXT / "sources.json").read_text(encoding="utf-8"))["models"][repo]
        # Forge redirects Hugging Face into its own models folder; the original
        # local download may still be in Hugging Face's standard user cache.
        for cache_dir in (None, Path.home() / ".cache" / "huggingface" / "hub"):
            try:
                extra = {} if cache_dir is None else {"cache_dir": cache_dir}
                candidate = Path(snapshot_download(repo_id=repo, revision=revision, local_files_only=True,
                                                   allow_patterns=["processor/*"], **extra)) / "processor"
            except LocalEntryNotFoundError:
                continue
            if assets.processor_complete(candidate):
                return candidate
    raise RuntimeError("Boogu tokenizer/processor is missing. Run setup-models.cmd, review its download plan, then type DOWNLOAD.")


def invalidate_components():
    """Rescan support weights after Forge's native checkpoint refresh."""
    with LOCK:
        _component_cache.clear()


def _components(portable):
    with LOCK:
        cached = _component_cache.get(portable)
        if cached is not None and cached[0].is_file() and cached[1].is_file() and (cached[2] / "tokenizer.json").is_file():
            return cached
        te = assets.component("te", unquantized_only=portable)
        vae = assets.component("vae")
        if not te or not vae:
            raise RuntimeError("Missing compatible Qwen3-VL-8B text encoder or Flux1 VAE. Compatibility mode requires unquantized weights. Run setup-models.cmd; downloads require confirmation.")
        result = (te[0], vae[0], processor_path())
        # processor_path returns a directory; validate its required file below.
        _component_cache[portable] = result
        return result


def _can_reuse(pipe, width, height, requested, device):
    from .runtime import choose_offload
    if requested not in ("auto", "model", "sequential"):
        raise ValueError("offload must be auto, model, or sequential")
    current = pipe._boogu_offload
    if current == "sequential" and requested != "model":
        return True
    if current == "model" and requested != "sequential" and width * height <= pipe._boogu_max_pixels:
        return True
    import torch
    free_bytes, _ = torch.cuda.mem_get_info(device)
    mode = choose_offload(free_bytes, pipe._boogu_component_bytes, width, height, requested)
    if mode != current:
        return False
    pipe._boogu_max_pixels = max(pipe._boogu_max_pixels, width * height)
    return True


def get(asset, adapters, width=1024, height=1024, offload=None):
    global _pipe, _identity
    import torch
    from .hardware import detect, validate_weights
    hardware = detect(torch)
    dtype = hardware.dtype
    full = asset.path.name == "model_index.json"
    te, vae, processor = (None, None, None) if full else _components(hardware.portable)
    requested = offload or assets.config().get("offload", "auto")
    identity = (assets.fingerprint(asset.path), asset.trunk,
                tuple((assets.fingerprint(p), w) for p, w in adapters),
                assets.fingerprint(te) if te else None,
                assets.fingerprint(vae) if vae else None,
        assets.fingerprint(processor / "tokenizer.json") if processor else None,
        hardware.device, str(dtype))
    with LOCK:
        if _pipe is not None and identity == _identity and _can_reuse(_pipe, width, height, requested, hardware.device):
            return _pipe
        release()
        validate_weights(asset.path, hardware)
        if not full:
            validate_weights(te, hardware)
            validate_weights(vae, hardware)
        # Free Forge's loaded models through its public memory lifecycle.
        from backend import memory_management
        memory_management.unload_all_models()
        memory_management.soft_empty_cache()
        previous = os.environ.get("device")
        # Upstream uses this only while importing optional NVIDIA fused ops.
        # invoke passes the actual execution device explicitly on every call.
        os.environ["device"] = "cpu" if hardware.portable else hardware.device
        try:
            Base, Turbo = official_classes(portable=hardware.portable)
            cls = Turbo if "turbo" in asset.trunk else Base
            if full:
                # Explicit class is essential: Edit-Turbo model_index incorrectly says Base.
                from boogu.models.transformers import BooguImageTransformer2DModel
                from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
                transformer = BooguImageTransformer2DModel.from_pretrained(str(asset.path.parent / "transformer"),
                                              local_files_only=True, torch_dtype=dtype)
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(asset.path.parent / "scheduler"),
                                                                           local_files_only=True)
                pipe = cls.from_pretrained(str(asset.path.parent), torch_dtype=dtype,
                                           local_files_only=True, transformer=transformer, scheduler=scheduler)
            else:
                from accelerate import init_empty_weights
                from diffusers import AutoencoderKL
                from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
                from boogu.models.transformers import BooguImageTransformer2DModel
                from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
                from .weights import load_single, te_key
                with init_empty_weights(include_buffers=False):
                    transformer = BooguImageTransformer2DModel.from_config(_json("transformer"))
                    mllm = Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(_json("mllm")))
                transformer = load_single(transformer, asset.path, dtype=dtype)
                mllm = load_single(mllm, te, te_key, dtype=dtype)
                # Diffusers has a native Flux AE conversion; local config forbids Hub fallback.
                autoencoder = AutoencoderKL.from_single_file(str(vae), config=str(CONFIGS / "vae"),
                                   local_files_only=True, torch_dtype=dtype)
                proc = Qwen3VLProcessor.from_pretrained(str(processor), local_files_only=True)
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(_json("scheduler"))
                pipe = cls(transformer=transformer, vae=autoencoder, scheduler=scheduler, mllm=mllm, processor=proc)
            from ..lora import apply
            apply(pipe, adapters)
            pipe._boogu_has_adapters = bool(adapters)
            from .runtime import choose_offload, module_bytes
            free_bytes, _ = torch.cuda.mem_get_info(hardware.device)
            pipe._boogu_component_bytes = [module_bytes(pipe.mllm), module_bytes(pipe.transformer), module_bytes(pipe.vae)]
            pipe._boogu_max_pixels = width * height
            mode = choose_offload(free_bytes, pipe._boogu_component_bytes, width, height, requested)
            pipe.enable_model_cpu_offload_flag = mode == "model"
            pipe.enable_sequential_cpu_offload_flag = mode == "sequential"
            if mode == "model":
                pipe.enable_model_cpu_offload(device=hardware.device)
            else:
                pipe.enable_sequential_cpu_offload(device=hardware.device)
            pipe._boogu_offload = mode
            pipe._boogu_hardware = hardware
            print(f"[Invisible-Boogu] Backend: {hardware.backend}; device: {hardware.device}; compute: {dtype}; portable kernels: {hardware.portable}")
            print(f"[Invisible-Boogu] Memory: {mode} CPU offload; {free_bytes / 2**30:.1f} GiB free")
            from .prompt_cache import install as install_prompt_cache
            install_prompt_cache(pipe, megabytes=assets.config().get("prompt_cache_mb", 64))
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            _pipe, _identity = pipe, identity
            return pipe
        finally:
            if previous is None:
                os.environ.pop("device", None)
            else:
                os.environ["device"] = previous
