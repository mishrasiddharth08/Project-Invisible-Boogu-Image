"""Memory planning and inexpensive previews; official sampling stays unchanged."""
from contextlib import contextmanager
from time import monotonic

def begin_preview(state, width, height):
    from PIL import Image
    scale = min(1.0, 512 / max(width, height))
    state.current_latent = None
    state.assign_current_image(Image.new('RGB', (max(1, round(width * scale)),
                                               max(1, round(height * scale))), (128, 128, 128)))

def progress_text(state, stage):
    step = max(0, int(getattr(state, "sampling_step", 0)))
    total = max(1, int(getattr(state, "sampling_steps", 1)))
    return f"Boogu: {stage} [Boogu steps {min(step, total)}/{total}]"


def choose_offload(free_bytes, component_bytes, width=1024, height=1024, requested="auto"):
    if requested not in ("auto", "model", "sequential"):
        raise ValueError("offload must be auto, model, or sequential")
    # Keep room for attention, dequantization scratch, VAE and other GPU users.
    reserve = (8 + 4 * max(0, width * height / 1024**2 - 1)) * 2**30
    fits = max(component_bytes, default=0) + reserve <= free_bytes
    return "model" if requested != "sequential" and fits else "sequential"


def module_bytes(module):
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))


def latent_preview(latents):
    import torch
    from PIL import Image
    from huggingface_guess.latent import Flux
    # Boogu shares Flux's 16-channel AE. Only its RGB preview projection is used.
    fmt = Flux()
    sample = latents[0].detach().float().cpu().movedim(0, -1)
    factors = torch.tensor(fmt.latent_rgb_factors).T
    bias = torch.tensor(fmt.latent_rgb_factors_bias)
    rgb = torch.nn.functional.linear(sample, factors, bias)
    pixels = ((rgb + 1) * 127.5).clamp(0, 255).byte().numpy()
    return Image.fromarray(pixels).resize((512, max(1, round(512 * pixels.shape[0] / pixels.shape[1]))))


@contextmanager
def progress_hooks(pipe, state, opts):
    """Observe official outputs without retaining GPU tensors or decoding the VAE."""
    handles, restores = [], []
    preview_failed = False
    last_preview = float("-inf")

    def check(stage):
        if state.interrupted or state.skipped:
            raise InterruptedError("Boogu generation interrupted")
        state.textinfo = progress_text(state, stage.removeprefix("Boogu: "))

    def preview(latents):
        nonlocal preview_failed, last_preview
        if preview_failed or not getattr(opts, "live_previews_enable", True):
            return
        now = monotonic()
        if now - last_preview < 1.0:
            return
        try:
            state.assign_current_image(latent_preview(latents))
            last_preview = now
        except Exception as error:
            preview_failed = True
            print(f"[Invisible-Boogu] Live preview unavailable: {error}")

    def wrap(owner, name, extract=None, stage="generating image"):
        original = getattr(owner, name)
        had_local = name in owner.__dict__
        local = owner.__dict__.get(name)
        def observed(*args, **kwargs):
            check("Boogu: " + stage)
            result = original(*args, **kwargs)
            if extract is not None:
                preview(extract(result))
            return result
        setattr(owner, name, observed)
        restores.append((owner, name, had_local, local))

    try:
        for module, label in ((pipe.mllm, "encoding prompt"), (pipe.transformer, "generating image")):
            handles.append(module.register_forward_pre_hook(lambda m, a, label=label: check("Boogu: " + label)))
        for name, stage in (("encode", "encoding reference"), ("decode", "decoding image")):
            if hasattr(pipe.vae, name):
                wrap(pipe.vae, name, stage=stage)
        if hasattr(pipe, "_predict_dmd_student_step"):
            wrap(pipe, "_predict_dmd_student_step", lambda result: result)
        else:
            wrap(pipe.scheduler, "step", lambda result: result[0])
        yield
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, had_local, local in reversed(restores):
            if had_local:
                setattr(owner, name, local)
            else:
                delattr(owner, name)
