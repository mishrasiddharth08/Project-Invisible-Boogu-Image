"""Scoped, chain-preserving Forge Neo integration. No core files are modified."""
from __future__ import annotations
from functools import wraps
import inspect
from pathlib import Path
import secrets
import sys
from .lib import assets, pipeline, policy

TAG = "[Invisible-Boogu]"
_registered = {}
_undo = []
_installed = False
_checkpoint_component = None
_scripts = []

def selected(p=None, name=None):
    from modules import shared, sd_models
    override = getattr(p, "override_settings", {}) or {}
    name = name or override.get("sd_model_checkpoint") or shared.opts.sd_model_checkpoint
    ci = sd_models.checkpoint_aliases.get(name) or sd_models.checkpoints_list.get(name)
    if ci is None:
        ci = next((c for c in sd_models.checkpoints_list.values() if name in
                   (c.name, c.title, getattr(c, "short_title", None))), None)
    if ci is None:
        return None
    path = Path(ci.filename)
    cached = _registered.get(str(path.resolve()).casefold())
    if cached:
        return cached
    if assets.is_boogu(path):
        return assets.Asset(path, assets.trunk(path), assets.quant(path))
    return None

def register_checkpoints():
    pipeline.invalidate_components()
    from modules import sd_models
    discovered = assets.scan()
    _registered.clear()
    for asset in discovered:
        path = str(asset.path.resolve())
        _registered[path.casefold()] = asset
        ci = next((c for c in sd_models.checkpoints_list.values() if
                   str(Path(c.filename).resolve()).casefold() == path.casefold()), None)
        if ci is None:
            ci = sd_models.CheckpointInfo(path)
        # Remove the old display key but retain path/hash aliases.
        for key, value in list(sd_models.checkpoints_list.items()):
            if value is ci:
                sd_models.checkpoints_list.pop(key)
        label = asset.path.parent.name if asset.path.name == "model_index.json" else asset.path.stem
        priority = "0 hotfix — " if "hotfix" in label.lower() else "1 legacy — " if "turbo" in asset.trunk else ""
        ci.name = f"Boogu — {asset.trunk} — {priority}{label}"
        ci.title = ci.name + (f" [{ci.shorthash}]" if ci.shorthash else "")
        ci.short_title = ci.title
        ci.ids += [ci.name, ci.title, path]
        ci.register()

def _replace(owner, name, replacement):
    previous = getattr(owner, name)
    setattr(owner, name, replacement)
    _undo.append((owner, name, previous, replacement))

def install():
    global _installed
    if _installed:
        return True
    try:
        from modules import scripts, processing, sd_models, script_callbacks, shared
        # Probe before changing anything. Unknown APIs leave stock callables intact.
        inspect.signature(scripts.ScriptRunner.run).bind(object(), object(), 0)
        inspect.signature(processing.process_images).bind(object())
        inspect.signature(sd_models.list_models).bind()
        run = scripts.ScriptRunner.run
        process = processing.process_images
        listing = sd_models.list_models

        @wraps(run)
        def wrapped_run(runner, p, *args, **kwargs):
            if not _installed:
                return run(runner, p, *args, **kwargs)
            asset = selected(p)
            if asset is not None:
                return generate(p, asset, runner, args)
            pipeline.release()
            return run(runner, p, *args, **kwargs)

        @wraps(process)
        def wrapped_process(p, *args, **kwargs):
            if not _installed:
                return process(p, *args, **kwargs)
            asset = selected(p)
            if asset is not None:
                return generate(p, asset, getattr(p, "scripts", None), getattr(p, "script_args", ()))
            pipeline.release()
            return process(p, *args, **kwargs)

        @wraps(listing)
        def wrapped_list(*args, **kwargs):
            result = listing(*args, **kwargs)
            if _installed:
                register_checkpoints()
            return result

        _replace(scripts.ScriptRunner, "run", wrapped_run)
        _replace(processing, "process_images", wrapped_process)
        _replace(sd_models, "list_models", wrapped_list)
        for name in ("modules.api.api", "modules.txt2img", "modules.img2img"):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "process_images", None) is process:
                _replace(module, "process_images", wrapped_process)
        register_checkpoints()
        option = shared.opts.data_labels["sd_model_checkpoint"]
        previous_change = option.onchange

        def checkpoint_changed():
            pipeline.release_if_changed(selected())
            if previous_change is not None:
                return previous_change()
        _replace(option, "onchange", checkpoint_changed)
        script_callbacks.on_script_unloaded(uninstall)
        script_callbacks.on_before_ui(install_cards)
        _installed = True
        return True
    except Exception as error:
        uninstall()
        print(f"{TAG} Neo hook drift: integration disabled, stock hooks restored: {error}")
        return False

def uninstall():
    global _installed
    pipeline.release()
    for owner, name, previous, replacement in reversed(_undo):
        # Never tear out a wrapper another engine installed after us.
        if getattr(owner, name, None) is replacement:
            setattr(owner, name, previous)
    _undo.clear()
    _installed = False

def controls(p, runner, args, asset):
    result = {"ceiling": 1024, "distill": "turbo" in asset.trunk}
    for script in getattr(runner, "alwayson_scripts", []):
        if not getattr(script, "is_boogu_invisible", False):
            continue
        values = args[script.args_from:script.args_to]
        if len(values) >= 3:
            # The read-only trunk field protects against stale controls after a switch.
            if values[0] == asset.trunk:
                result["ceiling"] = int(values[1])
                if len(values) >= 4:
                    result["reduce_grid"] = bool(values[3])
    return result

def generate(p, asset, runner=None, args=()):
    from modules import processing, shared, images
    import torch
    from .lib.runtime import SkipImage, check_cancel
    from .lora import resolve
    output, seeds, prompts, negatives, infotexts = [], [], [], [], []
    info = ""
    with pipeline.LOCK:
        cleanup = False
        try:
            is_img2img = isinstance(p, processing.StableDiffusionProcessingImg2Img)
            refs = getattr(p, "init_images", None) or []
            policy.validate_task(asset.trunk, is_img2img, len({id(ref) for ref in refs}))
            if getattr(p, "enable_hr", False):
                raise ValueError("Turn off Hires fix for Boogu; choose the Boogu resolution ceiling instead.")
            cfg = assets.config()
            opts = controls(p, runner, args, asset)
            grid_filter = opts.pop("reduce_grid", False)
            from .lib.hardware import detect
            hardware = detect(torch)
            vram = hardware.total_bytes / 2**30
            resolved = policy.settings(asset.trunk, p.width, p.height, vram_gb=vram,
                 steps=getattr(p, "steps", cfg["default_steps"]), cfg=getattr(p, "cfg_scale", cfg["default_cfg"]), **opts)
            p.width, p.height = resolved["width"], resolved["height"]
            p.steps, p.cfg_scale = resolved["num_inference_steps"], resolved["text_guidance_scale"]
            p.sd_model_name = asset.path.parent.name if asset.path.name == "model_index.json" else asset.path.stem
            p.sampler_name = "Boogu official FlowMatchEuler + time-shift" if "turbo" not in asset.trunk else "Boogu official DMD"
            count = int(p.batch_size) * int(p.n_iter)
            if not 1 <= count <= cfg["max_outputs"]:
                raise ValueError(f"Choose between 1 and {cfg['max_outputs']} images per request")
            positive = p.prompt if isinstance(p.prompt, list) else [p.prompt] * count
            negative = p.negative_prompt if isinstance(p.negative_prompt, list) else [p.negative_prompt] * count
            if len(positive) < count or len(negative) < count or (isinstance(p.seed, list) and len(p.seed) < count):
                raise ValueError("Prompt lists must contain one entry per output image")
            base_seed = int(p.seed) if p.seed is not None and not isinstance(p.seed, list) else -1
            if base_seed < 0:
                base_seed = secrets.randbits(32)
            shared.state.job_count = count
            shared.state.sampling_step = 0
            shared.state.sampling_steps = p.steps
            shared.state.current_latent = None
            shared.state.textinfo = "Boogu: loading model"
            reference = refs[0].convert("RGB") if is_img2img else None

            def step(index, total):
                check_cancel(shared.state)
                shared.state.sampling_step, shared.state.sampling_steps = index + 1, total
                from .lib.runtime import progress_text
                shared.state.textinfo = progress_text(shared.state, "generating image")

            for index in range(count):
                if shared.state.interrupted:
                    break
                prompt = shared.prompt_styles.apply_styles_to_prompt(positive[index], p.styles)
                neg = shared.prompt_styles.apply_negative_styles_to_prompt(negative[index], p.styles)
                instruction, adapters = resolve(prompt, asset.trunk)
                neg, negative_adapters = resolve(neg, asset.trunk)
                if negative_adapters:
                    raise ValueError("Place Boogu LoRAs in the positive prompt")
                seed = int(p.seed[index]) if isinstance(p.seed, list) else (base_seed + index) % 2**32
                if seed < 0:
                    seed = secrets.randbits(32)
                shared.state.sampling_step = 0
                shared.state.sampling_steps = p.steps
                shared.state.textinfo = "Boogu: loading model"
                from .lib.runtime import begin_preview
                begin_preview(shared.state, p.width, p.height)
                kwargs = policy.arguments(asset.trunk, instruction, neg, resolved, reference)
                kwargs.update(generator=torch.Generator(hardware.device).manual_seed(seed), step_func=step)
                try:
                    result, offload = pipeline.invoke_with_recovery(asset, adapters,
                        p.width, p.height, seed, kwargs, shared.state, shared.opts)
                except SkipImage:
                    shared.state.skipped = False
                    shared.state.nextjob()
                    if not output:
                        info = "Boogu image skipped"
                    continue
                image = result.images[0]
                if grid_filter:
                    from .lib.degrid import reduce_grid
                    image = reduce_grid(image)
                shared.state.textinfo = "Boogu: saving image"
                mask = getattr(p, "image_mask", None)
                if reference is not None and mask is not None:
                    from PIL import Image, ImageOps, ImageFilter
                    mask = mask.convert("L").resize(image.size)
                    if getattr(p, "inpainting_mask_invert", 0):
                        mask = ImageOps.invert(mask)
                    blur = float(getattr(p, "mask_blur", 0) or 0)
                    if blur:
                        mask = mask.filter(ImageFilter.GaussianBlur(blur))
                    image = Image.composite(image, reference.resize(image.size), mask)
                shared.state.assign_current_image(image)
                info = (f"{prompt}\nNegative prompt: {neg}\nSteps: {p.steps}, Sampler: {p.sampler_name}, "
                        f"CFG scale: {p.cfg_scale}, Seed: {seed}, Size: {p.width}x{p.height}, "
                        f"Model: {asset.path.name}, Boogu trunk: {asset.trunk}, Offload: {offload}")
                info += f", Boogu grid filter: {'DeGrid auto' if grid_filter else 'off'}"
                p.extra_generation_params.update({"Boogu trunk": asset.trunk, "Boogu precision": asset.quant,
                    "Boogu offload": offload, "Boogu grid filter": "DeGrid auto" if grid_filter else "off"})
                if not getattr(p, "do_not_save_samples", False) and shared.opts.samples_save:
                    images.save_image(image, p.outpath_samples, "", seed, prompt,
                                      shared.opts.samples_format, info=info, p=p)
                output.append(image)
                seeds.append(seed)
                prompts.append(prompt)
                negatives.append(neg)
                infotexts.append(info)
                shared.state.nextjob()
        except InterruptedError as error:
            info = str(error)
            cleanup = True
        except torch.cuda.OutOfMemoryError:
            cleanup = True
            info = "Boogu ran out of VRAM even with offloading. Try 512 or 768 pixels, shorten the prompt, and close other GPU jobs. Packed checkpoints require compatible kernels; AMD/older-GPU compatibility mode uses unquantized weights. Some GPUs cannot fit the working tensors of this model."
        except Exception as error:
            cleanup = True
            info = f"Boogu: {error}"
            print(f"{TAG} {info}")
        if cleanup:
            pipeline.release()
        if not output:
            # Never fall through into Forge's SD/Flux denoiser for a Boogu checkpoint.
            p.comments[info] = 1
        return processing.Processed(p, output, seeds[0] if seeds else -1, info,
             all_prompts=prompts or None, all_negative_prompts=negatives or None,
             all_seeds=seeds or None, infotexts=infotexts or None)

def install_cards():
    try:
        from modules import ui_extra_networks
        cls = ui_extra_networks.ExtraNetworksPage
        original = cls.create_item_html
        if getattr(original, "_boogu_cards", False):
            return

        @wraps(original)
        def card(page, tabname, item, *args, **kwargs):
            is_lora = str(getattr(page, "name", "")).lower() == "lora"
            if not is_lora:
                return original(page, tabname, item, *args, **kwargs)
            from .lora import classification
            try:
                target, distill = classification(item["filename"])
            except (OSError, ValueError, KeyError):
                target, distill = None, False
            result = original(page, tabname, item, *args, **kwargs)
            marker = f" boogu-lora boogu-target-{target or 'foreign'}" + (" boogu-distill" if distill else "")
            if isinstance(result, str):
                return result.replace('class="card', 'class="card' + marker, 1)
            if isinstance(result, dict):
                result["search_only"] = result.get("search_only", "") + marker
            return result
        card._boogu_cards = True
        _replace(cls, "create_item_html", card)
    except (ImportError, AttributeError) as error:
        print(f"{TAG} Extra Networks card hook unavailable: {error}; generation-time trunk lock remains active")
