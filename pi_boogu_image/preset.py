"""Add 'boogu' to Neo's existing UI Preset choices without changing its enum."""
from copy import copy, deepcopy
from functools import wraps
from pathlib import Path
from .lib import assets

NAME = "boogu"

def install():
    try:
        from modules import shared
        from modules_forge import presets
        arch = presets.PresetArch
        original = arch.choices
        if not getattr(original, "_boogu_invisible", False):
            @wraps(original)
            def choices(*args, **kwargs):
                result = list(original(*args, **kwargs))
                if NAME not in result:
                    result.append(NAME)
                return result
            choices._boogu_invisible = True
            arch.choices = staticmethod(choices)
        template = {}
        presets.register(template)
        for key, info in template.items():
            if key.startswith("sd_"):
                target = NAME + key[2:]
            elif key in ("forge_checkpoint_sd", "forge_additional_modules_sd", "forge_unet_storage_dtype_sd"):
                target = key[:-2] + NAME
            else:
                continue
            if target in shared.opts.data_labels:
                continue
            cloned = copy(info)
            cloned.default = deepcopy(info.default)
            if getattr(cloned, "section", (None,))[0] == "ui_sd":
                cloned.section = ("ui_boogu", "BOOGU")
            if target.endswith("_sampler"):
                cloned.default = "Euler"
            elif target.endswith("_scheduler"):
                cloned.default = "Simple"
            elif target.endswith("_step"):
                cloned.default = 4
            elif target.endswith("_cfg"):
                cloned.default = 1.0
            elif target.endswith(("_width", "_height")):
                cloned.default = 1024
            shared.opts.add_option(target, cloned)
        available = assets.scan()
        preferred = sorted(available, key=lambda a: (a.trunk != "turbo", "hotfix" not in str(a.path),
                                                     a.quant != "int8_convrot", a.quant != "fp8_scaled", str(a.path)))
        from .forge import selected
        current = selected()
        stored = selected(name=shared.opts.data.get("forge_checkpoint_boogu", "")) if shared.opts.data.get("forge_checkpoint_boogu") else None
        choice = stored or current or (preferred[0] if preferred else None)
        if choice is not None:
            from modules import sd_models
            ci = next((c for c in sd_models.checkpoints_list.values()
                       if Path(c.filename).resolve() == choice.path.resolve()), None)
            if ci is not None:
                shared.opts.data["forge_checkpoint_boogu"] = ci.name
            turbo = "turbo" in choice.trunk
            for mode in ("t2i", "i2i"):
                shared.opts.data.setdefault(f"boogu_{mode}_step", 4 if turbo else 28)
                shared.opts.data.setdefault(f"boogu_{mode}_cfg", 1.0 if turbo else 4.0)
        # The official loader resolves validated shared components itself.
        # Never carry an unrelated preset's Flux2 VAE or text encoder into Boogu.
        shared.opts.data.setdefault("forge_additional_modules_boogu", [])
        print("[Invisible-Boogu] UI Preset 'boogu' registered")
        return True
    except Exception as error:
        print(f"[Invisible-Boogu] UI Preset unavailable: {error}")
        return False
