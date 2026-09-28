"""Normal checkpoint dropdown + Generate. No selectable script or additional tab."""
from pathlib import Path
import sys

EXT = Path(__file__).resolve().parent.parent
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))

from pi_boogu_image import forge
from pi_boogu_image.lib.policy import task_hint
_stock_controls = {}
_bindings = []
try:
    import gradio as gr
    from modules import scripts, script_callbacks

    def checkpoint_component(component, **kwargs):
        from gradio.context import Context
        if Context.root_block is None:
            return  # Gradio also constructs transient update objects at runtime.
        if kwargs.get("elem_id") == "setting_sd_model_checkpoint":
            forge._checkpoint_component = component
            for bind in _bindings:
                bind(component)
        if kwargs.get("elem_id") in {f"{mode}_{field}" for mode in ("txt2img", "img2img")
                                      for field in ("steps", "cfg_scale", "width", "height")}:
            _stock_controls[kwargs["elem_id"]] = component
    script_callbacks.on_after_component(checkpoint_component)

    class Script(scripts.Script):
        is_boogu_invisible = True

        def title(self):
            return "PROJECT INVISIBLE — Boogu-Image-0.1"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            asset = forge.selected()
            trunk = asset.trunk if asset else "base"
            prefix = "boogu_img2img" if is_img2img else "boogu_txt2img"
            mode = "img2img" if is_img2img else "txt2img"
            fields = [field for field in ("steps", "cfg_scale", "width", "height")
                      if f"{mode}_{field}" in _stock_controls]
            with gr.Accordion("Boogu", open=True, visible=asset is not None, elem_id=prefix) as accordion:
                hint = gr.Markdown(task_hint(trunk, is_img2img))
                detected = gr.Textbox(value=trunk, visible=False, interactive=False)
                with gr.Accordion("Optional settings", open=False):
                    ceiling = gr.Dropdown(label="Maximum image size", choices=[512, 768, 1024, 1536, 2048], value=1024,
                                          info="1024 is the usual starting point. Smaller sizes use less memory.")
                    grid_filter = gr.Checkbox(label="Reduce fine grid patterns (DeGrid)", value=False,
                        info="Automatic strength; skips detected clean images. Optional: fine details can still change.")
                # Preserve positional script/API arguments; model mode is automatic.
                distill = gr.Checkbox(value="turbo" in trunk, visible=False, interactive=False)

            def update(name):
                # Neo can replace the option onchange handler after extensions load.
                # The dropdown event remains an independent release path.
                selected = forge.selected(name=name)
                forge.pipeline.release_if_changed(selected)
                current = selected.trunk if selected else "base"
                resolution = 1024
                if selected:
                    from pi_boogu_image.lib.hardware import detect
                    from pi_boogu_image.lib.policy import profile
                    try:
                        resolution = profile(detect().total_bytes / 2**30, current)["default"]
                    except RuntimeError:
                        pass  # Generate explains unsupported runtimes without breaking UI creation.
                defaults = {"steps": 4 if "turbo" in current else 28,
                            "cfg_scale": 1.0 if "turbo" in current else 4.0, "width": resolution, "height": resolution}
                stock = [gr.update(value=defaults[field]) if selected else gr.update() for field in fields]
                return [gr.update(visible=selected is not None), current,
                        gr.update(value="turbo" in current), task_hint(current, is_img2img),
                        gr.update(value=resolution) if selected else gr.update(), *stock]

            # Neo creates quicksettings AFTER script panels. Bind when either
            # side arrives, matching the reference extension's lifecycle.
            bound = set()

            def bind(component):
                if component._id in bound:
                    return
                component.change(fn=update, inputs=[component], show_progress=False, queue=False,
                    outputs=[accordion, detected, distill, hint, ceiling,
                             *[_stock_controls[f"{mode}_{field}"] for field in fields]])
                bound.add(component._id)

            _bindings.append(bind)
            if forge._checkpoint_component is not None:
                bind(forge._checkpoint_component)
            return [detected, ceiling, distill, grid_filter]

    forge.install()
    from pi_boogu_image import preset
    preset.install()
except (ImportError, AttributeError) as error:
    print(f"[Invisible-Boogu] Forge UI unavailable: {error}")
