import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from pi_boogu_image import forge
from pi_boogu_image.lib.assets import Asset
from pi_boogu_image.lib.runtime import check_cancel


class BatchIntegration(unittest.TestCase):
    def test_skip_first_edit_keeps_second_and_adapter_metadata(self):
        class Edit(SimpleNamespace):
            pass

        image = Image.new("RGB", (256, 256))
        p = Edit(init_images=[image, image], width=256, height=256, steps=4,
                 cfg_scale=1, batch_size=2, n_iter=1, seed=7, styles=[],
                 prompt="flower <lora:example:0.5>", negative_prompt="",
                 comments={}, extra_generation_params={}, outpath_samples=".")
        state = SimpleNamespace(interrupted=False, skipped=False, nextjob=Mock(),
                                assign_current_image=Mock())
        styles = SimpleNamespace(apply_styles_to_prompt=lambda value, _: value,
                                 apply_negative_styles_to_prompt=lambda value, _: value)
        shared = SimpleNamespace(state=state, prompt_styles=styles,
                                 opts=SimpleNamespace(samples_save=True, samples_format="png"))
        save = Mock()
        modules = SimpleNamespace(shared=shared, images=SimpleNamespace(save_image=save),
            processing=SimpleNamespace(StableDiffusionProcessingImg2Img=Edit, Processed=Mock()))
        calls = []

        def invoke(asset, adapters, width, height, seed, kwargs, *rest):
            calls.append(kwargs["instruction"])
            self.assertEqual(adapters, [("example", 0.5)])
            if seed == 7:
                state.skipped = True
            check_cancel(state)
            return SimpleNamespace(images=[image]), "model"

        def resolve(text, trunk):
            return ("flower", [("example", 0.5)]) if text else ("", [])

        hardware = SimpleNamespace(device="cpu", total_bytes=16 * 2**30)
        with patch.dict(sys.modules, {"modules": modules}), \
             patch("pi_boogu_image.lib.hardware.detect", return_value=hardware), \
             patch("pi_boogu_image.lora.resolve", side_effect=resolve), \
             patch.object(forge.pipeline, "invoke_with_recovery", side_effect=invoke), \
             patch.object(forge.pipeline, "release") as release:
            forge.generate(p, Asset(Path("edit.safetensors"), "edit", "bf16"))
        release.assert_not_called()
        self.assertEqual(calls, [["flower"], ["flower"]])
        self.assertEqual(state.nextjob.call_count, 2)
        save.assert_called_once()
        self.assertIn(p.prompt, save.call_args.kwargs["info"])
        self.assertEqual(modules.processing.Processed.call_args.kwargs["all_seeds"], [8])
