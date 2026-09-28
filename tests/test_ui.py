"""Construct both real Gradio panels without starting Forge or loading weights."""
from pathlib import Path
import runpy
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from pi_boogu_image import forge, preset
from pi_boogu_image.lib.assets import Asset


class UiTests(unittest.TestCase):
    def test_both_tabs_build_and_transient_components_do_not_bind(self):
        callbacks = []
        modules = ModuleType('modules')
        modules.scripts = SimpleNamespace(Script=object, AlwaysVisible=True)
        modules.script_callbacks = SimpleNamespace(on_after_component=callbacks.append)
        asset = Asset(Path('boogu_edit_turbo.safetensors'), 'edit-turbo', 'bf16')
        path = Path(__file__).resolve().parents[1] / 'scripts/engine.py'
        with patch.dict('sys.modules', {'modules': modules}), \
             patch.object(forge, 'install'), patch.object(preset, 'install'), \
             patch.object(forge, '_checkpoint_component', None), \
             patch.object(forge, 'selected', return_value=asset):
            engine = runpy.run_path(str(path))
            with gr.Blocks() as blocks:
                for editing in (False, True):
                    controls = engine['Script']().ui(editing)
                    self.assertEqual(len(controls), 4)
                    self.assertFalse(controls[3].value)
                    self.assertFalse(controls[2].interactive)
                    self.assertFalse(controls[2].visible)
                checkpoint = gr.Dropdown(choices=['Boogu'], value='Boogu')
                callbacks[0](checkpoint, elem_id='setting_sd_model_checkpoint')
                count = len(blocks.fns)
                callbacks[0](checkpoint, elem_id='setting_sd_model_checkpoint')
                self.assertEqual(len(blocks.fns), count)
            callbacks[0](gr.Dropdown(), elem_id='setting_sd_model_checkpoint')
            self.assertIs(forge._checkpoint_component, checkpoint)


if __name__ == '__main__':
    unittest.main()
