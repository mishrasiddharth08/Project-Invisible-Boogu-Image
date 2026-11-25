import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pi_boogu_image.download import setup
from pi_boogu_image.lib.assets import Asset


def fake_pipeline(folder):
    module = types.ModuleType("pi_boogu_image.lib.pipeline")
    module.CONFIGS = Path(folder)
    processor = module.CONFIGS / "processor"
    processor.mkdir()
    pins = json.loads((setup.assets.EXT / "sources.json").read_text(encoding="utf-8"))
    for name in pins["processor_files"]:
        (processor / Path(name).name).write_text("{}")
    return module


class DownloadTests(unittest.TestCase):
    def test_manual_route_never_builds_download_plan(self):
        with patch("builtins.input", return_value="1"), patch("builtins.print"), patch.object(setup, "plan_many") as planner:
            setup.main([])
        planner.assert_not_called()

    def test_menu_accepts_multiple_verified_models_and_removes_duplicates(self):
        with patch("builtins.input", return_value="1,4,1"), patch("builtins.print"):
            self.assertEqual(setup.choose_models(), [setup.OPTIONS[0], setup.OPTIONS[3]])

    def test_invalid_menu_selection_cancels(self):
        with patch("builtins.input", return_value="999"), patch("builtins.print"):
            self.assertEqual(setup.choose_models(), [])

    def test_existing_assets_need_no_download(self):
        checkpoint = Asset(Path("/existing/boogu_image_turbo_hotfix_bf16.safetensors"), "turbo", "bf16")
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(sys.modules, {"pi_boogu_image.lib.pipeline": fake_pipeline(folder)}), \
             patch.object(setup.assets, "forge_root", return_value=Path("/forge")), \
             patch.object(setup.assets, "scan", return_value=[checkpoint]), \
             patch.object(setup.assets, "component", side_effect=[[Path("/existing/encoder")], [Path("/existing/vae")]]):
            jobs, reused = setup.plan("turbo", "bf16")
        self.assertEqual(jobs, [])
        self.assertIn(str(checkpoint.path), reused)

    def test_edit_turbo_current_hotfix_is_preferred_over_legacy(self):
        legacy = Asset(Path("/existing/boogu_image_edit_turbo_hotfix_int8_convrot.safetensors"), "edit-turbo", "int8_convrot")
        current = Asset(Path("/existing/boogu_image_edit_turbo_hotfix_1k_20260708_int8_convrot.safetensors"), "edit-turbo", "int8_convrot")
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(sys.modules, {"pi_boogu_image.lib.pipeline": fake_pipeline(folder)}), \
             patch.object(setup.assets, "forge_root", return_value=Path("/forge")), \
             patch.object(setup.assets, "scan", return_value=[legacy, current]), \
             patch.object(setup.assets, "component", side_effect=[[Path("/existing/encoder")], [Path("/existing/vae")]]):
            jobs, reused = setup.plan("edit-turbo", "int8_convrot")
        self.assertEqual(jobs, [])
        self.assertIn(str(current.path), reused)
        self.assertNotIn(str(legacy.path), reused)

    def test_multi_plan_deduplicates_shared_files(self):
        shared = ("repo", "shared", Path("/target"))
        with patch.object(setup, "plan", side_effect=[([shared, ("repo", "one", Path("/target"))], ["cached"]), ([shared], ["cached"]) ]):
            jobs, reused = setup.plan_many([("base", "bf16"), ("turbo", "bf16")])
        self.assertEqual(jobs, [shared, ("repo", "one", Path("/target"))])
        self.assertEqual(reused, ["cached"])

    def test_plan_flag_alone_never_prompts(self):
        with patch.object(setup, "plan_many", return_value=([("repo", "file", Path("/target"))], [])), patch("builtins.input") as user_input, patch("builtins.print"):
            setup.main(["--plan"])
        user_input.assert_not_called()

    def test_cancel_does_not_download(self):
        sentinel = types.ModuleType("huggingface_hub")
        with patch.dict(sys.modules, {"huggingface_hub": sentinel}), \
             patch.object(setup, "plan_many", return_value=([("repo", "file", Path("/target"))], [])), \
             patch("builtins.input", return_value="no"), patch("builtins.print"):
            setup.main(["--trunk", "turbo", "--precision", "bf16"])
            self.assertIs(sys.modules["huggingface_hub"], sentinel)


if __name__ == "__main__":
    unittest.main()
