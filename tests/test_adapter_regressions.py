from pathlib import Path
import unittest
import tempfile
from unittest.mock import Mock, patch
import torch
from pi_boogu_image import lora
from pi_boogu_image.lib import assets
from pi_boogu_image.lora.lokr import _split_key, attach
from safetensors.torch import save_file


class AdapterRegressions(unittest.TestCase):
    def test_trunk_ignores_parent_and_partial_words(self):
        self.assertEqual(assets.trunk(Path("edit-turbo") / "boogu_image_base_bf16.safetensors"), "base")
        self.assertEqual(assets.trunk("boogu_editor_database.safetensors"), "base")
        self.assertEqual(assets.trunk("boogu_image_edit_turbo_hotfix_bf16.safetensors"), "edit-turbo")
        self.assertEqual(assets.trunk(Path("boogu_edit") / "model_index.json"), "edit")

    def test_adapter_partial_word_does_not_declare_trunk(self):
        data = {"__metadata__": {"ss_base_model": "Boogu"},
                "proj.lora_A.weight": {}, "proj.lora_B.weight": {}}
        with patch.object(lora, "header", return_value=data), patch.object(lora, "config", return_value={}):
            self.assertEqual(lora.classification("boogu_lora_editor.safetensors"), ("unknown", False))

    def test_lokr_optional_weight_suffix(self):
        for suffix in ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "alpha", "scale"):
            self.assertEqual(_split_key("transformer.proj." + suffix + ".weight"), ("transformer.proj", suffix))
            self.assertEqual(_split_key("transformer.proj." + suffix), ("transformer.proj", suffix))

    def test_unprefixed_lora_reaches_official_loader_with_prefix(self):
        pipe = Mock()
        target = torch.nn.Linear(4, 3, bias=False)
        pipe.transformer.named_modules.return_value = [("proj", target)]
        state = {"proj.lora_a.weight": torch.ones(2, 4), "proj.lora_b.weight": torch.ones(3, 2)}
        pipe.lora_state_dict.return_value = state
        with patch.object(lora, "header", return_value=state), patch.object(lora, "_load_official") as loader, patch("pi_boogu_image.lib.weights.materialize_lora_targets"):
            lora.apply(pipe, [("adapter.safetensors", 0.5)])
        normalized = loader.call_args.args[1]
        self.assertEqual(set(normalized), {"transformer.proj.lora_A.weight", "transformer.proj.lora_B.weight"})
        self.assertIs(normalized["transformer.proj.lora_A.weight"], state["proj.lora_a.weight"])

    def test_duplicate_lora_aliases_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            lora._normalize_lora_state({"proj.lora_A.weight": None, "transformer.proj.lora_A.weight": None})

    def test_weight_suffixed_lokr_attaches_and_changes_output(self):
        pipe = Mock()
        pipe.prompt_embedding = None
        pipe.transformer = torch.nn.ModuleDict({"proj": torch.nn.Linear(6, 4, bias=False)})
        pipe.transformer["proj"].weight.data.zero_()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file({"transformer.proj.lokr_w1.weight": torch.eye(2),
                       "transformer.proj.lokr_w2.weight": torch.ones(2, 3)}, str(path))
            attach(pipe, path)
        torch.testing.assert_close(pipe.transformer["proj"](torch.ones(1, 6)), torch.full((1, 4), 3.0))


if __name__ == "__main__":
    unittest.main()
