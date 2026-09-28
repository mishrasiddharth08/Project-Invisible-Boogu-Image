import sys
import types
import unittest
from unittest.mock import Mock, patch
import torch
from pi_boogu_image import lora

def hdr(*keys, trunk="base"):
    out = {key: {"shape": [4, 4]} for key in keys}
    out["__metadata__"] = {"boogu_trunk": trunk, "ss_base_model": "Boogu"}
    return out

class LoraTests(unittest.TestCase):
    def test_ai_toolkit_boogu_metadata_and_explicit_trunk(self):
        data = {'__metadata__': {'ss_base_model_version': 'boogu_image.0.1'},
                'diffusion_model.proj.lokr_w1': {'shape': [2, 2]},
                'diffusion_model.proj.lokr_w2': {'shape': [2, 2]}}
        with patch.object(lora, 'header', return_value=data), patch.object(lora, 'config', return_value={}):
            self.assertEqual(lora.classification('eva-BOOGU.safetensors'), ('unknown', False))
        with patch.object(lora, 'header', return_value=data), patch.object(lora, 'config', return_value={
                'adapter_trunks': {'eva-BOOGU.safetensors': 'turbo'}}):
            self.assertEqual(lora.classification('eva-BOOGU.safetensors'), ('turbo', False))

    def test_classifies_lora_and_lokr(self):
        with patch.object(lora, "header", return_value=hdr("transformer.x.lora_A.weight", "transformer.x.lora_B.weight")):
            self.assertEqual(lora.classification("style.safetensors"), ("base", False))
        with patch.object(lora, "header", return_value=hdr("x.lokr_w1", "x.lokr_w2")):
            self.assertEqual(lora.classification("style.safetensors"), ("base", False))

    def test_rejects_hybrid_and_unsupported(self):
        with patch.object(lora, "header", return_value=hdr("x.lora_A.weight", "x.lokr_w1")):
            with self.assertRaisesRegex(ValueError, "Hybrid"):
                lora.classification("style.safetensors")
        with patch.object(lora, "header", return_value=hdr("x.hada_w1_a")):
            with self.assertRaisesRegex(ValueError, "Linear LoRA and LoKr"):
                lora.classification("style.safetensors")

    def test_exact_official_targets_and_pairs(self):
        state = {"transformer.blocks.0.to_q.lora_A.weight": object(),
                 "transformer.blocks.0.to_q.lora_B.weight": object()}
        self.assertEqual(lora._lora_targets(state), {"blocks.0.to_q"})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            lora._lora_targets({"transformer.x.lora_A.weight": object()})

    def test_native_extra_network_resolution(self):
        params = types.SimpleNamespace(positional=["style", "0.7"], named={})
        item = types.SimpleNamespace(filename="C:/models/style.safetensors")
        modules = types.ModuleType("modules")
        modules.extra_networks = types.SimpleNamespace(parse_prompt=lambda prompt: ("clean", {"lora": [params]}))
        networks = types.SimpleNamespace(available_networks={"style": item}, available_network_aliases={})
        with patch.dict(sys.modules, {"modules": modules, "networks": networks}), patch.object(lora, "classification", return_value=("base", False)):
            clean, found = lora.resolve("x <lora:style:0.7>", "base")
        self.assertEqual((clean, found[0][1]), ("clean", 0.7))

    def test_apply_materializes_only_official_targets(self):
        pipe = Mock()
        target = Mock(in_features=4, out_features=4)
        pipe.transformer.named_modules.return_value = [("block.to_q", target)]
        state = {"transformer.block.to_q.lora_A.weight": torch.zeros(2, 4),
                 "transformer.block.to_q.lora_B.weight": torch.zeros(4, 2)}
        pipe.lora_state_dict.return_value = state
        with patch.object(lora, "header", return_value=hdr(*state)), \
             patch("pi_boogu_image.lib.weights.materialize_lora_targets") as materialize:
            lora.apply(pipe, [("style.safetensors", 0.5)])
        materialize.assert_called_once_with(pipe.transformer, {"block.to_q"})
        pipe.load_lora_weights.assert_called_once_with(state, adapter_name="boogu_0", local_files_only=True)
        pipe.set_adapters.assert_called_once_with(["boogu_0"], adapter_weights=[0.5])

    def test_official_loader_with_real_peft(self):
        try:
            import peft  # noqa: F401
            from diffusers import DiffusionPipeline, ModelMixin
            from diffusers.loaders import PeftAdapterMixin
            from pi_boogu_image.lib.vendor.boogu.pipelines.lora_pipeline import BooguImageLoraLoaderMixin
        except ImportError as error:
            self.skipTest(str(error))

        class TinyTransformer(ModelMixin, PeftAdapterMixin):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 3, bias=False)
            def forward(self, value):
                return self.proj(value)

        class TinyPipe(BooguImageLoraLoaderMixin, DiffusionPipeline):
            def __init__(self, transformer=None):
                super().__init__()
                transformer = transformer or TinyTransformer()
                transformer.proj.weight.data.zero_()
                self.register_modules(transformer=transformer)

        pipe = TinyPipe()
        state = {"transformer.proj.lora_A.weight": torch.ones(2, 4),
                 "transformer.proj.lora_B.weight": torch.ones(3, 2)}
        from peft.tuners.lora import model as peft_lora_model
        original_dispatch = peft_lora_model.dispatch_torchao
        lora._load_official(pipe, state, "real_peft", {pipe.transformer.proj})
        self.assertIs(peft_lora_model.dispatch_torchao, original_dispatch)
        self.assertIn("real_peft", pipe.transformer.peft_config)
        pipe.transformer.set_adapter("real_peft")
        self.assertTrue(torch.all(pipe.transformer(torch.ones(1, 4)) > 0))

if __name__ == "__main__": unittest.main()
