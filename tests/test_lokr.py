from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from safetensors.torch import save_file

from pi_boogu_image.lora.lokr import LokrLinear, attach
from pi_boogu_image.lib.weights import PackedLinear


class Pipe:
    def __init__(self):
        self.prompt_embedding = None
        self.transformer = nn.ModuleDict({"proj": nn.Linear(6, 4, bias=False)})


class LokrTests(unittest.TestCase):
    def _attach(self, tensors, strength=1.0):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "adapter.safetensors"
        save_file({"transformer.proj." + key: value for key, value in tensors.items()}, str(path))
        pipe = Pipe()
        pipe.transformer["proj"].weight.data.zero_()
        attach(pipe, path, strength)
        self.addCleanup(directory.cleanup)
        return pipe

    def test_factored_forward_matches_kron_without_materializing(self):
        from unittest.mock import patch
        for shape in [(6,), (3, 6), (2, 3, 6)]:
            for dtype in (torch.float32, torch.float64):
                base = nn.Linear(6, 8, bias=False).to(dtype)
                w1, w2 = torch.randn(2, 2, dtype=dtype), torch.randn(4, 3, dtype=dtype)
                layer = LokrLinear(base, {'lokr_w1': w1, 'lokr_w2': w2}, 0.7)
                x = torch.randn(*shape, dtype=dtype)
                expected = base(x) + nn.functional.linear(x, torch.kron(w1, w2)) * 0.7
                with patch('torch.kron', side_effect=AssertionError('full delta allocated')):
                    torch.testing.assert_close(layer(x), expected)

    def test_full_lokr_numeric(self):
        w1 = torch.tensor([[1., 2.], [3., 4.]])
        w2 = torch.tensor([[1., 0., -1.], [2., 1., 0.]])
        pipe = self._attach({"lokr_w1": w1, "lokr_w2": w2})
        value = torch.randn(3, 6)
        torch.testing.assert_close(pipe.transformer["proj"](value), nn.functional.linear(value, torch.kron(w1, w2)))

    def test_factorized_lokr_and_alpha(self):
        w1a, w1b = torch.randn(2, 1, dtype=torch.bfloat16), torch.randn(1, 2)
        w2a, w2b = torch.randn(2, 2), torch.randn(2, 3)
        pipe = self._attach({"lokr_w1_a": w1a, "lokr_w1_b": w1b,
                             "lokr_w2_a": w2a, "lokr_w2_b": w2b,
                             "alpha": torch.tensor(1.)}, strength=0.5)
        value = torch.randn(2, 6)
        delta = torch.kron(w1a.float() @ w1b, w2a @ w2b)
        torch.testing.assert_close(pipe.transformer["proj"](value), nn.functional.linear(value, delta) * 0.25)

    def test_zero_strength_is_exact_base(self):
        pipe = self._attach({"lokr_w1": torch.randn(2, 2), "lokr_w2": torch.randn(2, 3)}, 0)
        value = torch.randn(2, 6)
        self.assertIsInstance(pipe.transformer["proj"], LokrLinear)
        self.assertTrue(torch.equal(pipe.transformer["proj"](value), torch.zeros(2, 4)))

    def test_multiple_adapters_stack(self):
        pipe = self._attach({"lokr_w1": torch.eye(2), "lokr_w2": torch.ones(2, 3)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "second.safetensors"
            save_file({"lora_unet_proj.lokr_w1": torch.eye(2),
                       "lora_unet_proj.lokr_w2": torch.ones(2, 3)}, str(path))
            attach(pipe, path, 1)
        value = torch.randn(1, 6)
        expected = nn.functional.linear(value, torch.kron(torch.eye(2), torch.ones(2, 3))) * 2
        torch.testing.assert_close(pipe.transformer["proj"](value), expected)

    def test_accelerate_offload_sees_direct_factor_parameters(self):
        from accelerate.hooks import attach_align_device_hook
        pipe = self._attach({"lokr_w1": torch.eye(2), "lokr_w2": torch.ones(2, 3)})
        wrapper = pipe.transformer["proj"]
        attach_align_device_hook(wrapper, execution_device=torch.device("cpu"), offload=True)
        self.assertTrue(all(parameter.device.type == "meta" for parameter in wrapper.parameters()))
        wrapper(torch.randn(1, 6))
        self.assertTrue(all(parameter.device.type == "meta" for parameter in wrapper.parameters()))

    def test_packed_base_is_not_materialized(self):
        pipe = Pipe()
        source = pipe.transformer["proj"]
        packed = PackedLinear(source, torch.ones(4, 6).to(torch.float8_e4m3fn).view(torch.uint8),
                              {"format": "float8_e4m3fn"}, torch.tensor(1.0))
        pipe.transformer["proj"] = packed
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packed.safetensors"
            save_file({"transformer.proj.lokr_w1": torch.eye(2),
                       "transformer.proj.lokr_w2": torch.zeros(2, 3)}, str(path))
            attach(pipe, path)
        self.assertIs(pipe.transformer["proj"].base, packed)
        torch.testing.assert_close(pipe.transformer["proj"](torch.ones(1, 6)), torch.full((1, 4), 6.0))

    def test_invalid_keys_and_shapes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.safetensors"
            save_file({"transformer.proj.lokr_unknown": torch.ones(1)}, str(bad))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                attach(Pipe(), bad)
            save_file({"transformer.proj.lokr_w1": torch.ones(2, 2),
                       "transformer.proj.lokr_w2": torch.ones(3, 3)}, str(bad))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                attach(Pipe(), bad)


if __name__ == "__main__":
    unittest.main()
