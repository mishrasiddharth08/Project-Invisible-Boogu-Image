from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from pi_boogu_image.lib.hardware import detect, validate_weights
from pi_boogu_image.lib import policy


def fake_torch(hip=None, bf16=True, available=True, index=1):
    return SimpleNamespace(bfloat16='bf16',float32='fp32',version=SimpleNamespace(hip=hip),cuda=SimpleNamespace(
        is_available=lambda:available, current_device=lambda:index,
        is_bf16_supported=lambda **kw:bf16,
        get_device_properties=lambda i:SimpleNamespace(total_memory=6*2**30)))


class HardwareTests(unittest.TestCase):
    def test_selected_gpu_not_zero(self):
        result=detect(fake_torch())
        self.assertEqual(result.device,'cuda:1')
        self.assertEqual(result.dtype,'bf16')
        self.assertFalse(result.portable)

    def test_hip_uses_portable_path(self):
        result=detect(fake_torch(hip='test'))
        self.assertEqual(result.backend,'rocm')
        self.assertTrue(result.portable)

    def test_older_device_uses_fp32_not_fp16(self):
        result=detect(fake_torch(bf16=False))
        self.assertEqual(result.dtype,'fp32')
        self.assertTrue(result.portable)

    def test_no_usable_gpu_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError,'DirectML'):
            detect(fake_torch(available=False))

    def test_portable_quantization_guard_uses_header(self):
        hardware=detect(fake_torch(hip='test'))
        with patch('pi_boogu_image.lib.assets.header',return_value={'layer.weight':{'dtype':'F8_E4M3','shape':[4,4]}}):
            with self.assertRaisesRegex(RuntimeError,'unquantized'):
                validate_weights(Path('misleading_bf16.safetensors'),hardware)
        with patch('pi_boogu_image.lib.assets.header',return_value={'layer.weight':{'dtype':'BF16','shape':[4,4]}}):
            validate_weights(Path('valid.safetensors'),hardware)

    def test_small_memory_resolution_limits(self):
        for gb, expected in [(2,512),(3,512),(4,768),(6,768),(8,1024)]:
            self.assertEqual(policy.profile(gb,'turbo')['default'],expected)
            resolved=policy.settings('turbo',1024,1024,vram_gb=gb)
            self.assertEqual(resolved['width'],expected)
            self.assertEqual(resolved['num_inference_steps'],4)


if __name__ == '__main__':
    unittest.main()
