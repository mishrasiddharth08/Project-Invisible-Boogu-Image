import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from accelerate import init_empty_weights
from safetensors.torch import save_file
from pi_boogu_image.lib.weights import load_single, PackedLinear, materialize_lora_targets

class WeightTests(unittest.TestCase):
    def test_lora_materializes_only_targeted_layers_with_compute_dtype(self):
        def packed():
            layer = PackedLinear(nn.Linear(4, 3, bias=False), torch.ones(3, 4).to(torch.float8_e4m3fn),
                                 {'format': 'float8_e4m3fn'}, torch.tensor(1.0))
            layer.compute_dtype = torch.float32
            return layer
        model = nn.ModuleDict({'one': packed(), 'two': packed()})
        untouched = model['two']
        materialize_lora_targets(model, {'one'})
        self.assertIsInstance(model['one'], nn.Linear)
        self.assertEqual(model['one'].weight.dtype, torch.float32)
        self.assertIs(model['two'], untouched)
        torch.testing.assert_close(model['one'](torch.ones(1, 4)), torch.full((1, 3), 4.0))

    def test_unquantized_fp32_compatibility_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tiny.safetensors'
            weight=torch.randn(3,4,dtype=torch.bfloat16)
            save_file({'linear.weight':weight},str(path))
            with init_empty_weights():
                model=nn.ModuleDict({'linear':nn.Linear(4,3,bias=False)})
            load_single(model,path,dtype=torch.float32)
            self.assertEqual(model['linear'].weight.dtype,torch.float32)
            self.assertTrue(torch.equal(model['linear'].weight,weight.float()))

    def test_bf16_strict_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tiny.safetensors'
            weight=torch.randn(3,4,dtype=torch.bfloat16)
            save_file({'linear.weight':weight},str(path))
            with init_empty_weights():
                model=nn.ModuleDict({'linear':nn.Linear(4,3,bias=False)})
            load_single(model,path)
            self.assertTrue(torch.equal(model['linear'].weight,weight))

    def test_fp8_scale_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tiny_fp8.safetensors'
            weight=torch.randn(3,4).to(torch.float8_e4m3fn)
            scale=torch.tensor(0.125)
            save_file({'linear.weight':weight,'linear.weight_scale':scale},str(path),
                metadata={'_quantization_metadata':json.dumps({'layers':{'linear':{'format':'float8_e4m3fn'}}})})
            with init_empty_weights():
                model=nn.ModuleDict({'linear':nn.Linear(4,3,bias=False)})
            load_single(model,path)
            self.assertIsInstance(model['linear'],PackedLinear)
            x=torch.randn(2,4,dtype=torch.bfloat16)
            expected=torch.nn.functional.linear(x,(weight.float()*scale).bfloat16())
            torch.testing.assert_close(model['linear'](x),expected)

    def test_unknown_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.safetensors'
            save_file({'wrong.weight':torch.ones(3,4)},str(path))
            with init_empty_weights():
                model=nn.ModuleDict({'linear':nn.Linear(4,3,bias=False)})
            with self.assertRaises(ValueError): load_single(model,path)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the native NVFP4 kernel")
    def test_nvfp4_native_roundtrip(self):
        try:
            import comfy_kitchen as ck
        except ImportError:
            self.skipTest("Host Comfy-Kitchen is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tiny_nvfp4.safetensors'
            torch.manual_seed(17)
            weight=torch.randn(64,64,device='cuda',dtype=torch.bfloat16)
            scale2=weight.float().abs().max()/(448*6)
            packed,scale=ck.quantize_nvfp4(weight,scale2)
            save_file({'linear.weight':packed.cpu(),'linear.weight_scale':scale.cpu(),
                       'linear.weight_scale_2':scale2.cpu()},str(path),
                      metadata={'_quantization_metadata':json.dumps({'layers':{'linear':{'format':'nvfp4'}}})})
            with init_empty_weights():
                model=nn.ModuleDict({'linear':nn.Linear(64,64,bias=False)})
            load_single(model,path)
            recovered=model['linear'].unpack('cuda')
            relative_error=(recovered.float()-weight.float()).square().mean().sqrt()/weight.float().square().mean().sqrt()
            self.assertLess(relative_error.item(),0.25)
            self.assertEqual(model['linear'].packed.dtype,torch.uint8)

    def test_shape_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.safetensors'
            save_file({'linear.weight':torch.ones(4,3)},str(path))
            with init_empty_weights():
                model=nn.ModuleDict({'linear':nn.Linear(4,3,bias=False)})
            with self.assertRaises(ValueError): load_single(model,path)

if __name__=='__main__': unittest.main()
