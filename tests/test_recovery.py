from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import torch
from pi_boogu_image.lib import pipeline, runtime


class RecoveryTests(unittest.TestCase):
    def test_reference_randomness_is_seeded_and_restored(self):
        before = torch.get_rng_state().clone()
        def official(**kwargs):
            return torch.rand(4)
        first = pipeline.invoke(official, input_images=[['reference']], generator=torch.Generator('cpu').manual_seed(1234))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        second = pipeline.invoke(official, input_images=[['reference']], generator=torch.Generator('cpu').manual_seed(1234))
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def run_generation(self):
        return pipeline.invoke_with_recovery('asset', [], 1024, 1024, 1234,
            {'generator':torch.Generator('cpu').manual_seed(1234)},
            SimpleNamespace(sampling_step=3), SimpleNamespace())

    def test_retry_frees_old_model_and_restarts_same_seed(self):
        refs, modes = [], []
        class Model:
            pass
        def load(*args, **kwargs):
            if refs:
                self.assertIsNone(refs[0](), 'old model still retained during retry')
            model = Model()
            model._boogu_offload = kwargs.get('offload','model')
            pipeline._pipe = model
            refs.append(weakref.ref(model))
            modes.append(model._boogu_offload)
            return model
        def invoke(model, generator):
            value = torch.rand(1,generator=generator)
            if model._boogu_offload == 'model':
                raise torch.cuda.OutOfMemoryError('simulated allocation failure')
            return value
        def release():
            pipeline._pipe = None
        with patch.object(pipeline,'_pipe',None), patch.object(pipeline,'get',new=load), \
             patch.object(pipeline,'invoke',new=invoke), patch.object(pipeline,'release',new=release), \
             patch.object(runtime,'progress_hooks',new=lambda *args:nullcontext()):
            result, mode = self.run_generation()
            expected = torch.rand(1,generator=torch.Generator('cpu').manual_seed(1234))
            self.assertTrue(torch.equal(result,expected))
            self.assertEqual(mode,'sequential')
            self.assertEqual(modes,['model','sequential'])

    def test_sequential_oom_does_not_loop(self):
        with patch.object(pipeline,'get',return_value=SimpleNamespace(_boogu_offload='sequential')) as load, \
             patch.object(pipeline,'invoke',side_effect=torch.cuda.OutOfMemoryError()), \
             patch.object(runtime,'progress_hooks',new=lambda *args:nullcontext()):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                self.run_generation()
            self.assertEqual(load.call_count,1)

    def test_second_oom_is_propagated(self):
        with patch.object(pipeline,'get',side_effect=[SimpleNamespace(_boogu_offload='model'),SimpleNamespace(_boogu_offload='sequential')]) as load, \
             patch.object(pipeline,'invoke',side_effect=torch.cuda.OutOfMemoryError()), \
             patch.object(pipeline,'release') as release, \
             patch.object(runtime,'progress_hooks',new=lambda *args:nullcontext()):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                self.run_generation()
            self.assertEqual(load.call_count,2)
            self.assertEqual(release.call_count,1)


if __name__ == '__main__':
    unittest.main()
