import unittest
import weakref
import torch
from PIL import Image
from pi_boogu_image.lib.prompt_cache import PromptCache


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.calls = 0
        def encode(instruction, input_pil_images=None, device='cpu', truncate=False):
            self.calls += 1
            return [torch.arange(12, dtype=torch.float32).reshape(1,3,4)], torch.ones(1,3,dtype=torch.int64)
        self.cache = PromptCache(encode)

    def test_exact_hit_and_no_mutable_alias(self):
        original = self.cache(['hello'])
        original[0][0].zero_()
        hit = self.cache(['hello'])
        self.assertEqual(hit[0][0].sum().item(),66)
        hit[0][0].zero_()
        self.assertEqual(self.cache(['hello'])[0][0].sum().item(),66)
        self.assertEqual(self.calls,1)

    def test_changed_prompt_reference_and_settings_miss(self):
        image = Image.new('RGB',(4,4),'red')
        self.cache(['edit'], [[image]])
        self.cache(['edit'], [[image.copy()]])
        self.assertEqual(self.calls,1)
        image.putpixel((0,0),(0,0,255))
        self.cache(['edit'], [[image]])
        self.cache(['edit'], [[image]], truncate=True)
        self.cache(['different'], [[image]])
        self.assertEqual(self.calls,4)

    def test_strict_memory_bound_and_eviction(self):
        self.cache.max_bytes = 72
        self.cache(['one'])
        self.cache(['two'])
        self.assertEqual(self.cache.bytes,72)
        self.assertEqual(len(self.cache.entries),1)
        self.cache(['one'])
        self.assertEqual(self.calls,3)
        self.cache.clear()
        self.assertEqual(self.cache.bytes,0)
        self.assertFalse(self.cache.entries)

    def test_disabled_and_oversize_bypass(self):
        self.cache.max_bytes = 1
        self.cache(['one'])
        self.cache(['one'])
        self.assertFalse(self.cache.entries)
        self.cache.max_bytes = 0
        self.cache(['one'])
        self.cache(['one'])
        self.assertEqual(self.calls,4)

    def test_cache_does_not_keep_pipeline_alive(self):
        class Encoder:
            def encode(self, instruction, device='cpu'):
                return torch.ones(1)
        encoder = Encoder()
        ref = weakref.ref(encoder)
        cache = PromptCache(encoder.encode)
        encoder.encode = cache
        cache('hello')
        del encoder
        self.assertIsNone(ref())


if __name__ == '__main__':
    unittest.main()
