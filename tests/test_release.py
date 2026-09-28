import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pi_boogu_image.lib import pipeline


class ReleaseTests(unittest.TestCase):
    def test_release_drops_model_cache_hooks_and_allocator(self):
        class Pipe:
            pass
        cache = SimpleNamespace(clear=Mock())
        hooks = Mock()
        pipe = Pipe()
        pipe._boogu_prompt_cache = cache
        pipe.remove_all_hooks = hooks
        reference = weakref.ref(pipe)
        cuda = SimpleNamespace(is_available=lambda: True, empty_cache=Mock())
        old = pipeline._pipe
        self.addCleanup(setattr, pipeline, '_pipe', old)
        pipeline._pipe = pipe
        with patch.object(pipeline, '_identity', ('test',)), patch('torch.cuda', cuda):
            del pipe
            pipeline.release()
            self.assertIsNone(reference())
            self.assertIsNone(pipeline._identity)
            cache.clear.assert_called_once()
            hooks.assert_called_once()
            cuda.empty_cache.assert_called_once()
            pipeline.release()
            cuda.empty_cache.assert_called_once()

    def test_selection_events_keep_same_model_and_release_other_models(self):
        asset = SimpleNamespace(path='checkpoint')
        with patch.object(pipeline, '_pipe', object()), patch.object(pipeline, '_identity', ('same',)), patch.object(pipeline.assets, 'fingerprint', return_value='same'), patch.object(pipeline, 'release') as release:
            pipeline.release_if_changed(asset)
            release.assert_not_called()
            pipeline.release_if_changed(None)
            release.assert_called_once()
