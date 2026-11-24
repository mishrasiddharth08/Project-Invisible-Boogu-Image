import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import nullcontext

from pi_boogu_image.lib import pipeline


class CacheReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.te, self.vae = self.root / "te.safetensors", self.root / "vae.safetensors"
        self.processor = self.root / "processor"
        self.processor.mkdir()
        for path in (self.te, self.vae, self.processor / "tokenizer.json"):
            path.write_text("test")
        self.addCleanup(pipeline.invalidate_components)
        pipeline.invalidate_components()

    def test_component_discovery_is_cached_and_refreshable(self):
        def component(kind, **kwargs):
            return [self.te if kind == "te" else self.vae]
        with patch.object(pipeline.assets, "component", side_effect=component) as scan, patch.object(pipeline, "processor_path", return_value=self.processor):
            first = pipeline._components(False)
            self.assertEqual(first, pipeline._components(False))
            self.assertEqual(scan.call_count, 2)
            pipeline.invalidate_components()
            pipeline._components(False)
            self.assertEqual(scan.call_count, 4)
            pipeline._components(True)
            self.assertEqual(scan.call_count, 6)
            scan.assert_any_call("te", unquantized_only=True)

    def test_missing_component_rescans(self):
        with patch.object(pipeline.assets, "component", side_effect=[[self.te], [self.vae], [], []]) as scan, patch.object(pipeline, "processor_path", return_value=self.processor):
            pipeline._components(False)
            self.te.unlink()
            with self.assertRaises(RuntimeError):
                pipeline._components(False)
            self.assertEqual(scan.call_count, 4)

    def test_sequential_reuses_other_resolutions_and_oom_fallback(self):
        pipe = SimpleNamespace(_boogu_offload="sequential")
        with patch("torch.cuda.mem_get_info") as memory:
            self.assertTrue(pipeline._can_reuse(pipe, 2048, 2048, "auto", "cuda:0"))
            self.assertTrue(pipeline._can_reuse(pipe, 512, 512, "sequential", "cuda:0"))
            memory.assert_not_called()

    def test_larger_model_resolution_rechecks_memory(self):
        pipe = SimpleNamespace(_boogu_offload="model", _boogu_max_pixels=1024**2, _boogu_component_bytes=[16*2**30])
        with patch("torch.cuda.mem_get_info", return_value=(25*2**30, 32*2**30)) as memory:
            self.assertTrue(pipeline._can_reuse(pipe, 768, 1024, "auto", "cuda:0"))
            memory.assert_not_called()
            self.assertFalse(pipeline._can_reuse(pipe, 2048, 2048, "auto", "cuda:0"))
            memory.assert_called_once()

    def test_skip_frees_offload_hooks_before_returning(self):
        from pi_boogu_image.lib import runtime
        pipe = SimpleNamespace(_boogu_offload="model", maybe_free_model_hooks=Mock())
        with patch.object(pipeline, "get", return_value=pipe), patch.object(pipeline, "invoke", side_effect=runtime.SkipImage()), patch.object(runtime, "progress_hooks", return_value=nullcontext()):
            with self.assertRaises(runtime.SkipImage):
                pipeline.invoke_with_recovery(None, [], 512, 512, 1, {}, None, None)
        pipe.maybe_free_model_hooks.assert_called_once_with()

    def test_full_pipeline_warm_hit_does_not_scan_components(self):
        from pi_boogu_image.lib import hardware
        asset = SimpleNamespace(path=self.root / "model_index.json", trunk="base")
        asset.path.write_text("{}")
        device = SimpleNamespace(portable=False, dtype="dtype", device="cuda:0")
        pipe = SimpleNamespace(_boogu_offload="sequential")
        identity = (pipeline.assets.fingerprint(asset.path), "base", (), None, None, None, "cuda:0", "dtype")
        with patch.object(pipeline, "_pipe", pipe), patch.object(pipeline, "_identity", identity), patch.object(hardware, "detect", return_value=device), patch.object(hardware, "validate_weights") as validate, patch.object(pipeline.assets, "component") as scan, patch.object(pipeline.assets, "config", return_value={"offload": "auto"}):
            self.assertIs(pipeline.get(asset, [], 1536, 1024), pipe)
            scan.assert_not_called()
            validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
