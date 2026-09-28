import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from pi_boogu_image.lib.runtime import choose_offload, progress_hooks


class RuntimeTests(unittest.TestCase):
    def test_preview_starts_blank_and_clears_previous_latent(self):
        from pi_boogu_image.lib.runtime import begin_preview
        images = []
        state = SimpleNamespace(current_latent=object(), assign_current_image=images.append)
        begin_preview(state, 1024, 1536)
        self.assertIsNone(state.current_latent)
        self.assertEqual(images[0].size, (341, 512))
        self.assertEqual(images[0].getextrema(), ((128, 128),) * 3)

    def test_preview_uses_seconds_not_step_setting(self):
        pipe = SimpleNamespace(mllm=torch.nn.Identity(), transformer=torch.nn.Identity(),
                               vae=torch.nn.Identity(), _predict_dmd_student_step=lambda: "latents")
        images = []
        state = SimpleNamespace(interrupted=False, skipped=False, sampling_step=1,
                                assign_current_image=images.append)
        opts = SimpleNamespace(live_previews_enable=True, show_progress_every_n_steps=99)
        with patch('pi_boogu_image.lib.runtime.monotonic', side_effect=[0, .5, 1, 1.5, 2]), \
             patch('pi_boogu_image.lib.runtime.latent_preview', return_value='image'), \
             progress_hooks(pipe, state, opts):
            for _ in range(5):
                pipe._predict_dmd_student_step()
        self.assertEqual(images, ['image'] * 3)

    def test_memory_plan_preserves_headroom(self):
        gb = 2**30
        self.assertEqual(choose_offload(29*gb, [19*gb, 9*gb]), "model")
        self.assertEqual(choose_offload(12*gb, [9*gb]), "sequential")
        self.assertEqual(choose_offload(29*gb, [19*gb], 2048, 2048), "sequential")
        self.assertEqual(choose_offload(29*gb, [9*gb], requested="sequential"), "sequential")
        self.assertEqual(choose_offload(12*gb, [9*gb], requested="model"), "sequential")

    def test_observer_preserves_result_and_restores_on_cancel(self):
        class Pipe:
            def _predict_dmd_student_step(self):
                return value
        value = torch.ones(1, 16, 8, 8)
        pipe = Pipe()
        pipe.mllm = torch.nn.Identity()
        pipe.transformer = torch.nn.Identity()
        pipe.vae = torch.nn.Identity()
        state = SimpleNamespace(interrupted=False, skipped=False, sampling_step=0)
        with progress_hooks(pipe, state, SimpleNamespace(live_previews_enable=False)):
            self.assertIs(pipe._predict_dmd_student_step(), value)
            state.interrupted = True
            with self.assertRaises(InterruptedError):
                pipe._predict_dmd_student_step()
        self.assertNotIn("_predict_dmd_student_step", pipe.__dict__)
        self.assertEqual(len(pipe.transformer._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
