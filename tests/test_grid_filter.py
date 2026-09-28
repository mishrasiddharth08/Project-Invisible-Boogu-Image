import unittest
import numpy as np
from PIL import Image
from pi_boogu_image.lib.grid_filter import reduce_grid


class GridTests(unittest.TestCase):
    def test_constant_and_tiny_images_unchanged(self):
        for size in ((1, 1), (2, 3), (23, 19)):
            source = Image.new('RGB', size, (13, 128, 241))
            self.assertEqual(reduce_grid(source).tobytes(), source.tobytes())

    def test_horizontal_vertical_and_checker_patterns_removed(self):
        y, x = np.indices((32, 32))
        for pattern in (x % 2, y % 2, (x+y) % 2):
            source = np.repeat((64 + 128 * pattern)[..., None], 3, axis=2).astype(np.uint8)
            result = np.asarray(reduce_grid(Image.fromarray(source)))
            self.assertTrue(np.all(result[3:-3, 3:-3] == 128))

    def test_matches_original_shader_equation(self):
        source = np.random.default_rng(7).integers(0, 256, (11, 13, 3), dtype=np.uint8)
        center = source.astype(np.float32)
        pad = np.pad(center, ((3, 3), (3, 3), (0, 0)), mode='edge')
        b = np.array([-1, 6, -15, 20, -15, 6, -1], dtype=np.float32) / 64
        bx, by, bxy = [np.zeros_like(center) for _ in range(3)]
        for i, wi in enumerate(b):
            bx += wi * pad[3:14, i:i+13]
            by += wi * pad[i:i+11, 3:16]
            for j, wj in enumerate(b):
                bxy += wi * wj * pad[j:j+11, i:i+13]
        expected = np.rint(np.clip(center-bx-by+bxy, 0, 255)).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(reduce_grid(Image.fromarray(source))), expected)
