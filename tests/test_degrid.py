import unittest
import numpy as np
from PIL import Image
from pi_boogu_image.lib.degrid import reduce_grid


class DegridTests(unittest.TestCase):
    def test_clean_and_small_images_pass_through(self):
        for size in ((1, 1), (8, 5), (32, 32)):
            image = Image.new('RGB', size, (31, 128, 199))
            self.assertIs(reduce_grid(image), image)

    def test_mild_grid_reduced_and_alpha_preserved(self):
        y, x = np.indices((40, 40))
        for pattern in (x % 2, y % 2, (x+y) % 2):
            rgb = np.repeat((128 + 2 * (pattern*2-1))[..., None], 3, axis=2).astype(np.uint8)
            image = Image.fromarray(rgb).convert('RGBA')
            image.putalpha(Image.fromarray((x*6).astype(np.uint8)))
            result = reduce_grid(image)
            self.assertTrue(np.all(np.asarray(result)[5:-5, 5:-5, :3] == 128))
            self.assertEqual(result.getchannel('A').tobytes(), image.getchannel('A').tobytes())
