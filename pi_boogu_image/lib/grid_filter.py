"""CPU implementation of the user-linked 2px Grid Nyquist Notch shader.

Reference: https://pastebin.com/v7y1z0SH
B = [-1, 6, -15, 20, -15, 6, -1]/64; result = I-Bx-By+Bxy.
Two separable (identity-B) passes give the same filter. Borders clamp to edge.
Only the final result is clipped. This removes fine detail too; opt-in only.
"""
import numpy as np
from PIL import Image


def reduce_grid(image):
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    kernel = (1, -6, 15, 44, 15, -6, 1)
    for axis in (1, 0):
        pads = [(0, 0)] * 3
        pads[axis] = (3, 3)
        padded = np.pad(pixels, pads, mode='edge')
        result = np.zeros_like(pixels)
        for offset, coefficient in enumerate(kernel):
            section = [slice(None)] * 3
            section[axis] = slice(offset, offset + pixels.shape[axis])
            result += padded[tuple(section)] * (coefficient / 64.0)
        pixels = result
    return Image.fromarray(np.rint(np.clip(pixels, 0, 255)).astype(np.uint8))
