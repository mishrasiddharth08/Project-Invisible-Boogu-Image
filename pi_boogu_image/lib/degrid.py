"""CPU/PIL adapter for pinned, unmodified Apache-2.0 ComfyUI-DeGrid core."""
import numpy as np
import torch
from PIL import Image
from .vendor.degrid.degrid_core import degrid


def reduce_grid(image):
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    with torch.inference_mode():
        cleaned, _, stats = degrid(torch.from_numpy(pixels).unsqueeze(0), mode='auto', skip_when_clean=True)
    if stats[0]['skipped']:
        return image
    result = Image.fromarray(np.rint(cleaned[0].numpy().clip(0, 1) * 255).astype(np.uint8))
    if image.mode == 'RGBA':
        result.putalpha(image.getchannel('A'))
    return result
